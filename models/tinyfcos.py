import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from nn import Conv2d, fcos_loss


class myYOLO(nn.Module):
    def __init__(self,
                 device,
                 input_size=None,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.01,
                 nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.stride = [8, 16, 32, 64]
        self.scale_thresholds = [0, 49, 98, 196, 1e10]
        self.pixel_location = self.set_init()
        self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float()

        self.backbone = resnet('resnet18', pretrained=trainable)

        # neck:FPN P6 P5 P4 P3
        self.neck = FPN()

        self.pred_6 = nn.Sequential(Conv2d(512, 1024, 3, padding=1),
                                    nn.Conv2d(1024, 1 + self.num_classes + 1 + 4, 1))
        self.pred_5 = nn.Sequential(Conv2d(256, 512, 3, padding=1),
                                    nn.Conv2d(512, 1 + self.num_classes + 1 + 4, 1))
        self.pred_4 = nn.Sequential(Conv2d(128, 256, 3, padding=1),
                                    nn.Conv2d(256, 1 + self.num_classes + 1 + 4, 1))
        self.pred_3 = nn.Sequential(Conv2d(64, 128, 3, padding=1),
                                    nn.Conv2d(128, 1 + self.num_classes + 1 + 4, 1))

    def _create_grid(self, input_size):
        """Prepare matrix with shape [1, W*H, 2], each element corresponds to [grid_x, grid_y]

        Implemention:
            center_x of box on pic = (grid_x + t_x) * stride
        """
        w, h = (input_size, input_size)
        w, h = w // self.stride, h // self.stride
        grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h))
        # stack in the final dim
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # reshape from [W, H, 2] to [1, W*H, 2]
        grid_xy = grid_xy.view(1, -1, 2)

        return grid_xy.to(self.device)

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self._create_grid(input_size)

    def _decode_boxes(self, pred):
        """Convert pred{tx ty tw th} to bbox on input image{xmin, ymin, xmax, ymax}
        """
        # center_x = (grid_x + t_x) * stride; width = exp(t_w)
        pred[:, :, :2] = (self.grid_cell + torch.sigmoid(pred[:, :, :2])) * self.stride
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        out_pred = torch.zeros_like(pred)
        # x_min/max = center_x -/+ width/2
        out_pred[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        out_pred[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        out_pred[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        out_pred[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2

        return out_pred

    def _nms(self, dets, scores):
        # From faster rcnn nms
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0] #xmin
        y1 = dets[:, 1] #ymin
        x2 = dets[:, 2] #xmax
        y2 = dets[:, 3] #ymax

        areas = (x2 - x1) * (y2 - y1) # bbox的宽w和高h
        order = scores.argsort()[::-1] # 按照降序对bbox的得分进行排序

        keep = [] # 用于保存经过筛的最终bbox结果
        while order.size > 0:
            i = order[0] # 得到最高的那个bbox
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def _postprocess(self, bbox_pred, prob_pred):
        """
        Arguments:
            bbox_pred: (W*H, 4), bsize = 1
            prob_pred: (W*H, num_classes), bsize = 1
        """
        # Get the most possilble class pred in each grid
        class_idx = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), class_idx)]

        # filter out preds with low score
        pred_keep = np.where(prob_pred >= self.conf_thresh)
        bbox_pred = bbox_pred[pred_keep]
        prob_pred = prob_pred[pred_keep]
        class_idx = class_idx[pred_keep]

        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            idxs = np.where(class_idx == i)[0]
            if len(idxs) == 0:
                continue
            cls_bboxes = bbox_pred[idxs]
            cls_scores = prob_pred[idxs]
            cls_keep = self._nms(cls_bboxes, cls_scores)
            keep[idxs[cls_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        prob_pred = prob_pred[keep]
        class_idx = class_idx[keep]

        return bbox_pred, prob_pred, class_idx

    def forward(self, x, targets=None):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        P3, P4, P5, P6 = neck_out
        batch_size = P3.shape[0]

        pred_3 = self.pred_3(P3).view(batch_size, 1 + self.num_classes + 1 + 4, -1)
        pred_4 = self.pred_4(P4).view(batch_size, 1 + self.num_classes + 1 + 4, -1)
        pred_5 = self.pred_5(P5).view(batch_size, 1 + self.num_classes + 1 + 4, -1)
        pred_6 = self.pred_6(P6).view(batch_size, 1 + self.num_classes + 1 + 4, -1)

        total_pred = torch.cat([pred_3, pred_4, pred_5, pred_6], dim=-1).permute(0, 2, 1)

        if self.trainable:
            cls_loss, ctn_loss, box_loss, total_loss = fcos_loss(total_pred,
                                                                 targets,
                                                                 num_classes=self.num_classes)
            return cls_loss, ctn_loss, box_loss, total_loss
        else:
            with torch.no_grad():
                print('No done yet')
                pass


class FPN(nn.Module):
    """yolov3like
    """
    def __init__(self):
        super(FPN, self).__init__()
        # process c5 to c6
        self.conv_3x3_6 = Conv2d(512, 1024, 3, padding=1, stride=2)

        # c projects to p
        self.conv_set_6 = nn.Sequential(
            Conv2d(1024, 512, 1),
            Conv2d(512, 1024, 3, padding=1),
            Conv2d(1024, 512, 1),
        )
        self.conv_set_5 = nn.Sequential(Conv2d(512, 256, 1), Conv2d(256, 512, 3, padding=1),
                                        Conv2d(512, 256, 1))
        self.conv_set_4 = nn.Sequential(Conv2d(384, 128, 1), Conv2d(128, 256, 3, padding=1),
                                        Conv2d(256, 128, 1))
        self.conv_set_3 = nn.Sequential(Conv2d(192, 64, 1), Conv2d(64, 128, 3, padding=1),
                                        Conv2d(128, 64, 1))
        self.conv_1x1_5 = Conv2d(256, 128, 1)
        self.conv_1x1_4 = Conv2d(128, 64, 1)

    def upsamplelike(self, inputs, do_conv_fn=None):
        src, target = inputs
        if do_conv_fn is not None:
            src = do_conv_fn(src)
        return F.interpolate(src,
                             size=(target.shape[2], target.shape[3]),
                             mode='bilinear',
                             align_corners=True)

    def forward(self, x):
        C3, C4, C5 = x
        C6 = self.conv_3x3_6(C5)

        P6 = self.conv_set_6(C6)

        P5 = self.conv_set_5(C5)
        P5_up = self.upsamplelike([P5, C4], self.conv_1x1_5)

        P4 = torch.cat([C4, P5_up], dim=1)
        P4 = self.conv_set_4(C4)
        P4_up = self.upsamplelike([P4, C3], self.conv_1x1_5)

        P3 = torch.cat([C3, P4_up], dim=1)
        P3 = self.conv_set_3(C3)

        return [P3, P4, P5, P6]
