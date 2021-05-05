import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from nn import Conv2d, fcos_loss


class FCOS(nn.Module):
    def __init__(self,
                 device,
                 input_size=None,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.01,
                 nms_thresh=0.5):
        super(FCOS, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.stride = [8, 16, 32, 64]
        self.scale_thresholds = [0, 49, 98, 196, 1e10]
        self.pixel_location = self.set_grid(input_size)
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
        grid_length = sum([(input_size[0] // s) * (input_size[1] // s) for s in self.stride])
        grid_xy = torch.zeros(grid_length, 4)
        start_idx = 0

        for idx in range(len(self.stride)):
            s = self.stride[idx]
            w, h = input_size[0] // s, input_size[1] // s
            for y in range(h):
                for x in range(w):
                    x_y = y * w + x
                    index = x_y + start_idx
                    xx = x * s + s // 2
                    yy = y * s + s // 2
                    grid_xy[index, :] = torch.tensor([xx, yy, xx, yy]).float()
            start_idx += w * h
        return grid_xy.to(self.device)

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self._create_grid(input_size)

    def _clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes
        # assert x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

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
                # batch size = 1
                # Be careful, the index 0 in all_cls is background !!
                all_class_pred = torch.sigmoid(total_pred[0, :, 1:1 + self.num_classes])
                all_centerness = torch.sigmoid(total_pred[0, :, 1 + self.num_classes:1 +
                                                          self.num_classes + 1])
                all_bbox_pred = torch.exp(total_pred[
                    0, :, 1 + self.num_classes + 1:]) * self.location_weight + self.pixel_location
                # separate box pred and class conf
                all_class_pred = all_class_pred.to('cpu').numpy()
                all_centerness = all_centerness.to('cpu').numpy()
                all_bbox_pred = all_bbox_pred.to('cpu').numpy()

                bboxes, scores, cls_inds = self._postprocess(all_bbox_pred, all_class_pred)
                # clip the boxes
                bboxes = self._clip_boxes(bboxes, self.input_size) / self.scale

                return bboxes, scores, cls_inds


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
        P4 = self.conv_set_4(P4)
        P4_up = self.upsamplelike([P4, C3], self.conv_1x1_4)

        P3 = torch.cat([C3, P4_up], dim=1)
        P3 = self.conv_set_3(P3)

        return [P3, P4, P5, P6]
