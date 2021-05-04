import numpy as np
import torch
import torch.nn as nn

from .backbones import resnet
from nn import SpatialPyramidPool2d, Conv2d, loss


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
        self.stride = 32
        self.grid_cell = self._create_grid(input_size)
        self.input_size = input_size

        self.backbone = resnet('resnet18', pretrained=True)
        backbones_out_channels = 512

        self.neck = SpatialPyramidPool2d(in_channels=backbones_out_channels,
                                         out_channels=backbones_out_channels,
                                         k=(5, 9, 13))
        self.head = nn.Sequential(Conv2d(backbones_out_channels, 256, k_size=1),
                                  Conv2d(256, 512, k_size=3, padding=1), Conv2d(512, 256, k_size=1),
                                  Conv2d(256, 512, k_size=3, padding=1))

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def _create_grid(self, input_size):
        """Prepare matrix with shape [1, W*H, 2], each element corresponds to [grid_x, grid_y]

        Implemention:
            center_x of box on pic = (grid_x + t_x) * stride
        """
        w, h = (input_size, input_size)
        w, h = w / self.stride, h / self.stride
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
        pred[:, :, :2] = (self.grid_cell + torch.sigmoid(pred[:, :, 2])) * self.stride
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        out_pred = torch.zeros_like(pred)
        # x_min/max = center_x -/+ width/2
        out_pred[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        out_pred[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        out_pred[:, :, 2] = pred[:, :, 2] + pred[:, :, 2] / 2
        out_pred[:, :, 3] = pred[:, :, 3] + pred[:, :, 3] / 2

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
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), class_idx)].copy()

        # filter out preds with low score
        pred_keep = np.where(prob_pred >= self.conf_thresh)
        bbox_pred = bbox_pred[pred_keep]
        prob_pred = prob_pred[pred_keep]
        class_idx = class_idx[pred_keep]

        keep = np.zeros(len(prob_pred), dtype=np.int)
        for i in range(self.num_classes):
            idxs = np.where(class_idx == i)[0]
            if len(idxs) == 0:
                continue
            cls_bboxes = bbox_pred[idxs]
            cls_scores = prob_pred[idxs]
            cls_keep = self.nms(cls_bboxes, cls_scores)
            keep[idxs[cls_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        prob_pred = prob_pred[keep]
        class_idx = class_idx[keep]

        return bbox_pred, prob_pred, class_idx

    def forward(self, x, target=None):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        det_out = self.head(neck_out)
        pred = self.pred(det_out)

        # Reshape pred
        B, C, H, W = pred.size()
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        # objectness confedence: [B, H*W, 1]
        conf_pred = pred[:, :, :1]
        # class prediction: [B, H*W, num_classes]
        class_pred = pred[:, :, 1:1 + self.num_classes]
        # boundingbox pred: [B, H*W, 4]; tx ty tw th
        bbox_pred = pred[:, :, 1 + self.num_classes:]

        if self.trainable:
            conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                              pred_cls=class_pred,
                                                              pred_txtytwth=bbox_pred,
                                                              label=target)

            return conf_loss, cls_loss, bbox_loss, total_loss
        else:
            with torch.no_grad():
                conf_pred = torch.sigmoid(conf_pred)[0] # 0 is because that these is only 1 batch.
                class_scores = (torch.softmax(class_pred[0, :, :], dim=1) * conf_pred)
                bbox_pred = torch.clamp((self._decode_boxes(bbox_pred) / self.input_size)[0], 0., 1)

                class_scores = class_scores.to('cpu').numpy()
                bbox_pred = bbox_pred.to('cpu').numpy()

                bboxes, class_scores, class_idx = self.postprocess(bbox_pred, class_scores)

                return bboxes, class_scores, class_idx
