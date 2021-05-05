import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..utils import compute_iou


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        # We ignore those whose tarhets == -1.0.
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


class BCE_focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = (1.0-inputs)**self.gamma * (targets) * torch.log(inputs + 1e-14) + \
                (inputs)**self.gamma * (1.0 - targets) * torch.log(1.0 - inputs + 1e-14)
        loss = -torch.sum(torch.sum(loss, dim=-1), dim=-1)
        return loss


def loss(pred_conf, pred_cls, pred_txtytwth, label):
    # create loss_f
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = pred_conf[:, :, 0]
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    gt_obj = label[:, :, 0]
    gt_cls = label[:, :, 1].long()
    gt_txty = label[:, :, 2:4]
    gt_twth = label[:, :, 4:6]
    gt_box_scale_weight = label[:, :, 6]

    batch_size = pred_conf.size(0)
    # objectness loss
    conf_loss = conf_loss_function(pred_conf, gt_obj)

    # class loss
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_obj) / batch_size

    # box loss
    txty_loss = torch.sum(
        torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight *
        gt_obj) / batch_size
    twth_loss = torch.sum(
        torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight *
        gt_obj) / batch_size
    bbox_loss = txty_loss + twth_loss

    # total loss
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss


def fcos_loss(pred, label, num_classes):
    # define loss functions
    cls_w = 1.0
    ctn_w = 5.0
    box_w = 1.0
    cls_loss_func = BCE_focal_loss()
    ctn_loss_func = nn.BCELoss(reduction='none')
    box_loss_func = nn.BCELoss(reduction='none')

    pred_cls = torch.sigmoid(pred[:, :, :1 + num_classes])
    pred_ctn = torch.sigmoid(pred[:, :, 1 + num_classes])
    pred_box = torch.exp(pred[:, :, 1 + num_classes + 1:])

    gt_cls = label[:, :, :1 + num_classes].float()
    gt_ctn = label[:, :, 1 + num_classes].float()
    gt_box = label[:, :, 1 + num_classes + 1:-2].float()
    gt_pos = label[:, :, -2]
    gt_iou = label[:, :, -1]
    N_pos = torch.sum(gt_pos, dim=-1)
    N_pos = torch.max(N_pos, torch.ones(N_pos.size(), device=N_pos.device))

    # cls loss
    cls_loss = torch.mean(cls_loss_func(pred_cls, gt_cls) / N_pos)

    # ctn loss
    ctn_loss = torch.mean(torch.sum(ctn_loss_func(pred_ctn, gt_ctn) * gt_pos, dim=-1) / N_pos)

    # box loss
    iou = compute_iou(pred_box, gt_box)
    box_loss = torch.mean(torch.sum(box_loss_func(iou, gt_iou) * gt_pos, dim=-1) / N_pos)

    total_loss = cls_w * cls_loss + ctn_w * ctn_loss + box_w * box_loss
    return cls_loss, ctn_loss, box_loss, total_loss
