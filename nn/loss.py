import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from data import *


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


if __name__ == "__main__":
    pass
