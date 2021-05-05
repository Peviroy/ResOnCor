import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
""" Useful api dedicated to pytorch
"""


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def accuracy_calc(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, original_lr, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every step
    """
    lr = original_lr * (0.1 ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
