"""Customized nn Modules"""
import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(self, in_c, out_c, k_size, padding=0, stride=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SpatialPyramidPool2d(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super(SpatialPyramidPool2d, self).__init__()
        planes = in_channels // 2 # hidden channels
        self.cv1 = Conv2d(in_channels, planes, 1, 1)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv2 = Conv2d(planes * (len(k) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.pools], dim=1)
        x = self.cv2(x)
        return x
