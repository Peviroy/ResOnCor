"""Define the ResNet architecture"""
import torch.nn as nn
import math


class BasicBlock(nn.Module):
    """
    Description:
    ------------
    resnet can be divided into basic blocks;
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # replace original value,which can save memory
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residule = x # keep original input as residule part
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residule = self.downsample(x)
        out += residule
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers: list, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False) # size / 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # size / 4

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) # size / 4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # size / 8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # size / 16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # size / 32

        # output size of conv layer is (7, 7) using a avgpool with kernel size of 7 can get an output of size 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # concluded: dimension of final conv layer equal (expansion*512), where expansion means last block channel / first block channel (each layer is same)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1]
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, block_num, stride):
        """
        Description:
        ------------
            Key method
            ResNet can be diveded into different layers consisting of 'Basic Blocks'

        Parameter:
        ------------
            block: {Class} In resnet18 or resnet34, this blck type is called 'Basic Block'; while in resnet101, called 'Bottleneck Block'
            in_channels: {int} Channels of previous layer's output
            planes: {int} planes of blocks
            block_num: {int} Number of block
            stride: {int} stride function is executed by the first block of each layer

        """
        # dimension dismatch(will be happened in the first block of each layer, so we add downsample layer) used to serve residual part

        # *FIRST BLOCK -------------------
        downsample = None
        expanded_output = planes * block.expansion
        if stride != 1 or self.in_channels != expanded_output:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          expanded_output,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(expanded_output))

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        # *OTHER BLOCKS -------------------
        self.in_channels = expanded_output
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNet18(classes_num):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=classes_num)


def ResNet34(classes_num):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=classes_num)


def ResNet50(classes_num):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=classes_num)


def resnet(model: str, classes_num):
    switch = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50}
    if model.lower() in switch:
        return switch[model](classes_num)
    else:
        print('No model selected!')
