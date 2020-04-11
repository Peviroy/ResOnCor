import torch
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
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # replace original value,which can save memory
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample


    def forward(self, x):
        residule = x  # keep original input as residule part
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residule = self.downsample(x)
        out += residule
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers: dict, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=2, padding=3, bias=False)  # size / 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)  # size / 4

        self.layer1 = self._make_layer(
            block, layers["dimension"][0], layers["block_num"][0], stride=1)  # size / 4
        self.layer2 = self._make_layer(
            block, layers["dimension"][1], layers["block_num"][1], stride=2)  # size / 8
        self.layer3 = self._make_layer(
            block, layers["dimension"][2], layers["block_num"][2], stride=2)  # size / 16
        self.layer4 = self._make_layer(
            block, layers["dimension"][3], layers["block_num"][3], stride=2)  # size / 32

        # output size of conv layer is (7, 7) using a avgpool with kernel size of 7 can get an output of size 1
#         self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # concluded: dimension of final conv layer equal (expansion*512), where expansion means last block channel / first block channel (each layer is same)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1]
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, channels, block_num, stride):
        """
        Description:
        ------------
            ResNet can be diveded into different layers consisting of 'Basic Blocks'

        Parameter:
        ------------
            block: {Class} In resnet18 or resnet34, this blck type is called 'Basic Block'; while in resnet101, called 'Bottleneck Block'

            in_channels: {int} Channels of previous layer's output.

            block_num: {int} Number of block

            stride: {int} stride function is executed by the first block of each layer

        Note:
        -----
            The names of 'channels' and 'self.in_channels' are somehow confusing; 
            We define channels as the previous layer's output channel, 'in_channels' as the previous block's output channel
        """
        # dimension dismatch(will be happened in the first block of each layer, so we add downsample layer)
        # 简言之:第一个块要同上一层的输出进行衔接，这表现在维度的转换。所以需要降采样进行升维；
        #       而之后的块其输入输出维度是一致的，所以不需要进行改变， 因此使用for循环统一处理
        #
        # *FIRST BLOCK -------------------
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, block.expansion*channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        # *OTHER BLOCKS -------------------
        self.in_channels = channels * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channels, channels))

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


def ResNet18():
    layers = dict(
        {"dimension": [64, 128, 256, 512], "block_num": [2, 2, 2, 2]})
    return ResNet(BasicBlock, layers, num_classes=10)
