"""Define the ResNet architecture"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


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
    def __init__(self, block, layers: list):
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

        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet(name: str, pretrained=False):
    models = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101
    }

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    if name.lower() not in models:
        raise ValueError(f'No such model!\n\t{name.lower()}')
    model = models[name]()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[name.lower()]))
    return model
