import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from models.NewBayesianLayers.BBBlayers import BBBConv3d, BBBLinearFactorial
from models.NewBayesianLayers.BBBmodules import BBBSequential

__all__ = [
    'BBBResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def BBBconv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return BBBConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BBBBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BBBBasicBlock, self).__init__()
        self.conv1 = BBBconv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.soft = nn.ReLU()
        self.conv2 = BBBconv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        kl = 0
        residual = x

        out, _kl = self.conv1(x)
        kl += _kl
        out = self.bn1(out)
        out = self.soft(out)

        out, _kl = self.conv2(out)
        kl += _kl
        out = self.bn2(out)

        if self.downsample is not None:
            try:
                residual, _kl = self.downsample(x)
                kl += _kl
            except:
                residual = self.downsample(x)

        out += residual
        out = self.soft(out)

        return out, kl


class BBBBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BBBBottleneck, self).__init__()
        self.conv1 = BBBConv3d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = BBBConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = BBBConv3d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.soft = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        kl = 0
        residual = x

        out, _kl = self.conv1(x)
        kl += _kl
        out = self.bn1(out)
        out = self.soft(out)

        out, _kl = self.conv2(out)
        kl += _kl
        out = self.bn2(out)
        out = self.soft(out)

        out, _kl = self.conv3(out)
        kl += _kl
        out = self.bn3(out)

        if self.downsample is not None:
            try:
                residual, _kl = self.downsample(x)
                kl += _kl
            except:
                residual = self.downsample(x)

        out += residual
        out = self.soft(out)

        return out, kl


class BBBResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(BBBResNet, self).__init__()
        self.conv1 = BBBConv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.soft = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = BBBLinearFactorial(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = BBBSequential(
                    BBBConv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return BBBSequential(*layers)

    def forward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0

        x, _kl = self.conv1(x)
        kl += _kl
        x = self.bn1(x)
        x = self.soft(x)
        x = self.maxpool(x)

        x, _kl = self.layer1(x)
        kl += _kl
        x, _kl = self.layer2(x)
        kl += _kl
        x, _kl = self.layer3(x)
        kl += _kl
        x, _kl = self.layer4(x)
        kl += _kl

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x, _kl = self.fc(x)
        kl += _kl

        return x, kl


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = BBBResNet(BBBBasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = BBBResNet(BBBBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = BBBResNet(BBBBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = BBBResNet(BBBBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = BBBResNet(BBBBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = BBBResNet(BBBBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = BBBResNet(BBBBottleneck, [3, 24, 36, 3], **kwargs)
    return model
