from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.basic import DeepLearningBackbone

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNetS', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetS(DeepLearningBackbone):
    def __init__(
            self,
            block,
            layers: List[int],
            in_channels: int = 3,
            num_classes: int = 1000,
            zero_init_residual: bool = False
    ) -> None:
        super().__init__()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.num_feature_layers = 3  # conv1, layer2, layer3
        self.num_feature_channels = [self.in_planes]

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.num_feature_channels.append(32 * block.expansion)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.num_feature_channels.append(64 * block.expansion)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.sub_nets['feature_net'] = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.layer1, self.layer2, self.layer3)
        self.sub_nets['classifier'] = nn.Sequential(self.avgpool, self.flatten, self.fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> list:
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)

        return features

    def forward_classifier(self, x: Tensor):
        x = self.sub_nets['classifier'](x)
        return x

    def forward(self, x: Tensor):
        x = self.forward_features(x)[-1]
        x = self.forward_classifier(x)
        return x


def resnet20(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)


def resnet32(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [5, 5, 5], num_classes=num_classes, **kwargs)


def resnet44(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [7, 7, 7], num_classes=num_classes, **kwargs)


def resnet56(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [9, 9, 9], num_classes=num_classes, **kwargs)


def resnet110(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [18, 18, 18], num_classes=num_classes, **kwargs)


def resnet1202(num_classes=10, **kwargs):
    return ResNetS(BasicBlock, [200, 200, 200], num_classes=num_classes, **kwargs)
