import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic import DeepLearningModel
from models.resnet import resnet18, resnet50
from models.utils import get_nonlinear, get_norm


class UNet(DeepLearningModel):
    task = 'segmentation'

    def __init__(self, in_channels, num_classes, bilinear=False, backbone='conv', pretrain=False,
                 norm='bn', nl='leaky_relu'):
        super(UNet, self).__init__()
        assert backbone in ['conv', 'resnet18', 'resnet50']
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        if backbone == 'conv':
            hidden_channels = [64, 128, 256, 512, 1024]
            self.encoder = Encoder([in_channels] + hidden_channels[:-1], hidden_channels, norm=norm, nl=nl)
        else:
            if backbone == 'resnet18':
                self.encoder = resnet18(pretrain, in_channels=in_channels, num_classes=num_classes)
            elif backbone == 'resnet50':
                self.encoder = resnet50(pretrain, in_channels=in_channels, num_classes=num_classes)
            else:
                raise AttributeError
            for param in self.encoder.get_classifier().parameters():
                param.requires_grad = False
            hidden_channels = self.encoder.num_feature_channels
        self.up1 = Up(hidden_channels[-1], hidden_channels[-2], hidden_channels[-2], bilinear, norm=norm, nl=nl)
        self.up2 = Up(hidden_channels[-2], hidden_channels[-3], hidden_channels[-3], bilinear, norm=norm, nl=nl)
        self.up3 = Up(hidden_channels[-3], hidden_channels[-4], hidden_channels[-4], bilinear, norm=norm, nl=nl)
        self.up4 = Up(hidden_channels[-4], hidden_channels[-5], hidden_channels[-5], bilinear, norm=norm, nl=nl)
        self.up5 = Up(hidden_channels[-5], in_channels, hidden_channels[-5], bilinear, norm=norm, nl=nl)

        self.outc = OutConv(hidden_channels[-5], num_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder.forward_features(x)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.up5(out, x)
        logits = self.outc(out)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='bn', nl='leaky_relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            get_norm(norm, mid_channels),
            get_nonlinear(nl),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            get_norm(norm, out_channels),
            get_nonlinear(nl)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, norm='bn', nl='leaky_relu'):
        super().__init__()
        assert len(in_channels) == len(out_channels) == 5
        self.inc = DoubleConv(in_channels[0], in_channels[0], norm=norm, nl=nl)
        self.down1 = Down(in_channels[0], out_channels[0], norm=norm, nl=nl)
        self.down2 = Down(in_channels[1], out_channels[1], norm=norm, nl=nl)
        self.down3 = Down(in_channels[2], out_channels[2], norm=norm, nl=nl)
        self.down4 = Down(in_channels[3], out_channels[3], norm=norm, nl=nl)
        self.down5 = Down(in_channels[4], out_channels[4], norm=norm, nl=nl)

    def forward_features(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x):
        return self.forward_features(x)[-1]


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm='bn', nl='leaky_relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm, nl=nl)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_up, in_channels_skip, out_channels, bilinear=True, norm='bn', nl='leaky_relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_up + in_channels_skip, out_channels, in_channels_up, norm=norm, nl=nl)
        else:
            self.up = nn.ConvTranspose2d(in_channels_up, in_channels_up, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_up + in_channels_skip, out_channels, norm=norm, nl=nl)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
