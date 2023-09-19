import torch

from models.basic import DeepLearningBackbone
from models.utils import get_nonlinear

__all__ = ['OWMNet']


class OWMNet(DeepLearningBackbone):

    def __init__(self, in_channels: int = 3, num_classes: int = 1000, nl='relu'):
        super().__init__()

        self.nl = get_nonlinear(nl)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.num_feature_layers = 3  # maxpool1, maxpool2, maxpool3
        self.num_feature_channels = [64, 128, 256]

        self.c1 = torch.nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, num_classes, bias=False)

        self.sub_nets['feature_net'] = torch.nn.Sequential(
            self.padding, self.c1, self.nl, self.drop, self.maxpool,
            self.padding, self.c2, self.nl, self.drop, self.maxpool,
            self.padding, self.c3, self.nl, self.drop, self.maxpool)
        self.sub_nets['classifier'] = torch.nn.Sequential(
            self.flatten, self.fc1, self.nl, self.fc2, self.nl, self.fc3)

        torch.nn.init.xavier_normal(self.fc1.weight)
        torch.nn.init.xavier_normal(self.fc2.weight)
        torch.nn.init.xavier_normal(self.fc3.weight)

    def forward_features(self, x: torch.Tensor) -> list:
        x = self.padding(x)
        con1 = self.drop(self.nl(self.c1(x)))
        con1 = self.maxpool(con1)

        con2 = self.padding(con1)
        con2 = self.drop(self.nl(self.c2(con2)))
        con2 = self.maxpool(con2)

        con3 = self.padding(con2)
        con3 = self.drop(self.nl(self.c3(con3)))
        con3 = self.maxpool(con3)
        features = [con1, con2, con3]

        return features

    def forward_classifier(self, x: torch.Tensor):
        x = self.sub_nets['classifier'](x)
        return x

    def forward(self, x):
        # assert x.ndim == 4 and x.shape[2] == 32 and x.shape[3] == 32, 'Input to OWMNet should be [B, C, 32, 32]'
        # Gated
        x = self.forward_features(x)[-1]
        x = self.forward_classifier(x)
        return x
