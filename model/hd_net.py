# encoding:utf-8

from .modules import *


class HDNet(nn.Module):
    def __init__(self, n_channels, dilations, n_classes):
        """

        :type n_channels:
        """
        super(HDNet, self).__init__()
        self.layer1_head = head(n_channels[0], n_channels[1])
        self.layer1_res = nn.Sequential(
            hierarchical_dilated_module(n_channels[1], dilations[0]),
            tail(n_channels[1], 64, n_classes)
        )

        self.layer2_head = down(n_channels[1], n_channels[2])
        self.layer2_res = nn.Sequential(
            hierarchical_dilated_module(n_channels[2], dilations[1]),
            tail(n_channels[2], 64, n_classes, 2)
        )

        self.layer3 = nn.Sequential(
            down(n_channels[2], n_channels[3]),
            hierarchical_dilated_module(n_channels[3], dilations[2]),
            tail(n_channels[3], 64, n_classes, 4)
        )

    def forward(self, input):
        x1 = self.layer1_head(input)
        res1 = self.layer1_res(x1)
        x2 = self.layer2_head(x1)
        res2 = self.layer2_res(x2)
        res3 = self.layer3(x2)