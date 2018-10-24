# encoding:utf-8

from .modules import *


class HDNet(nn.Module):
    def __init__(self, n_channels, dilations, n_classes):
        """
        :param n_channels: list, (4), [input, L1, L2, L3]
        :param dilations: list, (3, 3), [[1, d1, d2], [1, d3, d4], [1, d5, d6]]
        :param n_classes: int
        """
        super(HDNet, self).__init__()
        self.layer1_head = Head(n_channels[0], n_channels[1])
        self.layer1_rest = nn.Sequential(
            HierarchicalDilatedModule(n_channels[1], dilations[0]),
            Tail(n_channels[1], 64, n_classes)
        )

        self.layer2_head = Down(n_channels[1], n_channels[2])
        self.layer2_rest = nn.Sequential(
            HierarchicalDilatedModule(n_channels[2], dilations[1]),
            Tail(n_channels[2], 64, n_classes, 2)
        )

        self.layer3 = nn.Sequential(
            Down(n_channels[2], n_channels[3]),
            HierarchicalDilatedModule(n_channels[3], dilations[2]),
            Tail(n_channels[3], 64, n_classes, 4)
        )

        self.fusion = Fusion(n_classes+n_classes+n_classes*3, 32, n_classes)    # C+C+C*multi_grains

    def forward(self, input):
        out1 = self.layer1_head(input)
        res1 = self.layer1_rest(out1)
        out2 = self.layer2_head(out1)
        res2 = self.layer2_rest(out2)
        res3 = self.layer3(out2)
        out = torch.stack([res1, res2, res3], dim=2)    #(N, C, multi_grained, H, W)
        return  self.fusion(out)