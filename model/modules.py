# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingBlock(nn.Module):
    """
    3*3 convolution -> Batch normalization -> Relu
    """
    def __init__(self, in_ch, out_ch, dilation=1):
        super(BuildingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return  self.process(input)


class Shortcut(nn.Module):
    """
    input -> 1*1 conv -> BN -> output
    """
    def __init__(self, in_ch, out_ch):
        super(Shortcut, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        return self.skip(input)


class DilatedResBlock(nn.Module):
    """
    input -> building_block -> building_block -> + -> output
          ↘ -----------(shortcut)------------- ↗
    """
    def __init__(self, in_ch, out_ch, dilation, shortcut=None):
        super(DilatedResBlock, self).__init__()
        self.left = nn.Sequential(
            BuildingBlock(in_ch, out_ch, dilation),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_ch),
        )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out += residual
        return F.relu(out)


class HierarchicalDilatedModule(nn.Module):
    """
    input -> dilated_res_block -> dilated_res_block -> dilated_res_block -> concat -> output
          ↘ -------> ↓ ------------------> ↓ -----------------> ↓ ------- ↗
    """
    def __init__(self, channel, dilation):
        super(HierarchicalDilatedModule, self).__init__()
        self.dilated_block_1 = DilatedResBlock(channel, channel, dilation[0])
        self.dilated_block_2 = DilatedResBlock(channel, channel, dilation[1])
        self.dilated_block_3 = DilatedResBlock(channel, channel, dilation[2])

    def forward(self, input):
        result_0 = self.dilated_block_1(input)
        result_1 = self.dilated_block_2(result_0)
        result_2 = self.dilated_block_3(result_1)
        return torch.cat([input, result_0, result_1, result_2], 1)


class Head(nn.Module):
    """
    input -> building_block -> dilated_res_block -> output
    """
    def __init__(self, in_ch, out_ch, dilation=1):
        super(Head, self).__init__()
        self.process = nn.Sequential(
            BuildingBlock(in_ch, out_ch, dilation),
            DilatedResBlock(out_ch, dilation)
        )

    def forward(self, input):
        return self.process(input)


class Down(nn.Module):
    """
    input -> max_pool -> head -> output
    """
    def __int__(self, in_ch, out_ch, dilation=1):
        super(Down, self).__int__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            Head(in_ch, out_ch, dilation)
        )

    def forward(self, input):
        return self.down_sample(input)


class Tail(nn.Module):
    """
    input -> building_block -> (upsampling ->) building_block -> 1*1 conv -> n_classes output
    """
    def __init__(self, in_ch, out_ch, K, scale=1, dilation=1):
        super(Tail, self).__init__()
        if scale == 1:
            self.process = nn.Sequential(
                BuildingBlock(in_ch, out_ch, dilation),
                BuildingBlock(out_ch, int(out_ch / 2), dilation),
                nn.Conv2d(int(out_ch / 2), K, 1)
            )
        else:
            self.process = nn.Sequential(
                BuildingBlock(in_ch, out_ch, dilation),
                nn.ConvTranspose2d(out_ch, out_ch, scale, stride=scale),
                BuildingBlock(out_ch, int(out_ch/2), dilation),
                nn.Conv2d(int(out_ch/2), K, 1)
            )

    def forward(self, input):
        return self.process(input)


class Fusion(nn.Module):
    """
          ↗  mean ↘
    input -> raw  -> concat -> building_block -> 1*1 conv -> output
          ↘  max  ↗
    """
    def __init__(self, in_ch, out_ch, K):
        super(Fusion, self).__init__()
        self.fuse = nn.Sequential(
            BuildingBlock(in_ch, out_ch),
            nn.Conv2d(out_ch, K, 1)
        )

    def forward(self, input):
        """
        :param input: (N, C, multi_grains, H, W)
        :return: (N, K, H, W)
        """
        input_mean = torch.mean(input, dim=2)   #(N, C, multi_grains, H, W) -> (N, C, H, W)
        input_max = torch.max(input, dim=2)
        input = input.view(input.shape[0], -1, input.shape[3], input.shape[4])    #(N, C, multi_grains, H, W) -> (N, C*multi_grains, H, W)
        out = torch.cat([input, input_max, input_mean], dim=1)  # (N, C+C+C*multi_grans, H, W)
        return self.fuse(out)
