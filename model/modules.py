# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class building_block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(building_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return  self.process(input)


class dilated_res_block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(dilated_res_block, self).__init__()
        self.build_blocks = nn.Sequential(
            building_block(in_ch, out_ch, dilation),
            building_block(out_ch, out_ch, dilation)
        )

    def forward(self, input):
        return torch.cat([input, self.build_blocks(input)], 1)


class hierarchical_dilated_module(nn.Module):
    def __init__(self, channel, dilation):
        super(hierarchical_dilated_module, self).__init__()
        self.dilated_block_1 = dilated_res_block(channel, channel, dilation[0])
        self.dilated_block_2 = dilated_res_block(channel, channel, dilation[1])
        self.dilated_block_3 = dilated_res_block(channel, channel, dilation[2])

    def forward(self, input):
        result_0 = self.dilated_block_1(input)
        result_1 = self.dilated_block_2(result_0)
        result_2 = self.dilated_block_3(result_1)
        return torch.cat([input, result_0, result_1, result_2], 1)


class head(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(head, self).__init__()
        self.process = nn.Sequential(
            building_block(in_ch, out_ch, dilation),
            dilated_res_block(out_ch, dilation)
        )

    def forward(self, input):
        return self.process(input)


class down(nn.Module):
    def __int__(self, in_ch, out_ch, dilation=1):
        super(down, self).__int__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            head(in_ch, out_ch, dilation)
        )

    def forward(self, input):
        return self.down_sample(input)


class tail(nn.Module):
    def __init__(self, in_ch, out_ch, K, scale=1, dilation=1):
        super(tail, self).__init__()
        if scale == 1:
            self.process = nn.Sequential(
                building_block(in_ch, out_ch, dilation),
                building_block(out_ch, int(out_ch / 2), dilation),
                nn.Conv2d(int(out_ch / 2), K, 1)
            )
        else:
            self.process = nn.Sequential(
                building_block(in_ch, out_ch, dilation),
                nn.ConvTranspose2d(out_ch, out_ch, scale, stride=scale),
                building_block(out_ch, int(out_ch/2), dilation),
                nn.Conv2d(int(out_ch/2), K, 1)
            )

    def forward(self, input):
        return self.process(input)
