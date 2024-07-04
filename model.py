#!/usr/bin/env python3
"""
UNet Architecture
"""
# Standard Library Imports

# Third-Party Imports
import torch
import torch.nn as nn


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True, ignore_pool=False):
        super(ContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if max_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

        self.ignore_pool = ignore_pool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x  # store the output for the skip connection

        if not self.ignore_pool:
            x = self.pool(x)

        return x, skip


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, transpose=True):
        super(ExpandingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.transpose = transpose
        if self.transpose:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv2d = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, skip):
        x = self.upsample(x)

        if not self.transpose:
            x = self.conv2d(x)

        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True, transpose=True):
        super(UNet, self).__init__()

        self.contract1 = ContractingBlock(in_channels, 64, max_pool)
        self.contract2 = ContractingBlock(64, 128, max_pool)
        self.contract3 = ContractingBlock(128, 256, max_pool)

        self.contract4 = ContractingBlock(256, 512, max_pool=False, ignore_pool=True)

        self.expand1 = ExpandingBlock(512, 256, transpose)
        self.expand2 = ExpandingBlock(256, 128, transpose)
        self.expand3 = ExpandingBlock(128, 64, transpose)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # # Contracting path
        x1, skip1 = self.contract1(x)    # [B, 64, 64, 64]    [B, 64, 128, 128]
        x2, skip2 = self.contract2(x1)   # [B, 128, 32, 32]   [B, 128, 64, 64]
        x3, skip3 = self.contract3(x2)   # [B, 256, 16, 16]   [B, 256, 32, 32]

        # Bottleneck layer
        x4, skip4 = self.contract4(x3)   # [B, 512, 16, 16]      [B, 512, 16, 16]

        # Expanding path
        x = self.expand1(x4, skip3)      # [B, 256, 32, 32]
        x = self.expand2(x, skip2)       # [B, 128, 64, 64]
        x = self.expand3(x, skip1)       # [B, 64, 128, 128]

        x = self.final_conv(x)           # [B, 3, 128, 128]
        return x
