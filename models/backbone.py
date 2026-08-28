"""Tiny Darknet-style backbone (~1.3M params) for EVM-nano.

Output strides/channels: P3 (s8), P4 (s16), P5 (s32).
Config (model_nano.yaml): stem 32 -> [64,128,256], depth [2,2,2].
"""
import torch.nn as nn

from .blocks import CSPBlock, Conv


class Backbone(nn.Module):
    def __init__(self, stem_channels=32, out_channels=(64, 128, 256), depth=(2, 2, 2)):
        super().__init__()
        c3, c4, c5 = out_channels
        self.feature_channels = (c4, c5, c5)   # P3, P4, P5 output channels
        self.stem = Conv(3, stem_channels, 3, 2)                    # s2
        self.stage1 = Conv(stem_channels, 64, 3, 2)                 # s4
        self.s4 = CSPBlock(64, c3, n=depth[0])                      # s4 out (used by downsample into P3)
        self.down1 = Conv(c3, c3, 3, 2)                             # s8
        self.s8 = CSPBlock(c3, c4, n=depth[1])                      # P3 out, s8
        self.down2 = Conv(c4, c4, 3, 2)                             # s16
        self.s16 = CSPBlock(c4, c5, n=depth[2])                     # P4 out, s16
        self.down3 = Conv(c5, c5, 3, 2)                             # s32
        self.s32 = CSPBlock(c5, c5, n=depth[2])                     # P5 out, s32

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.s4(x)
        x = self.down1(x)
        p3 = self.s8(x)
        x = self.down2(p3)
        p4 = self.s16(x)
        x = self.down3(p4)
        p5 = self.s32(x)
        return p3, p4, p5