"""PAN-lite neck: top-down FPN + bottom-up path over P3/P4/P5 (~0.7M params)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv


class PANNeck(nn.Module):
    def __init__(self, in_channels=(128, 256, 256), hidden=96, out=96):
        super().__init__()
        c3, c4, c5 = in_channels
        h = hidden
        # lateral 1x1 reduce
        self.l5 = Conv(c5, h, 1)
        self.l4 = Conv(c4, h, 1)
        self.l3 = Conv(c3, h, 1)
        # top-down fuse
        self.n4_td = Conv(h, h, 3)
        self.n3_td = Conv(h, h, 3)
        # bottom-up fuse
        self.n4_bu = Conv(h, h, 3)
        self.n5_bu = Conv(h, h, 3)
        # output smoothing
        self.o3 = Conv(h, out, 3)
        self.o4 = Conv(h, out, 3)
        self.o5 = Conv(h, out, 3)

    def forward(self, p3, p4, p5):
        t5 = self.l5(p5)                       # s32
        t4 = self.l4(p4) + F.interpolate(t5, scale_factor=2, mode="nearest")
        t4 = self.n4_td(t4)
        t3 = self.l3(p3) + F.interpolate(t4, scale_factor=2, mode="nearest")
        t3 = self.n3_td(t3)
        b4 = self.n4_bu(t4 + F.interpolate(t3, scale_factor=0.5, mode="nearest"))
        b5 = self.n5_bu(t5 + F.interpolate(b4, scale_factor=0.5, mode="nearest"))
        return self.o3(t3), self.o4(b4), self.o5(b5)   # (s8, s16, s32) each `out` ch