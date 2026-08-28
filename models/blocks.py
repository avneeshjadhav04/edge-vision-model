"""Convolution building blocks for EVM (from scratch, CPU-friendly ops only)."""
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Sequential):
    """conv-bn-silu, the workhorse block."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, bn=True, act=True):
        layers = [nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(c2))
        if act:
            layers.append(nn.SiLU())
        super().__init__(*layers)
        self.out_channels = c2


class CSPBlock(nn.Module):
    """CSP-lite: split, two 3x3 branches on one half, concat, fuse.

    Cheap gradient-flow highway (like a mini-C3 without bottleneck 1x1s) that
    keeps op count low for CPU.
    """

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2 // 2, 1)
        self.cv2 = Conv(c1, c2 // 2, 1)
        self.m = nn.Sequential(*[Conv(c2 // 2, c2 // 2, 3) for _ in range(n)])
        self.cv3 = Conv(2 * (c2 // 2), c2, 1)

    def forward(self, x):
        a = self.cv1(x)
        b = self.m(self.cv2(x))
        return self.cv3(torch.cat([a, b], dim=1))


class SPPF(nn.Module):
    """Sequential maxpool SPP (5/9/13 as three chained 5x5 pools). Cheap and effective."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        h = c1 // 2
        self.cv1 = Conv(c1, h, 1)
        self.cv2 = Conv(h * 4, c2, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = x
        y2 = nn.functional.max_pool2d(x, self.k, stride=1, padding=self.k // 2)
        y3 = nn.functional.max_pool2d(y2, self.k, stride=1, padding=self.k // 2)
        y4 = nn.functional.max_pool2d(y3, self.k, stride=1, padding=self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3, y4], dim=1))