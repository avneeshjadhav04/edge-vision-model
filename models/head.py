"""Decoupled anchor-free dual-assignment head (from scratch).

Main (one-to-one) head: box distribution (4*reg_max, DFL) + class logits +
a light objectness map; used at train and inference (NMS-free at decode time).
Aux (one-to-many) head: box + class branch that densifies gradients during
training (YOLOv10-style); discarded at export.
"""
import math

import torch
import torch.nn as nn

from .blocks import Conv


class BoxBranch(nn.Module):
    def __init__(self, c1, hidden, reg_max):
        super().__init__()
        self.box = nn.Sequential(
            Conv(c1, hidden, 3), Conv(hidden, hidden, 3),
            nn.Conv2d(hidden, 4 * reg_max, 1))
        m = self.box[-1]
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.box(x)


class ClsBranch(nn.Module):
    def __init__(self, c1, hidden, num_classes):
        super().__init__()
        self.cls = nn.Sequential(
            Conv(c1, hidden, 3), Conv(hidden, hidden, 3),
            nn.Conv2d(hidden, num_classes, 1))
        m = self.cls[-1]
        nn.init.normal_(m.weight, std=0.01)
        # prior: ~5 positive anchors among the 6400 anchors of the first grid
        nn.init.constant_(m.bias, math.log(5 / 6400))

    def forward(self, x):
        return self.cls(x)


class MainHead(nn.Module):
    """One-to-one head (train + inference)."""

    def __init__(self, c1, hidden, num_classes, reg_max):
        super().__init__()
        self.box = BoxBranch(c1, hidden, reg_max)
        self.cls = ClsBranch(c1, hidden, num_classes)
        h2 = max(hidden // 2, 16)
        self.obj = nn.Sequential(Conv(c1, h2, 3), nn.Conv2d(h2, 1, 1))
        nn.init.normal_(self.obj[-1].weight, std=0.01)
        nn.init.constant_(self.obj[-1].bias, 0.0)

    def forward(self, x):
        return self.box(x), self.cls(x), self.obj(x)


class AuxHead(nn.Module):
    """One-to-many auxiliary head (training only)."""

    def __init__(self, c1, hidden, num_classes, reg_max):
        super().__init__()
        self.box = BoxBranch(c1, hidden, reg_max)
        self.cls = ClsBranch(c1, hidden, num_classes)

    def forward(self, x):
        return self.box(x), self.cls(x)


class Detect(nn.Module):
    """Assembles main + aux heads over the 3 neck outputs."""

    def __init__(self, c1, hidden=96, num_classes=80, reg_max=8, aux_hidden=96):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.main = MainHead(c1, hidden, num_classes, reg_max)
        self.aux = AuxHead(c1, aux_hidden, num_classes, reg_max)

    def forward(self, feats, with_aux=True):
        """Returns list per level: dict(main=(box, cls, obj), aux=(box, cls)|None)."""
        out = []
        for f in feats:
            mb, mc, mo = self.main(f)
            level = {"main": (mb, mc, mo), "aux": None}
            if with_aux and self.aux is not None:
                level["aux"] = self.aux(f)
            out.append(level)
        return out