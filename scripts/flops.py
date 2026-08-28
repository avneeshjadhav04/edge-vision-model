"""Params + GFLOPs report (conv/GEMM MACs hook, deploy path only).

    python -m scripts.flops [--config model_nano] [--size 640]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_model, count_params
from scripts.common import load_config


def measure(model, size, device="cpu"):
    model = model.to(device).eval()
    macs = [0]
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, torch.nn.Conv2d):
            macs[0] += mod.weight.shape.numel() * out.shape[-2] * out.shape[-1]
        elif isinstance(mod, torch.nn.Linear):
            macs[0] += mod.weight.shape.numel()

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        model(x, with_aux=False)
    for h in hooks:
        h.remove()
    return macs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for nc, name in [(20, "VOC"), (80, "COCO")]:
        model = build_model(cfg, num_classes=nc)
        p = count_params(model)
        macs = measure(model, args.size)
        print(f"[{name}] deployable {p['deployable'] / 1e6:.2f}M | "
              f"{macs / 1e9:.2f} GMac ({2 * macs / 1e9:.1f} GFLOPs) @{args.size}px")


if __name__ == "__main__":
    main()