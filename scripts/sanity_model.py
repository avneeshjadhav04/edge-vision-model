"""Quick model sanity: forward/decode/inference on random input."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_model, count_params
from scripts.common import load_config


def main():
    cfg = load_config("model_nano")
    m = build_model(cfg, num_classes=80)
    print("params:", count_params(m))
    m.eval()
    with torch.no_grad():
        res = m.predict(torch.randn(1, 3, 640, 640), score_thresh=0.05)
    r = res[0]
    print("predict ->", {k: tuple(v.shape) for k, v in r.items()})
    print("SANITY OK")


if __name__ == "__main__":
    main()