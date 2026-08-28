"""Overfit sanity check: 20 VOC images -> target 90+ mAP on those images.

Usage:
    python -m scripts.overfit_test --root ./datasets/VOC --epochs 300 --device cuda
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.voc import OverfitSubset, VOC_CLASSES
from data.augment import TrainTransform
from engine.trainer import Trainer
from losses import DetectionLoss
from models import build_model, count_params
from scripts.common import load_config


def evaluate_overfit(model, ds, device, img_size=320):
    """Deterministic eval on the overfit set (letterboxed, no mosaic)."""
    from data.augment import EvalTransform
    from data.common import letterbox
    from engine.eval_voc import eval_voc
    model.eval()
    preds, targets = [], []
    et = EvalTransform(img_size)
    for i in range(len(ds)):
        img, tgt = ds[i]
        x, t2 = et(img, tgt)  # returns tensor + rescale info
        with torch.no_grad():
            res = model.predict(x[None].to(device), score_thresh=0.01, max_det=100)
        r, pw, ph = [float(v) for v in t2["rescale"]]
        bb = res[0]["pred_boxes"].clone()
        if bb.numel():
            bb[:, [0, 2]] = (bb[:, [0, 2]] - pw) / r
            bb[:, [1, 3]] = (bb[:, [1, 3]] - ph) / r
        preds.append({"pred_boxes": bb.cpu(), "scores": res[0]["scores"].cpu(),
                      "labels": res[0]["labels"].cpu()})
        targets.append(tgt)
    return eval_voc(preds, targets, num_classes=len(VOC_CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets/VOC")
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--save-dir", default="runs/overfit")
    args = ap.parse_args()

    mcfg = load_config(args.config)
    ds = OverfitSubset(args.root, n=args.n, transform=None)
    # light train transform (mosaic on for overfit robustness, small canvas)
    ds.transform = TrainTransform(args.img_size, mosaic_p=0.5, scale=0.2, translate=0.05,
                                  fliplr=0.5, hsv=(0.015, 0.5, 0.3), dataset_for_mosaic=None)
    model = build_model(mcfg, num_classes=20)
    params = count_params(model)
    print(f"params: {params['total'] / 1e6:.2f}M total / {params['deployable'] / 1e6:.2f}M deployable")
    crit = DetectionLoss(num_classes=20, reg_max=mcfg["head"]["reg_max"],
                         box_w=mcfg["loss"]["box_weight"], cls_w=mcfg["loss"]["cls_weight"],
                         dfl_w=mcfg["loss"]["dfl_weight"], obj_w=mcfg["loss"]["obj_weight"],
                         o2m_topk=mcfg["loss"]["o2m_topk"], alpha=mcfg["loss"]["alpha"],
                         beta=mcfg["loss"]["beta"])
    cfg = {"train": {"optimizer": "sgd", "lr0": 0.02, "lrf": 0.01, "warmup_epochs": 3,
                     "batch_size": args.batch_size, "workers": 2, "amp": True,
                     "ema_decay": 0.999, "val_interval": 10, "mosaic_close_epochs": 30,
                     "accum": 1}}
    tr = Trainer(model, crit, ds, cfg=cfg, device=args.device, save_dir=args.save_dir)
    tr.fit(args.epochs)
    ema_model = tr.ema.module
    # final eval at train size
    m = evaluate_overfit(ema_model, ds, args.device, args.img_size)
    print(f"OVERFIT mAP@0.5 = {m['mAP']:.4f} (target {args.target})")
    if m["mAP"] >= args.target:
        print("PASS")
    else:
        print("FAIL - investigate before full training")
    sys.exit(0 if m["mAP"] >= args.target else 1)


if __name__ == "__main__":
    main()