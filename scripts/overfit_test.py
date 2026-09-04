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
    all_scores, all_boxes, all_gt = [], [], []
    for i in range(len(ds)):
        img, tgt = ds[i]
        x, t2 = et(img, tgt)  # returns tensor + rescale info
        with torch.no_grad():
            # one-to-one head: no objectness (YOLOv10-style) - obj branch would
            # suppress scores; score = cls only
            res = model.predict(x[None].to(device), score_thresh=0.01, max_det=100,
                                use_obj=False)
        r, pw, ph = [float(v) for v in t2["rescale"]]
        bb = res[0]["pred_boxes"].clone()
        if bb.numel():
            bb[:, [0, 2]] = (bb[:, [0, 2]] - pw) / r
            bb[:, [1, 3]] = (bb[:, [1, 3]] - ph) / r
        preds.append({"pred_boxes": bb.cpu(), "scores": res[0]["scores"].cpu(),
                      "labels": res[0]["labels"].cpu()})
        targets.append(tgt)
        all_scores.append(res[0]["scores"].cpu())
        all_boxes.append(bb.cpu())
        all_gt.append(tgt["boxes"])
    # ---- diagnostics (why is mAP ~0?) ----
    import numpy as np
    sc = torch.cat(all_scores) if all_scores else torch.zeros(0)
    print(f"  [diag] preds/image: {[len(s) for s in all_scores][:5]}... "
          f"total={len(sc)}")
    if sc.numel():
        print(f"  [diag] score min/mean/max: {sc.min():.4f}/{sc.mean():.4f}/{sc.max():.4f}")
        print(f"  [diag] #score>0.5: {(sc > 0.5).sum().item()}, #score>0.1: {(sc > 0.1).sum().item()}")
    # best IoU of any pred box vs each GT (per image)
    ious = []
    for bb, gt in zip(all_boxes, all_gt):
        if bb.numel() == 0 or gt.numel() == 0:
            continue
        from models.decode import bbox_iou
        iou = bbox_iou(gt[:, None, :], bb[None, :, :])  # (M, P)
        ious.append(iou.max(dim=1).values)
    if ious:
        iou_all = torch.cat(ious)
        print(f"  [diag] best-pred-IoU vs GT: mean={iou_all.mean():.3f} "
              f"#IoU>0.5={(iou_all > 0.5).sum().item()}/{iou_all.numel()}")
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
    # light train transform for the gate: NO mosaic (dataset_for_mosaic=None would
    # fill 3/4 quadrants with empty gray, shrinking objects to near-invisible and
    # biasing the model toward tiny boxes). Affine/flip/HSV only.
    ds.transform = TrainTransform(args.img_size, mosaic_p=0.0, scale=0.2, translate=0.05,
                                  fliplr=0.5, hsv=(0.015, 0.5, 0.3), dataset_for_mosaic=None)
    model = build_model(mcfg, num_classes=20)
    params = count_params(model)
    print(f"params: {params['total'] / 1e6:.2f}M total / {params['deployable'] / 1e6:.2f}M deployable")
    crit = DetectionLoss(num_classes=20, reg_max=mcfg["head"]["reg_max"],
                         box_w=mcfg["loss"]["box_weight"], cls_w=mcfg["loss"]["cls_weight"],
                         dfl_w=mcfg["loss"]["dfl_weight"], obj_w=mcfg["loss"]["obj_weight"],
                         o2m_topk=mcfg["loss"]["o2m_topk"], alpha=mcfg["loss"]["alpha"],
                         beta=mcfg["loss"]["beta"])
    cfg = {"train": {"optimizer": "adamw", "lr0": 1e-3, "lrf": 0.01, "warmup_epochs": 3,
                     "batch_size": args.batch_size, "workers": 2, "amp": True,
                     "ema_decay": 0.99, "val_interval": 10, "mosaic_close_epochs": 30,
                     "mosaic": 0.0, "accum": 1}}
    tr = Trainer(model, crit, ds, cfg=cfg, device=args.device, save_dir=args.save_dir)
    tr.fit(args.epochs)
    # gate evals the RAW model: with only ~600 steps across 20 images the EMA
    # (decay 0.99) still lags; raw weights reflect actual learned fit.
    gate_model = tr.model
    # final eval on a raw (untransformed) view of the same images
    ds_eval = OverfitSubset(args.root, n=args.n, transform=None)
    m = evaluate_overfit(gate_model, ds_eval, args.device, args.img_size)
    print(f"OVERFIT mAP@0.5 = {m['mAP']:.4f} (target {args.target})")
    if m["mAP"] >= args.target:
        print("PASS")
    else:
        print("FAIL - investigate before full training")
    sys.exit(0 if m["mAP"] >= args.target else 1)


if __name__ == "__main__":
    main()