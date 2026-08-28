"""Standalone evaluation: mAP for a checkpoint on VOC test or COCO val.

    python -m scripts.eval --dataset voc --root ./datasets/VOC --weights runs/voc/best.pt
    python -m scripts.eval --dataset coco --root ./datasets/coco --weights runs/coco/best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from data.coco import CocoDataset
from data.common import collate_batch
from data.voc import VOCDataset, VOC_CLASSES
from engine.eval_voc import eval_voc
from engine.inference import run_inference
from models import build_model
from scripts.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["voc", "coco"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--score-thresh", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dcfg = load_config(args.dataset)
    mcfg = load_config(args.config)
    nc = dcfg["num_classes"]
    model = build_model(mcfg, num_classes=nc)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = sd.get("model", sd)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    if args.dataset == "voc":
        ds = VOCDataset(args.root, years=(dcfg["year_test"],), split=dcfg.get("test_set", "test"))
    else:
        ds = CocoDataset(args.root, split=f"val{dcfg['val_year']}")

    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=8,
                    collate_fn=collate_batch)
    # wrap eval transform via dataset transform attr
    from data.augment import EvalTransform
    ds.transform = EvalTransform(args.img_size)

    preds, tgts = [], []
    with torch.no_grad():
        from engine.inference import run_inference
        preds, tgts = run_inference(model, dl, args.device, args.img_size,
                                    score_thresh=args.score_thresh)
    if args.dataset == "voc":
        res = eval_voc(preds, tgts, num_classes=nc)
        print(f"VOC2007 test mAP@0.5: {res['mAP']:.4f}")
        for c, name in enumerate(VOC_CLASSES):
            print(f"  {name:14s} {res['per_class_ap'][c]:.3f}")
    else:
        from data.coco_io import CocoEvaluatorBridge
        for p, t in zip(preds, tgts):
            p["image_id"] = int(t["image_id"])
        bridge = CocoEvaluatorBridge(os.path.join(
            args.root, "annotations", f"instances_val{dcfg['val_year']}.json"))
        res = bridge.evaluate(preds)
        print(f"COCO val{dcfg['val_year']}: mAP={res['mAP']:.4f} "
              f"mAP50={res['mAP50']:.4f} mAP75={res['mAP75']:.4f}")


if __name__ == "__main__":
    main()