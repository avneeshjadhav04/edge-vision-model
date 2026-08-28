"""Main training script for VOC or COCO.

Usage:
    python -m scripts.train --dataset voc --root ./datasets/VOC --device cuda
    python -m scripts.train --dataset coco --root ./datasets/coco --epochs 300 --device cuda
    python -m scripts.train --resume runs/train/last.pt ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from data.augment import EvalTransform, TrainTransform
from data.coco import CocoDataset
from data.common import collate_batch
from data.voc import VOCDataset, VOC_CLASSES
from engine.eval_voc import eval_voc
from engine.inference import run_inference
from engine.trainer import Trainer
from losses import DetectionLoss
from models import build_model, count_params
from scripts.common import load_config


def make_voc_eval_fn(model_holder, root, img_size, device):
    ds = VOCDataset(root, years=("2007",), split="test",
                    transform=EvalTransform(img_size))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=collate_batch)

    def fn(model):
        preds, tgts = run_inference(model, loader, device, img_size)
        return eval_voc(preds, tgts, num_classes=20)

    return fn


def make_coco_eval_fn(root, img_size, device, year=2017):
    from data.coco_io import CocoEvaluatorBridge
    split = f"val{year}"
    ds = CocoDataset(root, split=split, year=year, transform=EvalTransform(img_size))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_batch)
    bridge = CocoEvaluatorBridge(os.path.join(root, "annotations",
                                              f"instances_{split}.json"))

    def fn(model):
        preds, tgts = run_inference(model, loader, device, img_size)
        for p, t in zip(preds, tgts):
            p["image_id"] = int(t["image_id"])
        return bridge.evaluate(preds)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["voc", "coco"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--init-from", default=None, help="weights to init from (VOC->COCO)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    dcfg = load_config(args.dataset)
    mcfg = load_config(args.config)
    nc = dcfg["num_classes"]
    epochs = args.epochs or dcfg.get("epochs", 120 if args.dataset == "voc" else 300)
    bs = args.batch_size or dcfg.get("batch_size", 32)
    save_dir = args.save_dir or f"runs/{args.dataset}"

    train_tf = TrainTransform(
        args.img_size, mosaic_p=mcfg["train"].get("mosaic", 1.0),
        scale=mcfg["train"].get("scale_jitter", 0.5),
        translate=mcfg["train"].get("translate", 0.1),
        fliplr=mcfg["train"].get("fliplr", 0.5),
        hsv=(mcfg["train"].get("hsv_h", 0.015), mcfg["train"].get("hsv_s", 0.7),
             mcfg["train"].get("hsv_v", 0.4)))
    eval_tf = EvalTransform(args.img_size)

    if args.dataset == "voc":
        train_ds = VOCDataset(args.root, years=tuple(dcfg["years_train"]),
                              split="trainval", transform=train_tf)
        eval_fn = make_voc_eval_fn(None, args.root, args.img_size, args.device)
    else:
        train_ds = CocoDataset(args.root, split=f"train{dcfg['train_year']}",
                               transform=train_tf)
        eval_fn = make_coco_eval_fn(args.root, args.img_size, args.device,
                                    dcfg["val_year"])

    model = build_model(mcfg, num_classes=nc)
    params = count_params(model)
    print(f"params: {params['total'] / 1e6:.2f}M total / {params['deployable'] / 1e6:.2f}M deployable")

    crit = DetectionLoss(
        num_classes=nc, reg_max=mcfg["head"]["reg_max"],
        box_w=mcfg["loss"]["box_weight"], cls_w=mcfg["loss"]["cls_weight"],
        dfl_w=mcfg["loss"]["dfl_weight"], obj_w=mcfg["loss"]["obj_weight"],
        o2m_topk=mcfg["loss"]["o2m_topk"], alpha=mcfg["loss"]["alpha"],
        beta=mcfg["loss"]["beta"])

    cfg = {"train": dict(mcfg["train"], batch_size=bs)}
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    tr = Trainer(model, crit, train_ds, val_eval_fn=eval_fn, cfg=cfg,
                 device=args.device, save_dir=save_dir)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {args.init_from} (missing {len(missing)}, unexpected {len(unexpected)})")
    if args.resume:
        tr.load(args.resume, resume=True)
        print(f"resumed at epoch {tr.start_epoch}")
    tr.fit(epochs)


if __name__ == "__main__":
    main()