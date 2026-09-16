"""Diagnostic eval for a finished run: separates detection / ranking / EMA questions.

Answers in one kernel run (no training):
  1. raw vs EMA weights - which checkpoint is actually better on test?
  2. cls-only vs cls*obj scoring (the gate favored cls*obj; full-run eval uses cls)
  3. per-class NMS on/off
  4. train-subset vs test mAP (generalization gap vs underfitting)
  5. recall@0.5 (score-agnostic) - does the model FIND objects at all?
  6. score distribution stats (calibration signal)

Usage (Kaggle only):
    python -m scripts.debug_eval --root ./datasets/VOC \
        --weights runs/voc/best.pt runs/voc/last.pt --device cuda
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from data.augment import EvalTransform
from data.common import collate_batch
from data.voc import VOC_CLASSES, VOCDataset
from engine.eval_voc import eval_voc
from engine.inference import run_inference
from models import build_model
from models.decode import bbox_iou
from scripts.common import load_config


def make_loader(ds, n_limit, batch_size=8, workers=4):
    if n_limit and n_limit < len(ds):
        ds = Subset(ds, list(range(n_limit)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, collate_fn=collate_batch)


def recall_stats(preds, tgts):
    """Score-agnostic detection quality: per GT, best same-class pred IoU (+score).

    recall@0.5 with NO threshold/NMS says whether the network places boxes on
    objects at all; the score attached there says whether ranking can find them.
    """
    ious_all, scores_at = [], []
    n_gt = 0
    n_hit = 0
    for pred, tgt in zip(preds, tgts):
        gb, gl = tgt["boxes"], tgt["labels"]
        pb, pl, ps = pred["pred_boxes"], pred["labels"], pred["scores"]
        for box, lbl in zip(gb, gl):
            n_gt += 1
            same = pl == lbl
            if not same.any() or not pb[same].numel():
                continue
            iou = bbox_iou(box[None, :].float(), pb[same].float())[0]
            j = int(iou.argmax())
            best = float(iou[j])
            ious_all.append(best)
            scores_at.append(float(ps[same][j]))
            if best > iou_thr:
                n_hit += 1
    ious_all = torch.tensor(ious_all) if ious_all else torch.zeros(0)
    return {
        "n_gt": n_gt,
        "recall@0.5": n_hit / max(1, n_gt),
        "best-iou mean": float(ious_all.mean()) if ious_all.numel() else float("nan"),
        "best-iou>0.5": float((ious_all > 0.5).float().mean()) if ious_all.numel() else float("nan"),
        "score_at_best_iou(mean)": (float(torch.tensor(scores_at).mean())
                                    if scores_at else float("nan")),
    }


def score_stats(preds):
    sc = torch.cat([p["scores"] for p in preds]) if preds else torch.zeros(0)
    if sc.numel() == 0:
        return "no preds"
    q = torch.quantile(sc, torch.tensor([0.5, 0.9, 0.99]))
    return (f"n/img~{len(sc) / len(preds):.0f} min={sc.min():.3f} mean={sc.mean():.3f} "
            f"max={sc.max():.3f} p50={q[0]:.3f} p90={q[1]:.3f} p99={q[2]:.3f} "
            f"#>0.5={(sc > 0.5).sum().item()}")


def evaluate_variant(model, loader, device, use_obj, nms_iou):
    preds, tgts = run_inference(model, loader, device, img_size=IMG_SIZE[0],
                                score_thresh=0.01, max_det=300,
                                use_obj=use_obj, nms_iou=nms_iou)
    res = eval_voc(preds, tgts, num_classes=NUM_CLASSES[0])
    tag = f"{'cls*obj' if use_obj else 'cls'} nms={'off' if nms_iou == 0 else nms_iou}"
    print(f"  [{tag}] mAP@0.5 = {res['mAP']:.4f} | {score_stats(preds)}")
    rec = recall_stats(preds, tgts)
    print(f"  [{tag}] recall@0.5={rec['recall@0.5']:.3f} "
          f"(frac bestIoU>0.5 {rec['best-iou>0.5']:.3f}, score@bestIoU {rec['score_at_best_iou']:.3f})")
    # per-class AP for the best variant
    aps = {VOC_CLASSES[c]: round(a, 3) for c, a in res['per_class_ap'].items() if a == a}
    print(f"  [{tag}] per-class: {aps}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", nargs="+", required=True,
                    help="checkpoint paths, each eval'd in order (e.g. best.pt last.pt)")
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--num-classes", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    IMG_SIZE[0] = args.img_size
    mcfg = load_config(args.config)
    num_classes = args.num_classes
    NUM_CLASSES[0] = num_classes

    test_ds = VOCDataset(args.root, years=("2007",), split="test",
                         transform=EvalTransform(args.img_size))
    train_ds = VOCDataset(args.root, years=("2007",), split="trainval",
                          transform=EvalTransform(args.img_size))
    test_loader = make_loader(test_ds, args.n_test, args.batch_size)
    train_loader = make_loader(train_ds, args.n_train, args.batch_size)

    for wpath in args.weights:
        print(f"\n===== weights: {wpath} =====")
        model = build_model(mcfg, num_classes=num_classes)
        sd = torch.load(wpath, map_location="cpu", weights_only=False)
        state = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded (missing {len(missing)}, unexpected {len(unexpected)}); "
              f"ckpt epoch {sd.get('epoch')} is_ema={sd.get('is_ema')}")
        model = model.to(args.device).eval()
        print("  --- VOC2007 test subset ---")
        evaluate_variant(model, test_loader, args.device, use_obj=False, nms_iou=0.45)
        evaluate_variant(model, test_loader, args.device, use_obj=True, nms_iou=0.45)
        evaluate_variant(model, test_loader, args.device, use_obj=False, nms_iou=0)
        print("  --- train subset (trainval, eval view) ---")
        evaluate_variant(model, train_loader, args.device, use_obj=False, nms_iou=0.45)
        evaluate_variant(model, train_loader, args.device, use_obj=True, nms_iou=0.45)


IMG_SIZE = [640]
NUM_CLASSES = [20]

if __name__ == "__main__":
    main()