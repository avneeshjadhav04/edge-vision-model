"""COCO JSON I/O: lightweight annotation index + pycocotools-backed evaluator bridge."""
import json
import os

import numpy as np
import torch


def load_coco_annotations(ann_file, skip_crowd_train=True):
    """Build img_id -> {file_name, boxes (M,4) xyxy float32 tensor, labels (M,) long}.

    Labels are contiguous 0..K-1 by sorted category id.
    skip_crowd_train: drop iscrowd boxes (they are handled in the evaluator instead).
    """
    with open(ann_file) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_contiguous = {c["id"]: i for i, c in enumerate(cats)}
    imgs = {im["id"]: im["file_name"] for im in coco["images"]}
    per_img = {i: {"file_name": imgs[i], "boxes": [], "labels": [], "crowd": []}
               for i in imgs}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        rec = per_img[a["image_id"]]
        rec["boxes"].append([x, y, x + w, y + h])
        rec["labels"].append(cat_to_contiguous[a["category_id"]])
        rec["crowd"].append(bool(a.get("iscrowd", 0)))
    index = {}
    for i, rec in per_img.items():
        if not rec["boxes"]:
            continue
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.tensor(rec["labels"], dtype=torch.long)
        crowd = torch.tensor(rec["crowd"], dtype=torch.bool)
        if skip_crowd_train and crowd.any():
            keep = ~crowd
            boxes, labels = boxes[keep], labels[keep]
        if boxes.numel():
            index[i] = {"file_name": rec["file_name"], "boxes": boxes, "labels": labels}
    return index


def coco_results_from_predictions(coco_gt, predictions):
    """Convert [{"image_id", "boxes", "scores", "labels"(contiguous)}] to COCO json format."""
    cat_ids = sorted(coco_gt.getCatIds())
    out = []
    for p in predictions:
        for box, score, label in zip(p["boxes"], p["scores"], p["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            out.append({
                "image_id": int(p["image_id"]),
                "category_id": cat_ids[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return out


class CocoEvaluatorBridge:
    """Thin wrapper: feed predictions, get AP metrics via pycocotools (Apache-2.0 dep)."""

    def __init__(self, ann_file):
        from pycocotools.coco import COCO
        self.coco_gt = COCO(ann_file)

    def evaluate(self, predictions):
        from pycocotools.cocoeval import COCOeval
        results = coco_results_from_predictions(self.coco_gt, predictions)
        if not results:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}
        dt = self.coco_gt.loadRes(results)
        E = COCOeval(self.coco_gt, dt, iouType="bbox")
        img_ids = sorted({int(p["image_id"]) for p in predictions})
        E.params.imgIds = img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        return {"mAP": float(E.stats[0]), "mAP50": float(E.stats[1]),
                "mAP75": float(E.stats[2])}