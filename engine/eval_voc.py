"""VOC mAP evaluation, from scratch (2007-style: 11-point interpolated AP by default).

Convention matches VOC2007 official protocol:
  - predictions ranked by confidence per class;
  - difficult GT boxes are ignored (neither TP nor FP);
  - AP = mean of 11-point interpolated precision (use_07_metric=True) or AUC (False).
"""
import numpy as np
import torch


def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
            ap += p / 11.0
        return ap
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_voc(predictions, targets, num_classes, iou_thresh=0.5, use_07_metric=True,
             difficult_key="difficult"):
    """predictions: list per image of dict(pred_boxes xyxy (in ORIGINAL image coords),
    scores, labels); targets: list per image of dict(boxes, labels, difficult).
    Returns dict(mAP, per_class_ap)."""
    aps = []
    per_class = {}
    for c in range(num_classes):
        # gather all GT and predictions for class c
        npos = 0
        gt_matched = []
        class_preds = []
        for i, (pred, tgt) in enumerate(zip(predictions, targets)):
            gb = tgt["boxes"]
            gl = tgt["labels"]
            diff = tgt.get(difficult_key)
            if diff is None:
                diff = torch.zeros(gl.shape[0], dtype=torch.bool)
            keep_c = gl == c
            npos += int((keep_c & ~diff).sum())
            matched = torch.zeros(int(keep_c.sum()), dtype=torch.bool)
            if keep_c.any():
                gt_matched.append((i, gb[keep_c], diff[keep_c], matched))
            pk = pred["labels"] == c
            for box, score in zip(pred["pred_boxes"][pk], pred["scores"][pk]):
                class_preds.append((float(score), i, box))
        if npos == 0:
            per_class[c] = float("nan")
            continue
        class_preds.sort(key=lambda x: -x[0])
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_index = {i: (b, d, m) for i, (b, d, m) in
                    [(g[0], (g[1], g[2], g[3])) for g in gt_matched]}
        for rank, (score, img_i, box) in enumerate(class_preds):
            if img_i in gt_index:
                gb, diff, matched = gt_index[img_i]
                ious = _iou_one(box, gb)
                j = int(ious.argmax()) if ious.numel() else -1
                if ious.numel() and ious[j] > iou_thresh:
                    if diff[j]:
                        # detection hits a difficult GT -> ignore (neither TP nor FP)
                        continue
                    if not matched[j]:
                        tp[rank] = 1
                        matched[j] = True
                    else:
                        fp[rank] = 1
                else:
                    fp[rank] = 1
            else:
                fp[rank] = 1
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / max(npos, 1e-9)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        ap = voc_ap(rec, prec, use_07_metric)
        aps.append(ap)
        per_class[c] = ap
    mAP = float(np.nanmean(aps)) if aps else float("nan")
    return {"mAP": mAP, "per_class_ap": per_class}


def _iou_one(box, boxes):
    """box (4,) vs boxes (M,4) -> (M,) IoU."""
    if boxes.numel() == 0:
        return torch.zeros(0)
    x1 = torch.maximum(boxes[:, 0], box[0])
    y1 = torch.maximum(boxes[:, 1], box[1])
    x2 = torch.minimum(boxes[:, 2], box[2])
    y2 = torch.minimum(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_g = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_p = max((box[2] - box[0]).item(), 0) * max((box[3] - box[1]).item(), 0)
    return inter / (area_g + area_p - inter + 1e-9)