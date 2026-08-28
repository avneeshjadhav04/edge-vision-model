"""Task-aligned matching: one-to-many (aux) and one-to-one (Hungarian) — from scratch.

SimOTA-lite cost:  cls_cost = BCE(cls_pred, onehot) weighted by (alpha * align + beta)
                   align_metric = (cls_score)^alpha * IoU^beta
No scipy: the o2o assignment uses a greedy Hungarian approximation on the cost
matrix (lowest cost first, one anchor per GT, one GT per anchor), which works
well in practice for nano models.
"""
import torch

from models.decode import bbox_iou


def _bbox_giou(g, p, eps=1e-9):
    """Generalized IoU between (M,4) and (N,4) xyxy tensors -> (M,N)."""
    x1 = torch.max(g[:, None, 0], p[None, :, 0])
    y1 = torch.max(g[:, None, 1], p[None, :, 1])
    x2 = torch.min(g[:, None, 2], p[None, :, 2])
    y2 = torch.min(g[:, None, 3], p[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = ((g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0))[:, None]
    a2 = ((p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0))[None, :]
    union = (a1 + a2 - inter).clamp(min=eps)
    iou = inter / union
    # smallest enclosing box
    ex1 = torch.min(g[:, None, 0], p[None, :, 0])
    ey1 = torch.min(g[:, None, 1], p[None, :, 1])
    ex2 = torch.max(g[:, None, 2], p[None, :, 2])
    ey2 = torch.max(g[:, None, 3], p[None, :, 3])
    enclose = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)
    return iou - (enclose - union) / (enclose + eps)


def align_metric(pred_boxes, pred_cls, gt_boxes, gt_cls, alpha=0.5, beta=6.0, eps=1e-9):
    """pred_boxes (N,4) xyxy pixels, pred_cls (N,nc) logits;
    gt_boxes (M,4), gt_cls (M,) long. Returns (M,N) alignment and (M,N) iou."""
    iou = bbox_iou(gt_boxes[:, None, :], pred_boxes[None, :, :])       # (M,N)
    with torch.no_grad():
        p_cls = pred_cls.sigmoid()                                     # (N,nc)
        cls_score = p_cls[:, gt_cls].T                                 # (M,N) GT-class score
    align = cls_score.pow(alpha) * iou.pow(beta)
    return align, iou


def select_candidates(align, iou, topk=10):
    """SimOTA-style candidate mask: topk by align, require IoU > iou_thresh.
    align/iou: (M,N). Returns (M,N) bool."""
    M, N = align.shape
    topk = min(max(topk, 1), N)
    topk_vals, _ = align.topk(topk, dim=1)                              # (M,topk)
    thresh = topk_vals[:, -1:].clamp(min=1e-6)
    mask_align = align >= thresh
    mask_iou = iou > 0.20
    return mask_align & mask_iou


def cost_matrix(align, iou, pred_cls, gt_cls, alpha=0.5, beta=6.0, eps=1e-9):
    """Task-aligned cost (lower is better). align/iou: (M,N); pred_cls (N,nc) logits."""
    M = gt_cls.numel()
    N, nc = pred_cls.shape
    p_cls = pred_cls.sigmoid().T                            # (nc,N)
    onehot = torch.zeros(M, nc, device=pred_cls.device)
    onehot.scatter_(1, gt_cls.view(-1, 1), 1.0)
    # BCE between every GT onehot and every anchor score -> (M,N)
    pt = p_cls.T[None]                                        # (1,N,nc)
    bce = (onehot[:, None, :] * -torch.log(pt + eps) +
           (1 - onehot[:, None, :]) * -torch.log(1 - pt + eps)).sum(-1)
    align_norm = align / (align.amax(dim=1, keepdim=True).clamp(min=eps))
    return bce * (alpha * align_norm + beta) - align


def greedy_hungarian(cost):
    """One-to-one assignment via greedy lowest-cost-first with mutual exclusion.
    cost: (M,N). Returns (M,) anchor index or -1."""
    M, N = cost.shape
    device = cost.device
    assign = torch.full((M,), -1, dtype=torch.long, device=device)
    used_anchors = torch.zeros(N, dtype=torch.bool, device=device)
    flat = cost.reshape(-1)
    order = torch.argsort(flat)
    for idx in order.tolist():
        gi, ni = divmod(idx, N)
        if assign[gi] != -1 or used_anchors[ni]:
            continue
        assign[gi] = ni
        used_anchors[ni] = True
        if (assign >= 0).all():
            break
    return assign