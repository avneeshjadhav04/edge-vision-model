"""Anchor grids, DFL decode, and NMS-free post-processing (from scratch)."""
import math

import torch
import torch.nn.functional as F


def make_anchors(feats, strides):
    """Center anchors per level. feats: list of tensors (B,C,H,W).
    Returns anchors (sum(HW), 2) in pixels and stride map (sum(HW), 1)."""
    anchors, strides_all = [], []
    for f, s in zip(feats, strides):
        _, _, H, W = f.shape
        sy, sx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=f.device),
            torch.arange(W, dtype=torch.float32, device=f.device), indexing="ij")
        # anchor center at cell center: (x + 0.5) * s
        ax = (sx + 0.5).reshape(-1) * s
        ay = (sy + 0.5).reshape(-1) * s
        anchors.append(torch.stack([ax, ay], dim=1))
        strides_all.append(torch.full((H * W, 1), float(s), device=f.device))
    return torch.cat(anchors, 0), torch.cat(strides_all, 0)


def _anchors_from_shapes(shapes, strides, device):
    """shapes: list of (H,W) per level -> anchor centers and stride map."""
    anchors, strides_all = [], []
    for (H, W), s in zip(shapes, strides):
        sy, sx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device), indexing="ij")
        ax = (sx + 0.5).reshape(-1) * s
        ay = (sy + 0.5).reshape(-1) * s
        anchors.append(torch.stack([ax, ay], dim=1))
        strides_all.append(torch.full((H * W, 1), float(s), device=device, dtype=torch.float32))
    return torch.cat(anchors, 0), torch.cat(strides_all, 0)


def dfl_decode(box_dist, reg_max, proj, anchors, strides):
    """box_dist: (B, N, 4*reg_max) -> decoded xyxy pixels (B, N, 4).

    proj: (reg_max,) projection vector (0..reg_max-1).
    ltrb are distances from the anchor center, in stride units (scaled by stride).
    """
    B, N, _ = box_dist.shape
    d = box_dist.view(B, N, 4, reg_max).softmax(-1)
    dist = (d * proj.view(1, 1, 1, reg_max)).sum(-1)          # (B,N,4) in [0, reg_max-1]
    dist = dist * strides.unsqueeze(0)                         # to pixels
    l, t, r, b = dist.unbind(-1)
    x1 = anchors[:, 0][None] - l
    y1 = anchors[:, 1][None] - t
    x2 = anchors[:, 0][None] + r
    y2 = anchors[:, 1][None] + b
    return torch.stack([x1, y1, x2, y2], dim=-1)


class Postprocessor:
    """NMS-free decoding: scores = sigmoid(cls); top-k over (obj * cls) if obj used.

    Because training assigns each GT to exactly one anchor, overlapping duplicates
    are rare; we still apply an optional light score-threshold + class-argmax.
    """

    def __init__(self, num_classes, strides=(8, 16, 32), reg_max=8, score_thresh=0.25,
                 max_det=300, use_obj=True):
        self.nc = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.score_thresh = score_thresh
        self.max_det = max_det
        self.use_obj = use_obj

    @torch.no_grad()
    def __call__(self, main_out, proj):
        """main_out: list of (box, cls, obj) per level, each (B,C,H,W).
        Returns list of dict(pred_boxes (M,4) xyxy, scores (M,), labels (M,))."""
        first = main_out[0][0]
        device = first.device
        shapes = [(int(b.shape[2]), int(b.shape[3])) for (b, _, _) in main_out]
        anchors, strides = _anchors_from_shapes(shapes, self.strides, device)

        box_flat, cls_flat, obj_flat = [], [], []
        for (bx, cl, ob) in main_out:
            B, _, H, W = bx.shape
            box_flat.append(bx.view(B, 4 * self.reg_max, H * W).permute(0, 2, 1))
            cls_flat.append(cl.view(B, -1, H * W).permute(0, 2, 1))
            obj_flat.append(ob.view(B, 1, H * W).permute(0, 2, 1))
        box = torch.cat(box_flat, 1)                     # (B,N,4r)
        cls = torch.cat(cls_flat, 1)                     # (B,N,nc)
        obj = torch.cat(obj_flat, 1)                     # (B,N,1)

        boxes = dfl_decode(box, self.reg_max, proj, anchors, strides)  # (B,N,4)
        cls_scores = cls.sigmoid()                       # (B,N,nc)
        if self.use_obj:
            scores = cls_scores * obj.sigmoid()          # (B,N,nc)
        else:
            scores = cls_scores

        results = []
        for bi in range(boxes.shape[0]):
            s, lbl = scores[bi].max(-1)                  # best class per anchor
            keep = s > self.score_thresh
            s, lbl, bb = s[keep], lbl[keep], boxes[bi][keep]
            if s.numel() > self.max_det:
                topv, topi = s.topk(self.max_det)
                s, lbl, bb = topv, lbl[topi], bb[topi]
            results.append({"pred_boxes": bb, "scores": s, "labels": lbl})
        return results


def bbox_iou(box1, box2, eps=1e-9):
    """IoU of two (..., 4) xyxy tensors, broadcastable."""
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    a2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + eps)