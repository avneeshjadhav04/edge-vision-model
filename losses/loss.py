"""EVM detection loss: CIoU + DFL + BCE cls + BCE obj, dual assignment (from scratch).

Pipeline per image:
  1. flatten main-head predictions over 3 levels (anchors 8400 @640);
  2. aux (one-to-many) targets via task-aligned top-k  -> train aux head + (optionally)
     warm-up signal to main head early in training;
  3. main (one-to-one) targets via greedy Hungarian on task-aligned cost -> the only
     source of positives for the main head at late training (NMS-free guarantee).
Losses:
  box:  1 - CIoU on assigned anchors (aux: GIoU, main: CIoU)
  dfl:  softmax divergence toward double-corner targets on assigned anchors
  cls:  BCE with soft target = normalized alignment metric (IoU-aware)
  obj:  BCE on objectness: positives per assignment, background elsewhere.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decode import make_anchors, bbox_iou, dfl_decode
from .matcher import (align_metric, select_candidates, select_candidates_in_gxy,
                      cost_matrix, greedy_hungarian)


def ciou(g, p, eps=1e-9):
    """(M,4) xyxy pairs -> CIoU penalty (1 - CIoU)."""
    x1 = torch.max(g[:, 0], p[:, 0]); y1 = torch.max(g[:, 1], p[:, 1])
    x2 = torch.min(g[:, 2], p[:, 2]); y2 = torch.min(g[:, 3], p[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    agw = (g[:, 2] - g[:, 0]).clamp(min=0); agh = (g[:, 3] - g[:, 1]).clamp(min=0)
    apw = (p[:, 2] - p[:, 0]).clamp(min=0); aph = (p[:, 3] - p[:, 1]).clamp(min=0)
    union = agw * agh + apw * aph - inter
    iou = inter / union.clamp(min=eps)
    ex1 = torch.min(g[:, 0], p[:, 0]); ey1 = torch.min(g[:, 1], p[:, 1])
    ex2 = torch.max(g[:, 2], p[:, 2]); ey2 = torch.max(g[:, 3], p[:, 3])
    cw = (ex2 - ex1).clamp(min=eps); ch = (ey2 - ey1).clamp(min=eps)
    rho2 = ((g[:, 0] + g[:, 2] - p[:, 0] - p[:, 2]) ** 2 +
            (g[:, 1] + g[:, 3] - p[:, 1] - p[:, 3]) ** 2) / 4
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(agw / agh.clamp(min=eps)) -
                                       torch.atan(apw / aph.clamp(min=eps)), 2)
    with torch.no_grad():
        alpha_ = v / (1 - iou + v + eps)
    return 1 - (iou - rho2 / (cw ** 2 + ch ** 2 + eps) - alpha_ * v)


def dfl_loss(pred_dist, target, reg_max, eps=1e-9):
    """pred_dist (K, reg_max) softmax logits; target (K,) continuous [0, reg_max-1]."""
    tl = target.floor().clamp(0, reg_max - 1).long()
    tr = (tl + 1).clamp(0, reg_max - 1)
    wl = (tr - target).clamp(0, 1)
    wr = 1 - wl
    logp = F.log_softmax(pred_dist, dim=-1)
    loss = -(wl * logp.gather(1, tl.view(-1, 1)).squeeze(1) +
             wr * logp.gather(1, tr.view(-1, 1)).squeeze(1))
    return loss.mean()


def _with_tiny_gt_fallback(cand, gboxes, anchors, strides, glabels, eps=1e-9):
    """Ensure every GT keeps at least one candidate anchor.

    GTs smaller than one stride cell can have zero center-inside anchors; for
    those (rows of `cand` all-False) assign the single nearest anchor by
    center distance in stride units. Returns (cand, glabels) unchanged otherwise.
    """
    empty = ~cand.any(dim=1)
    if not empty.any():
        return cand, glabels
    M, N = cand.shape
    gx = (gboxes[:, 0] + gboxes[:, 2]) / 2
    gy = (gboxes[:, 1] + gboxes[:, 3]) / 2
    gxy = torch.stack([gx, gy], 1)                                   # (M,2)
    d = ((gxy[:, None, :] - anchors[None, :, :]) / strides).norm(dim=2)  # (M,N)
    near = d.argmin(dim=1)                                           # (M,)
    cand[empty, near[empty]] = True
    return cand, glabels


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, reg_max=8, strides=(8, 16, 32),
                 box_w=7.5, cls_w=0.5, dfl_w=1.5, obj_w=1.0,
                 o2m_topk=10, alpha=0.5, beta=6.0, o2o_warmup_epochs=0, epoch=0,
                 focal_gamma=2.0, focal_alpha=0.5):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.box_w, self.cls_w, self.dfl_w, self.obj_w = box_w, cls_w, dfl_w, obj_w
        self.o2m_topk = o2m_topk
        self.alpha, self.beta = alpha, beta
        self.o2o_warmup_epochs = o2o_warmup_epochs
        self.epoch = epoch
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def set_epoch(self, e):
        self.epoch = e

    def forward(self, head_out, feats, targets):
        """head_out: per-level dict from Detect.forward (with_aux=True);
        feats: neck outputs (for shapes); targets: list per image of
               dict(boxes (M,4) xyxy pixels, labels (M,) long)."""
        device = feats[0].device
        B = len(targets)
        anchors, strides = make_anchors(feats, self.strides)
        N = anchors.shape[0]

        # flatten predictions
        box_flat, cls_flat, obj_flat, abox_flat, acls_flat = [], [], [], [], []
        for lvl, out in enumerate(head_out):
            mb, mc, mo = out["main"]
            B_, C4, H, W = mb.shape
            box_flat.append(mb.view(B_, 4, self.reg_max, H * W).permute(0, 3, 1, 2).reshape(B_, H * W, 4 * self.reg_max))
            cls_flat.append(mc.view(B_, -1, H * W).permute(0, 2, 1))
            obj_flat.append(mo.view(B_, 1, H * W).permute(0, 2, 1))
            ab, ac = out["aux"]
            abox_flat.append(ab.view(B_, 4, self.reg_max, H * W).permute(0, 3, 1, 2).reshape(B_, H * W, 4 * self.reg_max))
            acls_flat.append(ac.view(B_, -1, H * W).permute(0, 2, 1))
        pred_box = torch.cat(box_flat, 1)      # (B,N,4r)
        pred_cls = torch.cat(cls_flat, 1)      # (B,N,nc)
        pred_obj = torch.cat(obj_flat, 1)      # (B,N,1)
        aux_box = torch.cat(abox_flat, 1)
        aux_cls = torch.cat(acls_flat, 1)

        proj = torch.arange(self.reg_max, dtype=torch.float32, device=device)
        tot = pred_box.new_zeros(())
        stats = {}

        # accumulate target maps for obj
        obj_target = pred_obj.new_full((B, N, 1), 0.0)
        # full BCE cls target over all anchors (pos=one-hot, bg=zeros) so the
        # one-to-one head learns to suppress background (else it fires everywhere)
        cls_target = pred_cls.new_zeros((B, N, self.nc))
        loss_box = pred_box.new_zeros(())
        loss_dfl = pred_box.new_zeros(())
        loss_cls = pred_box.new_zeros(())
        loss_aux = pred_box.new_zeros(())
        n_pos_main = 0
        n_pos_aux = 0

        for bi in range(B):
            gt = targets[bi]
            if gt["boxes"].numel() == 0:
                continue
            gboxes, glabels = gt["boxes"].to(device), gt["labels"].to(device)
            M = gboxes.shape[0]

            pb = pred_box[bi]                      # (N,4r) logits
            pc = pred_cls[bi]                      # (N,nc)
            ab = aux_box[bi]
            ac = aux_cls[bi]

            # decoded pred boxes for matching
            pb_dec = dfl_decode(pb.unsqueeze(0), self.reg_max, proj,
                                anchors, strides.view(-1, 1)).squeeze(0)
            ab_dec = dfl_decode(ab.unsqueeze(0), self.reg_max, proj,
                                anchors, strides.view(-1, 1)).squeeze(0)

            # ---------- one-to-many (aux) ----------
            align_a, iou_a = align_metric(ab_dec, ac, gboxes, glabels, self.alpha, self.beta)
            cand = select_candidates(align_a, iou_a, self.o2m_topk)     # (M,N) bool
            cand = cand & select_candidates_in_gxy(gboxes, anchors, strides.view(-1, 1))
            # Tiny-GT fallback: a GT smaller than one stride cell can have zero
            # center-inside candidates (unsupervised in both heads) - give it its
            # nearest anchor by center distance so it still gets box/cls gradient.
            cand, _ = _with_tiny_gt_fallback(
                cand, gboxes, anchors, strides.view(-1, 1), glabels)
            align_norm = (align_a / (align_a.amax(dim=1, keepdim=True).clamp(min=1e-6))).detach()
            # aux positives: for each GT, its candidate anchors (dedup across GTs: keep best)
            flat_scores = torch.where(cand, align_norm, align_norm.new_full((), -1.0))
            top_vals, top_idx = flat_scores.max(dim=0)                   # (N,)
            pos_a = top_vals > 0                                         # anchors chosen by some GT
            gt_idx_a = torch.full((N,), -1, dtype=torch.long, device=device)
            gt_idx_a[pos_a] = top_idx[pos_a].clamp(0, M - 1)

            if pos_a.any():
                n_pos_aux += int(pos_a.sum())
                apb = ab_dec[pos_a]
                agb = gboxes[gt_idx_a[pos_a]]
                loss_aux = loss_aux + (1 - bbox_iou(agb, apb)).sum()
                # DFL on aux positives
                dist_t = self._dist_targets(agb, anchors[pos_a], strides.view(-1, 1)[pos_a])
                d_raw = ab[pos_a].view(-1, 4, self.reg_max)
                loss_aux = loss_aux + sum(dfl_loss(d_raw[:, k], dist_t[:, k], self.reg_max)
                                          for k in range(4))
                # soft cls targets from alignment (GT x its assigned anchors)
                aux_anchors = torch.nonzero(pos_a).squeeze(1)
                aux_gts = gt_idx_a[aux_anchors]
                tgt = align_norm[aux_gts, aux_anchors].clamp(0, 1)
                logp = F.logsigmoid(ac[pos_a])
                loss_aux = loss_aux + (-(tgt * logp.gather(1, glabels[aux_gts].view(-1, 1)).squeeze(1))).sum()

            # ---------- one-to-one (main) ----------
            # YOLOv10-style: initial o2o pick is the TOP-1 o2m candidate by the
            # task-aligned metric (IoU^beta * cls^alpha - dominated by IoU, so
            # targets are spatially stable). Only colliding GTs (two GTs wanting
            # the same anchor) are re-resolved by greedy Hungarian on the aligned
            # cost. Pure cost-matrix assignment (v31) is unstable early: the cls
            # BCE term dominates, cost is near-uniform over candidates, the pick
            # jumps every epoch and the box head chases a moving target
            # (observed: box=9.4 flat, best-pred-IoU 0.09).
            best_vals, best_anchor = align_a.masked_fill(~cand, -1e9).max(dim=1)  # (M,)
            valid = best_vals > -1e8
            fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
            gt_of_anchor = torch.full((N,), -1, dtype=torch.long, device=device)
            if valid.any():
                sel_gts = torch.nonzero(valid).squeeze(1)
                sel_anchors = best_anchor[sel_gts]
                # collision check: same anchor chosen by >1 GT
                uniq, counts = sel_anchors.unique(return_counts=True)
                dup = uniq[counts > 1]
                if dup.numel():
                    from .matcher import cost_matrix, greedy_hungarian
                    coll_gts = [gi for gi, a in zip(sel_gts.tolist(), sel_anchors.tolist())
                                if a in dup.tolist()]
                    coll = torch.tensor(coll_gts, dtype=torch.long, device=device)
                    sub = cost_matrix(align_a[coll], iou_a[coll], ac, glabels[coll],
                                      self.alpha, self.beta)
                    sub = sub[:, dup]                                  # (K, n_dup)
                    sub = sub.masked_fill(~cand[coll][:, dup], 1e9)
                    sub_assign = greedy_hungarian(sub)                 # local anchor idx
                    a_idx = torch.tensor(dup.tolist(), dtype=torch.long, device=device)
                    keep_local = sub_assign >= 0
                    # GTs that lost the collision fall back to their next-best
                    # non-colliding candidate
                    for k, gi in enumerate(coll.tolist()):
                        if keep_local[k]:
                            a = a_idx[sub_assign[k]]
                            sel_anchors[sel_gts == gi] = a
                        else:
                            masked = align_a[gi].clone()
                            masked[dup] = -1e9
                            masked[sel_anchors[sel_gts != gi]] = -1e9  # avoid other picks
                            if masked.max() > -1e8:
                                sel_anchors[sel_gts == gi] = masked.argmax()
                fg_mask[sel_anchors] = True
                gt_of_anchor[sel_anchors] = sel_gts
            n_pos_main += int(fg_mask.sum())

            if fg_mask.any():
                fg_anchors = torch.nonzero(fg_mask).squeeze(1)
                fg_gts = gt_of_anchor[fg_anchors]
                mpb = pb_dec[fg_anchors]
                mgb = gboxes[fg_gts]
                loss_box = loss_box + ciou(mgb, mpb).sum()
                dist_t = self._dist_targets(mgb, anchors[fg_anchors], strides.view(-1, 1)[fg_anchors])
                d_raw = pb[fg_anchors].view(-1, 4, self.reg_max)
                loss_dfl = loss_dfl + sum(dfl_loss(d_raw[:, k], dist_t[:, k], self.reg_max)
                                          for k in range(4))
                # one-to-one main head: cls target is QUALITY-AWARE (v45, YOLOv8/
                # TOOD style): the assigned anchor's normalized alignment metric
                # (IoU^beta-dominated, detached) instead of hard 1.0. A background
                # anchor can only reach a high cls score by producing a
                # high-quality box - photo-texture anchors (the v44 failure mode:
                # 1247/1934 bg anchors >0.5 with hard targets) cannot fake box
                # quality they don't have.
                lbl = glabels[fg_gts]
                q = align_norm[fg_gts, fg_anchors].clamp(0, 1)   # detached
                cls_target[bi, fg_anchors, lbl] = q
                obj_target[bi, fg_anchors, 0] = 1.0

        # cls: balanced BCE over ALL anchors with quality-aware soft targets
        # (v45). v44 (hard 1.0 targets, balanced BCE) converged (cls 0.048, pos
        # max 0.996, IoU 0.872) but 1247/1934 photo-texture bg anchors kept
        # p>0.5 - mAP capped at 0.505. Soft targets make the score box-quality
        # aware: bg anchors can't fake an IoU they can't produce.
        logits = pred_cls.view(B, N, self.nc)
        bce = F.binary_cross_entropy_with_logits(logits, cls_target, reduction="none")
        anchor_is_pos = cls_target.sum(dim=2) > 0                       # (B,N)
        pos_mask = anchor_is_pos.unsqueeze(2).expand_as(cls_target)
        n_pos = max(1, int(anchor_is_pos.sum()) * self.nc)
        n_neg = max(1, int((~anchor_is_pos).sum()) * self.nc)
        loss_cls = (bce[pos_mask].sum() / n_pos +
                    bce[~pos_mask].sum() / n_neg)

        # obj: balanced BCE over all anchors - pos=1 on o2o positives, neg=0 on
        # background, each normalized by its own count. The obj branch is SEPARATE
        # from cls/box (own conv weights), so this gives it real gradient to fire
        # on positives AND suppress background without freezing box/cls. Used at
        # inference (score = cls * obj) as the background-suppression signal.
        obj_logits = pred_obj.view(B, N)
        obj_bce = F.binary_cross_entropy_with_logits(obj_logits, obj_target.view(B, N),
                                                     reduction="none")
        obj_pos = obj_target.view(B, N) > 0
        n_obj_pos = max(1, int(obj_pos.sum()))
        n_obj_neg = max(1, int((~obj_pos).sum()))
        loss_obj = obj_bce[obj_pos].sum() / n_obj_pos + obj_bce[~obj_pos].sum() / n_obj_neg

        loss_box = self.box_w * loss_box / max(n_pos_main, 1)
        loss_dfl = self.dfl_w * loss_dfl / max(n_pos_main, 1)
        loss_cls = self.cls_w * loss_cls
        loss_obj = self.obj_w * loss_obj
        n_imgs_pos = max(1, sum(1 for t in targets if t["boxes"].numel() > 0))
        # aux is summed over ~500 dense positives - normalize by its own count so
        # it does not dwarf the o2o head (v26: aux=100 vs box=3.1, 30x, starving
        # the o2o head that is used at inference).
        loss_aux = self.box_w * loss_aux / max(n_pos_aux, 1)

        tot = loss_box + loss_dfl + loss_cls + loss_obj + loss_aux
        stats = {"loss": tot.detach(), "box": loss_box.detach(), "dfl": loss_dfl.detach(),
                 "cls": loss_cls.detach(), "obj": loss_obj.detach(), "aux": loss_aux.detach(),
                 "pos_main": float(n_pos_main)}
        return tot, stats

    def _dist_targets(self, gb, anchors, strides):
        """gt (K,4) xyxy pixels -> ltrb distances in stride units, clamped to reg_max-1."""
        l = (anchors[:, 0] - gb[:, 0]) / strides[:, 0]
        t = (anchors[:, 1] - gb[:, 1]) / strides[:, 0]
        r = (gb[:, 2] - anchors[:, 0]) / strides[:, 0]
        b = (gb[:, 3] - anchors[:, 1]) / strides[:, 0]
        return torch.stack([l, t, r, b], dim=1).clamp(0, self.reg_max - 1)