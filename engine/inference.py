"""Shared evaluation runner: model -> letterboxed batch -> rescale -> per-dataset AP."""
import torch

from data.common import collate_batch
from models.decode import Postprocessor


@torch.no_grad()
def run_inference(model, loader, device, img_size=640, score_thresh=0.01, max_det=300):
    """Runs the NMS-free postprocess on a dataset loader.
    Returns (predictions, targets_in_original_coords)."""
    model.eval()
    preds_out, tgts_out = [], []
    from models.decode import _anchors_from_shapes, dfl_decode
    proj = model.dfl_proj
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        raw = model(imgs, with_aux=False)
        # decode all images in batch (vectorized)
        shapes = [(int(b.shape[2]), int(b.shape[3])) for (b, _, _) in raw]
        anchors, strides = _anchors_from_shapes(shapes, model.strides, imgs.device)
        box_flat, cls_flat, obj_flat = [], [], []
        for (bx, cl, ob) in raw:
            B, _, H, W = bx.shape
            box_flat.append(bx.view(B, 4 * model.reg_max, H * W).permute(0, 2, 1))
            cls_flat.append(cl.view(B, -1, H * W).permute(0, 2, 1))
            obj_flat.append(ob.view(B, 1, H * W).permute(0, 2, 1))
        boxes = dfl_decode(torch.cat(box_flat, 1), model.reg_max, proj, anchors, strides)
        scores = torch.cat(cls_flat, 1).sigmoid() * torch.cat(obj_flat, 1).sigmoid()
        labels_all = scores.argmax(-1)                     # (B,N)
        best = scores.max(-1).values                       # (B,N)
        for bi in range(boxes.shape[0]):
            tgt = targets[bi]
            keep = best[bi] > score_thresh
            bb, ss, ll = boxes[bi][keep], best[bi][keep], labels_all[bi][keep]
            if ss.numel() > max_det:
                topv, topi = ss.topk(max_det)
                bb, ss, ll = bb[topi], topv, ll[topi]
                ll = ll
            else:
                ll = ll
            # rescale to original coords
            r, pw, ph = [float(v) for v in tgt["rescale"]]
            if bb.numel():
                bb = bb.clone()
                bb[:, [0, 2]] = (bb[:, [0, 2]] - pw) / r
                bb[:, [1, 3]] = (bb[:, [1, 3]] - ph) / r
            preds_out.append({"pred_boxes": bb.cpu(), "scores": ss.cpu(), "labels": ll.cpu()})
            tgts_out.append(tgt)
    return preds_out, tgts_out