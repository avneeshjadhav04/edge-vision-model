"""Shared dataset utilities: letterbox, collate, label format.

Label format throughout the repo: per-image dict(boxes (M,4) xyxy in pixels of the
ORIGINAL image, labels (M,) long, plus optional 'difficult' (M,) bool for VOC).
Datasets return (image_uint8_HWC_rgb, target) before transforms; after transforms
the image is a float tensor CHW in [0,1] and boxes are in network-input pixels.
"""
import numpy as np
import torch


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize with aspect preservation + padding. img: np.uint8 HWC.
    Returns (out, scale, (pad_w, pad_h)) where out is np.uint8 HWC."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pw, ph = new_shape - nw, new_shape - nh
    dw, dh = pw / 2, ph / 2
    import cv2
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def scale_boxes(boxes, r, pads, orig_shape, net_shape):
    """Map boxes from network-input coords back to original image coords."""
    boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    pads_w, pads_h = pads
    if isinstance(boxes, torch.Tensor):
        boxes[:, [0, 2]] -= pads_w
        boxes[:, [1, 3]] -= pads_h
        boxes[:, [0, 2]] /= r
        boxes[:, [1, 3]] /= r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_shape[0])
    return boxes


def collate_batch(batch):
    """Default collate for variable box counts: keeps list of targets."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    return imgs, targets


def clamp_boxes(boxes, w, h):
    """Clamp xyxy boxes to image bounds; drops degenerate boxes, returns keep mask."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    boxes[:, 0] = boxes[:, 0].clamp(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, h - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, w - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, h - 1)
    keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    return keep