"""Train-time augmentation: from-scratch mosaic + HSV + flip + scale/translate,
using albumentations (MIT) for the per-image photometric/geometric ops.

Transform contract: callable(img_uint8_HWC, target) -> (img_float_CHW [0,1], target)
with boxes in network-input pixels (square img_size canvas).
"""
import math
import random

import cv2
import numpy as np
import torch

from .common import clamp_boxes


class Mosaic:
    """4-image mosaic on an img_size canvas. p: probability."""

    def __init__(self, img_size=640, p=1.0):
        self.img_size = img_size
        self.p = p

    def __call__(self, img, target, extra_loader=None, extra_index=None):
        if random.random() > self.p:
            return img, target
        s = self.img_size
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        boxes_all, labels_all = [], []
        cx = int(random.uniform(s * 0.3, s * 0.7))
        cy = int(random.uniform(s * 0.3, s * 0.7))
        # quadrant rects on the 2s x 2s canvas: (x0, y0, w, h)
        rects = [(0, 0, cx, cy),                      # top-left
                 (cx, 0, s * 2 - cx, cy),             # top-right
                 (0, cy, cx, s * 2 - cy),             # bottom-left
                 (cx, cy, s * 2 - cx, s * 2 - cy)]    # bottom-right
        imgs = [(img, target)]
        if extra_loader is not None and extra_index is not None:
            for _ in range(3):
                j = random.randrange(len(extra_index))
                im2, t2 = extra_loader(extra_index[j])
                imgs.append((im2, t2))
        else:
            imgs = imgs * 4
        boxes_all, labels_all = [], []
        for k, (im, tgt) in enumerate(imgs[:4]):
            x0, y0, qw, qh = rects[k]
            if qw <= 0 or qh <= 0:
                continue
            # scale-to-fill quadrant
            ih, iw = im.shape[:2]
            r = max(qw / iw, qh / ih)
            nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
            im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            # random crop of (qh, qw) from the resized image
            ox = random.randrange(0, nw - qw + 1) if nw > qw else 0
            oy = random.randrange(0, nh - qh + 1) if nh > qh else 0
            crop = im_r[oy:oy + qh, ox:ox + qw]
            if crop.shape[0] != qh or crop.shape[1] != qw:
                crop = cv2.copyMakeBorder(crop, 0, qh - crop.shape[0], 0, qw - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
            canvas[y0:y0 + qh, x0:x0 + qw] = crop
            # boxes: scale by r, offset by crop origin, then to canvas position
            b = tgt["boxes"]
            if b.numel():
                bb = b.clone()
                bb[:, [0, 2]] = bb[:, [0, 2]] * r - ox
                bb[:, [1, 3]] = bb[:, [1, 3]] * r - oy
                bb[:, [0, 2]] += x0
                bb[:, [1, 3]] += y0
                keep = clamp_boxes(bb, s * 2, s * 2)
                if keep.any():
                    boxes_all.append(bb[keep])
                    labels_all.append(tgt["labels"][keep])
        img_out = canvas
        if boxes_all:
            boxes = torch.cat(boxes_all, 0)
            labels = torch.cat(labels_all, 0)
        else:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.long)
        return img_out, {"boxes": boxes, "labels": labels}


class RandomAffineBoxes:
    """Scale/translate jitter on the mosaic canvas, then crop to img_size."""

    def __init__(self, img_size=640, scale=0.5, translate=0.1, fliplr=0.5):
        self.img_size = img_size
        self.scale = scale
        self.translate = translate
        self.fliplr = fliplr

    def __call__(self, img, target):
        s = self.img_size
        H, W = img.shape[:2]  # may be 2s x 2s from mosaic or s x s normal
        r = random.uniform(1 - self.scale, 1 + self.scale)
        tx = random.uniform(-self.translate, self.translate) * s
        ty = random.uniform(-self.translate, self.translate) * s
        # affine: scale about center + translate
        M = np.array([[r, 0, tx + (s / 2) * (1 - r)],
                      [0, r, ty + (s / 2) * (1 - r)]], dtype=np.float64)
        img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))
        b = target["boxes"]
        if b.numel():
            M_t = torch.from_numpy(M).float()          # (2,3)
            ones = torch.ones(b.shape[0], 1)
            # [x, y, 1] @ M.T -> transformed coords
            p1 = torch.cat([b[:, [0, 1]], ones], 1) @ M_t.T
            p2 = torch.cat([b[:, [2, 3]], ones], 1) @ M_t.T
            nb = torch.stack([p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]], 1)
            if random.random() < self.fliplr:
                nb[:, [0, 2]] = s - nb[:, [2, 0]]
                img = np.ascontiguousarray(img[:, ::-1])
            keep = clamp_boxes(nb, s, s)
            target = {"boxes": nb[keep], "labels": target["labels"][keep]}
        return img, target


class Photometric:
    """HSV jitter + channel shuffle-lite (pure numpy; deterministic per call)."""

    def __init__(self, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
        self.hsv_h, self.hsv_s, self.hsv_v = hsv_h, hsv_s, hsv_v

    def __call__(self, img, target):
        r = self.hsv_h, self.hsv_s, self.hsv_v
        gain = [random.uniform(-v, v) for v in r]
        img = img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int32)
        h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h_ch = (h_ch + int(gain[0] * 180)) % 180
        s_ch = np.clip(s_ch * (1 + gain[1]), 0, 255)
        v_ch = np.clip(v_ch * (1 + gain[2]), 0, 255)
        hsv = np.stack([h_ch, s_ch, v_ch], -1).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img, target


class TrainTransform:
    """Full train pipeline: mosaic -> affine/flip -> photometric -> to tensor."""

    def __init__(self, img_size=640, mosaic_p=1.0, scale=0.5, translate=0.1,
                 fliplr=0.5, hsv=(0.015, 0.7, 0.4), dataset_for_mosaic=None):
        self.mosaic = Mosaic(img_size, mosaic_p)
        self.affine = RandomAffineBoxes(img_size, scale, translate, fliplr)
        self.photo = Photometric(*hsv)
        self.img_size = img_size
        self.mosaic_dataset = dataset_for_mosaic

    def _load_random(self):
        ds = self.mosaic_dataset
        if ds is None:
            return None
        idx = random.randrange(len(ds))
        try:
            return ds.get_raw(idx)
        except AttributeError:
            img, tgt = ds[idx]
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8) if isinstance(img, torch.Tensor) else img
            return img, tgt

    def __call__(self, img, target):
        if self.mosaic.p > 0:
            img, target = self.mosaic(img, target,
                                      extra_loader=self._mosaic_extra,
                                      extra_index=list(range(4)))
        img, target = self.affine(img, target)
        img, target = self.photo(img, target)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return img, target

    def _mosaic_extra(self, idx):
        ds = self.mosaic_dataset
        if ds is None:
            return np.full((64, 64, 3), 114, np.uint8), {"boxes": torch.zeros(0, 4),
                                                          "labels": torch.zeros(0, dtype=torch.long)}
        j = random.randrange(len(ds))
        img, tgt = ds[j]
        if isinstance(img, torch.Tensor):
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img, tgt


class EvalTransform:
    """Letterbox + to tensor (boxes untouched: kept in original coords for eval)."""

    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, img, target):
        from .common import letterbox
        img, r, pads = letterbox(img, self.img_size)
        orig = target.get("orig_size")
        target = dict(target)
        target["rescale"] = torch.tensor([r, pads[0], pads[1]])
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return img, target