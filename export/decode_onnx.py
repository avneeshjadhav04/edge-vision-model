"""Host-side decode of raw ONNX outputs -> detections (numpy, NMS-free)."""
import numpy as np


def make_anchors_np(shape_hw, stride):
    H, W = shape_hw
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    ax = (xs.reshape(-1) + 0.5) * stride
    ay = (ys.reshape(-1) + 0.5) * stride
    return np.stack([ax, ay], 1)


def decode_outputs(raw, img_size=640, strides=(8, 16, 32), reg_max=8,
                   num_classes=80, score_thresh=0.25, use_obj=True, max_det=300):
    """raw: 9 arrays (box_l*, cls_l*, obj_l*) each (B, C, H, W).
    Returns list per image of dict(boxes xyxy, scores, labels)."""
    boxes = raw[0::3]
    clss = raw[1::3]
    objs = raw[2::3]
    proj = np.arange(reg_max, dtype=np.float32)
    anchors_all, strides_all = [], []
    for (b, s) in zip(boxes, strides):
        anchors_all.append(make_anchors_np(b.shape[2:], s))
        strides_all.append(np.full((b.shape[2] * b.shape[3], 1), s, np.float32))
    anchors = np.concatenate(anchors_all, 0)         # (N,2)
    strides_n = np.concatenate(strides_all, 0)       # (N,1)

    results = []
    B = boxes[0].shape[0]
    for bi in range(B):
        bb = np.concatenate([b[bi].reshape(4 * reg_max, -1).T for b in boxes], 0)   # (N,4r)
        cc = np.concatenate([c[bi].reshape(c.shape[1], -1).T for c in clss], 0)     # (N,nc)
        oo = np.concatenate([o[bi].reshape(1, -1).T for o in objs], 0)              # (N,1)
        d = bb.reshape(-1, 4, reg_max)
        d = np.exp(d - d.max(-1, keepdims=True))
        d = d / d.sum(-1, keepdims=True)
        dist = (d * proj).sum(-1) * strides_n                                        # (N,4)
        l, t, r, bo = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        xyxy = np.stack([anchors[:, 0] - l, anchors[:, 1] - t,
                         anchors[:, 0] + r, anchors[:, 1] + bo], 1)
        s_cls = 1.0 / (1.0 + np.exp(-cc))
        s = s_cls * (1.0 / (1.0 + np.exp(-oo))) if use_obj else s_cls
        labels = s.argmax(1)
        best = s.max(1)
        keep = best > score_thresh
        xyxy, best, labels = xyxy[keep], best[keep], labels[keep]
        if best.size > max_det:
            top = np.argsort(-best)[:max_det]
            xyxy, best, labels = xyxy[top], best[top], labels[top]
        results.append({"pred_boxes": xyxy, "scores": best, "labels": labels})
    return results


def preprocess(img_rgb, img_size=640):
    """uint8 HWC -> float32 CHW normalized, with letterbox params for post-rescale."""
    import cv2
    h, w = img_rgb.shape[:2]
    r = min(img_size / h, img_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_rgb, (nw, nh))
    canvas = np.full((img_size, img_size, 3), 114, np.uint8)
    pw, ph = (img_size - nw) // 2, (img_size - nh) // 2
    canvas[ph:ph + nh, pw:pw + nw] = resized
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, (pw, ph), (h, w)


def rescale(xyxy, r, pads, orig_hw):
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pads[0]) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - pads[1]) / r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_hw[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_hw[0])
    return out