"""Live webcam demo: ONNX Runtime CPU inference, NMS-free, annotated feed.

    python -m benchmarks.webcam_demo --onnx runs/export/evm_nano.onnx \
        --num-classes 80 --names coco [--cam 0] [--score 0.35]
Press 'q' to quit.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def draw(img, boxes, scores, labels, names):
    import cv2
    for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 200, 0), 2)
        name = names[int(l)] if int(l) < len(names) else str(int(l))
        cv2.putText(img, f"{name} {s:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--names", default=None, help="comma-separated class names "
                    "(defaults: coco or voc based on --num-classes)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()

    if args.names:
        names = args.names.split(",")
    elif args.num_classes == 20:
        from data.voc import VOC_CLASSES as names
    else:
        from data.coco_names import COCO_CLASSES as names

    import cv2
    import onnxruntime as ort
    from export.decode_onnx import decode_outputs, preprocess, rescale

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("cannot open camera", args.cam)
        sys.exit(1)

    fps_t = time.perf_counter()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, r, pads, orig = preprocess(rgb, args.img_size)
        raw = sess.run(None, {inp: x})
        dets = decode_outputs(raw, args.img_size, num_classes=args.num_classes,
                              score_thresh=args.score, max_det=args.max_det)[0]
        bb = rescale(dets["pred_boxes"], r, pads, orig) if dets["pred_boxes"].size else dets["pred_boxes"]
        now = time.perf_counter()
        inst = 1.0 / max(now - fps_t, 1e-6)
        fps = fps * 0.9 + inst * 0.1
        fps_t = now
        draw(frame, bb, dets["scores"], dets["labels"], list(names))
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.imshow("EVM nano (NMS-free)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()