"""ONNX export of the deployable model (aux head stripped, pure decode outside graph).

The exported graph: input image (1,3,H,W) -> per-level (box_dist, cls_logits, obj).
Decode (DFL + anchors + top-k) stays in numpy/torch on the host, keeping the ONNX
graph INT8-quantization-friendly (conv/relu/add only).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_model, export_model
from scripts.common import load_config


def export_onnx(weights, out_path, num_classes, img_size=640, config="model_nano",
                simplify=True, dynamic_batch=False, opset=17, model=None):
    if model is None:
        cfg = load_config(config)
        model = build_model(cfg, num_classes=num_classes)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        state = sd.get("model", sd)
        model.load_state_dict(state, strict=True)
    deploy = export_model(model)  # strips aux head, sets eval
    deploy.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        deploy, dummy, out_path,
        input_names=["images"],
        output_names=["box_l1", "cls_l1", "obj_l1", "box_l2", "cls_l2", "obj_l2", "box_l3", "cls_l3", "obj_l3"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch"}} if dynamic_batch else None,
        do_constant_folding=True,
        dynamo=False,
    )
    if simplify:
        try:
            import onnx
            import onnxsim
            m = onnx.load(out_path)
            m, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m, out_path)
                print("simplified OK")
        except Exception as e:
            print(f"(simplify skipped: {e})")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"exported {out_path} ({size_mb:.1f} MB, opset {opset})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="runs/export/evm_nano.onnx")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--config", default="model_nano")
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_onnx(args.weights, args.out, args.num_classes, args.img_size, args.config,
                simplify=not args.no_simplify)