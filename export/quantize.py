"""INT8 static quantization of the exported ONNX via ONNX Runtime QDQ.

Calibration uses random tensors by default (sufficient for latency tables); pass
--calib-images to quantize with real preprocessing statistics instead.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def quantize_int8(onnx_path, out_path, img_size=640, calib_images=None, num_calib=64):
    from onnxruntime.quantization import QuantType, quantize_static, QuantFormat, CalibrationDataReader
    import numpy as np

    class Reader:
        def __init__(self, tensors):
            self.tensors = tensors
            self.i = 0

        def get_next(self):
            if self.i >= len(self.tensors):
                return None
            t = self.tensors[self.i]
            self.i += 1
            return {"images": t}

    rng = np.random.default_rng(0)
    tensors = []
    if calib_images:
        from export.decode_onnx import preprocess
        import cv2
        files = [os.path.join(calib_images, f) for f in sorted(os.listdir(calib_images))
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))][:num_calib]
        for f in files:
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            x, *_ = preprocess(img, img_size)
            tensors.append(x)
    if not tensors:
        tensors = [rng.random((1, 3, img_size, img_size), dtype=np.float32)
                   for _ in range(num_calib)]

    quantize_static(
        model_input=onnx_path,
        model_output=out_path,
        calibration_data_reader=Reader(tensors),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    mb = os.path.getsize(out_path) / 1e6
    print(f"quantized -> {out_path} ({mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--calib-images", default=None)
    args = ap.parse_args()
    out = args.out or args.onnx.replace(".onnx", "_int8.onnx")
    quantize_int8(args.onnx, out, args.img_size, args.calib_images)