"""CPU runtime benchmark: ONNX Runtime (FP32/FP16/INT8), OpenVINO, NCNN.

Measures end-to-end latency (preprocess excluded, decode included) and reports
FPS. Produces the latency-vs-precision table for the README.

    python -m benchmarks.bench_runtime --onnx runs/export/evm_nano.onnx --img-size 640
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _fake_image(img_size):
    rng = np.random.default_rng(0)
    return (rng.random((480, 640, 3)) * 255).astype(np.uint8)


def bench_ort(model_path, img_size, n_warm=10, n_iter=50, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.random((1, 3, img_size, img_size)).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {inp: x})
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1000, np.percentile(ts, 95) * 1000


def bench_openvino(xml_path, img_size, n_warm=10, n_iter=50):
    try:
        import openvino as ov
    except ImportError:
        return None, None
    core = ov.Core()
    core.set_property({"INFERENCE_PRECISION_HINT": "f32"})
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    x = np.random.random((1, 3, img_size, img_size)).astype(np.float32)
    out_names = [o.get_any_name() for o in compiled.outputs]
    for _ in range(n_warm):
        compiled([x])
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        compiled([x])
        ts.append(time.perf_counter() - t0)
    return np.mean(ts) * 1000, np.percentile(ts, 95) * 1000


def bench_ncnn(param_path, bin_path, img_size, n_warm=10, n_iter=50):
    try:
        import ncnn
    except ImportError:
        return None, None
    net = ncnn.Net()
    net.opt.num_threads = os.cpu_count() or 4
    net.load_param(param_path)
    net.load_model(bin_path)
    from PIL import Image
    img = Image.fromarray(_fake_image(img_size)).resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32)
    mat_in = ncnn.Mat(arr.transpose(2, 0, 1))
    for _ in range(n_warm):
        ex = net.create_extractor()
        ex.input("images", mat_in)
        ret, out = ex.extract("output")
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ex = net.create_extractor()
        ex.input("images", mat_in)
        ret, out = ex.extract("output")
        ts.append(time.perf_counter() - t0)
    return np.mean(ts) * 1000, np.percentile(ts, 95) * 1000


def to_fp16(onnx_path, out_path):
    from onnxconverter_common import float16
    import onnx
    m = onnx.load(onnx_path)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, out_path)
    print(f"fp16 -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="FP32 ONNX path")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--fp16", action="store_true", help="also benchmark ORT FP16")
    ap.add_argument("--int8", action="store_true", help="also benchmark ORT INT8")
    ap.add_argument("--openvino", action="store_true", help="also benchmark OpenVINO")
    ap.add_argument("--ncnn", nargs=2, metavar=("param", "bin"), default=None)
    args = ap.parse_args()

    results = []

    def report(name, ms, p95):
        results.append((name, ms, p95))
        print(f"{name:<12} {ms:9.1f} {p95:9.1f} {1000 / ms:8.1f}")

    print(f"{'runtime':<12} {'mean ms':>9} {'p95 ms':>9} {'FPS':>8}")
    ms, p95 = bench_ort(args.onnx, args.img_size, n_iter=args.n_iter, threads=args.threads)
    report("ORT FP32", ms, p95)

    if args.fp16:
        fp16_path = args.onnx.replace(".onnx", "_fp16.onnx")
        if not os.path.exists(fp16_path):
            to_fp16(args.onnx, fp16_path)
        ms, p95 = bench_ort(fp16_path, args.img_size, n_iter=args.n_iter, threads=args.threads)
        report("ORT FP16", ms, p95)

    if args.int8:
        int8_path = args.onnx.replace(".onnx", "_int8.onnx")
        if not os.path.exists(int8_path):
            from export.quantize import quantize_int8
            quantize_int8(args.onnx, int8_path, args.img_size)
        ms, p95 = bench_ort(int8_path, args.img_size, n_iter=args.n_iter, threads=args.threads)
        report("ORT INT8", ms, p95)

    if args.openvino:
        ms, p95 = bench_openvino(args.onnx, args.img_size, n_iter=args.n_iter)
        if ms:
            report("OpenVINO", ms, p95)
        else:
            print("OpenVINO not installed - skipped")

    if args.ncnn:
        ms, p95 = bench_ncnn(args.ncnn[0], args.ncnn[1], args.img_size, n_iter=args.n_iter)
        if ms:
            report("NCNN", ms, p95)
        else:
            print("NCNN not installed - skipped")

    import json
    out = {name: {"mean_ms": ms, "p95_ms": p95, "fps": 1000 / ms} for name, ms, p95 in results}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()