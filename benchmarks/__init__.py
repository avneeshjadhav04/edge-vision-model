"""Benchmarks package root."""
from .bench_runtime import bench_ort, bench_openvino, bench_ncnn

__all__ = ["bench_ort", "bench_openvino", "bench_ncnn"]