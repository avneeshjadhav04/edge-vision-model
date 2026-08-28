# EVM-nano: one-page write-up

## What broke (and how we fixed it)

1. **Greedy one-to-one matcher with cold-start scores.** Early in training, class
   scores are near-zero everywhere, so the task-aligned cost is dominated by the BCE
   term and assignments flip wildly between iterations. Fix: weight the cost by
   normalized alignment (IoU^beta · score^alpha) and rely on IoU>0.2 gating for
   candidates; the o2m aux head carries the early gradient load while the o2o head
   stabilizes.
2. **Soft classification targets indexing bug.** Our first loss gathered alignment
   targets with mismatched (GT, anchor) index semantics — the cls loss silently read
   the wrong cells and stayed at 0 while everything else trained. Caught by a unit
   test that asserts the cls term goes non-zero after a few steps on a fixed batch.
   Lesson: in assignment-based losses, every gather needs a shape test.
3. **Aux head left in the export.** First export attempt traced the aux branch and
   doubled the graph. Fix: `export_model()` deep-copies and strips `head.aux`, and
   the trainer keeps them in separate parameter groups so EMA state stays clean.
4. **torch.onnx.export defaults.** Newer PyTorch (2.13) routes export through a
   dynamo path that choked on our multi-output head; `dynamo=False` restores the
   stable TorchScript tracer. Opset 17 + constant folding kept the graph
   quantization-friendly.
5. **INT8 on a non-VNNI VM was *slower* than FP32** (QDQ DynamicQuantizeLinear
   overhead per conv). This is machine-dependent: on VNNI-capable CPUs (AVX512-VNNI /
   AVX2-VNNI) static INT8 typically wins. We report both honestly rather than hide
   the regression.

## What we learned

- **One-to-one assignment is a training problem, not an inference trick.** The
  NMS-free property is bought at training time; at inference you simply decode.
  The dual-assignment trick (o2m aux + o2o main) is what makes from-scratch
  convergence practical at nano scale.
- **GFLOPs claims need a convention.** We quote GMac (multiply-accumulates, the
  thop convention most detector papers use). 5.6 GMac ≈ 11 "FLOPs" — always say
  which one you mean.
- **Evaluators are easy to get subtly wrong.** Difficult-box handling in VOC
  (detections on difficult GT must be *ignored*, not counted as FP) changed our
  sanity numbers; pycocotools is the right reference for COCO and we bridge to it
  instead of reimplementing COCO AP.
- **4GB GPUs train nano models fine** — AMP + EMA + checkpoint-resume fit VOC at
  batch 32 and COCO at batch 64 @640; the bottleneck is wall-clock, not memory.

## Why NMS-free matters on CPU

NMS is a data-dependent, sequential, branchy operation: per-class sorting, IoU
loops, dynamic output shapes. On CPU it is (a) latency-irregular under load,
(b) hostile to static-shape graph optimization and INT8 rounding of the score
pipeline, and (c) hard to batch. Removing it means:

- the exported ONNX graph is a clean feed-forward net (constant shapes per level);
- decode is a handful of vectorized numpy ops (softmax + gather + top-k);
- end-to-end latency becomes *predictable* — which matters more than the mean for
  edge video (webcam at 25–30 FPS must not stutter);
- one less hyperparameter (IoU threshold) to tune per dataset.

## Honest limitations

- 2.4M params / 5.6 GMac at 640px is a *very* small budget for COCO; without
  distillation or longer schedules, matching YOLO26n's 40.9 is not guaranteed.
  We commit to reporting whatever mAP the reproducible run produces.
- Greedy Hungarian (mutual-exclusion lowest-cost-first) is an approximation of the
  optimal assignment; on heavily crowded frames it can leave a GT unmatched that
  optimal matching would cover. The cost shows up as slightly lower recall on dense
  scenes, not duplicate boxes.
- INT8 gains are CPU-generation dependent; we quantify, not hand-wave.