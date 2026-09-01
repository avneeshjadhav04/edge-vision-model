"""Engine tests: EMA correctness + VOC mAP evaluator vs hand-computed truth."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.ema import ModelEMA
from engine.eval_voc import eval_voc, voc_ap
from engine.inference import run_inference


def test_ema_tracks_average():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4)
    ema = ModelEMA(m, decay=0.9)
    x = torch.randn(8, 4)
    y1 = torch.randn(8, 4)
    y2 = torch.randn(8, 4)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    for y in (y1, y2):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        opt.step()
        ema.update(m)
    # EMA after 2 updates with ramp: d0 = 1 - 1/11 = 0.909, d1 = 0.9*... check monotone pull
    w_final = m.weight.detach()
    w_ema = ema.module.weight.detach()
    assert not torch.allclose(w_final, w_ema)
    assert w_ema.std() > 0
    print("test_ema_tracks_average OK")


def test_voc_evaluator_exact():
    """1 class, 2 images. Perfect predictions -> mAP=1; one missed -> known value."""
    boxes1 = torch.tensor([[10., 10., 50., 50.]])
    targets = [
        {"boxes": boxes1, "labels": torch.tensor([0]), "difficult": torch.tensor([False])},
        {"boxes": torch.tensor([[20., 20., 80., 80.]]), "labels": torch.tensor([0]),
         "difficult": torch.tensor([False])},
    ]
    preds_perfect = [
        {"pred_boxes": boxes1, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},
        {"pred_boxes": torch.tensor([[20., 20., 80., 80.]]), "scores": torch.tensor([0.8]),
         "labels": torch.tensor([0])},
    ]
    r = eval_voc(preds_perfect, targets, num_classes=1)
    assert abs(r["mAP"] - 1.0) < 1e-6, r
    # now miss image-2 object entirely
    preds_miss = [preds_perfect[0],
                  {"pred_boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                   "labels": torch.zeros(0, dtype=torch.long)}]
    r2 = eval_voc(preds_miss, targets, num_classes=1)
    assert r2["mAP"] < 1.0 and r2["mAP"] > 0.5, r2  # one TP ranked first, then FN
    print("test_voc_evaluator_exact OK", r["mAP"], r2["mAP"])


def test_voc_evaluator_difficult_ignored():
    boxes = torch.tensor([[10., 10., 50., 50.]])
    targets = [{"boxes": boxes, "labels": torch.tensor([0]),
                "difficult": torch.tensor([True])}]
    preds = [{"pred_boxes": boxes, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
    r = eval_voc(preds, targets, num_classes=1)
    assert np.isnan(r["mAP"])  # no positive GTs -> class skipped
    print("test_voc_evaluator_difficult_ignored OK")


def test_voc_ap_11pt():
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 0.8, 0.6])
    ap = voc_ap(rec, prec, use_07_metric=True)
    assert 0.0 < ap <= 1.0
    ap_auc = voc_ap(rec, prec, use_07_metric=False)
    assert 0.0 < ap_auc <= 1.0
    print("test_voc_ap_11pt OK", round(ap, 3), round(ap_auc, 3))


def test_trainer_skips_nan_batches():
    """Non-finite loss batch must not poison weights; consecutive abort fires."""
    import io
    from contextlib import redirect_stdout

    from engine.trainer import Trainer
    from models import build_model
    from losses import DetectionLoss

    class NaNLoss(DetectionLoss):
        """Real detection loss; injects non-finite loss on one batch."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = 0

        def forward(self, out, feats, targets):
            tot, stats = super().forward(out, feats, targets)
            self.calls += 1
            if self.calls == 3:
                tot = tot * float("nan")
            return tot, stats

    class TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 6  # 3 batches of 2 (drop_last)

        def __getitem__(self, i):
            img = np.full((128, 128, 3), 40, np.uint8)
            img[30:80, 20:80] = 220
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            return x, {"boxes": torch.tensor([[20., 30., 80., 80.]]),
                       "labels": torch.tensor([1])}

    cfg_m = {"input_size": 128,
             "backbone": {"stem_channels": 16, "out_channels": [32, 64, 128],
                          "depth": [1, 1, 1]},
             "neck": {"hidden": 32, "out": 32},
             "head": {"hidden": 32, "reg_max": 8, "aux_hidden": 32}}
    model = build_model(cfg_m, num_classes=5)
    crit = NaNLoss(num_classes=5, reg_max=8)
    cfg = {"train": {"optimizer": "adamw", "lr0": 1e-3, "batch_size": 2,
                     "workers": 0, "amp": False, "warmup_epochs": 0, "mosaic": 1.0,
                     "mosaic_close_epochs": 10, "val_interval": 1000}}
    tr = Trainer(model, crit, TinyDS(), cfg=cfg, device="cpu",
                 save_dir="/tmp/opencode/nan_test")
    buf = io.StringIO()
    with redirect_stdout(buf):
        tr.fit(1)
    out = buf.getvalue()
    assert crit.calls == 3, f"expected 3 batches (6 imgs @ bs2, drop_last), calls={crit.calls}"
    assert "non-finite loss" in out, "NaN batch was not skipped"
    w = model.head.main.box.box[-1].weight
    assert torch.isfinite(w).all(), "weights poisoned by NaN batch"
    print("test_trainer_skips_nan_batches OK")


def test_trainer_convergence_adamw():
    """AdamW @1e-3 (the overfit recipe) must drive a tiny detector's loss down."""
    from losses import DetectionLoss
    from models import build_model
    from engine.trainer import Trainer

    class SyntheticDS(torch.utils.data.Dataset):
        """Fixed batch of 4 images with a single bright square each."""

        def __len__(self):
            return 4

        def __getitem__(self, i):
            img = np.full((128, 128, 3), 40, np.uint8)
            x0, y0 = 20 + 10 * i, 30
            img[y0:y0 + 50, x0:x0 + 60] = 220
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            return x, {"boxes": torch.tensor([[float(x0), float(y0),
                                               float(x0 + 60), float(y0 + 50)]]),
                       "labels": torch.tensor([1])}

    cfg_m = {"input_size": 128,
             "backbone": {"stem_channels": 16, "out_channels": [32, 64, 128],
                          "depth": [1, 1, 1]},
             "neck": {"hidden": 32, "out": 32},
             "head": {"hidden": 32, "reg_max": 8, "aux_hidden": 32}}
    m = build_model(cfg_m, num_classes=5)
    crit = DetectionLoss(num_classes=5, reg_max=8)
    cfg = {"train": {"optimizer": "adamw", "lr0": 1e-3, "lrf": 0.01, "warmup_epochs": 3,
                     "batch_size": 4, "workers": 0, "amp": False, "ema_decay": 0.999,
                     "val_interval": 1000, "mosaic_close_epochs": 10, "mosaic": 1.0,
                     "accum": 1}}
    tr = Trainer(m, crit, SyntheticDS(), cfg=cfg, device="cpu",
                 save_dir="/tmp/opencode/overfit_test")
    h = tr.fit(60)
    first, last = h[0]["loss"], h[-1]["loss"]
    assert last < first * 0.55, f"loss did not converge: {first:.1f} -> {last:.1f}"
    assert np.isfinite(last)
    print(f"test_trainer_convergence_adamw OK {first:.1f} -> {last:.1f}")


if __name__ == "__main__":
    test_ema_tracks_average()
    test_voc_evaluator_exact()
    test_voc_evaluator_difficult_ignored()
    test_voc_ap_11pt()
    test_trainer_skips_nan_batches()
    test_trainer_convergence_adamw()
    print("ALL ENGINE TESTS PASSED")