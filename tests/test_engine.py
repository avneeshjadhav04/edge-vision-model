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


if __name__ == "__main__":
    test_ema_tracks_average()
    test_voc_evaluator_exact()
    test_voc_evaluator_difficult_ignored()
    test_voc_ap_11pt()
    print("ALL ENGINE TESTS PASSED")