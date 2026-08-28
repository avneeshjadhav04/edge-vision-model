"""Unit tests: matcher + loss + model forward/decode shapes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_model, count_params, make_anchors, dfl_decode, bbox_iou
from losses import DetectionLoss, align_metric, select_candidates, cost_matrix, greedy_hungarian


def _tiny_model(nc=5):
    cfg = {
        "input_size": 320,
        "backbone": {"stem_channels": 16, "out_channels": [32, 64, 128], "depth": [1, 1, 1]},
        "neck": {"hidden": 32, "out": 32},
        "head": {"reg_max": 8, "aux_hidden": 32},
    }
    return build_model(cfg, num_classes=nc)


def test_model_forward_shapes():
    m = _tiny_model()
    m.train()
    x = torch.randn(2, 3, 320, 320)
    feats = m.neck(*m.backbone(x))
    assert [f.shape[-1] for f in feats] == [40, 20, 10]
    out = m.head(feats, with_aux=True)
    assert len(out) == 3
    mb, mc, mo = out[0]["main"]
    assert mb.shape == (2, 32, 40, 40) and mc.shape == (2, 5, 40, 40) and mo.shape == (2, 1, 40, 40)
    params = count_params(m)
    assert params["deployable"] < params["total"]  # aux excluded from deploy count
    print("test_model_forward_shapes OK", params)


def test_decode_and_iou():
    m = _tiny_model()
    m.eval()
    x = torch.randn(1, 3, 320, 320)
    res = m.predict(x, score_thresh=0.01, max_det=50)
    assert set(res[0].keys()) == {"pred_boxes", "scores", "labels"}
    assert res[0]["pred_boxes"].shape[0] <= 50
    b1 = torch.tensor([[0., 0., 10., 10.]])
    b2 = torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.]])
    iou = bbox_iou(b1, b2)
    assert torch.isclose(iou[0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(iou[1], torch.tensor(50.0 / 150.0), atol=1e-6)
    print("test_decode_and_iou OK")


def test_matcher_one_to_one():
    torch.manual_seed(0)
    N, M, nc = 200, 4, 5
    pred_boxes = torch.rand(N, 4) * 100
    pred_boxes[:, 2:] += pred_boxes[:, :2] + 5
    pred_cls = torch.randn(N, nc)
    gt_boxes = torch.rand(M, 4) * 80
    gt_boxes[:, 2:] += gt_boxes[:, :2] + 10
    gt_cls = torch.randint(0, nc, (M,))
    align, iou = align_metric(pred_boxes, pred_cls, gt_boxes, gt_cls)
    assert align.shape == (M, N) and iou.shape == (M, N)
    cand = select_candidates(align, iou, topk=10)
    assert cand.shape == (M, N) and cand.sum() > 0
    cost = cost_matrix(align, iou, pred_cls, gt_cls)
    assign = greedy_hungarian(cost)
    assert assign.shape == (M,)
    valid = assign[assign >= 0]
    assert valid.numel() == torch.unique(valid).numel()  # strictly one anchor per GT
    print("test_matcher_one_to_one OK (assigned", int(valid.numel()), "of", M, ")")


def test_loss_decreases_on_fixed_batch():
    """Overfit a single fixed batch for a few steps: total loss must decrease."""
    torch.manual_seed(0)
    m = _tiny_model(nc=3)
    m.train()
    crit = DetectionLoss(num_classes=3, reg_max=8)
    x = torch.randn(2, 3, 320, 320)
    targets = [
        {"boxes": torch.tensor([[40., 50., 120., 160.], [150., 100., 260., 240.]]),
         "labels": torch.tensor([0, 2])},
        {"boxes": torch.tensor([[20., 30., 100., 140.]]),
         "labels": torch.tensor([1])},
    ]
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = None
    for step in range(200):
        opt.zero_grad()
        feats = m.neck(*m.backbone(x))
        out = m.head(feats, with_aux=True)
        loss, stats = crit(out, feats, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float(loss.detach()) < first * 0.1, f"loss did not decrease: {first} -> {float(loss)}"
    assert stats["pos_main"] >= 2  # each GT got exactly one anchor
    print(f"test_loss_decreases_on_fixed_batch OK  {first:.3f} -> {float(loss.detach()):.3f}")


def test_loss_empty_targets():
    m = _tiny_model(nc=3)
    m.train()
    crit = DetectionLoss(num_classes=3)
    x = torch.randn(1, 3, 320, 320)
    feats = m.neck(*m.backbone(x))
    out = m.head(feats, with_aux=True)
    loss, stats = crit(out, feats, [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}])
    assert torch.isfinite(loss)
    print("test_loss_empty_targets OK", float(loss))


if __name__ == "__main__":
    test_model_forward_shapes()
    test_decode_and_iou()
    test_matcher_one_to_one()
    test_loss_decreases_on_fixed_batch()
    test_loss_empty_targets()
    print("ALL TESTS PASSED")