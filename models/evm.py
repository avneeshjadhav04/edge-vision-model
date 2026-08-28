"""EVM-nano: from-scratch nano detector (~2.4M params, ~5 GFLOPs @ 640)."""
from .backbone import Backbone
from .neck import PANNeck
from .head import Detect
from .decode import Postprocessor, make_anchors, dfl_decode, bbox_iou
import torch
import torch.nn as nn


class EVM(nn.Module):
    """Full detector. Set with_aux=False (or drop aux params) for export."""

    def __init__(self, num_classes=80, input_size=640,
                 backbone=None, stem=32, out_channels=(128, 256, 256),
                 depth=(2, 2, 2),
                 neck_hidden=96, neck_out=96, head_hidden=96, reg_max=8,
                 aux_hidden=96, strides=(8, 16, 32)):
        super().__init__()
        self.nc = num_classes
        self.input_size = input_size
        self.strides = tuple(strides)
        self.reg_max = reg_max
        self.backbone = backbone if backbone is not None else Backbone(stem, out_channels, depth)
        self.neck = PANNeck(out_channels, neck_hidden, neck_out)
        self.head = Detect(neck_out, head_hidden, num_classes, reg_max, aux_hidden)
        # DFL projection register_buffer -> travels with checkpoints, fixed at eval
        self.register_buffer("dfl_proj", torch.arange(reg_max, dtype=torch.float32))
        self.post = Postprocessor(num_classes, self.strides, reg_max)

    def forward(self, x, with_aux=True):
        feats = self.neck(*self.backbone(x))
        if self.training or with_aux:
            return self.head(feats, with_aux=with_aux)
        # eval/export path: raw head maps, decode outside for cleaner ONNX graph
        out = []
        for f in feats:
            mb, mc, mo = self.head.main(f)
            out.append((mb, mc, mo))
        return out

    @torch.no_grad()
    def predict(self, x, score_thresh=0.25, max_det=300, use_obj=True):
        """NMS-free inference (torch path, used by demo/tests)."""
        self.post.score_thresh = score_thresh
        self.post.max_det = max_det
        self.post.use_obj = use_obj
        was_training = self.training
        self.eval()
        raw = self.forward(x, with_aux=False)
        res = self.post(raw, self.dfl_proj)
        if was_training:
            self.train()
        return res


def build_model(cfg_model, num_classes):
    """cfg_model: parsed configs/model_nano.yaml; num_classes from dataset config."""
    b = cfg_model["backbone"]
    n = cfg_model["neck"]
    h = cfg_model["head"]
    backbone = Backbone(
        stem_channels=b["stem_channels"],
        out_channels=tuple(b["out_channels"]),
        depth=tuple(b["depth"]),
    )
    return EVM(
        num_classes=num_classes,
        input_size=cfg_model.get("input_size", 640),
        backbone=backbone,
        out_channels=backbone.feature_channels,
        neck_hidden=n["hidden"],
        neck_out=n["out"],
        head_hidden=h.get("hidden", n["hidden"]),
        reg_max=h["reg_max"],
        aux_hidden=h.get("aux_hidden", n["hidden"]),
        strides=(8, 16, 32),
    )


def export_model(model: EVM) -> nn.Module:
    """Strip the aux head (and training-only params) for export/deployment."""
    import copy
    m = copy.deepcopy(model)
    m.eval()
    del m.head.aux
    m.head.aux = None
    return m


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train_only_aux = 0
    if getattr(model.head, "aux", None) is not None:
        train_only_aux = sum(p.numel() for p in model.head.aux.parameters())
    return {"total": total, "deployable": total - train_only_aux, "aux_train_only": train_only_aux}