"""Exponential moving average of model weights (from scratch)."""
import torch
import torch.nn as nn


class ModelEMA:
    """Standard EMA with warmup ramp: decay = min(d, (1 + updates) / (10 + updates))."""

    def __init__(self, model: nn.Module, decay=0.9999):
        self.module = _deepcopy_to_device(model)
        self.module.eval()
        self.decay = decay
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math_pow(self.updates)) if self.updates < 2000 else self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates,
                "decay": self.decay}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd.get("updates", 0)
        self.decay = sd.get("decay", self.decay)


def math_pow(x):
    # warmup ramp factor: min(decay, (1+x)/(10+x)) implemented here to keep formula local
    ramp = (1 + x) / (10 + x)
    return 1 - ramp if ramp < 0.9999 else 0.0


def _deepcopy_to_device(model):
    import copy
    device = next(model.parameters()).device
    m = copy.deepcopy(model).to(device)
    return m