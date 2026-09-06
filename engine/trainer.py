"""Trainer: SGD/AdamW + warmup + cosine decay, AMP, EMA, checkpoint/resume, logging."""
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ema import ModelEMA
from .inference import run_inference


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model, weight_decay):
    """No weight decay on BN params and biases (standard detector practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):          # BN/bias (incl. 1x1 conv bias)
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    def __init__(self, model, loss_fn, train_ds=None, val_eval_fn=None, val_loader=None,
                 cfg=None, device="cuda", save_dir="runs/train", log_name="log",
                 seed=0):
        seed_everything(seed)
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.cfg = cfg or {}
        tc = self.cfg.get("train", {})
        self.h = tc
        self.train_ds = train_ds
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.val_eval_fn = val_eval_fn
        self.epoch_hook = None
        self.val_loader = val_loader
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        wd = float(tc.get("weight_decay", 5e-4))
        groups = _param_groups(self.model, wd)
        opt_name = tc.get("optimizer", "sgd")
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(groups, lr=float(tc.get("lr0", 0.05)),
                                             momentum=float(tc.get("momentum", 0.937)),
                                             nesterov=True)
        else:
            self.optimizer = torch.optim.AdamW(groups, lr=float(tc.get("lr0", 1e-3)))
        self.ema = ModelEMA(self.model, float(tc.get("ema_decay", 0.9999)))
        self.amp = bool(tc.get("amp", True)) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch = 0
        self.best = -1.0
        self.history = []
        self.log_name = log_name

    # ----- LR schedule -----
    def lr_at(self, epoch):
        it, n_it = self.it, self.n_it
        tc = self.h
        warm = int(tc.get("warmup_epochs", 3))
        lr0, lrf = float(tc.get("lr0", 0.05)), float(tc.get("lrf", 0.01))
        total = self.total_epochs
        if epoch < warm:
            t = (epoch * n_it + it) / max(1, warm * n_it)
            return lr0 * (0.1 + 0.9 * t)
        t = (epoch - warm) / max(1, total - warm)
        return lrf * lr0 + (lr0 - lrf * lr0) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    total_epochs = 0
    n_it = 1
    it = 0

    # ----- checkpointing -----
    def save(self, name="last.pt", extra=None):
        sd = {"model": self.model.state_dict(), "ema": self.ema.state_dict(),
              "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
              "epoch": getattr(self, "epoch", 0), "history": self.history, "best": self.best}
        if extra:
            sd.update(extra)
        torch.save(sd, os.path.join(self.save_dir, name))

    def load(self, path, resume=True):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd["model"])
        if "ema" in sd:
            self.ema.load_state_dict(sd["ema"])
        if resume and "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
            self.scaler.load_state_dict(sd["scaler"])
            self.start_epoch = sd.get("epoch", 0) + 1
        self.history = sd.get("history", [])
        self.best = sd.get("best", self.best)

    # ----- train loop -----
    def fit(self, epochs):
        self.total_epochs = epochs
        accum = int(self.h.get("accum", 1))
        bs = int(self.h.get("batch_size", 32))
        nw = int(self.h.get("workers", 8))
        mosaic_close = int(self.h.get("mosaic_close_epochs", 10))
        mosaic_p0 = float(self.h.get("mosaic", 1.0))

        def make_loader(epoch=0):
            # dataset-side mosaic phase: worker processes fork/spawn AFTER this,
            # so the flag they snapshot is correct for the epochs this loader serves
            if hasattr(self.train_ds, "transform") and hasattr(self.train_ds.transform, "mosaic"):
                self.train_ds.transform.mosaic.p = (
                    0.0 if mosaic_close > 0 and epoch >= epochs - mosaic_close else mosaic_p0)
            return DataLoader(train_ds_wrap(self.train_ds), batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=_collate, pin_memory=True,
                              drop_last=True, persistent_workers=nw > 0,
                              generator=self.generator,
                              worker_init_fn=_worker_init_fn)

        dl = make_loader(self.start_epoch)
        n_it = len(dl)
        self.n_it = n_it
        for epoch in range(self.start_epoch, epochs):
            self.epoch = epoch
            self.model.train()
            self.loss_fn.set_epoch(epoch)
            # persistent workers snapshot the transform: rebuild the loader when
            # mosaic switches off (else the shutdown epoch is a no-op)
            mosaic_now = 0.0 if epoch >= epochs - mosaic_close else mosaic_p0
            if epoch > self.start_epoch and mosaic_now != getattr(self, "_mosaic_last", mosaic_p0):
                del dl
                dl = make_loader(epoch)
                self.n_it = len(dl)
            self._mosaic_last = mosaic_now
            m_it = 0.0
            m_comp = {}
            t0 = time.time()
            n_bad = 0
            self.optimizer.zero_grad(set_to_none=True)
            for it, (imgs, targets) in enumerate(dl):
                self.it = it
                lr = self.lr_at(epoch)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                imgs = imgs.to(self.device, non_blocking=True)
                with torch.autocast("cuda", enabled=self.amp):
                    feats = self.model.neck(*self.model.backbone(imgs))
                    out = self.model.head(feats, with_aux=True)
                    loss, stats = self.loss_fn(out, feats, targets)
                if not torch.isfinite(loss):
                    n_bad += 1
                    print(f"  [warn] epoch {epoch} it {it}: non-finite loss "
                          f"({n_bad} consecutive), batch skipped")
                    self.optimizer.zero_grad(set_to_none=True)
                    if n_bad >= int(self.h.get("nan_abort", 20)):
                        raise RuntimeError(
                            f"training diverged: {n_bad} consecutive non-finite losses")
                    continue
                n_bad = 0
                self.scaler.scale(loss / accum).backward()
                if (it + 1) % accum == 0 or it == n_it - 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        max_norm=float(self.h.get("grad_clip", 10.0)))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                m_it += float(loss.detach())
                for k in ("box", "dfl", "cls", "obj", "aux"):
                    if k in stats:
                        m_comp[k] = m_comp.get(k, 0.0) + float(stats[k].detach())
            tag = ""
            # epoch-hook (side-effect only: logging, probes). Runs OUTSIDE the
            # val_eval_fn contract, whose return value feeds history/best.
            if self.epoch_hook is not None:
                self.epoch_hook(self, epoch)
            if self.val_eval_fn is not None and (epoch + 1) % int(self.h.get("val_interval", 2)) == 0:
                metrics = self.val_eval_fn(self.ema.module)
                metrics["epoch"] = epoch
                metrics["lr"] = lr
                metrics["loss"] = m_it / max(1, n_it)
                for k in ("box", "dfl", "cls", "obj", "aux"):
                    metrics[k] = m_comp.get(k, 0.0) / max(1, n_it)
                self.history.append(metrics)
                tag = f"  mAP={metrics.get('mAP', float('nan')):.4f}"
                if metrics.get("mAP", -1) > self.best:
                    self.best = metrics["mAP"]
                    # best.pt carries the EMA weights (they are what mAP selected)
                    self.save("best.pt", extra={"model": self.ema.module.state_dict(),
                                                "is_ema": True})
                print(f"epoch {epoch:4d} loss={m_it / max(1, n_it):8.3f}{tag} "
                      f"box={metrics.get('box',0):7.3f} dfl={metrics.get('dfl',0):7.3f} "
                      f"cls={metrics.get('cls',0):7.3f} obj={metrics.get('obj',0):7.3f} "
                      f"aux={metrics.get('aux',0):7.3f} ({time.time() - t0:.0f}s)")
            else:
                self.history.append({"epoch": epoch, "loss": m_it / max(1, n_it), "lr": lr})
                print(f"epoch {epoch:4d} loss={m_it / max(1, n_it):8.3f}{tag} "
                      f"box={m_comp.get('box',0)/max(1,n_it):7.3f} dfl={m_comp.get('dfl',0)/max(1,n_it):7.3f} "
                      f"cls={m_comp.get('cls',0)/max(1,n_it):7.3f} obj={m_comp.get('obj',0)/max(1,n_it):7.3f} "
                      f"aux={m_comp.get('aux',0)/max(1,n_it):7.3f} ({time.time() - t0:.0f}s)")
            self.save("last.pt")
            with open(os.path.join(self.save_dir, f"{self.log_name}.json"), "w") as f:
                json.dump(self.history, f, indent=1)
        return self.history


def _collate(batch):
    from data.common import collate_batch
    return collate_batch(batch)


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2 ** 32)


def train_ds_wrap(ds):
    return ds