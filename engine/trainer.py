"""Trainer: SGD/AdamW + warmup + cosine decay, AMP, EMA, checkpoint/resume, logging."""
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .ema import ModelEMA
from .inference import run_inference


class Trainer:
    def __init__(self, model, loss_fn, train_ds=None, val_eval_fn=None, val_loader=None,
                 cfg=None, device="cuda", save_dir="runs/train", log_name="log"):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.cfg = cfg or {}
        tc = self.cfg.get("train", {})
        self.h = tc
        self.train_ds = train_ds
        self.h = tc
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.val_eval_fn = val_eval_fn
        self.val_loader = val_loader

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = tc.get("optimizer", "sgd")
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=float(tc.get("lr0", 0.05)),
                                             momentum=float(tc.get("momentum", 0.937)),
                                             weight_decay=float(tc.get("weight_decay", 5e-4)),
                                             nesterov=True)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=float(tc.get("lr0", 1e-3)),
                                               weight_decay=float(tc.get("weight_decay", 5e-4)))
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
              "epoch": self.epoch, "history": self.history}
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

    # ----- train loop -----
    def fit(self, epochs):
        self.total_epochs = epochs
        accum = int(self.h.get("accum", 1))
        bs = int(self.h.get("batch_size", 32))
        nw = int(self.h.get("workers", 8))
        dl = DataLoader(train_ds_wrap(self.train_ds), batch_size=bs, shuffle=True,
                        num_workers=nw, collate_fn=_collate, pin_memory=True,
                        drop_last=True, persistent_workers=nw > 0)
        n_it = len(dl)
        self.n_it = n_it
        mosaic_close = int(self.h.get("mosaic_close_epochs", 10))
        for epoch in range(self.start_epoch, epochs):
            self.epoch = epoch
            self.model.train()
            self.loss_fn.set_epoch(epoch)
            if hasattr(self.train_ds, "transform") and hasattr(self.train_ds.transform, "mosaic"):
                self.train_ds.transform.mosaic.p = 0.0 if epoch >= epochs - mosaic_close else float(self.h.get("mosaic", 1.0))
            m_it = 0.0
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
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                m_it += float(loss.detach())
            tag = ""
            if self.val_eval_fn is not None and (epoch + 1) % int(self.h.get("val_interval", 2)) == 0:
                metrics = self.val_eval_fn(self.ema.module)
                metrics["epoch"] = epoch
                metrics["lr"] = lr
                metrics["loss"] = m_it / max(1, n_it)
                self.history.append(metrics)
                tag = f"  mAP={metrics.get('mAP', float('nan')):.4f}"
                if metrics.get("mAP", -1) > self.best:
                    self.best = metrics["mAP"]
                    self.save("best.pt")
                print(f"epoch {epoch:4d} loss={m_it / max(1, n_it):8.3f}{tag} "
                      f"({time.time() - t0:.0f}s)")
            else:
                self.history.append({"epoch": epoch, "loss": m_it / max(1, n_it), "lr": lr})
                print(f"epoch {epoch:4d} loss={m_it / max(1, n_it):8.3f}{tag} "
                      f"({time.time() - t0:.0f}s)")
            self.save("last.pt")
            with open(os.path.join(self.save_dir, f"{self.log_name}.json"), "w") as f:
                json.dump(self.history, f, indent=1)
        return self.history


def _collate(batch):
    from data.common import collate_batch
    return collate_batch(batch)


def train_ds_wrap(ds):
    return ds