"""Training loop: AMP, gradient clipping, warmup + cosine LR, early stopping, TensorBoard."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from parsel.utils.metrics import ConfusionMatrix


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_with_warmup(total_steps: int, warmup_steps: int, min_ratio: float):
    """LR multiplier: linear warmup, then cosine decay to min_ratio."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return fn


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: list[str],
        out_dir: str | Path,
        cfg: dict,
        device: torch.device,
        ignore_index: int = 255,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.class_names, self.ignore_index = class_names, ignore_index
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg, self.device = cfg, device

        self.epochs = cfg["epochs"]
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.patience = cfg.get("early_stop_patience", 0)
        self.max_batches = cfg.get("max_train_batches")
        self.log_every = cfg.get("log_every", 50)
        amp = cfg.get("amp", "auto")
        # fp16 autocast gives no speedup on MPS (measured) and needs loss scaling on CUDA only.
        self.use_amp = (device.type == "cuda") if amp == "auto" else bool(amp)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and device.type == "cuda")

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
        )
        steps_per_epoch = min(len(train_loader), self.max_batches or len(train_loader))
        self.scheduler = LambdaLR(
            self.optimizer,
            cosine_with_warmup(
                self.epochs * steps_per_epoch,
                int(cfg.get("warmup_epochs", 1) * steps_per_epoch),
                cfg.get("min_lr", 1e-6) / cfg["lr"],
            ),
        )
        self.writer = SummaryWriter(self.out_dir / "tb")
        self.start_epoch, self.best_miou, self.bad_epochs, self.history = 0, -1.0, 0, []

    # ------------------------------------------------------------------ checkpoints
    def _state(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": self.best_miou,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": self.cfg,
            "class_names": self.class_names,
        }

    def resume(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch, self.best_miou = ckpt["epoch"] + 1, ckpt["best_miou"]
        self.bad_epochs, self.history = ckpt["bad_epochs"], ckpt["history"]

    # ------------------------------------------------------------------ loops
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        t0 = time.time()
        for i, batch in enumerate(self.train_loader):
            if self.max_batches and i >= self.max_batches:
                break
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["mask"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self.loss_fn(self.model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {i}: {loss.item()}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total, n = total + loss.item(), n + 1
            step = epoch * len(self.train_loader) + i
            if i % self.log_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], step)
                rate = (i + 1) * x.shape[0] / (time.time() - t0)
                print(f"  epoch {epoch} step {i}/{len(self.train_loader)} loss {loss.item():.4f} "
                      f"({rate:.1f} img/s)", flush=True)  # fmt: skip
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict:
        self.model.eval()
        cm = ConfusionMatrix(len(self.class_names), self.ignore_index)
        total, n = 0.0, 0
        for batch in loader or self.val_loader:
            x = batch["image"].to(self.device)
            y = batch["mask"].to(self.device)
            logits = self.model(x)
            total, n = total + self.loss_fn(logits, y).item(), n + 1
            cm.update(logits.argmax(1), y)
        out = cm.summary()
        out["loss"] = total / max(1, n)
        out["confusion"] = cm.mat
        return out

    def fit(self) -> dict:
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val = self.evaluate()
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_miou": val["miou"],
                "val_mf1": val["mf1"],
                "val_pixel_acc": val["pixel_acc"],
                "val_iou": {n: (None if np.isnan(v) else float(v)) for n, v in
                            zip(self.class_names, val["iou"], strict=True)},
                "seconds": time.time() - t0,
            }  # fmt: skip
            self.history.append(record)
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            for key in ("loss", "miou", "mf1", "pixel_acc"):
                self.writer.add_scalar(f"val/{key}", val[key], epoch)
            for name, v in record["val_iou"].items():
                if v is not None:
                    self.writer.add_scalar(f"val_iou/{name}", v, epoch)

            improved = self.best_miou < 0 or val["miou"] > self.best_miou + self.cfg.get(
                "min_delta", 0.0
            )
            if improved:
                self.best_miou, self.bad_epochs = val["miou"], 0
                torch.save(self._state(epoch), self.out_dir / "best.pt")
            else:
                self.bad_epochs += 1
            torch.save(self._state(epoch), self.out_dir / "last.pt")
            (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            print(
                f"epoch {epoch}: train_loss {train_loss:.4f} val_loss {val['loss']:.4f} "
                f"val_mIoU {val['miou']:.4f} pixel_acc {val['pixel_acc']:.4f} "
                f"{'*best*' if improved else f'(no gain {self.bad_epochs})'} "
                f"[{record['seconds']:.0f}s]",
                flush=True,
            )
            if self.patience and self.bad_epochs >= self.patience:
                print(f"Early stopping: no val mIoU gain for {self.patience} epochs", flush=True)
                break
        self.writer.close()
        return {"best_miou": self.best_miou, "epochs_run": len(self.history)}
