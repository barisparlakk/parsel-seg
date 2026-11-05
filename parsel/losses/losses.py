"""Segmentation losses: class-weighted cross-entropy, soft Dice and their combination."""

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Multi-class soft Dice over classes that occur in the batch's ground truth.

    Averaging only over present classes keeps classes absent from the data (and from a given
    batch) from contributing a constant loss. Pixels equal to ignore_index are excluded.
    """

    def __init__(self, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.ignore_index, self.eps = ignore_index, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = (target != self.ignore_index).unsqueeze(1)
        probs = logits.softmax(1) * valid
        onehot = F.one_hot(target.clamp(max=num_classes - 1), num_classes).permute(0, 3, 1, 2)
        onehot = onehot * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        present = onehot.sum(dims) > 0
        if not present.any():
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss: (1 - p_t)^gamma * -log p_t, with optional per-class weights.

    Normalised by the summed class weights of the valid pixels, like CrossEntropyLoss, so it
    sits on the same scale as CE and the two can be swapped without retuning the loss weights.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma, self.ignore_index = gamma, ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe = target.clamp(max=logits.shape[1] - 1)
        logp = logits.log_softmax(1).gather(1, safe.unsqueeze(1)).squeeze(1)
        loss = -((1 - logp.exp()).pow(self.gamma)) * logp
        w = valid.to(loss.dtype)
        if self.class_weights is not None:
            w = w * self.class_weights.to(loss.dtype)[safe]
        return (loss * w).sum() / w.sum().clamp_min(1e-12)


class CombinedLoss(nn.Module):
    """ce_weight * CrossEntropy + dice_weight * Dice.

    With focal_gamma set, the cross-entropy term is replaced by a focal loss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float | None = None,
    ):
        super().__init__()
        self.ce = (
            nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            if focal_gamma is None
            else FocalLoss(focal_gamma, class_weights, ignore_index)
        )
        self.dice = DiceLoss(ignore_index)
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = logits.sum() * 0.0
        # CrossEntropyLoss returns NaN (mean over zero pixels) when every pixel is ignored.
        if self.ce_weight and (target != self.ignore_index).any():
            loss = loss + self.ce_weight * self.ce(logits, target)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss


def build_loss(cfg: dict, class_weights: torch.Tensor | None, ignore_index: int = 255) -> nn.Module:
    """Build a loss from a config dict: type is "ce", "dice", "ce_dice" or "focal_dice"."""
    kind = cfg.get("type", "ce_dice")
    if kind == "ce":
        return CombinedLoss(class_weights, ignore_index, ce_weight=1.0, dice_weight=0.0)
    if kind == "dice":
        return CombinedLoss(None, ignore_index, ce_weight=0.0, dice_weight=1.0)
    if kind == "ce_dice":
        return CombinedLoss(
            class_weights,
            ignore_index,
            ce_weight=cfg.get("ce_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
        )
    if kind == "focal_dice":
        return CombinedLoss(
            class_weights,
            ignore_index,
            ce_weight=cfg.get("ce_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            focal_gamma=cfg.get("focal_gamma", 2.0),
        )
    raise ValueError(f"Unknown loss type {kind!r}")
