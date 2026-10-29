"""Segmentation metrics from a confusion matrix.

Rows are ground-truth classes, columns are predictions. Pixels equal to ignore_index in the
ground truth are excluded.
"""

import numpy as np
import torch
import torch.nn.functional as F


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate a batch of class-id predictions and targets of identical shape."""
        pred, target = pred.detach().cpu().flatten(), target.detach().cpu().flatten()
        valid = target != self.ignore_index
        idx = target[valid].long() * self.num_classes + pred[valid].long()
        counts = torch.bincount(idx, minlength=self.num_classes**2)
        self.mat += counts.reshape(self.num_classes, self.num_classes).numpy()

    def reset(self) -> None:
        self.mat[:] = 0

    def iou(self) -> np.ndarray:
        """Per-class IoU; NaN where a class has neither ground truth nor predictions."""
        tp = np.diag(self.mat).astype(np.float64)
        union = self.mat.sum(0) + self.mat.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, tp / union, np.nan)

    def f1(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        denom = self.mat.sum(0) + self.mat.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom > 0, 2 * tp / denom, np.nan)

    def present(self) -> np.ndarray:
        """Classes that have ground-truth pixels."""
        return self.mat.sum(1) > 0

    def summary(self) -> dict:
        """mIoU / mF1 average only over classes present in the ground truth."""
        present = self.present()
        iou, f1 = self.iou(), self.f1()
        total = self.mat.sum()
        return {
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "mf1": float(np.nanmean(f1[present])) if present.any() else float("nan"),
            "pixel_acc": float(np.diag(self.mat).sum() / total) if total else float("nan"),
            "iou": iou,
            "f1": f1,
        }


class BoundaryIoU:
    """Per-class IoU restricted to a band of `width` pixels around each region's contour.

    For every class the ground-truth and predicted masks are reduced to the pixels within
    `width` of their own boundary (mask minus its erosion); IoU is taken between the two bands.
    It rewards sharp edges, which plain IoU hides on large regions. Ignored pixels are removed
    from both masks before the bands are computed.
    """

    def __init__(self, num_classes: int, width: int = 3, ignore_index: int = 255):
        self.num_classes, self.width, self.ignore_index = num_classes, width, ignore_index
        self.inter = np.zeros(num_classes, dtype=np.int64)
        self.union = np.zeros(num_classes, dtype=np.int64)
        self.present = np.zeros(num_classes, dtype=bool)

    def _band(self, mask: torch.Tensor) -> torch.Tensor:
        """Pixels of a (N, 1, H, W) bool mask within `width` of its boundary."""
        k = 2 * self.width + 1
        eroded = 1 - F.max_pool2d((~mask).float(), k, stride=1, padding=self.width)
        return mask & ~eroded.bool()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate a batch of class-id predictions and targets of shape (N, H, W)."""
        valid = (target != self.ignore_index).unsqueeze(1)
        for c in range(self.num_classes):
            gt = (target == c).unsqueeze(1) & valid
            if not gt.any():
                continue
            self.present[c] = True
            pr = (pred == c).unsqueeze(1) & valid
            gb, pb = self._band(gt), self._band(pr)
            self.inter[c] += int((gb & pb).sum())
            self.union[c] += int((gb | pb).sum())

    def iou(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.union > 0, self.inter / self.union, np.nan)

    def miou(self) -> float:
        """Mean over classes present in the ground truth."""
        iou = self.iou()[self.present]
        return float(np.nanmean(iou)) if iou.size and not np.isnan(iou).all() else float("nan")
