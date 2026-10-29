import json

import pytest
import torch
from torch.utils.data import DataLoader

from parsel.data.dataset import TileDataset, build_transforms
from parsel.losses import build_loss
from parsel.models.factory import build_model
from parsel.train.trainer import Trainer, cosine_with_warmup, resolve_device

CFG = {"epochs": 2, "lr": 1e-3, "warmup_epochs": 1, "min_lr": 1e-5, "grad_clip": 1.0,
       "early_stop_patience": 0, "log_every": 1, "amp": False}  # fmt: skip


def _trainer(manifest, out_dir, cfg=None):
    train = TileDataset(manifest, "train", build_transforms(True, tile_size=64))
    val = TileDataset(manifest, "val", build_transforms(False))
    model = build_model({"arch": "unet", "encoder": "resnet18", "encoder_weights": None}, 9)
    return Trainer(
        model, build_loss({"type": "ce_dice"}, None),
        DataLoader(train, 2, shuffle=True), DataLoader(val, 1),
        [f"c{i}" for i in range(9)], out_dir, cfg or CFG, torch.device("cpu"),
    )  # fmt: skip


def test_fit_writes_checkpoints_and_history(tile_dataset, tmp_path):
    result = _trainer(tile_dataset, tmp_path / "run").fit()
    assert result["epochs_run"] == 2 and result["best_miou"] >= 0
    for f in ("best.pt", "last.pt", "history.json"):
        assert (tmp_path / "run" / f).exists()
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1]
    assert {"val_miou", "val_iou", "train_loss"} <= set(history[0])


def test_resume_continues_after_last_epoch(tile_dataset, tmp_path):
    _trainer(tile_dataset, tmp_path / "run").fit()
    t = _trainer(tile_dataset, tmp_path / "run2", {**CFG, "epochs": 3})
    t.resume(tmp_path / "run" / "last.pt")
    assert t.start_epoch == 2
    assert t.fit()["epochs_run"] == 3


def test_early_stopping(tile_dataset, tmp_path):
    t = _trainer(
        tile_dataset,
        tmp_path / "run",
        {**CFG, "epochs": 10, "early_stop_patience": 1, "min_delta": 10.0},
    )  # nothing can improve by 10
    assert t.fit()["epochs_run"] == 2  # epoch 0 sets the best, epoch 1 is the one bad epoch


def test_max_train_batches_limits_steps(tile_dataset, tmp_path):
    t = _trainer(tile_dataset, tmp_path / "run", {**CFG, "max_train_batches": 1})
    t.fit()
    assert t.scheduler.last_epoch == 2  # one step per epoch, two epochs


def test_non_finite_loss_raises(tile_dataset, tmp_path):
    t = _trainer(tile_dataset, tmp_path / "run")
    t.loss_fn = lambda logits, y: logits.sum() * float("nan")
    with pytest.raises(RuntimeError, match="Non-finite"):
        t.fit()


def test_cosine_schedule_shape():
    fn = cosine_with_warmup(total_steps=100, warmup_steps=10, min_ratio=0.01)
    assert fn(0) < fn(5) < fn(9) <= 1.0
    assert fn(10) == pytest.approx(1.0)
    assert fn(100) == pytest.approx(0.01)
    assert fn(55) < fn(10) and fn(55) > fn(100)


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"
