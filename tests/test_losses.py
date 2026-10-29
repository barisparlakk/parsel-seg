import pytest
import torch

from parsel.losses import CombinedLoss, DiceLoss, build_loss


def _logits_for(target, classes=3, confidence=20.0):
    """Logits that put almost all probability on the target class."""
    t = target.clone()
    t[t == 255] = 0
    return torch.nn.functional.one_hot(t, classes).permute(0, 3, 1, 2).float() * confidence


def test_dice_is_near_zero_for_perfect_prediction():
    y = torch.randint(0, 3, (2, 8, 8))
    assert DiceLoss()(_logits_for(y), y).item() < 1e-3


def test_dice_is_large_for_wrong_prediction():
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    wrong = _logits_for(torch.ones_like(y))
    assert DiceLoss()(wrong, y).item() > 0.9


def test_dice_ignores_ignore_index_pixels():
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    y[:, :4] = 255
    logits = _logits_for(torch.zeros_like(y))
    logits[:, :, :4] = _logits_for(torch.ones_like(y))[:, :, :4]  # wrong, but ignored
    assert DiceLoss()(logits, y).item() < 1e-3


def test_all_ignored_target_gives_zero_not_nan():
    y = torch.full((1, 4, 4), 255)
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    for loss in (DiceLoss(), CombinedLoss()):
        out = loss(logits, y)
        assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-6)


def test_class_weights_change_ce():
    y = torch.randint(0, 3, (2, 8, 8))
    logits = torch.randn(2, 3, 8, 8)
    plain = CombinedLoss(dice_weight=0.0)(logits, y)
    weighted = CombinedLoss(torch.tensor([1.0, 5.0, 0.1]), dice_weight=0.0)(logits, y)
    assert not torch.isclose(plain, weighted)


def test_combined_is_sum_of_parts():
    y = torch.randint(0, 3, (2, 8, 8))
    logits = torch.randn(2, 3, 8, 8)
    both = CombinedLoss(ce_weight=1.0, dice_weight=2.0)(logits, y)
    ce = CombinedLoss(ce_weight=1.0, dice_weight=0.0)(logits, y)
    dice = CombinedLoss(ce_weight=0.0, dice_weight=1.0)(logits, y)
    assert torch.isclose(both, ce + 2 * dice)


def test_gradients_flow():
    y = torch.randint(0, 3, (1, 8, 8))
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    CombinedLoss()(logits, y).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_build_loss_types_and_error():
    for kind in ("ce", "dice", "ce_dice"):
        assert build_loss({"type": kind}, None) is not None
    with pytest.raises(ValueError):
        build_loss({"type": "nope"}, None)
