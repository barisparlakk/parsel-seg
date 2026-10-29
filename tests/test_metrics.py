import numpy as np
import torch

from parsel.utils.metrics import ConfusionMatrix


def _cm(pred, target, n=3):
    cm = ConfusionMatrix(n)
    cm.update(torch.tensor(pred), torch.tensor(target))
    return cm


def test_perfect_prediction():
    s = _cm([0, 1, 2, 1], [0, 1, 2, 1]).summary()
    assert s["miou"] == 1.0 and s["mf1"] == 1.0 and s["pixel_acc"] == 1.0


def test_known_iou_values():
    # class 0: tp=1 fp=1 fn=0 -> 1/2; class 1: tp=1 fp=0 fn=1 -> 1/2; class 2 absent everywhere
    cm = _cm([0, 0, 0], [0, 0, 1], n=2) if False else _cm([0, 0, 1], [0, 1, 1], n=2)
    assert np.allclose(cm.iou(), [0.5, 0.5])
    assert np.allclose(cm.f1(), [2 / 3, 2 / 3])


def test_ignore_index_excluded():
    cm = _cm([0, 1, 2], [0, 255, 2])
    assert cm.mat.sum() == 2
    assert cm.summary()["pixel_acc"] == 1.0


def test_miou_only_over_classes_present_in_ground_truth():
    # class 2 has no ground truth but is predicted once: IoU 0, yet excluded from mIoU
    cm = _cm([0, 1, 2], [0, 1, 1])
    assert cm.iou()[2] == 0.0
    assert cm.summary()["miou"] == np.mean([1.0, 0.5])


def test_class_without_gt_or_predictions_is_nan():
    assert np.isnan(_cm([0, 1], [0, 1]).iou()[2])


def test_updates_accumulate_and_reset():
    cm = _cm([0, 1], [0, 1])
    cm.update(torch.tensor([1]), torch.tensor([0]))
    assert cm.mat.sum() == 3
    cm.reset()
    assert cm.mat.sum() == 0


def test_works_on_image_shaped_input():
    pred = torch.randint(0, 3, (2, 8, 8))
    cm = ConfusionMatrix(3)
    cm.update(pred, pred)
    assert cm.summary()["miou"] == 1.0
