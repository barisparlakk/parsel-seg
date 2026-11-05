import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def test_measure_speed_returns_positive_throughput():
    model = torch.nn.Conv2d(3, 2, 3, padding=1).eval()
    speed = evaluate.measure_speed(model, torch.device("cpu"), size=32, batch=2, iters=3)
    assert speed > 0
