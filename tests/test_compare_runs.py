import importlib.util
import json
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "compare_runs", Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def _make_run(root: Path, name: str, miou: float, boundary: float) -> Path:
    run = root / name
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"model": {"arch": "unet", "encoder": "r"}}))
    (run / "history.json").write_text(
        json.dumps([{"val_miou": miou - 0.02, "seconds": 60}, {"val_miou": miou, "seconds": 60}])
    )
    (run / "eval_test.json").write_text(json.dumps({
        "miou": miou, "mf1": 0.7, "pixel_acc": 0.8, "boundary_miou": boundary, "params_m": 24.4,
        "iou": {"a": miou, "b": miou - 0.1, "c": None},
        # class c never occurs in the ground truth but receives 10 of 110 predictions
        "confusion": [[80, 0, 10], [10, 10, 0], [0, 0, 0]],
    }))  # fmt: skip
    return run


def test_load_run_summarises_history_and_absent_class_predictions(tmp_path):
    r = compare.load_run(_make_run(tmp_path, "m1", 0.6, 0.4), "test")
    assert r["epochs"] == 2 and r["minutes"] == 2.0
    assert r["best_val_miou"] == 0.6
    assert abs(r["absent_fp"] - 10 / 110) < 1e-9


def test_load_run_returns_none_without_evaluation(tmp_path):
    (tmp_path / "x").mkdir()
    assert compare.load_run(tmp_path / "x", "test") is None


def test_tables_and_chart(tmp_path):
    rows = [
        compare.load_run(_make_run(tmp_path, "m1", 0.6, 0.4), "test"),
        compare.load_run(_make_run(tmp_path, "m2", 0.7, 0.5), "test"),
    ]
    text = compare.table(rows) + compare.class_table(rows)
    assert "m1" in text and "m2" in text and "0.7000" in text
    assert "| c |" not in text  # class absent from every run is dropped
    compare.plot_classes(rows, tmp_path / "c.png", "test")
    assert (tmp_path / "c.png").stat().st_size > 0
