"""Compare evaluated runs side by side: a Markdown table and a per-class IoU chart.

Reads <run>/eval_<split>.json (written by scripts/evaluate.py), <run>/config.yaml and
<run>/history.json for every run directory under --runs, or for the runs named explicitly.

Usage:
    python scripts/compare_runs.py --runs runs
    python scripts/compare_runs.py --runs /content/drive/MyDrive/parsel-runs --out docs/figures
    python scripts/compare_runs.py runs/unet_r34 runs/segformer_b2 --sort miou
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_run(run: Path, split: str) -> dict | None:
    """Summarise one run, or None if it has not been evaluated on `split`."""
    eval_path = run / f"eval_{split}.json"
    if not eval_path.exists():
        return None
    ev = json.loads(eval_path.read_text())
    cfg_path, hist_path = run / "config.yaml", run / "history.json"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    model = cfg.get("model", {})
    best_val = max((h["val_miou"] for h in hist), default=None)
    conf = np.array(ev["confusion"]) if "confusion" in ev else None
    absent_fp = None
    if conf is not None:
        absent = conf.sum(1) == 0
        absent_fp = float(conf[:, absent].sum() / conf.sum()) if conf.sum() else None
    return {
        "run": run.name,
        "model": f"{model.get('arch', '?')}/{model.get('encoder', '?')}",
        "params_m": ev.get("params_m"),
        "miou": ev["miou"],
        "mf1": ev["mf1"],
        "boundary_miou": ev.get("boundary_miou"),
        "pixel_acc": ev["pixel_acc"],
        "best_val_miou": best_val,
        "epochs": len(hist),
        "minutes": sum(h["seconds"] for h in hist) / 60 if hist else None,
        "absent_fp": absent_fp,
        "iou": ev["iou"],
    }


def fmt(v, spec: str = ".4f") -> str:
    return "-" if v is None else format(v, spec)


def table(rows: list[dict]) -> str:
    head = ("| Run | Model | Params (M) | mIoU | mF1 | Boundary mIoU | Pixel acc | Best val mIoU "
            "| Epochs | Train min | Pred. on absent classes |")  # fmt: skip
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        fp = "-" if r["absent_fp"] is None else f"{100 * r['absent_fp']:.2f}%"
        lines.append(
            f"| {r['run']} | {r['model']} | {fmt(r['params_m'], '.1f')} | {r['miou']:.4f} "
            f"| {r['mf1']:.4f} | {fmt(r['boundary_miou'])} | {r['pixel_acc']:.4f} "
            f"| {fmt(r['best_val_miou'])} | {r['epochs']} | {fmt(r['minutes'], '.0f')} | {fp} |"
        )
    return "\n".join(lines)


def class_table(rows: list[dict]) -> str:
    classes = [c for c in rows[0]["iou"] if any(r["iou"].get(c) is not None for r in rows)]
    lines = [
        "| Class | " + " | ".join(r["run"] for r in rows) + " |",
        "|" + "---|" * (len(rows) + 1),
    ]
    for c in classes:
        lines.append(f"| {c} | " + " | ".join(fmt(r["iou"].get(c)) for r in rows) + " |")
    return "\n".join(lines)


def plot_classes(rows: list[dict], path: Path, split: str) -> None:
    classes = [c for c in rows[0]["iou"] if any(r["iou"].get(c) is not None for r in rows)]
    x = np.arange(len(classes))
    w = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(classes)), 4.5))
    for i, r in enumerate(rows):
        vals = [r["iou"].get(c) or 0 for c in classes]
        ax.bar(x + (i - (len(rows) - 1) / 2) * w, vals, w, label=f"{r['run']} ({r['miou']:.3f})")
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylabel(f"IoU ({split})")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, title="run (mIoU)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs_list", nargs="*", type=Path, help="run dirs (default: all in --runs)")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sort", default="miou", choices=["miou", "run", "boundary_miou"])
    p.add_argument("--out", type=Path, help="directory for comparison_<split>.png and .md")
    args = p.parse_args()

    dirs = args.runs_list or sorted(d for d in args.runs.iterdir() if d.is_dir())
    rows = [r for d in dirs if (r := load_run(d, args.split))]
    if not rows:
        raise SystemExit(f"No evaluated runs (eval_{args.split}.json) found")
    rows.sort(key=lambda r: r["run"] if args.sort == "run" else -(r[args.sort] or 0))

    text = (
        f"### Overall ({args.split})\n\n{table(rows)}\n\n### IoU per class\n\n{class_table(rows)}\n"
    )
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"comparison_{args.split}.md").write_text(text)
        plot_classes(rows, args.out / f"comparison_{args.split}.png", args.split)
        print(f"Wrote {args.out}/comparison_{args.split}.[md|png]")


if __name__ == "__main__":
    main()
