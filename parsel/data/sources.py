"""Discovery of image/mask pairs in the raw downloads of each source dataset."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Ground sampling distance (m) for datasets whose files carry no georeference.
DEFAULT_GSD = {"openearthmap": None, "loveda": 0.3}


@dataclass(frozen=True)
class Sample:
    name: str  # file stem
    scene: str  # unit that must not be split across train/val/test
    image: Path
    mask: Path  # raw (source-encoded) mask
    split_hint: str | None = None  # set when the dataset ships an official split


def _openearthmap(raw: Path) -> list[Sample]:
    root = raw / "openearthmap" / "OpenEarthMap_wo_xBD"
    samples = []
    for img in sorted(root.glob("*/images/*.tif")):
        lab = img.parent.parent / "labels" / img.name
        if lab.exists():  # xBD imagery and test labels are not distributed
            samples.append(Sample(img.stem, img.parent.parent.name, img, lab))
    return samples


def _loveda(raw: Path) -> list[Sample]:
    root = raw / "loveda"
    samples = []
    for split in ("Train", "Val"):  # test masks are not public
        for img in sorted((root / split).glob("*/images_png/*.png")):
            lab = img.parent.parent / "masks_png" / img.name
            if lab.exists():
                scene = f"{split.lower()}_{img.parent.parent.name.lower()}"
                samples.append(Sample(img.stem, scene, img, lab, split.lower()))
    return samples


_LISTERS = {"openearthmap": _openearthmap, "loveda": _loveda}


def list_samples(dataset: str, raw_root: str | Path = DEFAULT_RAW) -> list[Sample]:
    if dataset not in _LISTERS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(_LISTERS)}")
    return _LISTERS[dataset](Path(raw_root))
