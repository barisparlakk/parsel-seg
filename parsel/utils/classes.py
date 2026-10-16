"""Loader for the class taxonomy defined in configs/classes.yaml."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "classes.yaml"


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    name_tr: str
    color: tuple[int, int, int]
    parent: int | None = None


class Taxonomy:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.ignore_index: int = cfg["ignore_index"]
        self.classes = [ClassInfo(**{**c, "color": tuple(c["color"])}) for c in cfg["classes"]]
        self.subclasses = [
            ClassInfo(**{**c, "color": tuple(c["color"])}) for c in cfg["subclasses"]
        ]

    def palette(self, level: str = "classes") -> np.ndarray:
        """Return an (N, 3) uint8 palette indexed by class id."""
        items = getattr(self, level)
        pal = np.zeros((max(c.id for c in items) + 1, 3), dtype=np.uint8)
        for c in items:
            pal[c.id] = c.color
        return pal

    def subclass_to_class(self) -> np.ndarray:
        """Lookup table mapping subclass id -> main class id."""
        lut = np.zeros(max(c.id for c in self.subclasses) + 1, dtype=np.uint8)
        for c in self.subclasses:
            lut[c.id] = c.parent
        return lut
