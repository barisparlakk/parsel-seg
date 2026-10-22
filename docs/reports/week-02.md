# Week 2 - Data pipeline and class mapping

## Done
- `configs/mappings/openearthmap.yaml`, `loveda.yaml`: raw mask value -> Parsel main class,
  with a rationale per class. OpenEarthMap is verified against sample overlays; LoveDA is not
  (marked `verified: false`).
- `scripts/remap_masks.py`: raw masks -> Parsel ids in `data/interim/`.
- `scripts/make_tiles.py`: 512 px tiles, optional overlap and GSD resampling, scene-based
  split, `manifest.csv`.
- `parsel/data/dataset.py`: `TileDataset` and `build_transforms` (flip, rotate90, scale,
  brightness/contrast, hue/saturation, ImageNet normalisation).
- `parsel/utils/viz.py`: mask colouring, overlays, de-normalisation.
- `scripts/class_stats.py`: class frequencies, median-frequency weights, channel statistics
  -> `configs/class_weights.yaml`.
- 30 tests (mapping, tiling, splits, dataset, viz, stats), all on synthetic data.

Run order:
```
python scripts/remap_masks.py --dataset openearthmap
python scripts/make_tiles.py --dataset openearthmap --tile-size 512
python scripts/class_stats.py --dataset openearthmap
```
On OpenEarthMap this takes about 11 s, 51 s and 44 s. Output is 6.6 GB in `data/processed/`.

## Class mapping (OpenEarthMap -> Parsel)

| OpenEarthMap | Parsel | Note |
|---|---|---|
| unlabeled | ignore (255) | not trained on |
| bareland | vacant land (3) | |
| rangeland | vegetation (7) | |
| developed | background/other (0) | paved yards, plazas; neither building nor road |
| road | transport (5) | |
| tree | vegetation (7) | |
| water | water (6) | |
| agriculture | agriculture (4) | |
| building | buildings (1) | |

"Developed" is 18.7% of the pixels and the least clear call. It was kept as other instead of
being forced into vacant land. Mapping to 255 is the alternative.

## Tiles and splits
- 2687 labelled scenes -> 10,596 tiles of 512x512, 75 regions. The other 22 regions are
  xBD-only (labels, no imagery).
- Split by region, seed 42, ratios 0.7 / 0.15 / 0.15. No region is in more than one split.

| Split | Tiles | Regions |
|---|---|---|
| train | 7693 | 53 |
| val | 1443 | 11 |
| test | 1460 | 11 |

Tiles with less than 50% labelled area are dropped; 1125 tiles contain some padding or
ignore pixels. Edge tiles are padded with black imagery and label 255.

## Class balance (train split, % of pixels) and weights

| Class | Train % | Weight |
|---|---|---|
| vegetation | 37.97 | 0.310 |
| background | 20.13 | 0.586 |
| buildings | 18.66 | 0.632 |
| agriculture | 11.79 | 1.000 |
| transport | 7.01 | 1.682 |
| water | 2.67 | 4.421 |
| vacant land | 1.78 | 6.612 |
| construction | 0 | 0 |
| energy | 0 | 0 |

Channel mean (RGB, 0-1): 0.4603, 0.4668, 0.4123; std: 0.1965, 0.1793, 0.1857. Training uses
ImageNet statistics by default because the encoders are ImageNet-pretrained.

## Checks
- A `DataLoader` batch has shape (16, 3, 512, 512) float32 with int64 masks (16, 512, 512).
  Mask ids are in {0, 1, 3, 4, 5, 6, 7, 255}.
- One tile takes about 8 ms to load and augment in a single process (about 127 tiles/s per worker).
- Overlay check: `docs/figures/week2_augmented_samples.png`.

## Caveats
- Construction and energy have no pixels, so the model cannot learn them yet and their weight
  is 0. Weights must be recomputed after the custom data in week 5.
- Val is not representative of train: vegetation 48.9% vs 38.0%, buildings 11.1% vs 18.7%.
  This comes from splitting by whole regions with few regions per split. Test is closer to
  train. Compare experiments on the same val set, and treat absolute val numbers with care.
- Subclass labels (for example paved vs dirt road) are not derivable from these datasets;
  only main classes are trained for now.
- LoveDA is not downloaded, so its mapping and pipeline are untested on real data.

## Problems
- The repo was in `~/Desktop`, which is managed by iCloud Drive. File reads stalled for
  minutes (imports, `uv run`, a venv move). Moved to `~/dev/parsel-seg`; data was re-downloaded.
- `DataLoader` with `num_workers > 0` hangs when run from a script read on stdin on macOS
  (spawn start method re-imports `__main__`). Use a real script file with a
  `if __name__ == "__main__":` guard.

## Open questions
1. Keep "developed" as background, or ignore it?
2. Download LoveDA (about 18 GB) for more training data and a second domain, or move on?
3. Overlap between tiles is 0 (edge tiles are shifted back to fit, so neighbours overlap a
   little). Add overlap for the training split?
4. Stratify the region split by class mix to make val more representative?
