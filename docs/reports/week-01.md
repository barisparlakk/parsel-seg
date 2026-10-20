# Week 1 - Setup and data exploration

## Done
- Repo skeleton, `uv` project (Python 3.11), dependencies, ruff and pre-commit.
- `configs/classes.yaml`: 9 main classes, 18 subclasses (+ background), colours, TR/EN names;
  loader in `parsel/utils/classes.py` with tests.
- `scripts/download_data.py`: resumable, MD5-verified download for OpenEarthMap and LoveDA.
- `docs/datasets.md`: sources, GSD, licenses, coverage against the taxonomy.
- `notebooks/01_eda.ipynb`: image size, GSD, class distribution, sample overlays. Run on
  OpenEarthMap; LoveDA has not been downloaded yet.

## Dataset comparison

| | OpenEarthMap | LoveDA |
|---|---|---|
| Images | 5000 listed; 2687 usable (see below) | Train / Val / Test, 1024x1024 tiles |
| GSD | 0.2 - 0.5 m, median 0.3 m (200-image sample) | 0.3 m, fixed |
| Coverage | 97 regions, 44 countries | 3 Chinese cities |
| Classes | 8 | 7 (+ no-data) |
| License | CC BY 4.0 record; labels follow source imagery (some CC BY-NC-SA 4.0) | not confirmed; Google Earth terms apply |

## OpenEarthMap: what is actually usable
The archive lists 5000 images, but only part of it can be used for supervised training.

| Split | Listed | Images present | Labels present | Usable pairs |
|---|---|---|---|---|
| train | 3000 | 2303 | 3000 | 2303 |
| val | 500 | 384 | 500 | 384 |
| test | 1500 | 1151 | 0 | 0 |

- The 813 missing train/val images are the xBD regions: imagery is not distributed, labels are.
- Test labels are not public, so the test split cannot be used for evaluation.
- Usable data: 2687 labelled tiles (2303 train + 384 val).
- Tile sizes are not uniform (1024, 1000, 900, 650, 438 px seen in a 200-image sample), so
  tiling in week 2 must not assume a fixed input size.
- The sampled GSD minimum (0.197 m) is slightly below the documented 0.25 m. It comes from
  converting geographic-CRS pixel sizes to metres at the tile latitude; treat GSD as approximate.

## Class imbalance (OpenEarthMap, all 2687 labelled tiles, % of pixels)

| Class | % |
|---|---|
| rangeland | 20.89 |
| developed | 18.74 |
| tree | 18.39 |
| building | 17.16 |
| agriculture | 12.52 |
| road | 6.74 |
| water | 3.09 |
| bareland | 1.75 |
| unlabeled | 0.72 |

Figure: `docs/figures/eda_class_distribution.png`. Samples: `docs/figures/eda_samples_openearthmap.png`.

Observations:
- Bareland (1.75%) and water (3.09%) are rare; road (6.74%) is thin. Expect low IoU on these
  without class weighting, which is planned for week 2.
- Rangeland and developed space are the two largest classes and are the two that map
  ambiguously onto the Parsel taxonomy (see open questions).
- Labels are coarse in dense areas: in the Khartoum sample, whole building blocks are one
  polygon, and open ground between them is "developed".
- Imagery varies a lot between regions (colour, sensor, post-disaster scenes such as Tonga),
  which is good for generalisation but makes per-region validation important.

## Coverage gaps (from class definitions, independent of pixel counts)
Construction, solar panels (GES), greenhouses, pools, parking lots and orchards exist in
neither dataset. These gaps drive the custom dataset in week 5.

## Problems
- Reading `.venv` and data files was very slow at times (a kernel start timed out, `uv run`
  hung once). The repo is in `~/Desktop`, which appears to be synced by iCloud Drive (the
  `bird` process was at 99% CPU). The 14 GB `data/raw` should be kept out of iCloud.
- `python -m jupyter nbconvert` resolves to Anaconda's Jupyter, which has no project
  packages. Use the venv's own `jupyter-nbconvert`, or `uv run jupyter nbconvert`.

## Open questions
1. LoveDA mask encoding (0 = no-data or background) can only be checked once LoveDA is downloaded.
2. The official OpenEarthMap split is per image, not per region: the same region appears in
   train, val and test (for example, Aachen has images in all three). Week 2 needs a
   region-disjoint split to avoid geographic leakage, since the 2687 usable tiles are all we
   have labels for.
3. xBD imagery would add 813 labelled tiles but needs a separate registration and download.
   Worth it, or skip?
4. How to map "rangeland" (vegetation vs vacant land) and "developed" (buildings vs transport
   vs vacant land) onto the Parsel taxonomy. Decide in week 2.
5. LoveDA license terms for any use beyond research.
6. Training hardware: local Apple Silicon (MPS) or a cloud GPU.
