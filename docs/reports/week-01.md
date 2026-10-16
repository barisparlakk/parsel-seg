# Week 1 - Setup and data exploration

## Done
- Repo skeleton, `uv` project (Python 3.11), dependencies, ruff and pre-commit.
- `configs/classes.yaml`: 9 main classes, 18 subclasses (+ background), colours, TR/EN names;
  loader in `parsel/utils/classes.py` with tests.
- `scripts/download_data.py`: resumable, MD5-verified download for OpenEarthMap and LoveDA.
- `docs/datasets.md`: sources, GSD, licenses, coverage against the taxonomy.
- `notebooks/01_eda.ipynb`: image size, GSD, class distribution, sample overlays.

## Dataset comparison

| | OpenEarthMap | LoveDA |
|---|---|---|
| Images | 5000 | Train / Val / Test, 1024x1024 tiles |
| GSD | 0.25 - 0.5 m, varies by region | 0.3 m, fixed |
| Coverage | 44 countries | 3 Chinese cities |
| Classes | 8 | 7 (+ no-data) |
| License | CC BY 4.0 record; labels follow source imagery (some CC BY-NC-SA 4.0) | not confirmed; Google Earth terms apply |

OpenEarthMap is the primary training source because of its geographic diversity. LoveDA adds
a second domain (urban vs rural) and a fixed GSD.

## Class imbalance
Pixel statistics are not in this report yet: the datasets have not been downloaded and the
notebook has only been run on synthetic data. Run the notebook on the real data and fill in
the dominant and rare classes here before week 2.

## Coverage gaps (from class definitions, independent of pixel counts)
Construction, solar panels (GES), greenhouses, pools, parking lots and orchards exist in
neither dataset. Roads and water are expected to be thin, low-frequency structures. These gaps
drive the custom dataset in week 5.

## Open questions
1. LoveDA mask encoding: is 0 no-data or background? Check with the notebook.
2. OpenEarthMap xBD regions: obtain the imagery or exclude those regions?
3. LoveDA license terms for any use beyond research.
4. OpenEarthMap "rangeland" and "developed space" map ambiguously onto the Parsel taxonomy
   (vegetation vs vacant land, buildings vs transport). Decide in week 2.
5. Training hardware: local Apple Silicon (MPS) or a cloud GPU.
