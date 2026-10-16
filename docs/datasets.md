# Datasets

Parsel starts from two public land-cover datasets and adds a custom Turkish set in week 5.
Raw data lives in `data/raw/` (git-ignored). Fetch it with `scripts/download_data.py`.

## OpenEarthMap

| | |
|---|---|
| Source | https://zenodo.org/record/7223446 (DOI 10.5281/zenodo.7223446) |
| Size | 5000 images, 97 regions, 44 countries; archive about 9.1 GB |
| GSD | 0.25 - 0.5 m |
| Classes | bareland, rangeland, developed space, road, tree, water, agriculture land, building |
| License | Zenodo record states CC BY 4.0. Labels follow the license of each source image set, which varies; labels for regions with public-domain or unstated imagery are CC BY-NC-SA 4.0. |

Caveat: imagery for the xBD regions is not distributed in the archive. It has to be obtained
from xview2.org and placed per `xbd_files.csv`. Until then those regions are label-only.

Citation: Xia et al., "OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land
Cover Mapping", WACV 2023.

## LoveDA

| | |
|---|---|
| Source | https://zenodo.org/record/5706578 (DOI 10.5281/zenodo.5706578) |
| Size | Train / Val / Test archives, roughly 6.7 / 8.9 / 2.9 GB |
| GSD | 0.3 m, 1024 x 1024 px PNG tiles (Google Earth imagery) |
| Domains | Urban and Rural scenes (three Chinese cities) |
| Classes | background, building, road, water, barren, forest, agriculture (+ no-data) |
| License | Not confirmed in this review. The authors (Wuhan University, RSIDEA) state the imagery is subject to the Google Earth terms of use. Check the Zenodo record before any non-research use. |

Caveats:
- Public test masks are not distributed; use Val as the held-out split.
- Sources disagree on the mask encoding (0 = no-data vs 0 = background). Verify on the raw
  masks in the EDA notebook before writing the class mapping in week 2.

Citation: Wang et al., "LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive
Semantic Segmentation", NeurIPS Datasets and Benchmarks 2021.

## Coverage against the Parsel taxonomy

| Parsel main class | OpenEarthMap | LoveDA |
|---|---|---|
| Buildings | building | building |
| Construction | none | none |
| Vacant land | bareland | barren |
| Agriculture | agriculture land | agriculture |
| Transport | road | road |
| Water | water | water |
| Vegetation | tree, rangeland | forest |
| Energy (solar) | none | none |

Neither dataset has construction sites, solar panels, greenhouses, pools as a separate class,
parking lots or orchards. These need the custom dataset (week 5). The final class mapping
files are written in week 2 under `configs/mappings/`.
