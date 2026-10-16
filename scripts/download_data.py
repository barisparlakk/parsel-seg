"""Download the public datasets used by Parsel into data/raw/.

Usage:
    uv run python scripts/download_data.py --list
    uv run python scripts/download_data.py --dataset openearthmap
    uv run python scripts/download_data.py --dataset loveda --splits Train Val

Downloads are resumable and verified against the published MD5 checksums.
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "openearthmap": {
        "doi": "10.5281/zenodo.7223446",
        "size": "9.1 GB",
        "files": {
            "OpenEarthMap": (
                "https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1",
                "64155d1dc9d3b68536063f79878e1a67",
            ),
        },
        "note": (
            "xBD imagery is not bundled (license). Download it from xview2.org and place it "
            "as described in xbd_files.csv inside the archive; until then those regions have "
            "labels only and must be skipped."
        ),
    },
    "loveda": {
        "doi": "10.5281/zenodo.5706578",
        "size": "~6.7 GB (Train) + ~8.9 GB (Val) + ~2.9 GB (Test)",
        "files": {
            "Train": (
                "https://zenodo.org/record/5706578/files/Train.zip?download=1",
                "de2b196043ed9b4af1690b3f9a7d558f",
            ),
            "Val": (
                "https://zenodo.org/record/5706578/files/Val.zip?download=1",
                "84cae2577468ff0b5386758bb386d31d",
            ),
            "Test": (
                "https://zenodo.org/record/5706578/files/Test.zip?download=1",
                "a489be0090465e01fb067795d24e6b47",
            ),
        },
        "note": "Test masks are not public; use Val as the held-out set for evaluation.",
    },
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download url to dest, resuming a partial `.part` file when possible."""
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req) as resp:
        if start and resp.status != 206:  # server ignored Range; restart
            start = 0
        total = start + int(resp.headers.get("Content-Length", 0))
        done = start
        with open(part, "ab" if start else "wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    part.rename(dest)


def fetch(name: str, splits: list[str] | None) -> None:
    info = DATASETS[name]
    out = RAW_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{name} (DOI {info['doi']}): {info['note']}")
    for key, (url, md5) in info["files"].items():
        if splits and key not in splits:
            continue
        archive = out / f"{key}.zip"
        if not archive.exists():
            print(f"Downloading {key} -> {archive}")
            download(url, archive)
        if md5sum(archive) != md5:
            sys.exit(f"Checksum mismatch for {archive}; delete it and retry.")
        print(f"Extracting {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=[*DATASETS, "all"])
    p.add_argument("--splits", nargs="+", help="LoveDA only: Train / Val / Test")
    p.add_argument("--list", action="store_true", help="show datasets and exit")
    args = p.parse_args()

    if args.list or not args.dataset:
        for name, info in DATASETS.items():
            print(f"{name}: {info['size']}  doi:{info['doi']}\n  {info['note']}")
        return
    for name in DATASETS if args.dataset == "all" else [args.dataset]:
        fetch(name, args.splits if name == "loveda" else None)


if __name__ == "__main__":
    main()
