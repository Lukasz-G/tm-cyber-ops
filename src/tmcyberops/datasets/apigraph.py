"""APIGraph Android malware, Drebin features from Chen, Ding and Wagner (USENIX Security 2023).

    uv run python -m tmcyberops.datasets.apigraph download [--from-file drebin_features.zip]
    uv run python -m tmcyberops.datasets.apigraph prepare

`download` fetches the authors' archive from Google Drive (or copies a local copy), checks its SHA-256
and extracts the APIGraph part. `prepare` writes one `.tmx` matrix with an Arrow metadata sidecar per
month, one for the 2012 training pool, and `features.arrow`. Card: datasets/apigraph/DATASET_CARD.md.

Two properties of the source files:

* `y_train` holds family indices, not a binary flag: 0 is benign and every positive value is a malware
  family. Binary detection is `y > 0`; `y == 1` selects a single family.
* The training pool and the monthly test files differ only in name: `2012-01to2012-12_selected.npz`
  spans months, `2013-01_selected.npz` does not.

The archive stores malware family names as pickled object arrays. They are not read: `np.load` runs with
`allow_pickle=False`, so loading the archive cannot execute code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa

from tmcyberops import tmx
from tmcyberops.paths import data_dir

DRIVE_FILE_ID = "1O0upEcTolGyyvasCPkZFY86FNclk29XO"
DRIVE_PAGE = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view"
DRIVE_DIRECT = f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}&export=download&confirm=t"
ARCHIVE_NAME = "drebin_features.zip"
ARCHIVE_SHA256 = "827f36fc5affd58dd31b6393042d92cbe3bae012290a03b6fcccc6355266384d"
SUBSET = "gen_apigraph_drebin"

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})_selected\.npz$")
POOL_RE = re.compile(r"^(\d{4})-(\d{2})to(\d{4})-(\d{2})_selected\.npz$")
TEST_YEARS = range(2013, 2019)


def root() -> Path:
    return data_dir() / "apigraph"


def raw_dir() -> Path:
    return root() / "raw" / SUBSET


def tmx_dir() -> Path:
    return root() / "tmx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(from_file: str | None = None) -> Path:
    archive = root() / ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    if from_file:
        shutil.copyfile(from_file, archive)
    elif not archive.exists():
        print(f"fetching {DRIVE_PAGE}")
        with urllib.request.urlopen(DRIVE_DIRECT) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        archive.unlink()
        raise RuntimeError(
            f"SHA-256 {digest} does not match {ARCHIVE_SHA256}. If Google Drive returned a page instead of "
            f"the file, download it by hand from {DRIVE_PAGE} and pass --from-file."
        )
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if f"{SUBSET}/" in m and not m.endswith("/")]
        for m in members:
            name = Path(m).name
            if not (MONTH_RE.match(name) or POOL_RE.match(name) or name.endswith("_features.json")):
                continue
            with z.open(m) as src, open(_out(name), "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"archive ok ({digest[:12]}...), extracted to {raw_dir()}")
    return raw_dir()


def _out(name: str) -> Path:
    raw_dir().mkdir(parents=True, exist_ok=True)
    return raw_dir() / name


def feature_names(src: Path) -> list[str]:
    """Column names from the pool's `*_selected_training_features.json`, a name -> index mapping."""
    cands = sorted(src.glob("*to*_selected_training_features.json"))
    if not cands:
        raise RuntimeError(f"no *to*_selected_training_features.json in {src}")
    mapping = json.loads(cands[-1].read_text(encoding="utf-8"))
    names: list[str | None] = [None] * len(mapping)
    for name, idx in mapping.items():
        names[int(idx)] = name
    if any(n is None for n in names):
        raise RuntimeError(f"{cands[-1]}: index mapping has gaps")
    return names  # type: ignore[return-value]


def _load_npz(path: Path, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        X = f["X_train"]
        fam = f["y_train"].astype(np.int64)
    if X.shape[1] != n_cols:
        raise RuntimeError(f"{path.name}: width {X.shape[1]}, expected {n_cols}")
    if not set(np.unique(X).tolist()) <= {0, 1}:
        raise RuntimeError(f"{path.name}: features are not binary")
    return X.astype(bool), fam


def _write(X: np.ndarray, fam: np.ndarray, stem: str, year: int, month: int, split: str) -> int:
    n = len(fam)
    label = fam > 0
    tmx.write_tmx(tmx_dir() / f"{stem}.tmx", X)
    tmx.write_arrow(
        tmx_dir() / f"{stem}.meta.arrow",
        pa.table(
            {
                "label": pa.array(label, type=pa.bool_()),
                "year": pa.array(np.full(n, year, np.int16), type=pa.int16()),
                "month": pa.array(np.full(n, month, np.int8), type=pa.int8()),
                "split": pa.array([split] * n, type=pa.string()),
                "family_index": pa.array(fam, type=pa.int32()),
            }
        ),
    )
    tmx.check_consistent(tmx_dir() / f"{stem}.tmx", tmx_dir() / f"{stem}.meta.arrow")
    return int(label.sum())


def prepare() -> None:
    src = raw_dir()
    names = feature_names(src)
    tmx.write_features(tmx_dir() / "features.arrow", names)
    files = sorted(src.glob("*_selected.npz"))
    pools = [p for p in files if POOL_RE.match(p.name)]
    if len(pools) != 1:
        raise RuntimeError(f"expected one training pool, found {[p.name for p in pools]}")
    X, fam = _load_npz(pools[0], len(names))
    mal = _write(X, fam, "train", int(POOL_RE.match(pools[0].name).group(1)), 0, "train")
    print(f"width {len(names)}")
    print(f"train   {pools[0].name}: {len(fam):>7,} rows, {mal:>6,} malware ({100 * mal / len(fam):.1f}%)")
    for p in files:
        m = MONTH_RE.match(p.name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        X, fam = _load_npz(p, len(names))
        mal = _write(X, fam, f"{year}-{month:02d}", year, month, "test")
        print(
            f"month   {year}-{month:02d}: {len(fam):>7,} rows, {mal:>6,} malware ({100 * mal / max(1, len(fam)):.1f}%)"
        )


def load(stem: str) -> tuple[np.ndarray, np.ndarray]:
    X = tmx.read_tmx(tmx_dir() / f"{stem}.tmx")
    y = tmx.read_arrow(tmx_dir() / f"{stem}.meta.arrow").column("label").to_numpy(zero_copy_only=False)
    return X, y.astype(bool)


def load_train() -> tuple[np.ndarray, np.ndarray]:
    return load("train")


def load_year(year: int) -> tuple[np.ndarray, np.ndarray]:
    parts = [load(f"{year}-{m:02d}") for m in range(1, 13) if (tmx_dir() / f"{year}-{m:02d}.tmx").exists()]
    if not parts:
        raise FileNotFoundError(f"no months for {year} in {tmx_dir()}")
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def features() -> list[str]:
    return tmx.read_features(tmx_dir() / "features.arrow")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tmcyberops.datasets.apigraph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--from-file", help="a local copy of the archive; its SHA-256 is checked")
    sub.add_parser("prepare")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.from_file)
    else:
        prepare()
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
