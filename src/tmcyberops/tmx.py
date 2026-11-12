"""Write and read the `.tmx` packed binary matrix, the boundary between Python and Julia.

Format: docs/matrix-format.md. The on-disk bit layout is the layout TMCore's `TMInput` uses in memory,
so Julia loads a row by copying words. Ported from tm-cyber-lab (python/tmcyber/tmx.py, commit e6f8fb8).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

MAGIC = b"TMCYBERX"
VERSION = 1
HEADER_BYTES = 64
HEADER_STRUCT = "<8sIIQQ32x"
REQUIRED_META_COLUMNS = ("label", "year", "month", "split")


@dataclass(frozen=True)
class TmxHeader:
    version: int
    flags: int
    n_rows: int
    n_cols: int

    @property
    def words_per_row(self) -> int:
        return (self.n_cols + 63) // 64


def pack(rows: np.ndarray) -> np.ndarray:
    """Pack a bool matrix into little-endian uint64 words, row-major, least significant bit first.

    Padding bits in the last word of each row are zero. TMCore requires clause include-masks to be zero
    there too, so non-zero padding would train a different model.
    """
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {rows.shape}")
    rows = np.ascontiguousarray(rows, dtype=bool)
    n_rows, n_cols = rows.shape
    pad = (-n_cols) % 64
    if pad:
        rows = np.hstack([rows, np.zeros((n_rows, pad), dtype=bool)])
    return np.packbits(rows, axis=1, bitorder="little").view("<u8")


def write_tmx(path: str | Path, rows: np.ndarray) -> TmxHeader:
    """Write a bool matrix as `.tmx` and return the header written."""
    path = Path(path)
    n_rows, n_cols = rows.shape
    words = pack(rows)
    header = struct.pack(HEADER_STRUCT, MAGIC, VERSION, 0, n_rows, n_cols)
    assert len(header) == HEADER_BYTES, len(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        words.tofile(f)
    return TmxHeader(VERSION, 0, n_rows, n_cols)


def read_header(path: str | Path) -> TmxHeader:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: shorter than a {HEADER_BYTES}-byte header")
    magic, version, flags, n_rows, n_cols = struct.unpack(HEADER_STRUCT, raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, this reader reads version {VERSION}")
    return TmxHeader(version, flags, n_rows, n_cols)


def read_tmx(path: str | Path) -> np.ndarray:
    """Read a `.tmx` into a bool matrix (the same bits the Tsetlin machine reads)."""
    h = read_header(path)
    words = np.fromfile(path, dtype="<u8", offset=HEADER_BYTES)
    expected = h.n_rows * h.words_per_row
    if words.size != expected:
        raise ValueError(f"{path}: {words.size} words, expected {expected}")
    bits = np.unpackbits(words.reshape(h.n_rows, h.words_per_row).view(np.uint8), axis=1, bitorder="little")
    return bits[:, : h.n_cols].astype(bool)


def write_arrow(path: str | Path, table: pa.Table) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(str(path), table.schema) as w:
        w.write_table(table)


def read_arrow(path: str | Path) -> pa.Table:
    with ipc.open_file(str(path)) as r:
        return r.read_all()


def write_features(path: str | Path, names: list[str]) -> None:
    """Feature names, one row per matrix column, in column order."""
    write_arrow(path, pa.table({"name": pa.array(list(names), type=pa.string())}))


def read_features(path: str | Path) -> list[str]:
    return read_arrow(path).column("name").to_pylist()


def check_consistent(
    tmx_path: str | Path, meta_path: str | Path, features_path: str | Path | None = None
) -> TmxHeader:
    """Check the invariants the Julia reader assumes. Cheap; run after every write."""
    h = read_header(tmx_path)
    meta = read_arrow(meta_path)
    if meta.num_rows != h.n_rows:
        raise ValueError(f"metadata has {meta.num_rows} rows, matrix has {h.n_rows}")
    for col in REQUIRED_META_COLUMNS:
        if col not in meta.column_names:
            raise ValueError(f"metadata is missing required column {col}")
    if features_path is not None:
        n_names = read_arrow(features_path).num_rows
        if n_names != h.n_cols:
            raise ValueError(f"{n_names} feature names for {h.n_cols} matrix columns")
    return h
