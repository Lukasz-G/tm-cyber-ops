"""The `.tmx` boundary: Python round trip, and Python writer against the Julia reader that feeds TMCore."""

from __future__ import annotations

import subprocess

import numpy as np
import pyarrow as pa
import pytest

from tmcyberops import tmx
from tmcyberops.paths import JULIA_PROJECT, REPO, julia_exe

# Widths that catch padding and word-boundary errors: 1 and 64 are degenerate, 65 spills one bit into a
# second word, 1159 is APIGraph's width, 4561 is not a multiple of 64.
SHAPES = ((1, 1), (3, 64), (5, 65), (7, 1159), (9, 4561))


def fixtures(tmp_path):
    rng = np.random.default_rng(20261112)
    out = {}
    for n_rows, n_cols in SHAPES:
        rows = rng.random((n_rows, n_cols)) < 0.05
        rows[0, 0] = True  # first bit of the first word
        rows[-1, n_cols - 1] = True  # last real bit, next to the zero padding
        stem = f"b_{n_rows}x{n_cols}"
        tmx.write_tmx(tmp_path / f"{stem}.tmx", rows)
        meta = pa.table(
            {
                "label": pa.array(rows[:, 0], type=pa.bool_()),
                "year": pa.array([2013] * n_rows, type=pa.int16()),
                "month": pa.array([1] * n_rows, type=pa.int8()),
                "split": pa.array(["train"] * n_rows, type=pa.string()),
            }
        )
        tmx.write_arrow(tmp_path / f"{stem}.meta.arrow", meta)
        tmx.write_features(tmp_path / f"{stem}.features.arrow", [f"f{j}" for j in range(n_cols)])
        tmx.check_consistent(
            tmp_path / f"{stem}.tmx", tmp_path / f"{stem}.meta.arrow", tmp_path / f"{stem}.features.arrow"
        )
        out[stem] = rows
    return out


def row_line(r: np.ndarray) -> str:
    return " ".join(str(i + 1) for i in np.flatnonzero(r))


def test_python_round_trip(tmp_path):
    for stem, rows in fixtures(tmp_path).items():
        back = tmx.read_tmx(tmp_path / f"{stem}.tmx")
        assert back.shape == rows.shape and np.array_equal(back, rows), stem


def test_padding_is_zero(tmp_path):
    rows = np.ones((2, 65), bool)
    tmx.write_tmx(tmp_path / "p.tmx", rows)
    words = np.fromfile(tmp_path / "p.tmx", dtype="<u8", offset=tmx.HEADER_BYTES).reshape(2, 2)
    assert np.all(words[:, 1] == 1)


def test_julia_reads_the_same_bits(tmp_path):
    try:
        julia = julia_exe()
    except RuntimeError:
        pytest.skip("julia not installed")
    rows_by_stem = fixtures(tmp_path)
    (tmp_path / "manifest.txt").write_text("\n".join(rows_by_stem) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [julia, f"--project={JULIA_PROJECT}", str(REPO / "tests" / "boundary.jl"), str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    for stem, rows in rows_by_stem.items():
        actual = (tmp_path / f"{stem}.actual.txt").read_text(encoding="utf-8").splitlines()
        assert actual == [row_line(r) for r in rows], stem
