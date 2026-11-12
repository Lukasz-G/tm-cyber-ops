"""Fuzzy-Pattern Tsetlin machine (TMCore), run in a Julia subprocess.

Matrices cross to Julia as `.tmx` files and results come back as Arrow or raw Float64 files
(julia/tm.jl). Each call starts one Julia process.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

from tmcyberops import tmx
from tmcyberops.paths import JULIA_PROJECT, julia_exe

TM_SCRIPT = JULIA_PROJECT / "tm.jl"


def run_julia(*args: str | Path, threads: str = "auto") -> subprocess.CompletedProcess:
    cmd = [julia_exe(), f"--project={JULIA_PROJECT}", "-t", threads, str(TM_SCRIPT), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"julia failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc


class TsetlinDetector:
    name = "tm"
    threshold_rule = "class score of malicious > class score of benign (margin > 0; ties benign)"

    def __init__(
        self,
        clauses_per_class: int = 20,
        T: int = 10,
        S: int = 100,
        L: int = 64,
        LF: int = 10,
        epochs: int = 30,
        seed: int = 1,
        workdir: str | Path | None = None,
    ):
        self.config = dict(
            clauses_per_class=clauses_per_class, T=T, S=S, L=L, LF=LF, epochs=epochs, seed=seed
        )
        self._tmp = None if workdir else tempfile.TemporaryDirectory(prefix="tmco-")
        self.workdir = Path(workdir) if workdir else Path(self._tmp.name)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.workdir / f"tm-s{S}-seed{seed}.tmcore"
        self.fit_log = ""

    def _matrix(self, X: np.ndarray, stem: str) -> Path:
        p = self.workdir / f"{stem}.tmx"
        tmx.write_tmx(p, np.asarray(X, bool))
        return p

    def fit(self, X: np.ndarray, y: np.ndarray) -> TsetlinDetector:
        xp = self._matrix(X, "fit")
        yp = self.workdir / "fit.labels.arrow"
        tmx.write_arrow(yp, pa.table({"label": pa.array(np.asarray(y, bool), type=pa.bool_())}))
        cp = self.workdir / "fit.config.json"
        cp.write_text(json.dumps(self.config), encoding="utf-8")
        self.fit_log = run_julia("fit", xp, yp, self.model_path, cp).stderr.strip()
        return self

    def scores(self, X: np.ndarray) -> pa.Table:
        xp = self._matrix(X, "score")
        out = self.workdir / "score.arrow"
        run_julia("score", self.model_path, xp, out)
        return tmx.read_arrow(out)

    def margin(self, X: np.ndarray) -> np.ndarray:
        t = self.scores(X)
        return (t.column("score_true").to_numpy() - t.column("score_false").to_numpy()).astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X).column("pred").to_numpy(zero_copy_only=False).astype(bool)

    def attribute(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        xp = self._matrix(X, "explain")
        bp = self._matrix(background, "background")
        out = self.workdir / "phi.f64"
        run_julia("shapley", self.model_path, xp, bp, out)
        return np.fromfile(out, dtype="<f8").reshape(len(X), np.asarray(X).shape[1])
