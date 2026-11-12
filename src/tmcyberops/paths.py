"""Where data, the Julia project and scratch files live.

Datasets go under `data/` in the repository by default, which git ignores. Set `TMCO_DATA` to keep them
elsewhere.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JULIA_PROJECT = REPO / "julia"


def data_dir() -> Path:
    return Path(os.environ.get("TMCO_DATA", REPO / "data"))


def julia_exe() -> str:
    exe = os.environ.get("JULIA") or shutil.which("julia")
    if not exe:
        raise RuntimeError("julia not found on PATH; install Julia 1.11 or set JULIA to its executable")
    return exe
