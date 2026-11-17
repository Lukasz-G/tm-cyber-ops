"""Platform configuration (ops/platform.yaml) and the platform's working directory."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml

from tmcyberops.paths import REPO

MODEL_LABELS = {"fptm": "FPTM", "xgboost": "XGBoost", "logreg": "Logistic regression"}
ARM_LABELS = {"frozen": "frozen", "retrain": "retrain", "incremental": "incremental"}


def config_path() -> Path:
    return Path(os.environ.get("TMCO_PLATFORM_CONFIG", REPO / "ops" / "platform.yaml"))


@cache
def load() -> dict:
    return yaml.safe_load(config_path().read_text(encoding="utf-8"))


def home() -> Path:
    """Models, per-month arrays, the state database and Dagster's run storage (not committed)."""
    p = Path(os.environ.get("TMCO_PLATFORM_DIR", REPO / "work" / "platform"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def months(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load()
    y0, m0 = map(int, cfg["replay"]["first_month"].split("-"))
    y1, m1 = map(int, cfg["replay"]["last_month"].split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def previous_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"


def pairs(cfg: dict | None = None) -> list[tuple[str, str]]:
    """Every (model, arm) the platform runs."""
    cfg = cfg or load()
    return [(m, a) for m in cfg["models"] for a in cfg["arms"]]
