"""Attribution drift: which features carry a model's margin, and how much that set changes.

Importance is the mean |phi| over a set of explained rows. The top-k set takes the k most important
features among those with nonzero importance (ties by index), and two sets are compared by the Jaccard
distance 1 - |A and B| / |A or B|. This is the measure of tm-cyber-lab research/b2-drift (commit
277531a).
"""

from __future__ import annotations

import numpy as np


def importance(phi: np.ndarray) -> np.ndarray:
    """Mean absolute attribution per feature over the explained rows."""
    return np.abs(np.asarray(phi, np.float64)).mean(axis=0)


def topk(imp: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k most important features among those with nonzero importance (ties by index)."""
    nz = np.flatnonzero(imp > 0)
    if len(nz) > k:
        nz = nz[np.argsort(-imp[nz], kind="stable")[:k]]
    mask = np.zeros(imp.shape[0], bool)
    mask[nz] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard distance between two boolean feature masks (0 when both are empty)."""
    union = int(np.sum(a | b))
    return 0.0 if union == 0 else 1.0 - int(np.sum(a & b)) / union


def ranks(imp: np.ndarray) -> np.ndarray:
    """Rank of each feature by importance, 1 = most important (ties by index)."""
    order = np.argsort(-imp, kind="stable")
    r = np.empty(len(imp), np.int64)
    r[order] = np.arange(1, len(imp) + 1)
    return r


def pick(rng: np.random.Generator, idx: np.ndarray, y: np.ndarray, n_clean: int, n_mal: int) -> np.ndarray:
    """n_clean clean and n_mal malware rows drawn from idx without replacement (clean rows first)."""
    idx = np.asarray(idx)
    return np.concatenate(
        [
            rng.choice(idx[~y[idx]], n_clean, replace=False),
            rng.choice(idx[y[idx]], n_mal, replace=False),
        ]
    )
