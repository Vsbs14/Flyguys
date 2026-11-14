"""Read the Phase 3 response tables and turn senses into sampled actions. NumPy only.

For intensities between grid points, the action probabilities of the 2**5
surrounding grid points are blended (multilinear interpolation), then one
action is drawn per colony. At a grid point this reproduces exactly the share
of replicate brains that chose each action.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
SENSES = ("sugar", "bitter", "looming", "obstacle", "target")
ACTIONS = ("idle", "feed", "escape", "retreat", "advance")
OUTPUTS = ("feed", "escape", "retreat", "advance")
NEURONS = {"feed": "MN9", "escape": "GF", "retreat": "MDN", "advance": "DNp09"}


class ResponseTable:
    def __init__(self, levels: np.ndarray, counts: np.ndarray, name: str = "", rates_hz: np.ndarray | None = None):
        self.name = name
        self.levels = np.asarray(levels, dtype=np.float64)
        L, D = len(self.levels), len(SENSES)
        counts = np.asarray(counts, dtype=np.float64).reshape((L,) * D + (len(ACTIONS),))
        self.p = counts / counts.sum(-1, keepdims=True)                      # (L,)*D + (A,)
        self.r = None if rates_hz is None else np.asarray(rates_hz, np.float64).reshape((L,) * D + (len(OUTPUTS),))
        self._corners = np.array(list(itertools.product((0, 1), repeat=D)))  # (2**D, D)

    @classmethod
    def load(cls, name: str, directory: Path = TABLES) -> "ResponseTable":
        z = np.load(directory / f"{name}.npz")
        if tuple(z["senses"]) != SENSES or tuple(z["actions"]) != ACTIONS:
            raise ValueError(f"{name}.npz was built for different senses/actions")
        return cls(z["levels"], z["counts"], name, z["rates_hz"] if "rates_hz" in z else None)

    def _interp(self, arr: np.ndarray, x) -> np.ndarray:
        x = np.clip(np.atleast_2d(np.asarray(x, dtype=np.float64)), 0.0, 1.0)
        lv = self.levels
        i0 = np.clip(np.searchsorted(lv, x, side="right") - 1, 0, len(lv) - 2)    # (C, D)
        w = (x - lv[i0]) / (lv[i0 + 1] - lv[i0])                                   # (C, D) in [0, 1]
        out = np.zeros((len(x), arr.shape[-1]))
        for c in self._corners:
            weight = np.prod(np.where(c, w, 1.0 - w), axis=1)                      # (C,)
            out += weight[:, None] * arr[tuple((i0 + c).T)]
        return out

    def probs(self, x) -> np.ndarray:
        """(C, senses) intensities in [0, 1] -> (C, actions) probabilities."""
        return self._interp(self.p, x)

    def rates(self, x) -> np.ndarray | None:
        """(C, senses) -> (C, outputs) mean output firing rates in Hz, or None if the table has none."""
        return None if self.r is None else self._interp(self.r, x)

    def sample(self, x, rng: np.random.Generator) -> np.ndarray:
        """(C, senses) -> (C,) action indices drawn from the interpolated probabilities."""
        p = self.probs(x)
        cum = np.cumsum(p, axis=1)
        u = rng.random((len(p), 1)) * cum[:, -1:]
        return (u > cum).sum(1)


def available() -> dict:
    """Variant name -> index entry, for every table that has been built."""
    path = TABLES / "index.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text(encoding="utf8"))["tables"].items()
            if (TABLES / v["file"]).exists()}


def load_all(names=None) -> dict[str, ResponseTable]:
    names = names or list(available())
    if not names:
        raise FileNotFoundError("no response tables in tables/; run `python -m brain.tables` (Phase 3)")
    return {n: ResponseTable.load(n) for n in names}
