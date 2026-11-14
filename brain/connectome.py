"""Load Shiu's connectivity file into a sparse matrix sorted by presynaptic neuron.

The weight of a connection is `synapse count x (+1 or -1)`, read straight from
the `Excitatory x Connectivity` column, so the transmitter signs are Shiu's and
are never re-derived from raw predictions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    env = os.environ.get("FLY_DATA_DIR")
    return Path(env) if env else ROOT / "external" / "Drosophila_brain_model"


@dataclass
class Connectome:
    """W in compressed form, one row per *presynaptic* neuron.

    Outgoing connections of neuron i are `post[ptr[i]:ptr[i+1]]` with signed
    synapse counts `signed[ptr[i]:ptr[i+1]]`.
    """

    ids: np.ndarray      # (N,)   int64  FlyWire root ID of each model index
    ptr: np.ndarray      # (N+1,) int64
    post: np.ndarray     # (nnz,) int32  postsynaptic model index
    signed: np.ndarray   # (nnz,) int32  synapse count x sign

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def nnz(self) -> int:
        return len(self.post)

    @classmethod
    def from_edges(cls, ids: np.ndarray, pre: np.ndarray, post: np.ndarray, signed: np.ndarray) -> "Connectome":
        n = len(ids)
        order = np.argsort(pre, kind="stable")
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=ptr[1:])
        return cls(
            ids=np.asarray(ids, dtype=np.int64),
            ptr=ptr,
            post=np.ascontiguousarray(post[order], dtype=np.int32),
            signed=np.ascontiguousarray(signed[order], dtype=np.int32),
        )

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> "Connectome":
        d = Path(data_dir) if data_dir else default_data_dir()
        comp = pd.read_csv(d / "Completeness_783.csv", index_col=0)
        con = pd.read_parquet(
            d / "Connectivity_783.parquet",
            columns=["Presynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
        )
        ids = comp.index.to_numpy(dtype=np.int64)
        pre = con["Presynaptic_Index"].to_numpy()
        # The model index must be the row number in the completeness file.
        if not np.array_equal(ids[pre], con["Presynaptic_ID"].to_numpy()):
            raise ValueError("Connectivity indices do not match the completeness file order")
        return cls.from_edges(ids, pre, con["Postsynaptic_Index"].to_numpy(),
                              con["Excitatory x Connectivity"].to_numpy())

    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pre, post, signed) arrays, one entry per connection."""
        pre = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.ptr))
        return pre, self.post, self.signed

    def index_of(self, flywire_ids) -> np.ndarray:
        """Model indices for FlyWire root IDs. Raises if any ID is not in the model."""
        want = np.asarray(list(flywire_ids), dtype=np.int64)
        order = np.argsort(self.ids)
        pos = np.searchsorted(self.ids, want, sorter=order)
        pos = np.clip(pos, 0, self.n - 1)
        idx = order[pos]
        missing = want[self.ids[idx] != want]
        if len(missing):
            raise KeyError(f"{len(missing)} FlyWire IDs are not in the model, e.g. {missing[:3].tolist()}")
        return idx.astype(np.int64)
