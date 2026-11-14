"""Response tables: the fly's answers, cached.

For every point on a grid of (sugar, bitter, looming) input rates we run the
brain several times from rest and store what the output neurons did. The galaxy
simulation then looks decisions up instead of simulating a brain per colony.

Stored per grid point, over the seeds:
  mean and SD of each output's firing rate (feed, escape, retreat, advance)
  probability that the Giant Fibre spiked at least once

The looming axis is unevenly spaced. The Giant Fibre goes from never firing to
always firing between about 1 and 5 Hz of looming input, so most of the grid
points sit there.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .connectome import ROOT
from .neurons import NeuronGroups
from .reduce import DECISION_MS, LOOMING_LEVELS, random_conditions, sense_inputs
from .sim import BrainSim

TABLE_DIR = ROOT / "tables"
SUGAR_HZ = np.linspace(0.0, 200.0, 10)
BITTER_HZ = np.linspace(0.0, 200.0, 10)
LOOMING_HZ = np.asarray(LOOMING_LEVELS, dtype=float)
OUTPUTS = ["feed", "escape", "retreat", "advance"]      # MN9, Giant Fibre, MDN, DNp09


def output_slices(groups: NeuronGroups) -> dict[str, slice]:
    slices, at = {}, 0
    for o in groups.outputs:
        n = len(groups.output(o))
        slices[o] = slice(at, at + n)
        at += n
    return slices


def run_decisions(sim: BrainSim, groups: NeuronGroups, conds: np.ndarray, alive: np.ndarray | None, seed: int,
                  batch: int = 256, progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One decision per row of `conds`. Returns output rates (B, 4) in Hz and Giant Fibre spiked (B,)."""
    monitor = torch.from_numpy(groups.all_outputs())
    slices = output_slices(groups)
    rates = np.zeros((len(conds), len(OUTPUTS)), dtype=np.float32)
    gf = np.zeros(len(conds), dtype=bool)
    t0 = time.time()
    for k, start in enumerate(range(0, len(conds), batch)):
        chunk = conds[start:start + batch]
        state = sim.new_state(len(chunk), alive=alive, seed=seed * 100_003 + k)
        res = sim.run(state, DECISION_MS, inputs=sense_inputs(groups, chunk), monitor=monitor)
        r = res.rates_hz.cpu().numpy()
        for j, o in enumerate(OUTPUTS):
            rates[start:start + len(chunk), j] = r[slices[o]].mean(axis=0)
        gf[start:start + len(chunk)] = res.counts[slices["escape"]].sum(dim=0).cpu().numpy() > 0
        if progress:
            done = start + len(chunk)
            print(f"    {done}/{len(conds)} decisions, {done / (time.time() - t0):.1f} per second", flush=True)
    return rates, gf


def build_table(sim: BrainSim, groups: NeuronGroups, alive: np.ndarray | None, seeds: int = 5, seed: int = 0,
                batch: int = 256, progress: bool = False) -> dict[str, np.ndarray]:
    grid = np.stack(np.meshgrid(SUGAR_HZ, BITTER_HZ, LOOMING_HZ, indexing="ij"), axis=-1).reshape(-1, 3)
    conds = np.repeat(grid, seeds, axis=0).astype(np.float32)
    rates, gf = run_decisions(sim, groups, conds, alive, seed, batch, progress)
    shape = (len(SUGAR_HZ), len(BITTER_HZ), len(LOOMING_HZ))
    rates = rates.reshape(*shape, seeds, len(OUTPUTS))
    return {
        "sugar_hz": SUGAR_HZ, "bitter_hz": BITTER_HZ, "looming_hz": LOOMING_HZ,
        "mean": rates.mean(axis=3), "sd": rates.std(axis=3),
        "p_gf": gf.reshape(*shape, seeds).mean(axis=3).astype(np.float32),
        "seeds": np.asarray(seeds), "decision_ms": np.asarray(DECISION_MS),
    }


@dataclass
class ResponseTable:
    """A cached brain. `lookup` interpolates between grid points; `sample` draws one decision."""

    name: str
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    mean: np.ndarray      # (S, B, L, 4) Hz
    sd: np.ndarray        # (S, B, L, 4) Hz
    p_gf: np.ndarray      # (S, B, L)

    @classmethod
    def load(cls, name: str, directory: str | Path = TABLE_DIR) -> "ResponseTable":
        z = np.load(Path(directory) / f"{name}.npz")
        return cls(name, (z["sugar_hz"], z["bitter_hz"], z["looming_hz"]), z["mean"], z["sd"], z["p_gf"])

    def _weights(self, sugar, bitter, looming):
        out = []
        for axis, x in zip(self.axes, (sugar, bitter, looming)):
            x = np.clip(np.asarray(x, dtype=float), axis[0], axis[-1])
            i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2)
            out.append((i, (x - axis[i]) / (axis[i + 1] - axis[i])))
        return out

    def _interp(self, values: np.ndarray, w) -> np.ndarray:
        (i, ti), (j, tj), (k, tk) = w
        total = 0.0
        for di, wi in ((0, 1 - ti), (1, ti)):
            for dj, wj in ((0, 1 - tj), (1, tj)):
                for dk, wk in ((0, 1 - tk), (1, tk)):
                    weight = wi * wj * wk
                    corner = values[i + di, j + dj, k + dk]
                    total = total + (weight[..., None] * corner if corner.ndim > weight.ndim else weight * corner)
        return total

    def lookup(self, sugar, bitter, looming) -> dict[str, np.ndarray]:
        """Expected response at any input rates (Hz). Inputs broadcast; rates come back as (..., 4)."""
        sugar, bitter, looming = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (sugar, bitter, looming)))
        w = self._weights(sugar, bitter, looming)
        return {"mean": self._interp(self.mean, w), "sd": self._interp(self.sd, w), "p_gf": self._interp(self.p_gf, w)}

    def sample(self, sugar, bitter, looming, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """One noisy decision per input: output rates (..., 4) in Hz and whether the Giant Fibre spiked."""
        r = self.lookup(sugar, bitter, looming)
        rates = np.maximum(rng.normal(r["mean"], r["sd"]), 0.0)
        return {"rates": rates, "gf": rng.random(r["p_gf"].shape) < r["p_gf"]}


def validate_table(table: ResponseTable, sim: BrainSim, groups: NeuronGroups, alive: np.ndarray | None,
                   n: int = 200, seed: int = 12345, batch: int = 256) -> dict:
    """Step 3.3: run the live brain on random inputs and compare with the table's interpolation."""
    conds = random_conditions(seed, n)
    # off-grid looming: jitter within the neighbouring levels so interpolation is really tested
    rng = np.random.default_rng(seed + 1)
    conds[:, 2] = np.clip(conds[:, 2] * rng.uniform(0.6, 1.4, n), 0, 200)
    live_rates, live_gf = run_decisions(sim, groups, conds, alive, seed, batch)
    pred = table.lookup(conds[:, 0], conds[:, 1], conds[:, 2])
    out = {"n": n, "seed": seed, "outputs": {}}
    for j, o in enumerate(OUTPUTS):
        err = live_rates[:, j] - pred["mean"][:, j]
        spread = float(pred["sd"][:, j].mean())
        live_sd, pred_sd = live_rates[:, j].std(), pred["mean"][:, j].std()
        corr = float(np.corrcoef(live_rates[:, j], pred["mean"][:, j])[0, 1]) if live_sd > 0 and pred_sd > 0 else None
        out["outputs"][o] = {
            "live_mean_hz": float(live_rates[:, j].mean()),
            "table_mean_hz": float(pred["mean"][:, j].mean()),
            "mean_abs_error_hz": float(np.abs(err).mean()),
            "bias_hz": float(err.mean()),
            "trial_to_trial_sd_hz": spread,
            "correlation": corr,
        }
    p = pred["p_gf"]
    out["giant_fibre"] = {
        "live_spike_fraction": float(live_gf.mean()),
        "table_mean_probability": float(p.mean()),
        "agreement": float(((p >= 0.5) == live_gf).mean()),
        "brier_score": float(((p - live_gf) ** 2).mean()),
    }
    return out
