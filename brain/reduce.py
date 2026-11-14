"""Shrink a brain to the neurons that matter, and measure what that costs.

A neuron only influences others by spiking. Under our three senses about 2,400
of the 138,639 neurons ever spike, so the rest can be dropped: a sub-network
that contains every neuron that spikes produces exactly the same spikes as the
full brain, far faster.

We cannot know that set in advance, so we measure it and then check it:

1. `find_active`  runs the full brain over a spread of inputs and records each
   neuron's highest membrane potential. Every neuron that rose at least
   `margin_mv` above rest (threshold is 7 mV above rest) is kept, a wide margin
   around the ones that actually spiked.
2. `verify`  runs full and reduced brains on new inputs with the same random
   seed and compares them. Any dropped neuron that spiked in the full brain is
   reported, and the caller adds it to the kept set.

The reduction is near-exact, not exact. Now and then a brain under heavy input
has a brief cascade that recruits a few normally silent neurons for one spike
each, and no margin predicts those. `verify` therefore reports how many spikes
were missed and how far the output neurons' counts moved, and the response
table is checked again against the live full brain (`validate_table`).

Each variant gets its own measurement, because a lesion can change which
neurons are active.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .connectome import Connectome
from .neurons import CORE_SENSES, NeuronGroups
from .sim import BrainSim
from .variants import Variant

MARGIN_MV = 2.0            # keep every neuron whose potential rose at least this far above rest
DECISION_MS = 200.0
LOOMING_LEVELS = [0, 0.5, 1, 2, 3, 5, 10, 30, 100, 200]


def calibration_conditions(seed: int, n_random: int = 40) -> np.ndarray:
    """(B, 3) sugar, bitter, looming rates covering the corners and the bulk of the input space."""
    rng = np.random.default_rng(seed)
    ends = [0.0, 200.0]
    conds = [(s, b, l) for s in ends for b in ends for l in ends]
    conds += [(s, b, l) for s in (100.0, 200.0) for b in (0.0, 60.0, 120.0) for l in (0.0, 2.0, 8.0, 30.0)]
    conds += list(zip(rng.uniform(0, 200, n_random), rng.uniform(0, 200, n_random),
                      rng.choice(LOOMING_LEVELS, n_random)))
    return np.asarray(conds, dtype=np.float32)


def random_conditions(seed: int, n: int) -> np.ndarray:
    """(n, 3) random inputs. Looming is drawn from the table levels so low rates are well covered."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 200, n), rng.uniform(0, 200, n),
                            rng.choice(LOOMING_LEVELS, n)]).astype(np.float32)


def sense_inputs(groups: NeuronGroups, conds: np.ndarray) -> list:
    return [(groups.sense(s), torch.from_numpy(np.ascontiguousarray(conds[:, i]))) for i, s in enumerate(CORE_SENSES)]


def find_active(variant: Variant, groups: NeuronGroups, device: str = "cpu", margin_mv: float = MARGIN_MV,
                seed: int = 0) -> tuple[np.ndarray, dict]:
    """Indices of the neurons to keep for this variant, measured on the full brain."""
    conds = calibration_conditions(seed)
    sim = BrainSim(variant.conn, device=device)
    state = sim.new_state(len(conds), alive=variant.alive, seed=seed)
    t0 = time.time()
    res = sim.run(state, DECISION_MS, inputs=sense_inputs(groups, conds), record_all=True, track_peak=True)
    peak = res.peak_u.amax(dim=1).cpu().numpy()
    spiked = (res.all_counts.sum(dim=1) > 0).cpu().numpy()
    near = (peak >= margin_mv) & variant.alive          # silenced neurons never matter
    keep = np.union1d(np.flatnonzero(near), np.concatenate([groups.core_inputs(), groups.all_outputs()]))
    info = {
        "margin_mv": margin_mv,
        "calibration_brains": int(len(conds)),
        "calibration_seed": seed,
        "neurons_full": int(variant.conn.n),
        "neurons_that_spiked": int(spiked.sum()),
        "neurons_kept": int(len(keep)),
        "calibration_seconds": round(time.time() - t0, 1),
    }
    return keep, info


def reduce(variant: Variant, keep: np.ndarray) -> tuple[Connectome, np.ndarray]:
    """The variant's sub-network on `keep`, and its mask of silenced neurons."""
    keep = np.unique(keep)
    return variant.conn.subset(keep), variant.alive[keep]


def verify(variant: Variant, groups: NeuronGroups, keep: np.ndarray, device: str = "cpu", seed: int = 1,
           n_brains: int = 64) -> tuple[dict, np.ndarray]:
    """Run full and reduced brains on fresh inputs with one seed and compare them.

    Returns the comparison and the indices of dropped neurons that spiked in the full brain.
    """
    conds = random_conditions(seed, n_brains)
    keep = np.unique(keep)

    full_sim = BrainSim(variant.conn, device=device)
    full = full_sim.run(full_sim.new_state(n_brains, alive=variant.alive, seed=seed), DECISION_MS,
                        inputs=sense_inputs(groups, conds), record_all=True)
    full_counts = full.all_counts.cpu().numpy()
    del full_sim

    sub, sub_alive = reduce(variant, keep)
    sub_groups = NeuronGroups(sub)
    red_sim = BrainSim(sub, device=device)
    red = red_sim.run(red_sim.new_state(n_brains, alive=sub_alive, seed=seed), DECISION_MS,
                      inputs=sense_inputs(sub_groups, conds), record_all=True)
    red_counts = red.all_counts.cpu().numpy()

    dropped = np.ones(variant.conn.n, dtype=bool)
    dropped[keep] = False
    missed = np.flatnonzero(dropped & (full_counts.sum(axis=1) > 0))
    dropped_spikes = int(full_counts[missed].sum())
    total = int(full_counts.sum())
    identical = bool(np.array_equal(full_counts[keep], red_counts))

    # how far did the output neurons move?
    outputs = groups.all_outputs()
    full_out = full_counts[outputs].astype(float)
    red_out = red_counts[np.searchsorted(keep, outputs)].astype(float)
    to_hz = 1000.0 / DECISION_MS
    return {
        "verification_brains": n_brains,
        "verification_seed": seed,
        "spikes_compared": total,
        "spikes_from_dropped_neurons": dropped_spikes,
        "dropped_spike_fraction": dropped_spikes / max(total, 1),
        "brains_with_identical_spikes": int((full_counts[keep] == red_counts).all(axis=0).sum()),
        "brains_with_identical_outputs": int((full_out == red_out).all(axis=0).sum()),
        "output_rate_mean_abs_diff_hz": float(np.abs(full_out - red_out).mean() * to_hz),
        "output_rate_max_abs_diff_hz": float(np.abs(full_out - red_out).max() * to_hz),
        "neurons_added_after_check": int(len(missed)),
        "exact": identical and dropped_spikes == 0,
    }, missed
