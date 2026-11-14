"""Check the PyTorch port against Brian2, spike for spike.

`scripts/brian2_equivalence.py` runs a small random recurrent network in Shiu's
original Brian2 model with no random input, so its spike trains are fully
determined. This replays the same network in the port (float64) and compares
every spike's neuron and time step.

  python -m brain.test_equivalence
"""
from __future__ import annotations

import numpy as np
import torch

from .connectome import ROOT, Connectome
from .sim import BrainSim

CASE = ROOT / "data" / "brian2_equivalence_case.npz"


def run_port(case, n_brains: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Spikes (neuron, step) from the port for brain 0. All brains are identical copies."""
    n = len(case["v_init"])
    conn = Connectome.from_edges(np.arange(n), case["pre"], case["post"], case["signed"])
    sim = BrainSim(conn, dtype=torch.float64)
    state = sim.new_state(n_brains)
    state.u[:] = torch.from_numpy(case["v_init"] - sim.p.v_rest)[:, None]
    state.exempt[:] = torch.from_numpy(case["no_refrac"])
    neurons, steps = [], []
    for k in range(round(float(case["t_ms"]) / sim.p.dt)):
        res = sim.run(state, sim.p.dt, record_all=True)
        counts = res.all_counts
        assert bool((counts == counts[:, :1]).all()), "brains in one batch diverged"
        fired = counts[:, 0].nonzero().reshape(-1).numpy()
        neurons.append(fired)
        steps.append(np.full(len(fired), k))
    return np.concatenate(neurons), np.concatenate(steps)


def main() -> None:
    case = np.load(CASE)
    ref = set(zip(case["spike_neuron"].tolist(), case["spike_step"].tolist()))
    neurons, steps = run_port(case)
    ours = set(zip(neurons.tolist(), steps.tolist()))
    print(f"Brian2: {len(ref)} spikes. Port: {len(ours)} spikes.")
    if ours == ref:
        print("PASS: every spike matches in neuron and time step.")
        return
    # Brian2 counts steps from 0 at t = 0; report the first disagreement to help debugging.
    for shift in (-1, 1):
        if {(n, s + shift) for n, s in ours} == ref:
            print(f"FAIL: identical spike trains, but shifted by {shift} step(s).")
            raise SystemExit(1)
    diff = sorted(ours ^ ref, key=lambda x: x[1])
    first = diff[0][1]
    print(f"FAIL: {len(ours - ref)} extra and {len(ref - ours)} missing spikes; first disagreement at step {first}.")
    print("  port only :", sorted((s, n) for n, s in ours - ref)[:8])
    print("  brian only:", sorted((s, n) for n, s in ref - ours)[:8])
    raise SystemExit(1)


if __name__ == "__main__":
    main()
