"""Deterministic reference for the port: a small random network in Shiu's Brian2 model.

No random input is used. Neurons start at random voltages (some above threshold)
and the recurrent network runs on its own, so the PyTorch port must reproduce the
spike trains exactly. Run with the Brian2 environment (see brian2_reference.py):

  external/.venv-brian/Scripts/python scripts/brian2_equivalence.py

Writes data/brian2_equivalence_case.npz (committed, so the test needs no Brian2). Compare with `python -m brain.test_equivalence`.
"""
from pathlib import Path
from textwrap import dedent

import numpy as np
from brian2 import Network, NeuronGroup, SpikeMonitor, Synapses, defaultclock, ms, mV, prefs

prefs.codegen.target = "numpy"
ROOT = Path(__file__).resolve().parents[1]

params = {
    "v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms,
    "tau": 5 * ms, "t_rfc": 2.2 * ms, "t_dly": 1.8 * ms, "w_syn": 0.275 * mV,
}
eqs = dedent("""
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """)

rng = np.random.default_rng(42)
N, K, T_MS = 400, 40, 300
pre = np.repeat(np.arange(N), K)
post = np.concatenate([rng.choice(np.delete(np.arange(N), i), K, replace=False) for i in range(N)])
signed = rng.integers(-25, 46, size=N * K)
signed[signed == 0] = 1
v_init = rng.uniform(-52.0, -40.0, size=N)
no_refrac = rng.random(N) < 0.05          # like Shiu's Poisson targets: refractory period 0

neu = NeuronGroup(N=N, model=eqs, method="linear", threshold="v > v_th",
                  reset="v = v_rst; w = 0; g = 0 * mV", refractory="rfc", namespace=params)
neu.v = v_init * mV
neu.g = 0
neu.rfc = params["t_rfc"]
neu.rfc[no_refrac] = 0 * ms
syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"])
syn.connect(i=pre, j=post)
syn.w = signed * params["w_syn"]
mon = SpikeMonitor(neu)
Network(neu, syn, mon).run(T_MS * ms)

step = np.rint(np.asarray(mon.t / defaultclock.dt)).astype(np.int64)
out = ROOT / "data" / "brian2_equivalence_case.npz"
np.savez_compressed(out, pre=pre, post=post, signed=signed, v_init=v_init, no_refrac=no_refrac,
         spike_neuron=np.asarray(mon.i), spike_step=step, t_ms=T_MS)
print(f"{len(step)} spikes from {len(np.unique(mon.i))} neurons, last at step {step.max()}; wrote {out}")
