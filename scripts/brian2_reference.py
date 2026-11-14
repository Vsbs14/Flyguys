"""Reference run of Shiu et al.'s original Brian2 model, for validating the PyTorch port.

The network construction below is copied from Shiu's `model.py` (`create_model`,
`poi`) with the same equations and constants. It needs its own environment,
because Brian2 is not a dependency of this project:

  py -3.12 -m venv external/.venv-brian
  external/.venv-brian/Scripts/python -m pip install brian2 pandas pyarrow
  external/.venv-brian/Scripts/python scripts/brian2_reference.py

Writes results/brian2_reference.json. Compare with `python -m brain.validate_port`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from brian2 import Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, ms, mV, prefs, seed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external" / "Drosophila_brain_model"
OUT = ROOT / "results" / "brian2_reference.json"

prefs.codegen.target = "numpy"

params = {
    "v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms,
    "tau": 5 * ms, "t_rfc": 2.2 * ms, "t_dly": 1.8 * ms, "w_syn": 0.275 * mV, "f_poi": 250,
    "eqs": dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
        """),
    "eq_th": "v > v_th",
    "eq_rst": "v = v_rst; w = 0; g = 0 * mV",
}

# name -> {sense: rate in Hz}
# The same stimuli as the sanity checks (and the starter notebook's section 7).
CONDITIONS = {
    "sugar 200 Hz": {"sugar": 200},
    "sugar 200 Hz + bitter 200 Hz": {"sugar": 200, "bitter": 200},
    "looming 150 Hz": {"looming": 150},
}
T_RUN_MS = 1000
N_TRIALS = 2


def main() -> None:
    doc = json.loads((ROOT / "data" / "neuron_groups.json").read_text(encoding="utf8"))
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(DATA / "Connectivity_783.parquet")
    idx_of = {int(f): i for i, f in enumerate(comp.index)}
    group = lambda name: [idx_of[i] for i in doc["groups"][name]["ids"]]
    sense = {s: sum((group(g) for g in gs), []) for s, gs in doc["roles"]["inputs"].items()}
    outputs = {o: group(g) for o, g in doc["roles"]["outputs"].items()}

    results = {}
    for cond, rates in CONDITIONS.items():
        trials = []
        for trial in range(N_TRIALS):
            seed(1000 + trial)
            t0 = time.time()
            neu = NeuronGroup(N=len(comp), model=params["eqs"], method="linear", threshold=params["eq_th"],
                              reset=params["eq_rst"], refractory="rfc", namespace=params)
            neu.v = params["v_0"]
            neu.g = 0
            neu.rfc = params["t_rfc"]
            syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"])
            syn.connect(i=con["Presynaptic_Index"].values, j=con["Postsynaptic_Index"].values)
            syn.w = con["Excitatory x Connectivity"].values * params["w_syn"]
            mon = SpikeMonitor(neu)
            pois = []
            for s, rate in rates.items():
                for i in sense[s]:
                    pois.append(PoissonInput(target=neu[i], target_var="v", N=1, rate=rate * Hz,
                                             weight=params["w_syn"] * params["f_poi"]))
                    neu[i].rfc = 0 * ms
            net = Network(neu, syn, mon, *pois)
            net.run(T_RUN_MS * ms)
            counts = np.bincount(np.asarray(mon.i), minlength=len(comp))
            trials.append({
                "total_spikes": int(counts.sum()),
                "n_spiking_neurons": int((counts > 0).sum()),
                "rates_hz": {o: [float(counts[i]) * 1000 / T_RUN_MS for i in idx] for o, idx in outputs.items()},
                "input_spikes": {s: int(counts[sense[s]].sum()) for s in rates},
            })
            print(cond, "trial", trial, f"{time.time() - t0:.0f}s", trials[-1], flush=True)
        results[cond] = {"input_rates_hz": rates, "trials": trials}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"t_run_ms": T_RUN_MS, "conditions": results}, indent=1) + "\n", encoding="utf8")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
