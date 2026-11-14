"""Compare the PyTorch port against Shiu's original Brian2 model.

Runs the same conditions as `scripts/brian2_reference.py` and puts the output
firing rates side by side. The two simulators use different random number
streams, so the comparison is statistical: the port should land within the
trial-to-trial spread of the reference.

  python -m brain.validate_port
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .runlog import run_record, save_json
from .sim import BrainSim

REFERENCE = ROOT / "results" / "brian2_reference.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=4)
    args = ap.parse_args()

    ref = json.loads(REFERENCE.read_text(encoding="utf8"))
    conn = Connectome.load()
    groups = NeuronGroups(conn)
    sim = BrainSim(conn, device=args.device)

    conds = list(ref["conditions"])
    R = args.replicates
    rate = lambda sense: torch.tensor([float(ref["conditions"][c]["input_rates_hz"].get(sense, 0)) for c in conds]
                                      ).repeat_interleave(R)
    state = sim.new_state(len(conds) * R, seed=args.seed)
    res = sim.run(state, float(ref["t_run_ms"]), monitor=torch.from_numpy(groups.all_outputs()),
                  inputs=[(groups.sense(s), rate(s)) for s in groups.senses], record_all=True)
    rates = res.rates_hz.cpu().numpy()
    total = res.total_spikes.cpu().numpy()
    n_active = (res.all_counts > 0).sum(0).cpu().numpy()

    rows, at = [], 0
    for o in groups.outputs:
        k = len(groups.output(o))
        for j, c in enumerate(conds):
            ours = rates[at:at + k, j * R:(j + 1) * R].mean(axis=0)            # per replicate, mean over neurons
            theirs = np.array([np.mean(t["rates_hz"][o]) for t in ref["conditions"][c]["trials"]])
            rows.append((c, f"{o} rate (Hz)", theirs, ours))
        at += k
    for j, c in enumerate(conds):
        trials = ref["conditions"][c]["trials"]
        rows.append((c, "whole-brain spikes", np.array([t["total_spikes"] for t in trials]), total[j * R:(j + 1) * R]))
        rows.append((c, "neurons that spiked", np.array([t["n_spiking_neurons"] for t in trials]),
                     n_active[j * R:(j + 1) * R]))
    rows.sort(key=lambda r: conds.index(r[0]))

    lines = [
        "# Port validation: PyTorch simulator vs Shiu's Brian2 model",
        "",
        f"Same connectome, same parameters, {ref['t_run_ms']} ms per trial. Values are mean ± SD over trials "
        f"({len(next(iter(ref['conditions'].values()))['trials'])} Brian2, {R} PyTorch).",
        "",
        "| Condition | Measure | Brian2 (reference) | PyTorch port | Difference |",
        "|---|---|---|---|---|",
    ]
    table = []
    for c, measure, theirs, ours in rows:
        a, b = float(np.mean(theirs)), float(np.mean(ours))
        if a == 0 and b == 0:
            diff = "both 0"
        elif a == 0:
            diff = "reference is 0"
        else:
            diff = f"{(b - a) / a:+.1%}"
        lines.append(f"| {c} | {measure} | {a:.1f} ± {np.std(theirs):.1f} | {b:.1f} ± {np.std(ours):.1f} | {diff} |")
        table.append({"condition": c, "measure": measure, "brian2_mean": a, "brian2_sd": float(np.std(theirs)),
                      "torch_mean": b, "torch_sd": float(np.std(ours))})
    md = "\n".join(lines) + "\n"
    record = run_record({"seed": args.seed, "replicates": R}, device=args.device)
    save_json(ROOT / "results" / "port_validation.json", {"run": record, "rows": table})
    (ROOT / "results" / "port_validation.md").write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
