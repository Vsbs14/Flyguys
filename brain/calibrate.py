"""Calibrate the Phase 2 decision rule and check it on fixed scenarios.

1. Reference rates: each output's own sense at full intensity, every other
   sense off. The mean rate over replicates becomes that output's reference
   in data/decision_rule.json.
2. Scenario checks: fixed sense combinations with the action a fly should
   pick. A check passes when that action wins in at least PASS_FRACTION of the
   replicate brains. Expectations are fixed here and never tuned per run.
3. Dose-response: each sense alone from 0 to 1, so the world side can see
   where each action starts to happen.

  python -m brain.calibrate --device cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .connectome import ROOT, Connectome
from .interface import ACTIONS, DECISION_MS, DRIVER, OUTPUTS, SENSES, DecisionRule, FlyBrain
from .neurons import NeuronGroups
from .runlog import run_record, save_json
from .sim import BrainSim

THRESHOLD = 0.25
PASS_FRACTION = 0.75
DOSES = np.round(np.linspace(0.0, 1.0, 11), 2)


def vec(**kw) -> list[float]:
    return [float(kw.get(s, 0.0)) for s in SENSES]


# name, intensities, expected action (None = recorded, no expectation)
SCENARIOS = [
    ("nothing", vec(), "idle"),
    ("sugar", vec(sugar=1), "feed"),
    ("bitter", vec(bitter=1), "idle"),
    ("looming", vec(looming=1), "escape"),
    ("obstacle", vec(obstacle=1), "retreat"),
    ("target", vec(target=1), "advance"),
    ("weak sugar (0.1)", vec(sugar=0.1), "idle"),
    ("sugar + bitter", vec(sugar=1, bitter=1), "idle"),
    ("sugar + looming", vec(sugar=1, looming=1), "escape"),
    ("target + looming", vec(target=1, looming=1), "escape"),
    ("obstacle + looming", vec(obstacle=1, looming=1), "escape"),
    ("sugar + target", vec(sugar=1, target=1), None),
    ("sugar + obstacle", vec(sugar=1, obstacle=1), None),
    ("target + obstacle", vec(target=1, obstacle=1), None),
    ("half of everything", vec(sugar=.5, bitter=.5, looming=.5, obstacle=.5, target=.5), None),
]


def calibrate(sim: BrainSim, groups: NeuronGroups, replicates: int = 16, seed: int = 0) -> dict:
    brain = FlyBrain(sim, groups)
    t0 = time.time()

    # 1. reference rates
    ref_in = np.repeat(np.array([vec(**{DRIVER[o]: 1.0}) for o in OUTPUTS]), replicates, axis=0)
    ref_rates = brain.output_rates(ref_in, seed=seed).numpy().reshape(len(OUTPUTS), replicates, len(OUTPUTS))
    ref_hz = {o: float(ref_rates[i, :, i].mean()) for i, o in enumerate(OUTPUTS)}
    ref_sd = {o: float(ref_rates[i, :, i].std()) for i, o in enumerate(OUTPUTS)}
    rule = DecisionRule(ref_hz=ref_hz, threshold=THRESHOLD)

    # 2 + 3. scenarios and dose-response in one batch, on a different seed than the reference
    dose_rows = [(s, d, vec(**{s: d})) for s in SENSES for d in DOSES]
    inputs = [x for _, x, _ in SCENARIOS] + [x for _, _, x in dose_rows]
    batch = np.repeat(np.array(inputs, dtype=np.float32), replicates, axis=0)
    rates = brain.output_rates(batch, seed=seed + 1)
    acts = rule.choose(rates).numpy().reshape(len(inputs), replicates)
    scores = rule.scores(rates).numpy().reshape(len(inputs), replicates, len(OUTPUTS))
    rates = rates.numpy().reshape(len(inputs), replicates, len(OUTPUTS))

    def summary(k: int) -> dict:
        frac = {a: float((acts[k] == i).mean()) for i, a in enumerate(ACTIONS)}
        return {
            "action_fraction": frac,
            "majority": max(frac, key=frac.get),
            "rate_hz_mean": {o: float(rates[k, :, j].mean()) for j, o in enumerate(OUTPUTS)},
            "score_mean": {o: float(scores[k, :, j].mean()) for j, o in enumerate(OUTPUTS)},
        }

    checks = []
    for k, (name, x, expected) in enumerate(SCENARIOS):
        s = summary(k)
        row = {"scenario": name, "intensity": dict(zip(SENSES, x)), "expected": expected, **s}
        if expected is not None:
            row["passed"] = bool(s["action_fraction"][expected] >= PASS_FRACTION)
        checks.append(row)
    dose = [{"sense": s, "dose": float(d), **summary(len(SCENARIOS) + k)} for k, (s, d, _) in enumerate(dose_rows)]

    return {
        "rule": rule,
        "ref_hz_sd": ref_sd,
        "replicates": replicates,
        "checks": checks,
        "all_passed": all(c.get("passed", True) for c in checks),
        "dose_response": dose,
        "wall_seconds": round(time.time() - t0, 1),
    }


def to_markdown(res: dict, record: dict) -> str:
    rule: DecisionRule = res["rule"]
    R = res["replicates"]
    L = [
        "# Decision rule: calibration and checks",
        "",
        f"Device: {record['device_name']} ({record['device']}). Seed {record['config']['seed']}, {R} replicate "
        f"brains per row, {record['config']['decision_ms']:.0f} ms decision window from rest. "
        "Intensity 1 = 200 Hz Poisson input on every neuron of that sense.",
        "",
        "## Reference rates",
        "",
        "Each output's own sense at full intensity, everything else off. An output's score is its rate divided by "
        f"this reference; the action is the highest score, or `idle` if no score reaches {rule.threshold}.",
        "",
        "| Output | Neurons | Driven by | Reference rate (Hz) |",
        "|---|---|---|---|",
    ]
    names = {"feed": "MN9", "escape": "Giant Fibre (DNp01)", "retreat": "MDN", "advance": "DNp09"}
    for o in OUTPUTS:
        L.append(f"| {o} | {names[o]} | {DRIVER[o]} | {rule.ref_hz[o]:.1f} ± {res['ref_hz_sd'][o]:.1f} |")
    L += ["", "## Scenario checks", "",
          f"Pass = the expected action wins in at least {PASS_FRACTION:.0%} of the {R} brains. "
          "Rows without an expectation are recorded only.", "",
          "| Scenario | Expected | Majority | Action split | Result |", "|---|---|---|---|---|"]
    for c in res["checks"]:
        split = ", ".join(f"{a} {f:.0%}" for a, f in c["action_fraction"].items() if f > 0)
        result = "-" if c["expected"] is None else ("PASS" if c["passed"] else "FAIL")
        L.append(f"| {c['scenario']} | {c['expected'] or '-'} | {c['majority']} | {split} | {result} |")
    L += ["", "## Dose-response", "",
          "One sense alone. Cells: mean score of the output that sense drives / share of brains that chose its action.",
          "", "| Dose | " + " | ".join(f"{s}" for s in SENSES) + " |", "|---|" + "---|" * len(SENSES)]
    inv = {s: o for o, s in DRIVER.items()}
    by = {(d["sense"], d["dose"]): d for d in res["dose_response"]}
    for dose in DOSES:
        cells = []
        for s in SENSES:
            d = by[(s, float(dose))]
            if s in inv:
                o = inv[s]
                cells.append(f"{d['score_mean'][o]:.2f} / {d['action_fraction'][o]:.0%}")
            else:
                cells.append(f"majority {d['majority']}")
        L.append(f"| {dose:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=16)
    args = ap.parse_args()

    conn = Connectome.load()
    sim = BrainSim(conn, device=args.device)
    res = calibrate(sim, NeuronGroups(conn), replicates=args.replicates, seed=args.seed)
    config = {"seed": args.seed, "replicates": args.replicates, "decision_ms": DECISION_MS,
              "threshold": THRESHOLD, "pass_fraction": PASS_FRACTION}
    record = run_record(config, device=args.device)
    rule: DecisionRule = res["rule"]
    rule.provenance = {"time_utc": record["time_utc"], "git_commit": record["git_commit"],
                       "device": record["device_name"], "seed": args.seed, "replicates": args.replicates,
                       "made_by": "python -m brain.calibrate"}
    rule.save()
    md = to_markdown(res, record)
    doc = {"run": record, "result": {**res, "rule": {"ref_hz": rule.ref_hz, "threshold": rule.threshold}}}
    save_json(ROOT / "results" / "decision_calibration.json", doc)
    (ROOT / "results" / "decision_calibration.md").write_text(md, encoding="utf8")
    print(md)
    print(f"{res['wall_seconds']} s. Saved data/decision_rule.json and results/decision_calibration.json/.md")
    raise SystemExit(0 if res["all_passed"] else 1)


if __name__ == "__main__":
    main()
