"""The six sanity checks every brain has to pass before it is used.

  1. No input              -> silence
  2. Sugar                 -> MN9 fires at about 50 Hz
  3. Sugar + bitter        -> MN9 drops a lot
  4. Looming               -> Giant Fibre fires
  5. Giant Fibre silenced  -> no escape
  6. Input stopped         -> activity dies out

All conditions run side by side as columns of one batch, each with several
replicate brains that differ only in their random input spikes.

  python -m brain.sanity                 # intact brain on CPU
  python -m brain.sanity --device cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .runlog import run_record, save_json
from .sim import BrainSim

# Stimuli and duration follow the starter notebook (fly_brain_starter.ipynb, section 7):
# taste at 200 Hz, looming at 150 Hz, one 200 ms decision window from rest.
TASTE_HZ = 200.0
LOOM_HZ = 150.0
ON_MS = 200.0          # stimulus on, rates are measured here
OFF_MS = 300.0         # stimulus off
QUIET_MS = 100.0       # the last part of the off period, which must be silent

# Pass criteria. These are fixed here and never tuned per variant.
MN9_BAND_HZ = (25.0, 125.0)    # tens of Hz: brackets the handbook's ~50 Hz and the starter notebook's 100 Hz
BITTER_MAX_FRACTION = 0.5      # "drops a lot": MN9 falls below half its sugar-only rate

CONDITIONS = [
    # name,             sugar,    bitter,   looming, Giant Fibre silenced
    ("none",            0.0,      0.0,      0.0,     False),
    ("sugar",           TASTE_HZ, 0.0,      0.0,     False),
    ("sugar+bitter",    TASTE_HZ, TASTE_HZ, 0.0,     False),
    ("looming",         0.0,      0.0,      LOOM_HZ, False),
    ("looming, GF off", 0.0,      0.0,      LOOM_HZ, True),
]
CHECK_NAMES = ["silence", "sugar_feeds", "bitter_suppresses", "looming_escapes", "gf_off_no_escape", "dies_out"]


def run_sanity(
    sim: BrainSim,
    groups: NeuronGroups,
    alive: np.ndarray | None = None,
    replicates: int = 4,
    seed: int = 0,
    expected_fail: tuple[str, ...] = (),
    variant: str = "intact",
) -> dict:
    """Run the six checks on one brain variant.

    alive         : (N,) bool mask of the variant; False = silenced neuron.
    expected_fail : checks the variant's lesion is supposed to break. They are
                    still run and recorded, and flagged if they pass anyway.
    """
    n_cond = len(CONDITIONS)
    B = n_cond * replicates
    col = lambda k: torch.tensor([c[k] for c in CONDITIONS], dtype=torch.float32).repeat_interleave(replicates)
    gf = groups.output("escape")

    mask = np.ones((sim.n, B), dtype=bool) if alive is None else np.repeat(np.asarray(alive, bool)[:, None], B, axis=1)
    for j, cond in enumerate(CONDITIONS):
        if cond[4]:
            mask[gf, j * replicates:(j + 1) * replicates] = False

    monitor = torch.from_numpy(groups.all_outputs())
    slices, at = {}, 0
    for o in groups.outputs:
        slices[o] = slice(at, at + len(groups.output(o)))
        at += len(groups.output(o))

    state = sim.new_state(B, alive=mask, seed=seed)
    senses = [groups.sense("sugar"), groups.sense("bitter"), groups.sense("looming")]
    t0 = time.time()
    on = sim.run(state, ON_MS, inputs=[(senses[0], col(1)), (senses[1], col(2)), (senses[2], col(3))], monitor=monitor)
    zero = [(s, 0.0) for s in senses]
    sim.run(state, OFF_MS - QUIET_MS, inputs=zero, monitor=monitor)
    quiet = sim.run(state, QUIET_MS, inputs=zero, monitor=monitor)
    wall = time.time() - t0

    rates = on.rates_hz.cpu().numpy()                       # (n_outputs_total, B)
    per_cond = lambda x: x.reshape(n_cond, replicates)      # brains are grouped by condition
    out_rate = {o: per_cond(rates[sl].mean(axis=0)) for o, sl in slices.items()}   # mean over the neurons of a type
    gf_spiked = per_cond((on.counts[slices["escape"]].sum(0) > 0).cpu().numpy())
    first = on.first_spike_ms[slices["escape"]].cpu().numpy()
    gf_latency = per_cond(np.where(np.isnan(first).all(axis=0), np.nan, np.nanmin(np.nan_to_num(first, nan=np.inf), axis=0)))
    total_on = per_cond(on.total_spikes.cpu().numpy())
    total_quiet = per_cond(quiet.total_spikes.cpu().numpy())
    ci = {c[0]: i for i, c in enumerate(CONDITIONS)}

    mn9_sugar = float(out_rate["feed"][ci["sugar"]].mean())
    mn9_mixed = float(out_rate["feed"][ci["sugar+bitter"]].mean())
    frac = mn9_mixed / mn9_sugar if mn9_sugar > 0 else None
    suppression = f"{frac:.0%} of sugar-only" if frac is not None else "no sugar-only response"
    n_loom = int(gf_spiked[ci["looming"]].sum())
    loom_measured = f"spiked in {n_loom}/{replicates}"
    if n_loom:
        loom_measured += f", first spike at {np.nanmean(gf_latency[ci['looming']]):.1f} ms"
    checks = [
        ("silence", "No input -> silence", "0 spikes in the whole brain",
         f"{int(total_on[ci['none']].sum())} spikes", bool(total_on[ci["none"]].sum() == 0)),
        ("sugar_feeds", f"Sugar ({TASTE_HZ:.0f} Hz) -> MN9 fires at about 50 Hz",
         f"MN9 rate within {MN9_BAND_HZ[0]:.0f}-{MN9_BAND_HZ[1]:.0f} Hz",
         f"MN9 = {mn9_sugar:.1f} Hz", bool(MN9_BAND_HZ[0] <= mn9_sugar <= MN9_BAND_HZ[1])),
        ("bitter_suppresses", "Sugar + bitter -> MN9 drops a lot",
         f"MN9 below {BITTER_MAX_FRACTION:.0%} of its sugar-only rate",
         f"MN9 = {mn9_mixed:.1f} Hz ({suppression})", frac is not None and frac < BITTER_MAX_FRACTION),
        ("looming_escapes", f"Looming ({LOOM_HZ:.0f} Hz) -> Giant Fibre fires",
         "Giant Fibre spikes in every replicate", loom_measured, bool(n_loom == replicates)),
        ("gf_off_no_escape", "Giant Fibre silenced -> no escape",
         "0 Giant Fibre spikes under the same looming input",
         f"spiked in {int(gf_spiked[ci['looming, GF off']].sum())}/{replicates}",
         bool(not gf_spiked[ci["looming, GF off"]].any())),
        ("dies_out", "Input stopped -> activity dies out",
         f"0 spikes in the whole brain {OFF_MS - QUIET_MS:.0f}-{OFF_MS:.0f} ms after input ends",
         f"{int(total_quiet.sum())} spikes", bool(total_quiet.sum() == 0)),
    ]
    rows = []
    for key, name, criterion, measured, passed in checks:
        expect_pass = key not in expected_fail
        rows.append({
            "check": key, "name": name, "criterion": criterion, "measured": measured,
            "passed": passed, "expected_to_pass": expect_pass, "as_expected": passed == expect_pass,
        })

    return {
        "variant": variant,
        "all_as_expected": all(r["as_expected"] for r in rows),
        "checks": rows,
        "conditions": {
            c[0]: {
                "inputs_hz": {"sugar": c[1], "bitter": c[2], "looming": c[3]},
                "giant_fibre_silenced": c[4],
                "output_rate_hz_mean": {o: float(out_rate[o][i].mean()) for o in groups.outputs},
                "output_rate_hz_sd": {o: float(out_rate[o][i].std()) for o in groups.outputs},
                "giant_fibre_spiked": int(gf_spiked[i].sum()),
                "whole_brain_spikes_on": float(total_on[i].mean()),
                "whole_brain_spikes_quiet": float(total_quiet[i].sum()),
            } for i, c in enumerate(CONDITIONS)
        },
        "replicates": replicates,
        "wall_seconds": round(wall, 1),
    }


def sanity_config(seed: int, replicates: int) -> dict:
    return {"seed": seed, "replicates": replicates, "taste_hz": TASTE_HZ, "looming_hz": LOOM_HZ,
            "on_ms": ON_MS, "off_ms": OFF_MS}


def to_markdown(result: dict, record: dict) -> str:
    mark = lambda r: ("PASS" if r["passed"] else "FAIL") + ("" if r["as_expected"] else " (unexpected)")
    lines = [
        f"# Sanity checks: {result['variant']} brain",
        "",
        f"Device: {record['device_name']} ({record['device']}). Seed {record['config']['seed']}, "
        f"{result['replicates']} replicate brains per condition, stimulus on for {ON_MS:.0f} ms.",
        "",
        "| # | Check | Pass criterion | Measured | Result |",
        "|---|-------|----------------|----------|--------|",
    ]
    for i, r in enumerate(result["checks"], 1):
        lines.append(f"| {i} | {r['name']} | {r['criterion']} | {r['measured']} | {mark(r)} |")
    outs = list(next(iter(result["conditions"].values()))["output_rate_hz_mean"])
    lines += ["", "Output firing rates (Hz, mean over the neurons of each type and over replicates):", "",
              "| Condition | " + " | ".join(outs) + " | Giant Fibre spiked |",
              "|---|" + "---|" * (len(outs) + 1)]
    for name, c in result["conditions"].items():
        cells = " | ".join(f"{c['output_rate_hz_mean'][o]:.1f}" for o in outs)
        lines.append(f"| {name} | {cells} | {c['giant_fibre_spiked']}/{result['replicates']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=4)
    from galaxy.responses import LESIONS, alive_mask, shuffled

    ap.add_argument("--variant", choices=LESIONS, default="intact")
    args = ap.parse_args()

    conn = Connectome.load()
    groups = NeuronGroups(conn)
    if args.variant == "shuffle":
        conn = shuffled(conn, groups, args.seed)
    sim = BrainSim(conn, device=args.device)
    expected_fail = {
        "fearless": ("looming_escapes",),
        "fearless_no_retreat": ("looming_escapes",),
        "hungry": ("bitter_suppresses",),
    }
    result = run_sanity(sim, groups, alive=alive_mask(conn, groups, LESIONS[args.variant]),
                        replicates=args.replicates, seed=args.seed, variant=args.variant,
                        expected_fail=expected_fail.get(args.variant, ()))
    record = run_record(sanity_config(args.seed, args.replicates), device=args.device)
    tag = f"sanity_{result['variant']}_{sim.device.type}"
    save_json(ROOT / "results" / f"{tag}.json", {"run": record, "result": result})
    md = to_markdown(result, record)
    (ROOT / "results" / f"{tag}.md").write_text(md, encoding="utf8")
    print(md)
    print(f"{result['wall_seconds']} s. Saved results/{tag}.json and .md")
    raise SystemExit(0 if result["all_as_expected"] else 1)


if __name__ == "__main__":
    main()
