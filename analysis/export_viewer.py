"""Feed the viewer's Baselines and Fly Brain pages from the main experiments.

Runs every available brain and baseline on the same galaxy seeds at three
sense-mapping gains (all three gains scaled together) and writes
results/galaxy_experiment.json in the format `viz/index.html` reads:

  gains[gain][policy]            survivors, detectable_colonies, settled_fraction, extinction_fraction (mean, se)
  paired_survivor_difference_from_intact[gain][policy]   per-seed difference from the intact fly
  escape_response_at_zero_bitter  P(Giant Fibre spikes), sugar x looming, from the intact table
  escape_axes                     the sugar and looming rates of that map (the looming steps are uneven)

  python -m analysis.export_viewer --seeds 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from brain.connectome import ROOT
from brain.response import TABLE_DIR, ResponseTable
from galaxy.run import run_batch
from galaxy.sim import SimConfig
from galaxy.world import GalaxyConfig

OUT = ROOT / "results" / "galaxy_experiment.json"
GAINS = (0.5, 1.0, 1.5)
# our policy name -> the name the viewer knows
POLICIES = {"intact": "intact", "no_gf": "fearless", "no_gf_no_mdn": "fearless_no_retreat", "no_bitter": "hungry",
            "shuffled": "shuffle", "hand_rule": "threshold", "neural_net": "neural_net"}


def stat(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args()

    have = [p for p in POLICIES if p in ("hand_rule", "neural_net") or (TABLE_DIR / f"{p}.npz").exists()]
    missing = [p for p in POLICIES if p not in have]
    print("policies:", have, "missing tables:", missing, flush=True)
    seeds = list(range(args.seeds))
    rows = run_batch(have, seeds, gains_list=[(g, g, g) for g in GAINS], processes=args.processes)

    n_stars = GalaxyConfig().n_stars
    report, paired = {}, {}
    for g in GAINS:
        key = f"{g:.1f}"
        by = {p: sorted((r for r in rows if r["policy"] == p and r["gains"][0] == g), key=lambda r: r["seed"])
              for p in have}
        intact = np.array([r["final_colonies"] for r in by["intact"]], dtype=float)
        report[key], paired[key] = {}, {}
        for p in have:
            rs = by[p]
            report[key][POLICIES[p]] = {
                "survivors": stat([r["final_colonies"] for r in rs]),
                "detectable_colonies": stat([r["mean_detectable_civilizations_last_50"] for r in rs]),
                "settled_fraction": stat([r["final_colonies"] / n_stars for r in rs]),
                "extinction_fraction": stat([r["final_civilizations"] == 0 for r in rs]),
                "civilizations": stat([r["final_civilizations"] for r in rs]),
                "flights": stat([r["flights"] for r in rs]),
            }
            if p != "intact":
                paired[key][POLICIES[p]] = stat(np.array([r["final_colonies"] for r in rs]) - intact)

    table = ResponseTable.load("intact")
    doc = {
        "config": {
            "seeds": args.seeds, "stars": n_stars, "epochs": GalaxyConfig().n_epochs, "gains": list(GAINS),
            "policies": {POLICIES[p]: p for p in have}, "missing": missing,
            "survivors_note": "Colonies alive at the final epoch",
            "detector_note": "Detected is the number of civilizations an observer at the Sun's position sees: colonies "
                             "within 10,000 light-years at 30% of capacity or more, averaged over the last 25 Myr.",
            "galaxy": GalaxyConfig().as_dict(), "sim": SimConfig().as_dict(),
        },
        "gains": report,
        "paired_survivor_difference_from_intact": paired,
        "escape_response_at_zero_bitter": table.p_gf[:, 0, :].tolist(),
        "escape_axes": {"sugar_hz": table.axes[0].tolist(), "looming_hz": table.axes[2].tolist()},
        "rows": [{k: v for k, v in r.items() if k != "series"} for r in rows],
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    print("wrote", OUT.relative_to(ROOT))
    for key in report:
        print(key, {p: round(r["survivors"]["mean"], 1) for p, r in report[key].items()})


if __name__ == "__main__":
    main()
