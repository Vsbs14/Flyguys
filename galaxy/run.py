"""Run galaxies: one for a look, many for statistics.

Every policy is given the same galaxies. For a given seed the star map, the
worlds, the threat schedule and the home worlds are identical, so the brain is
the only thing that differs.

  python -m galaxy.run one intact --seed 0            # print one run's summary
  python -m galaxy.run three --seed 42                # normal, fearless and always-hungry flies -> galaxy/frames.json
  python -m galaxy.run variants --seeds 100           # every brain variant
  python -m galaxy.run ladder --seeds 100             # fly vs hand rule vs neural net vs shuffled wiring
  python -m galaxy.run sensitivity --seeds 30         # mapping gains varied by +-50%
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from brain.connectome import ROOT
from brain.response import TABLE_DIR, ResponseTable

from .mapping import Decoder
from .policy import load_policy
from .sim import GalaxySim, SimConfig
from .world import GalaxyConfig, describe, make_galaxy

OUT = ROOT / "results" / "experiments"
DECODER_FILE = TABLE_DIR / "decoder.json"

FLY_VARIANTS = ["intact", "no_gf", "no_mdn", "no_gf_no_mdn", "no_bitter", "size_50", "size_20", "size_05"]
LADDER = ["intact", "hand_rule", "neural_net", "shuffled"]
THREE = {"intact": "intact", "fearless": "no_gf", "hungry": "no_bitter"}     # names the viewer uses -> tables
SERIES_KEPT = ["colonies", "civilizations", "detectable_civilizations"]


def frozen_decoder() -> Decoder:
    """Fitted on the intact table the first time, then read from disk for every variant and baseline."""
    if DECODER_FILE.exists():
        return Decoder.load(DECODER_FILE)
    decoder = Decoder.fit(ResponseTable.load("intact"))
    decoder.save(DECODER_FILE)
    return decoder


_policies: dict = {}


def _policy(name: str):
    if name not in _policies:
        _policies[name] = load_policy(name)
    return _policies[name]


def run_one(policy: str, seed: int, gains: tuple = (1.0, 1.0, 1.0), sim_cfg: SimConfig = SimConfig(),
            galaxy_cfg: GalaxyConfig = GalaxyConfig(), frames=None, label: str = "") -> dict:
    galaxy = make_galaxy(seed, galaxy_cfg)
    senses = galaxy.senses.with_gains(*gains)
    sim = GalaxySim(galaxy, _policy(policy), frozen_decoder(), sim_cfg, seed=seed, senses=senses, frames=frames,
                    label=label or policy)
    return sim.run()


def _job(args) -> dict:
    policy, seed, gains = args
    out = run_one(policy, seed, gains)
    series = out.pop("series")
    out["series"] = {k: series[k][::4] for k in SERIES_KEPT}          # every 4th epoch is plenty for curves
    return {"policy": policy, "seed": seed, "gains": list(gains), **out}


def run_batch(policies: list[str], seeds: list[int], gains_list: list[tuple] = ((1.0, 1.0, 1.0),),
              processes: int = 8) -> list[dict]:
    jobs = list(itertools.product(policies, seeds, gains_list))
    t0 = time.time()
    if processes > 1:
        with Pool(processes) as pool:
            rows = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (processes * 8)))
    else:
        rows = [_job(j) for j in jobs]
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s", flush=True)
    return rows


def save(name: str, rows: list[dict], extra: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"experiment": name, "galaxy": GalaxyConfig().as_dict(), "sim": SimConfig().as_dict(),
           "decoder": frozen_decoder().__dict__, **(extra or {}), "rows": rows}
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(doc) + "\n", encoding="utf8")
    print("wrote", path.relative_to(ROOT))
    return path


def brief(rows: list[dict], keys=("final_colonies", "final_civilizations", "effective_p", "flights",
                                  "mean_detectable_civilizations_last_50")) -> None:
    """Mean +- standard error per policy, printed."""
    names = list(dict.fromkeys((r["policy"], tuple(r["gains"])) for r in rows))
    print("policy".ljust(16) + "gains".ljust(18) + "".join(k[:22].rjust(24) for k in keys))
    for name, gains in names:
        sel = [r for r in rows if r["policy"] == name and tuple(r["gains"]) == gains]
        cells = ""
        for k in keys:
            v = np.array([r[k] for r in sel], dtype=float)
            cells += f"{v.mean():14.2f} +- {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:5.2f}".rjust(24)
        print(name.ljust(16) + str(gains).ljust(18) + cells)


def three_galaxies(seed: int, every: int) -> None:
    """Normal, fearless and always-hungry flies on one seed, recorded for the viewer."""
    try:
        from viz.frames import FrameWriter                      # the viewer's own writer, when that branch is present
    except ImportError:
        from .frames import FrameWriter
    galaxy = make_galaxy(seed)
    stars = [{"id": i, "x": float(x), "y": float(y)} for i, (x, y) in enumerate(galaxy.xy)]
    writer = FrameWriter(variants=list(THREE), seed=seed, stars=stars, every=every)
    rows = []
    for label, table in THREE.items():
        out = run_one(table, seed, frames=writer, label=label)
        out.pop("series")
        rows.append({"policy": label, "seed": seed, "gains": [1.0, 1.0, 1.0], **out})
        print(f"{label:9s} final colonies {out['final_colonies']:4d}  civilizations {out['final_civilizations']}  "
              f"flights {out['flights']:4d}  encounters {out['encounters']}")
    writer.save(ROOT / "galaxy" / "frames.json")
    save("three_galaxies", rows, {"seed": seed})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["one", "three", "variants", "ladder", "sensitivity", "describe"])
    ap.add_argument("policy", nargs="?", default="intact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--every", type=int, default=2, help="record every Nth epoch in the three-galaxy run")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    if args.what == "describe":
        print(json.dumps(describe(make_galaxy(args.seed)), indent=1))
    elif args.what == "one":
        out = run_one(args.policy, args.seed)
        series = out.pop("series")
        print(json.dumps(out, indent=1))
        print("colonies every 40 epochs:", series["colonies"][::40])
    elif args.what == "three":
        three_galaxies(args.seed, args.every)
    elif args.what == "variants":
        rows = run_batch(FLY_VARIANTS, seeds, processes=args.processes)
        brief(rows)
        save("variants", rows)
    elif args.what == "ladder":
        rows = run_batch(LADDER, seeds, processes=args.processes)
        brief(rows)
        save("ladder", rows)
    elif args.what == "sensitivity":
        # each mapping gain moved by +-50% on its own, plus all three together
        levels = [0.5, 1.0, 1.5]
        gains = [(1.0, 1.0, 1.0)]
        gains += [tuple(g if i == k else 1.0 for i in range(3)) for k in range(3) for g in (0.5, 1.5)]
        gains += [(g, g, g) for g in (0.5, 1.5)]
        rows = run_batch(["intact", "no_gf", "no_bitter", "hand_rule"], seeds, gains_list=gains,
                         processes=args.processes)
        brief(rows)
        save("sensitivity", rows, {"gain_levels": levels})


if __name__ == "__main__":
    main()
