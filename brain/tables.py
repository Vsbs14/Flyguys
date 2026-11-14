"""Phase 3: response tables. Precompute every brain variant's decisions over a grid of senses.

The galaxy needs thousands of decisions per tick; simulating each one live
would be far too slow. Instead, each variant (data/variants.json) is run once
over a grid: every sense at LEVELS intensities (5 senses -> LEVELS**5 points),
several replicate brains per point. The table stores, per grid point, how many
replicates chose each action and the mean output rates. `galaxy.brains`
interpolates between grid points and samples an action, with no GPU needed.

Writes tables/<variant>.npz, tables/index.json and results/response_tables.md.

  python -m brain.tables --device cuda                    # every variant in data/variants.json
  python -m brain.tables --device cuda --variants intact
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from .calibrate import SCENARIOS
from .connectome import ROOT, Connectome
from .interface import ACTIONS, OUTPUTS, SENSES, DecisionRule, FlyBrain
from .neurons import NeuronGroups
from .runlog import run_record
from .sim import BrainSim

TABLES = ROOT / "tables"
VARIANTS_FILE = ROOT / "data" / "variants.json"
LEVELS = np.round(np.linspace(0.0, 1.0, 5), 3)
MAX_BRAINS = 1024


# Controls added to every variant list. "shuffled" asks whether the specific wiring matters,
# or whether any brain with the same number of connections per neuron would do as well.
CONTROLS = {
    "shuffled": {"description": "Degree-preserving shuffle: interneuron wiring permuted (seed 0), every neuron "
                                "keeps its number of inputs and outputs; sense and output neurons untouched.",
                 "silenced_cell_types": [], "silenced_root_ids": [], "shuffle_seed": 0},
}


def load_variants() -> dict:
    if VARIANTS_FILE.exists():
        variants = json.loads(VARIANTS_FILE.read_text(encoding="utf8"))["variants"]
    else:
        variants = {"intact": {"description": "The unmodified FlyWire v783 brain.", "silenced_root_ids": [],
                               "silenced_cell_types": []}}
    return {**variants, **CONTROLS}


def degree_preserving_shuffle(conn: Connectome, protected: np.ndarray, seed: int) -> Connectome:
    """Relabel interneurons within groups of equal in-degree and equal out-degree.

    Every edge keeps its weight; the neuron at each end is swapped for another
    interneuron with the same in- and out-degree, so the degree sequence is
    unchanged but who-talks-to-whom is scrambled. Protected neurons (senses and
    outputs) keep their identity.
    """
    pre, post, signed = conn.edges()
    indeg = np.bincount(post, minlength=conn.n)
    outdeg = np.diff(conn.ptr)
    free = np.setdiff1d(np.arange(conn.n), protected)
    key = indeg[free].astype(np.int64) * (outdeg.max() + 1) + outdeg[free]
    order = np.argsort(key, kind="stable")
    free, key = free[order], key[order]
    mapping = np.arange(conn.n)
    rng = np.random.default_rng(seed)
    for cohort in np.split(free, np.flatnonzero(np.diff(key)) + 1):
        mapping[cohort] = rng.permutation(cohort)
    return Connectome.from_edges(conn.ids, mapping[pre], mapping[post], signed)


def build_table(brain: FlyBrain, conn: Connectome, silenced_ids: list[int], replicates: int, seed: int,
                levels=LEVELS) -> dict:
    alive = None
    if silenced_ids:
        alive = np.ones(conn.n, dtype=bool)
        alive[conn.index_of(silenced_ids)] = False
        alive = torch.from_numpy(alive)
    grid = np.array(list(itertools.product(levels, repeat=len(SENSES))), dtype=np.float32)   # (P, senses)
    per = max(1, MAX_BRAINS // replicates)
    counts = np.zeros((len(grid), len(ACTIONS)), dtype=np.uint8)
    rates = np.zeros((len(grid), len(OUTPUTS)), dtype=np.float32)
    for c0 in range(0, len(grid), per):
        x = np.repeat(grid[c0:c0 + per], replicates, axis=0)
        r = brain.output_rates(x, alive=alive, seed=seed + c0)                    # (B, outputs)
        a = brain.rule.choose(r).numpy().reshape(-1, replicates)
        for k in range(len(ACTIONS)):
            counts[c0:c0 + len(a), k] = (a == k).sum(1)
        rates[c0:c0 + len(a)] = r.numpy().reshape(-1, replicates, len(OUTPUTS)).mean(1)
    return {"levels": np.asarray(levels, dtype=np.float32), "grid": grid, "counts": counts, "rates_hz": rates,
            "replicates": replicates}


def nearest_point(levels: np.ndarray, x) -> int:
    """Flat grid index of the point closest to intensities x."""
    idx = [int(np.abs(levels - v).argmin()) for v in x]
    return int(np.ravel_multi_index(idx, (len(levels),) * len(SENSES)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=None)
    args = ap.parse_args()

    variants = load_variants()
    names = args.variants or list(variants)
    conn = Connectome.load()
    groups = NeuronGroups(conn)
    rule = DecisionRule.load()
    brain = FlyBrain(BrainSim(conn, device=args.device), groups, rule)
    TABLES.mkdir(exist_ok=True)
    index_path = TABLES / "index.json"
    index = json.loads(index_path.read_text(encoding="utf8")) if index_path.exists() else {"tables": {}}
    index["tables"] = {k: v for k, v in index["tables"].items() if k in variants}   # drop stale variants

    for name in names:
        t0 = time.time()
        v = variants[name]
        if "shuffle_seed" in v:
            protected = np.concatenate([groups.all_inputs(), groups.all_outputs()])
            sconn = degree_preserving_shuffle(conn, protected, v["shuffle_seed"])
            tab = build_table(FlyBrain(BrainSim(sconn, device=args.device), groups, rule), sconn,
                              v["silenced_root_ids"], args.replicates, args.seed)
        else:
            tab = build_table(brain, conn, v["silenced_root_ids"], args.replicates, args.seed)
        np.savez_compressed(TABLES / f"{name}.npz", levels=tab["levels"], counts=tab["counts"],
                            rates_hz=tab["rates_hz"], replicates=tab["replicates"],
                            senses=np.array(SENSES), actions=np.array(ACTIONS), outputs=np.array(OUTPUTS))
        frac = tab["counts"].sum(0) / tab["counts"].sum()
        record = run_record({"seed": args.seed, "replicates": args.replicates, "levels": LEVELS.tolist(),
                             "variant": name}, device=args.device)
        index["tables"][name] = {
            "file": f"{name}.npz", "description": v["description"], "silenced_cell_types": v["silenced_cell_types"],
            "levels": LEVELS.tolist(), "replicates": args.replicates, "grid_points": len(tab["grid"]),
            "action_share_over_grid": dict(zip(ACTIONS, np.round(frac, 4).tolist())),
            "scenarios": {s[0]: ACTIONS[int(tab["counts"][nearest_point(LEVELS, s[1])].argmax())]
                          for s in SCENARIOS if all(val in (0.0, 1.0) for val in s[1])},
            "run": record, "wall_seconds": round(time.time() - t0, 1),
        }
        index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf8")
        print(f"{name}: {len(tab['grid'])} points x {args.replicates} in {time.time() - t0:.0f}s", flush=True)

    (ROOT / "results" / "response_tables.md").write_text(to_markdown(index), encoding="utf8")
    print((ROOT / "results" / "response_tables.md").read_text(encoding="utf8"))


def to_markdown(index: dict) -> str:
    tabs = index["tables"]
    any_t = next(iter(tabs.values()))
    L = ["# Response tables", "",
         f"Each variant run over a {len(any_t['levels'])}-level grid on all {len(SENSES)} senses "
         f"({any_t['grid_points']:,} points, levels {any_t['levels']}), {any_t['replicates']} replicate brains "
         "per point, 200 ms from rest. Files: `tables/<variant>.npz`.", "",
         "## Share of the grid where each action wins", "",
         "| Variant | Silenced | " + " | ".join(ACTIONS) + " |", "|---|---|" + "---|" * len(ACTIONS)]
    for n, t in tabs.items():
        L.append(f"| {n} | {', '.join(t['silenced_cell_types']) or '-'} | "
                 + " | ".join(f"{t['action_share_over_grid'][a]:.0%}" for a in ACTIONS) + " |")
    scen = list(any_t["scenarios"])
    L += ["", "## Majority action in the probe scenarios (read from the tables)", "",
          "| Scenario | " + " | ".join(tabs) + " |", "|---|" + "---|" * len(tabs)]
    for s in scen:
        cells = []
        for t in tabs.values():
            a, b = t["scenarios"][s], tabs.get("intact", t)["scenarios"][s]
            cells.append(f"**{a}**" if a != b else a)
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Bold = differs from the intact brain."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
