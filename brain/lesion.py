"""Phase 4: the lesion screen. Silence one cell type at a time and see what breaks.

Candidates (Step 4.1)
    A cell type is a candidate if at least one of its neurons
      * lies within 3 synaptic hops downstream of an input neuron and within
        3 hops upstream of an output neuron, counting connections of at least
        5 synapses, and
      * actually spikes in the intact brain under some input.
    The second condition is what keeps the list to a few hundred: silencing a
    neuron that never spikes changes nothing, because neurons only act on each
    other through spikes. Hops come from Shiu's connectivity file and cell
    types from the public FlyWire annotation table, so no token is needed.

Screen (Step 4.2)
    Every candidate gets its own response table on a coarse 4 x 4 x 4 grid.
    Lesioned brains run side by side in one batch, each brain with its own mask.
    Entry 0 of the result is the intact brain on the same grid, the reference.

  python -m brain.lesion candidates
  python -m brain.lesion screen                         # reduced network, CPU
  python -m brain.lesion screen --device cuda --full-brain --batch 1024
  python -m brain.lesion confirm --types TYPE1 TYPE2    # re-run chosen lesions on the full brain
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from . import reduce as red
from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .response import OUTPUTS, TABLE_DIR, ResponseTable, run_decisions
from .runlog import run_record, save_json
from .sim import BrainSim
from .variants import make_variant

ANNOTATIONS = ROOT / "external" / "flywire_annotations" / "neuron_annotations.tsv"
CANDIDATES_FILE = ROOT / "data" / "lesion_candidates.json"
SCREEN_FILE = TABLE_DIR / "lesion_screen.npz"

MAX_HOPS = 3
MIN_SYNAPSES = 5
COARSE_SUGAR = np.array([0.0, 67.0, 133.0, 200.0])
COARSE_BITTER = np.array([0.0, 67.0, 133.0, 200.0])
COARSE_LOOMING = np.array([0.0, 2.0, 10.0, 200.0])


def hops_from(conn: Connectome, sources: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Fewest synaptic hops from `sources` to every neuron (or to `sources`, if reverse). MAX_HOPS + 1 = farther."""
    pre, post, signed = conn.edges()
    strong = np.abs(signed) >= MIN_SYNAPSES
    rows, cols = (post[strong], pre[strong]) if reverse else (pre[strong], post[strong])
    adj = sp.csr_matrix((np.ones(strong.sum(), dtype=np.int8), (rows, cols)), shape=(conn.n, conn.n))
    hops = np.full(conn.n, MAX_HOPS + 1)
    frontier = np.zeros(conn.n, dtype=bool)
    frontier[sources] = True
    hops[sources] = 0
    for h in range(1, MAX_HOPS + 1):
        reached = (adj.T @ frontier.astype(np.int8)) > 0
        frontier = reached & (hops > MAX_HOPS)
        hops[frontier] = h
    return hops


def find_candidates(device: str = "cpu", seed: int = 0) -> dict:
    conn = Connectome.load()
    groups = NeuronGroups(conn)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False, usecols=["root_id", "cell_type", "super_class", "top_nt"])
    ann = ann[ann.root_id.isin(conn.ids)].drop_duplicates("root_id")
    cell_type = np.full(conn.n, None, dtype=object)
    super_class = np.full(conn.n, None, dtype=object)
    transmitter = np.full(conn.n, None, dtype=object)
    idx = conn.index_of(ann.root_id.to_numpy())
    cell_type[idx], super_class[idx], transmitter[idx] = ann.cell_type.to_numpy(), ann.super_class.to_numpy(), ann.top_nt.to_numpy()

    down = hops_from(conn, groups.core_inputs())
    up = hops_from(conn, groups.all_outputs(), reverse=True)
    on_path = (down <= MAX_HOPS) & (up <= MAX_HOPS)

    # which neurons spike in the intact brain, over a spread of inputs
    sim = BrainSim(conn, device=device)
    conds = np.concatenate([red.calibration_conditions(seed), red.random_conditions(seed + 1, 56)])
    res = sim.run(sim.new_state(len(conds), seed=seed), red.DECISION_MS, inputs=red.sense_inputs(groups, conds),
                  record_all=True)
    spikes = res.all_counts.sum(dim=1).cpu().numpy()
    active = spikes > 0

    is_io = np.zeros(conn.n, dtype=bool)
    is_io[groups.core_inputs()] = True
    is_io[groups.all_outputs()] = True
    typed = np.array([isinstance(c, str) for c in cell_type])
    eligible = on_path & active & typed & ~is_io

    rows = []
    for name in sorted(set(cell_type[eligible])):
        members = np.flatnonzero(cell_type == name)
        hit = members[eligible[members]]
        if is_io[members].any():
            continue                                    # never lesion a type that contains our inputs or outputs
        nts = [t for t in transmitter[members] if isinstance(t, str)]
        rows.append({
            "cell_type": name,
            "super_class": next((s for s in super_class[members] if isinstance(s, str)), None),
            "transmitter": max(set(nts), key=nts.count) if nts else None,
            "neurons": int(len(members)),
            "active_neurons": int(active[members].sum()),
            "spikes_in_calibration": int(spikes[members].sum()),
            "hops_from_input": int(down[hit].min()),
            "hops_to_output": int(up[hit].min()),
            "ids": conn.ids[members].tolist(),
        })
    rows.sort(key=lambda r: -r["spikes_in_calibration"])
    doc = {
        "max_hops": MAX_HOPS, "min_synapses": MIN_SYNAPSES,
        "counts": {
            "neurons": int(conn.n),
            "neurons_within_hops_of_inputs_and_outputs": int(on_path.sum()),
            "cell_types_within_hops": int(len(set(cell_type[on_path & typed]))),
            "neurons_that_spike": int(active.sum()),
            "neurons_that_spike_and_on_path": int((active & on_path).sum()),
            "candidate_cell_types": len(rows),
        },
        "calibration_brains": int(len(conds)),
        "candidates": rows,
    }
    CANDIDATES_FILE.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    return doc


def load_candidates() -> list[dict]:
    return json.loads(CANDIDATES_FILE.read_text(encoding="utf8"))["candidates"]


def coarse_grid() -> np.ndarray:
    return np.stack(np.meshgrid(COARSE_SUGAR, COARSE_BITTER, COARSE_LOOMING, indexing="ij"), axis=-1).reshape(-1, 3)


def run_screen(lesions: list[dict], full_brain: bool, device: str, seeds: int, batch: int, seed: int) -> dict:
    """Coarse response tables for the intact brain (entry 0) and every lesion."""
    conn = Connectome.load()
    groups = NeuronGroups(conn)
    intact = make_variant("intact", conn, groups)
    if full_brain:
        net, net_groups, reduction = conn, groups, None
    else:
        keep, info = red.find_active(intact, groups, device=device, seed=seed)
        check, missed = red.verify(intact, groups, keep, device=device, seed=seed + 1)
        keep = np.union1d(keep, missed)
        net, _ = red.reduce(intact, keep)
        net_groups, reduction = NeuronGroups(net), {**info, **check, "neurons_kept": int(len(keep))}
    sim = BrainSim(net, device=device)

    grid = coarse_grid()
    per = len(grid) * seeds                                   # brains per lesion
    names = ["none"] + [l["cell_type"] for l in lesions]
    members = [np.array([], dtype=np.int64)]
    for l in lesions:
        ids = np.asarray(l["ids"], dtype=np.int64)
        members.append(net.index_of(ids[np.isin(ids, net.ids)]))

    rates = np.zeros((len(names), per, len(OUTPUTS)), dtype=np.float32)
    gf = np.zeros((len(names), per), dtype=bool)
    lesions_per_batch = max(1, batch // per)
    conds_one = np.repeat(grid, seeds, axis=0).astype(np.float32)
    t0 = time.time()
    for start in range(0, len(names), lesions_per_batch):
        chunk = range(start, min(start + lesions_per_batch, len(names)))
        alive = np.ones((net.n, len(chunk) * per), dtype=bool)
        for j, li in enumerate(chunk):
            alive[members[li], j * per:(j + 1) * per] = False
        conds = np.tile(conds_one, (len(chunk), 1))
        r, g = run_decisions(sim, net_groups, conds, alive, seed=seed * 7919 + start, batch=len(conds))
        rates[list(chunk)] = r.reshape(len(chunk), per, len(OUTPUTS))
        gf[list(chunk)] = g.reshape(len(chunk), per)
        done = chunk[-1] + 1
        rate = done * per / (time.time() - t0)
        print(f"  {done}/{len(names)} lesions, {rate:.1f} decisions per second, "
              f"about {(len(names) - done) * per / rate / 60:.0f} min left", flush=True)

    shape = (len(names), len(COARSE_SUGAR), len(COARSE_BITTER), len(COARSE_LOOMING))
    rates = rates.reshape(*shape, seeds, len(OUTPUTS))
    return {
        "names": np.array(names), "sugar_hz": COARSE_SUGAR, "bitter_hz": COARSE_BITTER, "looming_hz": COARSE_LOOMING,
        "mean": rates.mean(axis=4), "sd": rates.std(axis=4), "p_gf": gf.reshape(*shape, seeds).mean(axis=4).astype(np.float32),
        "seeds": np.asarray(seeds), "reduction": json.dumps(reduction),
    }


def lesion_tables(path=SCREEN_FILE) -> dict[str, ResponseTable]:
    """Every lesion's coarse table as a policy, keyed by cell type. 'none' is the intact reference."""
    z = np.load(path)
    axes = (z["sugar_hz"], z["bitter_hz"], z["looming_hz"])
    return {str(n): ResponseTable(str(n), axes, z["mean"][i], z["sd"][i], z["p_gf"][i]) for i, n in enumerate(z["names"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["candidates", "screen", "confirm"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="runs per grid point")
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--full-brain", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the N most active candidates")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.what == "candidates":
        doc = find_candidates(device=args.device, seed=args.seed)
        print(json.dumps(doc["counts"], indent=1))
        for r in doc["candidates"][:15]:
            print(f"  {r['cell_type']:14s} {str(r['super_class']):18s} {r['neurons']:4d} neurons, "
                  f"{r['active_neurons']:3d} active, {r['spikes_in_calibration']:7d} spikes")
        return

    lesions = load_candidates()
    if args.what == "screen":
        lesions = lesions[:args.limit] if args.limit else lesions
        out = run_screen(lesions, args.full_brain, args.device, args.seeds, args.batch, args.seed)
        np.savez_compressed(SCREEN_FILE, **out)
        save_json(TABLE_DIR / "lesion_screen.json", {
            "run": run_record({"seed": args.seed, "seeds_per_point": args.seeds, "full_brain": args.full_brain},
                              device=args.device),
            "lesions": len(lesions), "reduction": json.loads(str(out["reduction"]))})
        print("wrote", SCREEN_FILE.relative_to(ROOT))
    else:
        # the lesions that matter most are re-run on the full brain, so the ranking does not rest on the reduction
        chosen = [l for l in lesions if l["cell_type"] in set(args.types or [])]
        out = run_screen(chosen, True, args.device, args.seeds, args.batch, args.seed)
        np.savez_compressed(TABLE_DIR / "lesion_confirm.npz", **out)
        print("wrote tables/lesion_confirm.npz")


if __name__ == "__main__":
    main()
