"""Build a reusable lookup table from full-connectome simulations."""
import argparse
import os
import tempfile
from itertools import product
from pathlib import Path

import numpy as np

SENSES = ("sugar", "bitter", "looming")
OUTPUTS = {"feed": "MN9", "escape": "GF", "retreat": "MDN", "advance": "DNp09"}
LESIONS = {
    "intact": (),
    "fearless": ("escape",),
    "fearless_no_retreat": ("escape", "retreat"),
    "hungry": ("bitter",),
    "shuffle": (),
}
TABLE = Path(__file__).resolve().parents[1] / "tables" / "responses.npz"


def shuffled(conn, groups, seed):
    """Permute interneuron targets within classes of equal incoming degree."""
    from brain import Connectome

    pre, post, weights = conn.edges()
    protected = np.concatenate((groups.all_inputs(), groups.all_outputs()))
    candidates = np.setdiff1d(np.arange(conn.n), protected)
    indegree = np.bincount(post, minlength=conn.n)
    candidates = candidates[np.argsort(indegree[candidates])]
    mapping = np.arange(conn.n)
    rng = np.random.default_rng(seed)
    for cohort in np.split(candidates, np.flatnonzero(np.diff(indegree[candidates])) + 1):
        mapping[cohort] = rng.permutation(cohort)
    return Connectome.from_edges(conn.ids, pre, mapping[post], weights)


def alive_mask(conn, groups, lesions):
    alive = np.ones(conn.n, dtype=bool)
    for lesion in lesions:
        indices = groups.sense(lesion) if lesion in SENSES else groups.output(lesion)
        alive[indices] = False
    return alive


def build(path, device, levels, replicates, batch, seed):
    import torch
    from brain import BrainSim, Connectome, NeuronGroups

    conn = Connectome.load()
    groups = NeuronGroups(conn)
    monitor = np.concatenate([groups.output(out) for out in OUTPUTS])
    splits = np.cumsum([len(groups.output(out)) for out in OUTPUTS])[:-1]
    grid = np.array(list(product(np.linspace(0, 1, levels), repeat=3)))
    stimuli = np.repeat(grid, replicates, axis=0)
    tables = {}
    sim = BrainSim(conn, device=device)
    for variant, lesions in LESIONS.items():
        if variant == "shuffle":
            del sim
            sim = BrainSim(shuffled(conn, groups, seed), device=device)
        alive = alive_mask(conn, groups, lesions)
        rows = []
        for start in range(0, len(stimuli), batch):
            chunk = stimuli[start:start + batch]
            state = sim.new_state(len(chunk), alive=torch.from_numpy(alive), seed=seed + start)
            inputs = [(groups.sense(s), chunk[:, i] * 200) for i, s in enumerate(SENSES)]
            result = sim.run(state, 200, inputs=inputs, monitor=monitor)
            rates = np.split(result.rates_hz.cpu().numpy(), splits)
            rows.append(np.column_stack([r.mean(axis=0) for r in rates] + [(rates[1] > 0).any(axis=0)]))
        tables[variant] = np.concatenate(rows).reshape(levels, levels, levels, replicates, 5)
        print(f"{variant}: {len(stimuli)} decisions", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, seed=seed, batch=batch, device=device, **tables)
            handle.close()
            temporary.chmod(0o644)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=TABLE)
    args = parser.parse_args()
    if args.levels < 2 or min(args.replicates, args.batch) < 1:
        parser.error("use at least two levels and positive replicates and batch size")
    build(args.output, args.device, args.levels, args.replicates, args.batch, args.seed)


if __name__ == "__main__":
    main()
