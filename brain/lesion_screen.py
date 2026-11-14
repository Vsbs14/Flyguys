"""Phase 4: lesion screen. Silence one cell type at a time and see which decisions change.

1. Find the neurons that matter: run the probe scenarios (the 15 scenarios of
   `brain.calibrate`) on the intact brain and keep every neuron that spikes.
   Silencing a neuron that never fires changes nothing, so only cell types
   with at least one active neuron are screened. Sense neurons are skipped
   (silencing them only removes a sense).
2. For each candidate type, silence *every* neuron of that type and rerun the
   probe scenarios with several replicate brains. A silenced neuron still
   receives input but never emits a spike, as in `BrainSim`.
3. Score = mean over scenarios of the total variation distance between the
   lesioned and the intact action distributions (0 = identical behaviour,
   1 = completely different). Unlesioned "null" entries run alongside the
   real ones; a lesion counts as an effect only if it beats every null.
4. Pick the variants for the galaxy: the strongest lesion of each distinct
   phenotype (the pattern of majority actions across scenarios), plus the
   intact brain, and run the six sanity checks on each.

Writes results/lesion_screen.{json,md} and data/variants.json.

  python -m brain.lesion_screen --device cuda
  python -m brain.lesion_screen --max-types 20 --replicates 2      # quick test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import sanity
from .calibrate import SCENARIOS
from .connectome import ROOT, Connectome
from .interface import ACTIONS, OUTPUTS, SENSES, DecisionRule, FlyBrain, intensity_to_hz
from .neurons import NeuronGroups
from .runlog import run_record, save_json
from .sim import BrainSim

ANNOT = ROOT / "external" / "flywire_annotations" / "neuron_annotations.tsv"
VARIANTS_FILE = ROOT / "data" / "variants.json"
N_NULL = 12
N_VARIANTS = 5          # lesion variants picked for the galaxy, besides the intact brain
MAX_BRAINS = 1024       # columns per GPU batch


def tv_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation distance along the last axis."""
    return 0.5 * np.abs(p - q).sum(-1)


class Screen:
    def __init__(self, brain: FlyBrain, replicates: int, seed: int):
        self.brain, self.R, self.seed = brain, replicates, seed
        self.sim, self.groups, self.rule = brain.sim, brain.groups, brain.rule
        self.x = np.array([s[1] for s in SCENARIOS], dtype=np.float32)        # (S, senses)
        self.S = len(SCENARIOS)

    def active_neurons(self, replicates: int = 4) -> np.ndarray:
        """Indices of neurons that spike in at least one probe scenario (intact brain)."""
        x = np.repeat(self.x, replicates, axis=0)
        hz = intensity_to_hz(x)
        state = self.sim.new_state(len(x), seed=self.seed + 99)
        res = self.sim.run(state, self.brain.decision_ms, record_all=True,
                           inputs=[(self.groups.sense(s), hz[i]) for i, s in enumerate(SENSES)])
        return torch.nonzero(res.all_counts.sum(1) > 0).flatten().cpu().numpy()

    def run(self, lesions: list[np.ndarray | None], seed: int) -> np.ndarray:
        """Action fractions for each lesion (None = intact): (n_lesions, S, n_actions)."""
        per = self.S * self.R
        chunk = max(1, MAX_BRAINS // per)
        out = []
        for c0 in range(0, len(lesions), chunk):
            part = lesions[c0:c0 + chunk]
            B = len(part) * per
            alive = torch.ones(self.sim.n, B, dtype=torch.bool, device=self.sim.device)
            for k, idx in enumerate(part):
                if idx is not None and len(idx):
                    alive[torch.as_tensor(idx, device=self.sim.device), k * per:(k + 1) * per] = False
            x = np.tile(np.repeat(self.x, self.R, axis=0), (len(part), 1))
            acts = self.rule.choose(self.brain.output_rates(x, alive=alive, seed=seed + c0)).numpy()
            acts = acts.reshape(len(part), self.S, self.R)
            out.append(np.stack([(acts == a).mean(-1) for a in range(len(ACTIONS))], axis=-1))
            del alive
        return np.concatenate(out)


def candidate_types(conn: Connectome, groups: NeuronGroups, active: np.ndarray) -> pd.DataFrame:
    ann = pd.read_csv(ANNOT, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type"])
    ann = ann[ann.cell_type.notna() & ann.root_id.isin(set(conn.ids.tolist()))]
    ann["idx"] = conn.index_of(ann.root_id)
    inputs = set(groups.all_inputs().tolist())
    outputs = {int(i): o for o in groups.outputs for i in groups.output(o)}
    active_set = set(active.tolist())
    rows = []
    for ct, d in ann.groupby("cell_type"):
        idx = d.idx.to_numpy()
        n_act = sum(int(i) in active_set for i in idx)
        if n_act == 0 or any(int(i) in inputs for i in idx):
            continue
        out_of = sorted({outputs[int(i)] for i in idx if int(i) in outputs})
        rows.append({"cell_type": ct, "super_class": d.super_class.mode().iloc[0], "n_neurons": len(idx),
                     "n_active": n_act, "is_output": ",".join(out_of), "idx": np.sort(idx),
                     "root_ids": sorted(int(r) for r in d.root_id)})
    return pd.DataFrame(rows)


def describe(frac: np.ndarray, base: np.ndarray) -> list[str]:
    """Human-readable changes in majority action vs the intact brain."""
    changes = []
    for s, (name, _, _) in enumerate(SCENARIOS):
        a, b = ACTIONS[base[s].argmax()], ACTIONS[frac[s].argmax()]
        if a != b and tv_distance(frac[s], base[s]) >= 0.5:
            changes.append(f"{name}: {a} -> {b}")
    return changes


def slug(cell_type: str) -> str:
    return "no-" + "".join(ch if ch.isalnum() else "-" for ch in cell_type).strip("-").lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=6)
    ap.add_argument("--max-types", type=int, default=None, help="screen only the first N candidates (testing)")
    ap.add_argument("--no-write-variants", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    conn = Connectome.load()
    groups = NeuronGroups(conn)
    sim = BrainSim(conn, device=args.device)
    brain = FlyBrain(sim, groups, DecisionRule.load())
    screen = Screen(brain, args.replicates, args.seed)

    active = screen.active_neurons()
    cands = candidate_types(conn, groups, active)
    if args.max_types:
        cands = cands.sort_values("n_active", ascending=False).head(args.max_types).reset_index(drop=True)
    print(f"{len(active)} active neurons, {len(cands)} candidate cell types", flush=True)

    # intact reference with more replicates, then nulls + lesions
    ref_screen = Screen(brain, max(16, args.replicates), args.seed + 1)
    base = ref_screen.run([None], seed=args.seed + 1)[0]                          # (S, A)
    fr = screen.run([None] * N_NULL + list(cands.idx), seed=args.seed + 2)
    null, les = fr[:N_NULL], fr[N_NULL:]
    null_scores = tv_distance(null, base[None]).mean(-1)
    scores = tv_distance(les, base[None]).mean(-1)
    cutoff = float(null_scores.max())
    cands["score"] = scores
    cands["effect"] = scores > cutoff
    cands["changes"] = [describe(f, base) for f in les]
    cands["phenotype"] = [tuple(ACTIONS[a] for a in f.argmax(-1)) for f in les]
    print(f"null scores max {cutoff:.3f}; {int(cands.effect.sum())} lesions beat it", flush=True)

    # pick variants: strongest lesion per distinct phenotype that changes at least one majority action
    base_pheno = tuple(ACTIONS[a] for a in base.argmax(-1))
    picked, seen = [], {base_pheno}
    for _, r in cands[cands.effect].sort_values("score", ascending=False).iterrows():
        if r.phenotype in seen or not r.changes:
            continue
        seen.add(r.phenotype)
        picked.append(r)
        if len(picked) == N_VARIANTS:
            break

    # sanity checks on each picked variant (recorded, not required to pass)
    sanity_rows = []
    for r in picked:
        alive = np.ones(conn.n, dtype=bool)
        alive[r.idx] = False
        res = sanity.run_sanity(sim, groups, alive=alive, replicates=4, seed=args.seed, variant=slug(r.cell_type))
        sanity_rows.append({c["check"]: c["passed"] for c in res["checks"]})

    record = run_record({"seed": args.seed, "replicates": args.replicates, "n_null": N_NULL,
                         "n_candidates": len(cands), "scenarios": [s[0] for s in SCENARIOS]}, device=args.device)
    variants = {"intact": {"description": "The unmodified FlyWire v783 brain.", "silenced_cell_types": [],
                           "silenced_root_ids": [], "screen_score": 0.0, "changes": []}}
    for r in picked:
        variants[slug(r.cell_type)] = {
            "description": f"{r.cell_type} ({r.n_neurons} neurons, {r.super_class}) silenced.",
            "silenced_cell_types": [r.cell_type], "silenced_root_ids": r.root_ids,
            "screen_score": round(float(r.score), 4), "changes": r.changes,
        }
    if not args.no_write_variants:
        doc = {"made_by": "python -m brain.lesion_screen", "time_utc": record["time_utc"],
               "git_commit": record["git_commit"], "variants": variants}
        VARIANTS_FILE.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")

    table = cands.drop(columns=["idx", "root_ids"]).sort_values("score", ascending=False)
    save_json(ROOT / "results" / "lesion_screen.json", {
        "run": record, "intact_action_fraction": {s[0]: dict(zip(ACTIONS, base[i].round(3).tolist()))
                                                  for i, s in enumerate(SCENARIOS)},
        "null_scores": null_scores.round(4).tolist(), "cutoff": cutoff,
        "picked": list(variants), "sanity": dict(zip([slug(r.cell_type) for r in picked], sanity_rows)),
        "lesions": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
                    for row in table.to_dict("records")],
    })
    Path(ROOT / "results" / "lesion_screen.md").write_text(
        to_markdown(table, picked, sanity_rows, null_scores, record, args, len(active), time.time() - t0), encoding="utf8")
    print((ROOT / "results" / "lesion_screen.md").read_text(encoding="utf8"))


def to_markdown(table, picked, sanity_rows, null_scores, record, args, n_active, wall) -> str:
    L = [
        "# Lesion screen",
        "",
        f"Device: {record['device_name']} ({record['device']}). Seed {args.seed}, {args.replicates} replicate brains "
        f"per scenario per lesion, {len(SCENARIOS)} probe scenarios (from `brain.calibrate`). {wall / 60:.0f} min.",
        "",
        f"{n_active} neurons spike in at least one scenario. Every cell type with at least one of them, "
        f"apart from the sense neurons, was silenced in full: {len(table)} cell types.",
        "",
        "Score = mean over scenarios of the total variation distance between the lesioned and intact action "
        f"distributions. {N_NULL} unlesioned null entries scored at most {null_scores.max():.3f} "
        f"(mean {null_scores.mean():.3f}); **{int(table.effect.sum())} lesions beat every null.**",
        "",
        "## Variants picked for the galaxy",
        "",
        "The strongest lesion of each distinct behaviour pattern. Sanity checks: 1 silence, 2 sugar feeds, "
        "3 bitter suppresses, 4 looming escapes, 5 GF-off no escape, 6 dies out.",
        "",
        "| Variant | Neurons silenced | Score | What changes | Sanity 1-6 |",
        "|---|---|---|---|---|",
    ]
    for r, s in zip(picked, sanity_rows):
        checks = "".join("✓" if v else "✗" for v in s.values())
        L.append(f"| {slug(r.cell_type)} | {r.cell_type} ({r.n_neurons}, {r.super_class}) | {r.score:.2f} | "
                 f"{'; '.join(r.changes)} | {checks} |")
    L += ["", "## Top 30 lesions", "", "| Cell type | Class | Neurons (active) | Output neuron? | Score | Effect | What changes |",
          "|---|---|---|---|---|---|---|"]
    for _, r in table.head(30).iterrows():
        L.append(f"| {r.cell_type} | {r.super_class} | {r.n_neurons} ({r.n_active}) | {r.is_output or '-'} | "
                 f"{r.score:.2f} | {'yes' if r.effect else 'no'} | {'; '.join(r.changes) or '-'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
