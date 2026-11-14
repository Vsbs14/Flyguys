"""Phase 3: build, check and cache every brain variant.

For each variant this
  1. finds the neurons that matter and measures how closely the reduced brain
     matches the full one (see `brain.reduce`), unless --full-brain is given,
  2. builds the response table on the 10 x 10 x 10 input grid with 5 seeds,
  3. runs the six sanity checks on the full brain,
and writes  tables/<variant>.npz  tables/<variant>.json  results/sanity_<variant>_<device>.*

  python -m brain.build_tables                         # every variant, on CPU
  python -m brain.build_tables --variants intact no_gf
  python -m brain.build_tables --device cuda --full-brain --batch 1024
  python -m brain.build_tables --validate              # Step 3.3 for the intact table
  python -m brain.build_tables --summary               # tables of what exists
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import reduce as red
from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .response import OUTPUTS, TABLE_DIR, ResponseTable, build_table, validate_table
from .runlog import run_record, save_json
from .sanity import run_sanity, sanity_config, to_markdown
from .sim import BrainSim
from .variants import DESCRIPTIONS, EXPECTED_FAIL, VARIANTS, make_variant

RESULTS = ROOT / "results"

# The reduced brain is used only if, on fresh inputs, it misses at most this share of the full
# brain's spikes and moves the output neurons' rates by at most this much on average.
MAX_DROPPED_FRACTION = 1e-3
MAX_OUTPUT_DIFF_HZ = 0.5


def build_variant(name: str, conn: Connectome, groups: NeuronGroups, args) -> None:
    t0 = time.time()
    variant = make_variant(name, conn, groups)
    print(f"\n=== {name}: {variant.description} ({variant.n_silenced} neurons silenced) ===", flush=True)
    meta = {"variant": name, "description": variant.description, "neurons_silenced": variant.n_silenced}

    if args.full_brain:
        sim, sim_groups, alive = BrainSim(variant.conn, device=args.device), groups, variant.alive
        meta["reduction"] = None
    else:
        keep, info = red.find_active(variant, groups, device=args.device, seed=args.seed)
        print(f"  kept {info['neurons_kept']} of {info['neurons_full']} neurons "
              f"({info['neurons_that_spiked']} spiked in calibration)", flush=True)
        check, missed = red.verify(variant, groups, keep, device=args.device, seed=args.seed + 1)
        print(f"  reduced vs full on {check['verification_brains']} new brains: "
              f"{check['spikes_from_dropped_neurons']} of {check['spikes_compared']} spikes came from dropped neurons; "
              f"outputs identical in {check['brains_with_identical_outputs']}/{check['verification_brains']} brains, "
              f"mean output difference {check['output_rate_mean_abs_diff_hz']:.3f} Hz", flush=True)
        if (check["dropped_spike_fraction"] > MAX_DROPPED_FRACTION
                or check["output_rate_mean_abs_diff_hz"] > MAX_OUTPUT_DIFF_HZ):
            raise SystemExit(f"{name}: the reduced brain is too far from the full brain. Use --full-brain.")
        keep = np.union1d(keep, missed)                 # keep whatever the check caught
        info["neurons_kept"] = int(len(keep))
        sub, alive = red.reduce(variant, keep)
        sim, sim_groups = BrainSim(sub, device=args.device), NeuronGroups(sub)
        meta["reduction"] = {**info, **check}

    table = build_table(sim, sim_groups, alive, seeds=args.seeds, seed=args.seed, batch=args.batch, progress=True)
    TABLE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(TABLE_DIR / f"{name}.npz", **table)
    meta["grid"] = {k: table[k].tolist() for k in ("sugar_hz", "bitter_hz", "looming_hz")}
    meta["outputs"] = OUTPUTS
    meta["table_seconds"] = round(time.time() - t0, 1)
    del sim

    if not args.skip_sanity:
        full = BrainSim(variant.conn, device=args.device)
        result = run_sanity(full, groups, alive=variant.alive, replicates=args.replicates, seed=args.seed,
                            expected_fail=EXPECTED_FAIL[name], variant=name)
        record = run_record(sanity_config(args.seed, args.replicates), device=args.device)
        tag = f"sanity_{name}_{full.device.type}"
        save_json(RESULTS / f"{tag}.json", {"run": record, "result": result})
        (RESULTS / f"{tag}.md").write_text(to_markdown(result, record), encoding="utf8")
        meta["sanity"] = {r["check"]: r["passed"] for r in result["checks"]}
        print("  sanity: " + ", ".join(f"{r['check']}={'pass' if r['passed'] else 'FAIL'}" for r in result["checks"]),
              flush=True)
        del full

    record = run_record({"seed": args.seed, "seeds_per_point": args.seeds, "batch": args.batch,
                         "decision_ms": red.DECISION_MS, "full_brain": args.full_brain}, device=args.device)
    save_json(TABLE_DIR / f"{name}.json", {"run": record, **meta})
    print(f"  done in {time.time() - t0:.0f} s", flush=True)


def validate(conn: Connectome, groups: NeuronGroups, args) -> None:
    """Step 3.3: live full brain vs table interpolation on random inputs."""
    name = "intact"
    variant = make_variant(name, conn, groups)
    table = ResponseTable.load(name)
    sim = BrainSim(variant.conn, device=args.device)
    out = validate_table(table, sim, groups, variant.alive, n=args.validate_n, seed=args.seed + 12345, batch=args.batch)
    record = run_record({"seed": args.seed + 12345, "n": args.validate_n}, device=args.device)
    save_json(RESULTS / "table_validation_intact.json", {"run": record, "result": out})
    gf = out["giant_fibre"]
    lines = [
        "# Response table check: live brain vs cached table",
        "",
        f"{out['n']} random inputs were run once each on the live full brain ({record['device_name']}) and compared "
        "with the intact table's interpolation. One live run is noisy, so the error should be read against the "
        "trial-to-trial spread stored in the table.",
        "",
        "| Output | Live mean (Hz) | Table mean (Hz) | Mean abs. error (Hz) | Bias (Hz) | Trial-to-trial SD (Hz) | Correlation |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"feed": "MN9 (feed)", "escape": "Giant Fibre (escape)", "retreat": "MDN (retreat)", "advance": "DNp09 (advance)"}
    for o in OUTPUTS:
        r = out["outputs"][o]
        corr = "-" if r["correlation"] is None else f"{r['correlation']:.3f}"
        lines.append(f"| {names[o]} | {r['live_mean_hz']:.2f} | {r['table_mean_hz']:.2f} | {r['mean_abs_error_hz']:.2f} | "
                     f"{r['bias_hz']:+.2f} | {r['trial_to_trial_sd_hz']:.2f} | {corr} |")
    lines += ["",
              f"Giant Fibre spike: the live brain spiked in {gf['live_spike_fraction']:.1%} of runs; the table predicted "
              f"{gf['table_mean_probability']:.1%}. The table's call (probability at least 0.5) matched the live brain in "
              f"{gf['agreement']:.1%} of runs. Brier score {gf['brier_score']:.3f}."]
    md = "\n".join(lines) + "\n"
    (RESULTS / "table_validation_intact.md").write_text(md, encoding="utf8")
    print(md)


def summary() -> None:
    """Write results/variants_summary.md from whatever tables and sanity results exist."""
    checks = ["silence", "sugar_feeds", "bitter_suppresses", "looming_escapes", "gf_off_no_escape", "dies_out"]
    heads = ["No input: silence", "Sugar: MN9 fires", "Bitter: MN9 drops", "Looming: GF fires", "GF off: no escape",
             "Input off: dies out"]
    lines = ["# Brain variants", "", "## Sanity checks", "",
             "PASS* and FAIL* mark results the lesion is designed to produce. "
             "For the size and shuffled variants there is no prediction; the result is just recorded.", "",
             "| Variant | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
    rows2 = []
    for name in VARIANTS:
        meta_path = TABLE_DIR / f"{name}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        sanity_files = sorted(RESULTS.glob(f"sanity_{name}_*.json"))
        if sanity_files:
            res = json.loads(sanity_files[0].read_text(encoding="utf8"))["result"]
            cells = []
            for c in checks:
                r = next(x for x in res["checks"] if x["check"] == c)
                word = "PASS" if r["passed"] else "FAIL"
                if r["as_expected"] is False:
                    word += " (unexpected)"
                elif r["expected_to_pass"] is False:
                    word += "*"
                cells.append(word)
            lines.append(f"| {DESCRIPTIONS[name]} | " + " | ".join(cells) + " |")
        t = ResponseTable.load(name)
        red_info = meta.get("reduction")
        rows2.append(
            f"| {DESCRIPTIONS[name]} | {meta['neurons_silenced']:,} | "
            + (f"{red_info['neurons_kept']:,} | {red_info['dropped_spike_fraction']:.4%} | "
               f"{red_info['brains_with_identical_outputs']}/{red_info['verification_brains']}"
               if red_info else "full brain | - | -")
            + f" | {t.mean[..., 0].max():.0f} | {t.p_gf.max():.2f} | {t.mean[..., 2].max():.1f} | {t.mean[..., 3].max():.1f} |")
    lines += ["", "## Response tables", "",
              "Largest mean response anywhere on the 10 x 10 x 10 input grid. Tables are built on a reduced network "
              "of the neurons that matter. The two check columns compare it with the full brain on 64 fresh inputs: "
              "the share of the full brain's spikes that came from neurons the reduction dropped, and the number of "
              "brains whose output neurons fired exactly the same number of spikes.", "",
              "| Variant | Neurons silenced | Neurons simulated | Spikes missed | Brains with identical outputs | Max MN9 (Hz) | Max P(Giant Fibre spike) | Max MDN (Hz) | Max DNp09 (Hz) |",
              "|---|---|---|---|---|---|---|---|---|"] + rows2
    md = "\n".join(lines) + "\n"
    (RESULTS / "variants_summary.md").write_text(md, encoding="utf8")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=None, choices=VARIANTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="runs per grid point")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replicates", type=int, default=4, help="replicate brains per sanity condition")
    ap.add_argument("--full-brain", action="store_true", help="simulate all neurons instead of the reduced network")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--validate", action="store_true", help="only run the Step 3.3 check on the intact table")
    ap.add_argument("--validate-n", type=int, default=200)
    ap.add_argument("--summary", action="store_true", help="only write results/variants_summary.md")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.summary:
        return summary()

    conn = Connectome.load()
    groups = NeuronGroups(conn)
    if args.validate:
        return validate(conn, groups, args)
    for name in args.variants or VARIANTS:
        if args.skip_existing and (TABLE_DIR / f"{name}.json").exists():
            print(f"skip {name}: table exists")
            continue
        build_variant(name, conn, groups, args)
    summary()


if __name__ == "__main__":
    main()
