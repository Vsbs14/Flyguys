"""Time one batch of decisions at several batch sizes.

The workload is the one the response tables use: every brain starts from rest,
gets its own random sugar / bitter / looming rates in 0-200 Hz, and runs for one
decision window.

  python -m brain.bench --device cpu --sizes 16 64
  python -m brain.bench --device cuda            # 64, 256, 1024
"""
from __future__ import annotations

import argparse
import time

import torch

from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .runlog import run_record, save_json
from .sim import BrainSim

DECISION_MS = 200.0


def run_benchmark(sim: BrainSim, groups: NeuronGroups, sizes: list[int], decision_ms: float = DECISION_MS,
                  seed: int = 0) -> list[dict]:
    rows = []
    monitor = torch.from_numpy(groups.all_outputs())
    cuda = sim.device.type == "cuda"
    for B in sizes:
        gen = torch.Generator().manual_seed(seed)
        rates = torch.rand(3, B, generator=gen) * 200.0
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(sim.device)
            torch.cuda.synchronize(sim.device)
        t0 = time.perf_counter()
        state = sim.new_state(B, seed=seed)
        res = sim.run(state, decision_ms, monitor=monitor,
                      inputs=[(groups.sense(s), rates[i]) for i, s in enumerate(groups.senses)])
        if cuda:
            torch.cuda.synchronize(sim.device)
        wall = time.perf_counter() - t0
        n_steps = round(decision_ms / sim.p.dt)
        rows.append({
            "brains": B,
            "decision_ms": decision_ms,
            "wall_s": round(wall, 2),
            "ms_per_step": round(wall / n_steps * 1e3, 3),
            "decisions_per_s": round(B / wall, 1),
            "spikes_per_brain": round(float(res.total_spikes.mean()), 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated(sim.device) / 2**30, 2) if cuda else None,
        })
        print(rows[-1], flush=True)
        del state, res
    return rows


def to_markdown(rows: list[dict], record: dict) -> str:
    lines = [
        "# Benchmark: one batch of decisions",
        "",
        f"Device: {record['device_name']} ({record['device']}). Each brain runs {rows[0]['decision_ms']:.0f} ms "
        "from rest with random 0-200 Hz sugar, bitter and looming input.",
        "",
        "| Brains | Wall time (s) | ms per step | Decisions per second | Peak GPU memory (GB) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mem = "-" if r["peak_gpu_gb"] is None else f"{r['peak_gpu_gb']:.2f}"
        lines.append(f"| {r['brains']} | {r['wall_s']:.2f} | {r['ms_per_step']:.2f} | {r['decisions_per_s']:.1f} | {mem} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sizes = args.sizes or ([64, 256, 1024] if args.device.startswith("cuda") else [16, 64])

    conn = Connectome.load()
    sim = BrainSim(conn, device=args.device)
    rows = run_benchmark(sim, NeuronGroups(conn), sizes, seed=args.seed)
    record = run_record({"seed": args.seed, "sizes": sizes, "decision_ms": DECISION_MS}, device=args.device)
    tag = f"benchmark_{sim.device.type}"
    save_json(ROOT / "results" / f"{tag}.json", {"run": record, "rows": rows})
    md = to_markdown(rows, record)
    (ROOT / "results" / f"{tag}.md").write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
