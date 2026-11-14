"""Run the brain engine on a Modal GPU.

One-time setup (opens a browser to log in; the token is stored in your home
directory, never in this repo):

  pip install modal
  modal setup

Then, from the repo root:

  modal run -m brain.modal_app                    # sanity checks + benchmark on a T4
  modal run -m brain.modal_app --task sanity
  modal run -m brain.modal_app --task benchmark

Results are written to results/ on your machine.
"""
from __future__ import annotations

import modal

SHIU_RAW = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("torch==2.9.0", "numpy", "pandas", "pyarrow")
    # Shiu's connectome files are baked into the image, so containers start with the data in place.
    .run_commands(
        "mkdir -p /fly_data",
        f"curl -fsSL -o /fly_data/Connectivity_783.parquet {SHIU_RAW}/Connectivity_783.parquet",
        f"curl -fsSL -o /fly_data/Completeness_783.csv {SHIU_RAW}/Completeness_783.csv",
    )
    .env({"FLY_DATA_DIR": "/fly_data"})
    .add_local_dir("data", "/root/data")          # neuron_groups.json
    .add_local_python_source("brain")
)

app = modal.App("flyguys-brain", image=image)
GPU = "T4"


def _engine():
    from brain.connectome import Connectome
    from brain.neurons import NeuronGroups
    from brain.sim import BrainSim

    conn = Connectome.load()
    return BrainSim(conn, device="cuda"), NeuronGroups(conn)


@app.function(gpu=GPU, timeout=1800)
def sanity_gpu(seed: int = 0, replicates: int = 4) -> dict:
    from brain import sanity
    from brain.runlog import run_record

    sim, groups = _engine()
    result = sanity.run_sanity(sim, groups, replicates=replicates, seed=seed)
    record = run_record(sanity.sanity_config(seed, replicates), device="cuda")
    return {"run": record, "result": result, "markdown": sanity.to_markdown(result, record)}


@app.function(gpu=GPU, timeout=3600)
def benchmark_gpu(sizes: list[int], seed: int = 0) -> dict:
    from brain import bench
    from brain.runlog import run_record

    sim, groups = _engine()
    bench.run_benchmark(sim, groups, [sizes[0]], decision_ms=10.0, seed=seed)     # warm up CUDA
    rows = bench.run_benchmark(sim, groups, sizes, seed=seed)
    record = run_record({"seed": seed, "sizes": sizes, "decision_ms": bench.DECISION_MS}, device="cuda")
    return {"run": record, "rows": rows, "markdown": bench.to_markdown(rows, record)}


@app.local_entrypoint()
def main(task: str = "all", seed: int = 0):
    from pathlib import Path

    from brain.runlog import _git_commit, save_json

    out = Path(__file__).resolve().parents[1] / "results"
    commit = _git_commit()
    jobs = {}
    if task in ("all", "sanity"):
        jobs["sanity_intact_cuda"] = sanity_gpu.spawn(seed=seed)
    if task in ("all", "benchmark"):
        jobs["benchmark_cuda"] = benchmark_gpu.spawn([64, 256, 1024], seed=seed)
    if not jobs:
        raise SystemExit(f"unknown task {task!r}; use all, sanity or benchmark")
    for tag, job in jobs.items():
        doc = job.get()
        md = doc.pop("markdown")
        doc["run"]["git_commit"] = commit       # the container has no git checkout
        save_json(out / f"{tag}.json", doc)
        (out / f"{tag}.md").write_text(md, encoding="utf8")
        print(md)
        print(f"saved results/{tag}.json and .md\n")
