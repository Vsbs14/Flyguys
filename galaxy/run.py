"""Run the galaxy with every brain variant that has a response table, over several seeds.

  python -m galaxy.run                         # 300 ticks x 5 seeds, all variants
  python -m galaxy.run --ticks 100 --seeds 0   # quick look

Writes results/galaxy_summary.{json,md} and, for the first seed, a full
playback file galaxy/runs/seed<k>.json (map + one frame per tick) for Phase 7.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from .brains import ACTIONS, ROOT, SENSES, load_all
from .world import ESCAPE, Galaxy, GalaxyConfig

RUNS = ROOT / "galaxy" / "runs"


def run_one(brains, cfg: GalaxyConfig, seed: int, ticks: int, frames: bool) -> dict:
    g = Galaxy(cfg, brains, seed=seed)
    names = g.variants
    pop = np.zeros((ticks + 1, len(names)))
    pop[0] = [g.census()[n]["population"] for n in names]
    acts = {n: np.zeros(len(ACTIONS)) for n in names}
    threat = {n: [0, 0] for n in names}           # (decisions under a raider, of which escape)
    toxic_feed = {n: [0, 0] for n in names}       # (decisions with bitter >= 0.5 and sugar >= 0.5, of which feed)
    out_frames = [g.frame()] if frames else None
    for t in range(ticks):
        info = g.step()
        for v, n in enumerate(names):
            m = info["var"] == v
            acts[n] += np.bincount(info["act"][m], minlength=len(ACTIONS))
            loom = m & (info["x"][:, 2] >= 0.99)
            threat[n][0] += int(loom.sum())
            threat[n][1] += int((info["act"][loom] == ESCAPE).sum())
            tox = m & (info["x"][:, 1] >= 0.5) & (info["x"][:, 0] >= 0.5)
            toxic_feed[n][0] += int(tox.sum())
            toxic_feed[n][1] += int((info["act"][tox] == 1).sum())
        c = g.census()
        pop[t + 1] = [c[n]["population"] for n in names]
        if frames:
            out_frames.append(g.frame())
    result = {"seed": seed, "variants": names, "population": pop.round(2).tolist(), "census": g.census(),
              "actions": {n: (a / max(a.sum(), 1)).round(4).tolist() for n, a in acts.items()},
              "escape_when_raided": {n: (e / d if d else None) for n, (d, e) in threat.items()},
              "feed_on_toxic": {n: (f / d if d else None) for n, (d, f) in toxic_feed.items()}}
    if frames:
        result["playback"] = {
            "systems": [{"x": round(float(x), 4), "y": round(float(y), 4), "capacity": round(float(c), 1),
                         "toxicity": round(float(tx), 3)}
                        for (x, y), c, tx in zip(g.pos, g.capacity, g.toxicity)],
            "lanes": [list(map(int, e)) for e in g.lanes], "variants": names, "actions": list(ACTIONS),
            "frame_columns": ["colony_id", "system", "population", "variant", "action"], "frames": out_frames}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=None)
    args = ap.parse_args()
    t0 = time.time()

    brains = load_all(args.variants)
    cfg = GalaxyConfig()
    runs = []
    for k, seed in enumerate(args.seeds):
        r = run_one(brains, cfg, seed, args.ticks, frames=(k == 0))
        if "playback" in r:
            RUNS.mkdir(parents=True, exist_ok=True)
            pb = {"seed": seed, "ticks": args.ticks, "config": cfg.as_dict(), **r.pop("playback")}
            (RUNS / f"seed{seed}.json").write_text(json.dumps(pb, separators=(",", ":")), encoding="utf8")
        runs.append(r)
        print(f"seed {seed}: " + ", ".join(f"{n} {r['census'][n]['population']:.0f}" for n in r["variants"]),
              flush=True)

    names = runs[0]["variants"]
    final = np.array([[r["census"][n]["population"] for n in names] for r in runs])
    share = final / np.maximum(final.sum(1, keepdims=True), 1e-9)
    summary = {
        n: {"final_population_mean": float(final[:, i].mean()), "final_population_sd": float(final[:, i].std()),
            "share_mean": float(share[:, i].mean()), "survived_seeds": int((final[:, i] > 0).sum()),
            "wins": int((final.argmax(1) == i).sum()),
            "actions": dict(zip(ACTIONS, np.mean([r["actions"][n] for r in runs], 0).round(3).tolist())),
            "escape_when_raided": _mean([r["escape_when_raided"][n] for r in runs]),
            "feed_on_toxic": _mean([r["feed_on_toxic"][n] for r in runs])}
        for i, n in enumerate(names)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit or None,
            "ticks": args.ticks, "seeds": args.seeds, "config": cfg.as_dict()}
    (ROOT / "results" / "galaxy_summary.json").write_text(
        json.dumps({"run": meta, "summary": summary,
                    "population_by_seed": {r["seed"]: r["population"] for r in runs}}, indent=1) + "\n",
        encoding="utf8")
    md = to_markdown(summary, meta, time.time() - t0)
    (ROOT / "results" / "galaxy_summary.md").write_text(md, encoding="utf8")
    print(md)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def to_markdown(summary: dict, meta: dict, wall: float) -> str:
    cfg = meta["config"]
    fmt = lambda v: "-" if v is None else f"{v:.0%}"
    L = ["# Galaxy run", "",
         f"{len(meta['seeds'])} seeds x {meta['ticks']} ticks, {cfg['n_systems']} systems, {cfg['n_raiders']} raiders, "
         f"{cfg['colonies_per_variant']} starting colonies per variant. Every colony decides with its variant's "
         f"response table. {wall:.0f} s.", "",
         "| Variant | Final population (mean ± SD) | Share of all life | Survived | Won | Escapes when raided | Feeds on toxic food |",
         "|---|---|---|---|---|---|---|"]
    order = sorted(summary, key=lambda n: -summary[n]["final_population_mean"])
    for n in order:
        s = summary[n]
        L.append(f"| {n} | {s['final_population_mean']:.0f} ± {s['final_population_sd']:.0f} | {s['share_mean']:.0%} | "
                 f"{s['survived_seeds']}/{len(meta['seeds'])} | {s['wins']} | {fmt(s['escape_when_raided'])} | "
                 f"{fmt(s['feed_on_toxic'])} |")
    L += ["", "Action mix over the whole run:", "", "| Variant | " + " | ".join(ACTIONS) + " |",
          "|---|" + "---|" * len(ACTIONS)]
    for n in order:
        L.append(f"| {n} | " + " | ".join(f"{summary[n]['actions'][a]:.0%}" for a in ACTIONS) + " |")
    L += ["", "\"Escapes when raided\": share of decisions taken with a raider in the system that were `escape`. "
          "\"Feeds on toxic food\": share of decisions with sugar and bitter both at least 0.5 that were `feed`."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
