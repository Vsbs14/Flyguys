"""Phase 6: experiments and baselines. Does the fly brain actually help a colony survive?

Every decision maker plays the *same* galaxies alone: same map, resources,
toxic systems and raider start for a given seed. Comparisons are paired by
seed, so map luck cancels out. Decision makers:

  intact          the real FlyWire brain (response table)
  no-<type>       lesioned brains picked by the Phase 4 screen
  shuffled        the same brain with its interneuron wiring scrambled
  threshold       a hand-written rule, no brain
  random          uniform random actions

The whole comparison is repeated in a mild, a normal and a harsh galaxy
(fewer or more raiders) to see whether the ranking holds.

A second experiment puts every decision maker in one shared galaxy, where
they compete for the same resources and draw the same raiders.

  python -m galaxy.experiments              # 40 seeds x 3 hazard levels, 200 ticks
  python -m galaxy.experiments --seeds 5    # quick look
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from .brains import ACTIONS, ROOT, available, load_all
from .policies import BASELINES
from .world import ESCAPE, FEED, Galaxy, GalaxyConfig

HAZARDS = {"mild": {"n_raiders": 2}, "normal": {"n_raiders": 4}, "harsh": {"n_raiders": 8}}


def deciders(names=None) -> dict:
    d = dict(load_all(names))
    for name, cls in BASELINES.items():
        d[name] = cls()
    return d


def play_solo(decider, cfg: GalaxyConfig, seed: int, ticks: int) -> dict:
    g = Galaxy(cfg, {"x": decider}, seed=seed)
    start = g.census()["x"]["population"]
    acts = np.zeros(len(ACTIONS))
    raids = escapes = toxic = toxic_fed = 0
    extinct_at = None
    pop_curve = [start]
    for t in range(ticks):
        info = g.step()
        acts += info["act_by_variant"]["x"]
        loom = info["x"][:, 2] >= 0.99
        raids += int(loom.sum())
        escapes += int((info["act"][loom] == ESCAPE).sum())
        tox = (info["x"][:, 1] >= 0.5) & (info["x"][:, 0] >= 0.5)
        toxic += int(tox.sum())
        toxic_fed += int((info["act"][tox] == FEED).sum())
        c = g.census()["x"]
        pop_curve.append(c["population"])
        if c["colonies"] == 0 and extinct_at is None:
            extinct_at = t + 1
            break
    c = g.census()["x"]
    return {"final_population": c["population"], "systems": c["systems"], "survived": c["colonies"] > 0,
            "extinct_at": extinct_at, "actions": acts, "escape_when_raided": escapes / raids if raids else np.nan,
            "feed_on_toxic": toxic_fed / toxic if toxic else np.nan,
            "mean_population": float(np.mean(pop_curve + [0.0] * (ticks + 1 - len(pop_curve))))}


def mean_se(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
            "n": int(len(x))}


def solo_experiment(ds: dict, seeds: list[int], ticks: int) -> dict:
    out = {}
    for hz, over in HAZARDS.items():
        cfg = replace(GalaxyConfig(), **over)
        runs = {n: [play_solo(d, cfg, s, ticks) for s in seeds] for n, d in ds.items()}
        ref = np.array([r["mean_population"] for r in runs["intact"]])
        res = {}
        for n, rs in runs.items():
            mp = np.array([r["mean_population"] for r in rs])
            acts = np.sum([r["actions"] for r in rs], 0)
            res[n] = {
                "mean_population": mean_se(mp),
                "final_population": mean_se([r["final_population"] for r in rs]),
                "systems_held": mean_se([r["systems"] for r in rs]),
                "survival": float(np.mean([r["survived"] for r in rs])),
                "vs_intact": mean_se(mp - ref) if n != "intact" else None,
                "beats_intact": float(np.mean(mp > ref)) if n != "intact" else None,
                "escape_when_raided": mean_se([r["escape_when_raided"] for r in rs]),
                "feed_on_toxic": mean_se([r["feed_on_toxic"] for r in rs]),
                "actions": dict(zip(ACTIONS, (acts / max(acts.sum(), 1)).round(4).tolist())),
            }
        out[hz] = res
        print(f"{hz}: " + ", ".join(f"{n} {res[n]['mean_population']['mean']:.0f}" for n in res), flush=True)
    return out


def shared_experiment(ds: dict, seeds: list[int], ticks: int) -> dict:
    """Everyone in one galaxy: share of all population at the end, and wins."""
    names = list(ds)
    shares = []
    for s in seeds:
        g = Galaxy(GalaxyConfig(colonies_per_variant=6), ds, seed=s)
        for _ in range(ticks):
            g.step()
        c = g.census()
        pop = np.array([c[n]["population"] for n in names])
        shares.append(pop / pop.sum() if pop.sum() > 0 else pop)
    shares = np.array(shares)
    return {n: {"share": mean_se(shares[:, i]), "wins": int((shares.argmax(1) == i).sum()),
                "survived": float((shares[:, i] > 0).mean())} for i, n in enumerate(names)}


def to_markdown(doc: dict) -> str:
    solo, shared, meta = doc["solo"], doc["shared"], doc["run"]
    desc = doc["descriptions"]
    L = ["# Experiments and baselines", "",
         f"{len(meta['seeds'])} paired galaxy seeds, {meta['ticks']} ticks each. Every decision maker plays the same "
         "galaxies alone. Score = mean population over the run (higher is better), mean ± standard error over seeds. "
         "\"vs intact\" is the paired difference on the same seeds; \"beats intact\" is the share of seeds where it did "
         "better.", ""]
    for hz, res in solo.items():
        L += [f"## {hz.capitalize()} galaxy ({HAZARDS[hz]['n_raiders']} raiders)", "",
              "| Decision maker | Mean population | vs intact | Beats intact | Survived | Escapes when raided | "
              "Feeds on toxic food |", "|---|---|---|---|---|---|---|"]
        for n in sorted(res, key=lambda k: -res[k]["mean_population"]["mean"]):
            r = res[n]
            vs = "reference" if r["vs_intact"] is None else f"{r['vs_intact']['mean']:+.0f} ± {r['vs_intact']['se']:.0f}"
            bi = "-" if r["beats_intact"] is None else f"{r['beats_intact']:.0%}"
            pct = lambda m: "-" if m["mean"] is None else f"{m['mean']:.0%}"
            L.append(f"| {n} | {r['mean_population']['mean']:.0f} ± {r['mean_population']['se']:.0f} | {vs} | {bi} | "
                     f"{r['survival']:.0%} | {pct(r['escape_when_raided'])} | {pct(r['feed_on_toxic'])} |")
        L.append("")
    L += ["## Shared galaxy", "",
          "All decision makers in one galaxy, competing for the same resources. Share = share of all population at the "
          "end.", "", "| Decision maker | Share of population | Won | Survived |", "|---|---|---|---|"]
    for n in sorted(shared, key=lambda k: -shared[k]["share"]["mean"]):
        s = shared[n]
        L.append(f"| {n} | {s['share']['mean']:.0%} ± {s['share']['se']:.0%} | {s['wins']}/{len(meta['seeds'])} | "
                 f"{s['survived']:.0%} |")
    L += ["", "## Decision makers", "", "| Name | What it is |", "|---|---|"]
    L += [f"| {n} | {d} |" for n, d in desc.items()]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    seeds = list(range(1000, 1000 + args.seeds))
    ds = deciders()
    if "intact" not in ds:
        raise SystemExit("no intact response table; run `python -m brain.tables` first")
    desc = {n: v["description"] for n, v in available().items()}
    desc["threshold"] = "Hand-written rule: flee looming, back off when crowded, eat plentiful clean food, follow a better neighbour."
    desc["random"] = "Uniformly random action every tick."
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = {"run": {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit or None,
                   "seeds": seeds, "ticks": args.ticks, "hazards": HAZARDS, "config": GalaxyConfig().as_dict()},
           "descriptions": desc,
           "solo": solo_experiment(ds, seeds, args.ticks),
           "shared": shared_experiment(ds, seeds, args.ticks)}
    doc["run"]["wall_seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "experiments.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    md = to_markdown(doc)
    (ROOT / "results" / "experiments.md").write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
