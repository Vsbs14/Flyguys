"""Phase 7: record matched galaxy runs for the browser UI.

Writes galaxy/frames.json in the frame format the galaxy UI on the VSTemp/UI
branches plays (viz/frames.py there): one galaxy per decision maker, all on
the same map and seed, one frame per tick.

  {"meta":   {"variants": [...], "seed": 7, "epochs": 150, "source": "flyguys-phase-7", ...},
   "stars":  [{"id": 0, "x": .41, "y": -.22, "capacity": 120.0, "toxicity": 0.05}, ...],
   "lanes":  [[0, 5], ...],
   "frames": [{"t": 0, "galaxies": {"intact": {
                 "colonies": [{"id": 7, "star": 0, "pop": 0.6, "action": "settle",
                               "brain_action": "feed", "population": 72.1,
                               "inputs": {"sugar": .8, "bitter": .1, "looming": 0, "obstacle": .3, "target": 0},
                               "rates": {"MN9": 48, "GF": 0, "MDN": 3, "DNp09": 0}}],
                 "ships":  [{"from": 0, "to": 5, "progress": 0.5, "kind": "advance"}],
                 "events": [{"type": "raid", "star": 12}]}}}, ...]}

`action` uses the UI's four words: feed and idle -> settle, advance -> expand,
retreat -> retreat, escape -> flee. `brain_action` keeps the brain's own word.
`pop` is population / the star's capacity (0-1), as the UI expects; the raw
number is in `population`. Extra keys (`lanes`, `capacity`, `brain_action`,
`kind`, raid events, the two extra senses) are ignored by the current UI.

  python -m galaxy.playback                     # all decision makers, seed 7, 150 ticks
  python -m galaxy.playback --check galaxy/frames.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .brains import ACTIONS, NEURONS, OUTPUTS, ROOT, SENSES
from .experiments import deciders
from .world import Galaxy, GalaxyConfig

FRAMES = ROOT / "galaxy" / "frames.json"
UI_ACTION = {"idle": "settle", "feed": "settle", "advance": "expand", "retreat": "retreat", "escape": "flee"}
UI_ACTIONS = {"settle", "expand", "retreat", "flee"}


def record(names=None, seed: int = 7, ticks: int = 150, cfg: GalaxyConfig | None = None) -> dict:
    cfg = cfg or GalaxyConfig()
    ds = deciders()
    names = names or list(ds)
    galaxies = {n: Galaxy(cfg, {n: ds[n]}, seed=seed) for n in names}
    g0 = galaxies[names[0]]
    stars = [{"id": i, "x": round(float(x), 4), "y": round(float(y), 4), "capacity": round(float(c), 1),
              "toxicity": round(float(t), 3)} for i, ((x, y), c, t) in enumerate(zip(g0.pos, g0.capacity, g0.toxicity))]
    frames = []
    for t in range(ticks + 1):
        frame = {"t": t, "galaxies": {}}
        for n, g in galaxies.items():
            if t > 0:
                g.step()
            frame["galaxies"][n] = snapshot(g, ds[n], first=(t == 0))
        frames.append(frame)
    final = {n: g.census()[n] for n, g in galaxies.items()}
    return {"meta": {"variants": names, "seed": seed, "epochs": len(frames), "source": "flyguys-phase-7",
                     "senses": list(SENSES), "brain_actions": list(ACTIONS), "config": cfg.as_dict(),
                     "final_census": final},
            "stars": stars, "lanes": [list(map(int, e)) for e in g0.lanes], "frames": frames}


def snapshot(g: Galaxy, decider, first: bool) -> dict:
    x = g.senses()                       # what each colony senses now (inputs for its next decision)
    rates = decider.rates(x) if hasattr(decider, "rates") else None
    cols = []
    for k, (cid, s, p, a) in enumerate(zip(g.c_id, g.c_sys, g.c_pop, g.c_act)):
        brain_action = "idle" if first else ACTIONS[int(a)]
        c = {"id": int(cid), "star": int(s), "pop": round(float(min(1.0, p / g.capacity[s])), 3),
             "population": round(float(p), 1), "action": UI_ACTION[brain_action], "brain_action": brain_action,
             "inputs": {n: round(float(v), 3) for n, v in zip(SENSES, x[k])}}
        if rates is not None:
            c["rates"] = {NEURONS[o]: round(float(r), 1) for o, r in zip(OUTPUTS, rates[k])}
        cols.append(c)
    moves = [] if first else getattr(g, "last_moves", [])
    raids = [] if first else getattr(g, "last_raided_systems", [])
    return {"colonies": cols,
            "ships": [{"from": a, "to": b, "progress": 0.5, "kind": kind} for a, b, kind, _ in moves],
            "events": [{"type": "raid", "star": s} for s in raids]
                      + [{"type": "raider", "star": int(r)} for r in g.raiders]}


def check(doc: dict) -> list[str]:
    """Problems that would stop the UI playing the file (same rules as the UI's viz/frames.py)."""
    p = []
    variants = doc.get("meta", {}).get("variants") or []
    if not variants:
        p.append("meta.variants is empty")
    star_ids = {s.get("id") for s in doc.get("stars", [])}
    if not star_ids:
        p.append("no stars")
    if not doc.get("frames"):
        p.append("no frames")
    for f in doc.get("frames", []):
        for v, gal in f.get("galaxies", {}).items():
            if v not in variants:
                p.append(f"frame {f.get('t')}: galaxy {v!r} is not in meta.variants")
            for c in gal.get("colonies", []):
                if c.get("star") not in star_ids:
                    p.append(f"frame {f.get('t')} {v}: colony on unknown star {c.get('star')}")
                if c.get("action") not in UI_ACTIONS:
                    p.append(f"frame {f.get('t')} {v}: action {c.get('action')!r} not in {sorted(UI_ACTIONS)}")
            for sh in gal.get("ships", []):
                if sh.get("from") not in star_ids or sh.get("to") not in star_ids:
                    p.append(f"frame {f.get('t')} {v}: ship between unknown stars")
    return p


def save(doc: dict, path: Path = FRAMES) -> Path:
    problems = check(doc)
    if problems:
        raise ValueError("frames.json would not play:\n  " + "\n  ".join(problems[:10]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--check", type=Path, default=None, help="only check an existing frames file")
    args = ap.parse_args()
    if args.check:
        problems = check(json.loads(args.check.read_text(encoding="utf8")))
        print("\n".join(problems) if problems else f"{args.check} is fine")
        raise SystemExit(1 if problems else 0)
    doc = record(args.variants, args.seed, args.ticks)
    path = save(doc)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB, {len(doc['frames'])} frames, "
          f"{len(doc['meta']['variants'])} galaxies)")
    for n, c in doc["meta"]["final_census"].items():
        print(f"  {n:24s} population {c['population']:7.0f}  colonies {c['colonies']:3d}")


if __name__ == "__main__":
    main()
