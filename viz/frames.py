"""The frame format the galaxy UI plays, plus helpers to write and check it.

The galaxy sim writes one file, galaxy/frames.json:

{
  "meta":   {"variants": ["intact", "fearless", "hungry"], "seed": 42, "epochs": 360},
  "stars":  [{"id": 0, "x": 0.41, "y": -0.22}, ...],          # any units, centred on the galaxy
  "frames": [
    {"t": 0,
     "galaxies": {
       "intact": {
         "colonies": [{"id": 7, "star": 0, "pop": 0.6, "action": "settle",        # settle|expand|retreat|flee
                       "inputs": {"sugar": 0.8, "bitter": 0.2, "looming": 0.1},   # 0-1, optional
                       "rates":  {"MN9": 48, "DNp09": 12, "MDN": 3, "GF": 0}}],  # Hz, GF = 1 if it spiked; optional
         "ships":    [{"from": 0, "to": 5, "progress": 0.3}],
         "events":   [{"type": "supernova", "star": 12},
                      {"type": "encounter", "a": 7, "b": 9, "outcome": "mutual_flight"}]}}},
    ...
  ]
}

Usage in the galaxy sim:

  from viz.frames import FrameWriter
  fw = FrameWriter(variants=["intact", "fearless", "hungry"], seed=42, stars=star_list)
  for epoch in range(E):
      for v in variants:
          fw.add(epoch, v, colonies=[...], ships=[...], events=[...])
  fw.save()                      # galaxy/frames.json, checked first

Keep the file under about 15 MB if it will go on the hosted site: log every Nth epoch with every=N.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {"settle", "expand", "retreat", "flee"}


class FrameWriter:
    def __init__(self, variants: list[str], seed: int, stars: list[dict], every: int = 1):
        self.meta = {"variants": list(variants), "seed": int(seed)}
        self.stars = [{"id": int(s["id"]), "x": round(float(s["x"]), 4), "y": round(float(s["y"]), 4)} for s in stars]
        self.every = every
        self.frames: dict[int, dict] = {}

    def add(self, t: int, variant: str, colonies: list[dict], ships: list[dict] = (), events: list[dict] = ()) -> None:
        if t % self.every:
            return
        f = self.frames.setdefault(t, {"t": t, "galaxies": {}})
        f["galaxies"][variant] = {"colonies": [_round(c) for c in colonies],
                                  "ships": [_round(s) for s in ships],
                                  "events": list(events)}

    def doc(self) -> dict:
        frames = [self.frames[t] for t in sorted(self.frames)]
        return {"meta": {**self.meta, "epochs": len(frames)}, "stars": self.stars, "frames": frames}

    def save(self, path: str | Path = ROOT / "galaxy" / "frames.json") -> Path:
        doc = self.doc()
        problems = check(doc)
        if problems:
            raise ValueError("frames.json would not play:\n  " + "\n  ".join(problems[:10]))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf8")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(doc['frames'])} frames)")
        return path


def _round(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 3)
        elif isinstance(v, dict):
            out[k] = _round(v)
        elif hasattr(v, "item"):          # numpy / torch scalars
            out[k] = round(float(v.item()), 3)
        else:
            out[k] = v
    return out


def check(doc: dict) -> list[str]:
    """Return a list of problems; empty means the UI can play it."""
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
        for v, g in f.get("galaxies", {}).items():
            if v not in variants:
                p.append(f"frame {f.get('t')}: galaxy {v!r} is not in meta.variants")
            for c in g.get("colonies", []):
                if c.get("star") not in star_ids:
                    p.append(f"frame {f.get('t')} {v}: colony on unknown star {c.get('star')}")
                if c.get("action") not in ACTIONS:
                    p.append(f"frame {f.get('t')} {v}: action {c.get('action')!r} should be one of {sorted(ACTIONS)}")
    return p


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "galaxy" / "frames.json")
    problems = check(json.loads(path.read_text(encoding="utf8")))
    print("\n".join(problems) if problems else f"{path} is fine")
