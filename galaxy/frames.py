"""Writer for galaxy/frames.json, the file the galaxy viewer (viz/) plays.

The format is defined by the viewer's `viz/frames.py`. This is a stand-alone
writer of the same format, used when that module is not present:

{
  "meta":   {"variants": [...], "seed": 42, "epochs": 200},
  "stars":  [{"id": 0, "x": 0.41, "y": -0.22}, ...],
  "frames": [{"t": 0, "galaxies": {"intact": {"colonies": [...], "ships": [...], "events": [...]}}}, ...]
}
"""
from __future__ import annotations

import json
from pathlib import Path


def _round(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 3)
        elif isinstance(v, dict):
            out[k] = _round(v)
        elif hasattr(v, "item"):
            out[k] = round(float(v.item()), 3)
        else:
            out[k] = v
    return out


class FrameWriter:
    def __init__(self, variants: list[str], seed: int, stars: list[dict], every: int = 1):
        self.meta = {"variants": list(variants), "seed": int(seed)}
        self.stars = [{"id": int(s["id"]), "x": round(float(s["x"]), 4), "y": round(float(s["y"]), 4)} for s in stars]
        self.every = every
        self.frames: dict[int, dict] = {}

    def add(self, t: int, variant: str, colonies: list[dict], ships: list[dict] = (), events: list[dict] = ()) -> None:
        if t % self.every:
            return
        frame = self.frames.setdefault(t, {"t": t, "galaxies": {}})
        frame["galaxies"][variant] = {"colonies": [_round(c) for c in colonies],
                                      "ships": [_round(s) for s in ships], "events": list(events)}

    def save(self, path: str | Path) -> Path:
        frames = [self.frames[t] for t in sorted(self.frames)]
        doc = {"meta": {**self.meta, "epochs": len(frames)}, "stars": self.stars, "frames": frames}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf8")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
        return path
