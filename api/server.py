"""Phase 7 backend: every result, the galaxy playback, and the live fly brain over HTTP.

  python -m api.server                  # http://localhost:8000, live brain on the CPU
  python -m api.server --device cuda

Endpoint names and response shapes follow the galaxy UI's own server
(viz/server.py on the VSTemp/UI branches), so that page can point here. Our
additions are marked "new".

  GET  /api/health            is the backend up, what is available
  GET  /api/sanity            newest results/sanity_intact_*.json
  GET  /api/frames            galaxy/frames.json (python -m galaxy.playback)
  GET  /api/experiment        results/experiments.json (python -m galaxy.experiments)
  GET  /api/lesions           new: results/lesion_screen.json
  GET  /api/calibration       new: results/decision_calibration.json
  GET  /api/variants          new: every decision maker the galaxy can use
  POST /api/preview           instant decision from the response tables, any variant (no torch)
  POST /api/decide            run real fly brains on the senses (loads the connectome on first call)
  POST /api/galaxy            new: record a fresh matched galaxy run and return its frames
  POST /api/loom              push a looming level to every open page
  POST /api/push              stream galaxy frames to every open page
  WS   /ws                    live messages {"type": "loom" | "start" | "frame", ...}

All sense values are intensities in [0, 1]; 1 = 200 Hz on every neuron of that sense.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRAMES = ROOT / "galaxy" / "frames.json"
VIZ_PAGE = ROOT / "viz" / "index.html"
NEURON = {"feed": "MN9", "escape": "GF", "retreat": "MDN", "advance": "DNp09"}

app = FastAPI(title="Flyguys backend")
DEVICE = "cpu"
_engine = None
_engine_lock = threading.Lock()


def _json_file(path: Path, hint: str):
    if not path.exists():
        raise HTTPException(404, f"{path.relative_to(ROOT)} not found; run {hint}")
    return FileResponse(path, media_type="application/json")


# ---------------------------------------------------------------- request models
class Senses(BaseModel):
    sugar: float = Field(0.0, ge=0, le=1)
    bitter: float = Field(0.0, ge=0, le=1)
    looming: float = Field(0.0, ge=0, le=1)
    obstacle: float = Field(0.0, ge=0, le=1)
    target: float = Field(0.0, ge=0, le=1)

    def vector(self) -> list[float]:
        return [self.sugar, self.bitter, self.looming, self.obstacle, self.target]


class DecideRequest(Senses):
    brains: int = Field(4, ge=1, le=64)
    variant: str = "intact"
    silence: list[str] = Field(default_factory=list, description="outputs to silence: feed, escape, retreat, advance")
    seed: int = 0


class PreviewRequest(Senses):
    variant: str = "intact"
    silence: list[str] = Field(default_factory=list)


class GalaxyRequest(BaseModel):
    variants: list[str] | None = None
    seed: int = Field(7, ge=0)
    ticks: int = Field(100, ge=1, le=400)


# ---------------------------------------------------------------- live brain
def engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from brain.connectome import Connectome
            from brain.interface import DecisionRule, FlyBrain
            from brain.neurons import NeuronGroups
            from brain.sim import BrainSim

            conn = Connectome.load()
            _engine = {"conn": conn, "brain": FlyBrain(BrainSim(conn, device=DEVICE), NeuronGroups(conn),
                                                       DecisionRule.load())}
        return _engine


def _variants_doc() -> dict:
    path = ROOT / "data" / "variants.json"
    return json.loads(path.read_text(encoding="utf8"))["variants"] if path.exists() else {}


def _decide(req: DecideRequest) -> dict:
    import numpy as np
    import torch

    from brain.interface import ACTIONS, OUTPUTS

    e = engine()
    brain, conn = e["brain"], e["conn"]
    alive = np.ones(conn.n, dtype=bool)
    if req.variant != "intact":
        v = _variants_doc().get(req.variant)
        if v is None or "shuffle_seed" in v:
            raise ValueError(f"live decisions support 'intact' and the lesion variants {sorted(_variants_doc())}")
        alive[conn.index_of(v["silenced_root_ids"])] = False
    for o in req.silence:
        if o not in OUTPUTS:
            raise ValueError(f"cannot silence {o!r}; choose from {list(OUTPUTS)}")
        alive[brain.groups.output(o)] = False
    t0 = time.time()
    d = brain.decide([req.vector()] * req.brains, alive=torch.from_numpy(alive), seed=req.seed)
    r = d.rates_hz.numpy()
    return {
        "inputs": dict(zip(("sugar", "bitter", "looming", "obstacle", "target"), req.vector())),
        "inputs_hz": {k: v * 200.0 for k, v in zip(("sugar", "bitter", "looming", "obstacle", "target"), req.vector())},
        "variant": req.variant, "silenced": req.silence, "brains": req.brains,
        "rates": {NEURON[o]: {"mean_hz": round(float(r[:, j].mean()), 2), "sd_hz": round(float(r[:, j].std()), 2)}
                  for j, o in enumerate(OUTPUTS)},
        "gf_spiked": int((r[:, OUTPUTS.index("escape")] > 0).sum()),
        "actions": d.actions,
        "action_share": {a: d.actions.count(a) / len(d.actions) for a in ACTIONS},
        "decision_ms": brain.decision_ms, "wall_seconds": round(time.time() - t0, 2), "device": DEVICE,
    }


@app.post("/api/decide")
async def decide(req: DecideRequest):
    try:
        return await asyncio.to_thread(_decide, req)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, f"connectome data missing ({e}); run python scripts/fetch_data.py")


# ---------------------------------------------------------------- tables, no torch
@app.post("/api/preview")
def preview(req: PreviewRequest):
    """Decision probabilities and output rates from the response tables. Instant."""
    from galaxy.brains import ACTIONS, NEURONS, OUTPUTS, available
    from galaxy.experiments import deciders

    variant = req.variant
    if req.silence:                      # UI compatibility: silence ["escape"] means the Giant Fibre lesion
        lesion = {"escape": "no-dnp01", "feed": "no-cb0701", "retreat": "no-mdn", "advance": "no-dnp09"}
        if len(req.silence) != 1 or req.silence[0] not in lesion:
            raise HTTPException(400, "preview can silence one output at a time")
        variant = lesion[req.silence[0]]
    ds = deciders()
    if variant not in ds:
        raise HTTPException(404, f"no decision maker {variant!r}; available: {sorted(ds)}")
    d = ds[variant]
    p = d.probs([req.vector()])[0]
    rates = d.rates([req.vector()]) if hasattr(d, "rates") else None
    out = {"variant": variant, "action_probabilities": dict(zip(ACTIONS, map(float, p))),
           "escape_probability": float(p[ACTIONS.index("escape")]),
           "source": "response table" if variant in available() else "baseline rule"}
    if rates is not None:
        out["rates"] = {NEURONS[o]: float(r) for o, r in zip(OUTPUTS, rates[0])}
    return out


@app.get("/api/variants")
def variants():
    from galaxy.brains import available
    from galaxy.policies import BASELINES

    tabs = available()
    lesions = _variants_doc()
    return {"brains": {n: {"description": t["description"], "silenced_cell_types": t["silenced_cell_types"],
                           "changes": lesions.get(n, {}).get("changes", []),
                           "action_share_over_grid": t["action_share_over_grid"],
                           "live": n == "intact" or (n in lesions and "shuffle_seed" not in lesions[n])}
                       for n, t in tabs.items()},
            "baselines": {n: (cls.__doc__ or "").strip().splitlines()[0] for n, cls in BASELINES.items()}}


@app.post("/api/galaxy")
def galaxy(req: GalaxyRequest):
    from galaxy.playback import check, record

    try:
        doc = record(req.variants, req.seed, req.ticks)
    except KeyError as e:
        raise HTTPException(404, f"unknown decision maker {e}")
    problems = check(doc)
    if problems:
        raise HTTPException(500, "; ".join(problems[:5]))
    return JSONResponse(doc)


# ---------------------------------------------------------------- results
@app.get("/api/health")
def health():
    from galaxy.brains import available

    return {"ok": True, "device": DEVICE, "brain_loaded": _engine is not None, "has_frames": FRAMES.exists(),
            "has_sanity": any(RESULTS.glob("sanity_intact_*.json")),
            "has_experiment": (RESULTS / "experiments.json").exists(),
            "has_lesions": (RESULTS / "lesion_screen.json").exists(),
            "tables": sorted(available()), "senses": ["sugar", "bitter", "looming", "obstacle", "target"]}


@app.get("/api/sanity")
def sanity():
    files = sorted(RESULTS.glob("sanity_intact_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise HTTPException(404, "no sanity results yet; run python -m brain.sanity")
    doc = json.loads(files[-1].read_text(encoding="utf8"))
    doc["file"] = files[-1].name
    return doc


@app.get("/api/frames")
def frames():
    return _json_file(FRAMES, "python -m galaxy.playback")


@app.get("/api/experiment")
def experiment():
    return _json_file(RESULTS / "experiments.json", "python -m galaxy.experiments")


@app.get("/api/lesions")
def lesions():
    return _json_file(RESULTS / "lesion_screen.json", "python -m brain.lesion_screen")


@app.get("/api/calibration")
def calibration():
    return _json_file(RESULTS / "decision_calibration.json", "python -m brain.calibrate")


# ---------------------------------------------------------------- live messages
clients: set[WebSocket] = set()


async def broadcast(msg: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.discard(ws)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


class Loom(BaseModel):
    value: float = Field(..., ge=0, le=1)


@app.post("/api/loom")
async def loom(msg: Loom):
    await broadcast({"type": "loom", "value": msg.value})
    return {"sent_to": len(clients)}


class FrameMsg(BaseModel):
    type: str
    data: dict


@app.post("/api/push")
async def push(msg: FrameMsg):
    await broadcast({"type": msg.type, **msg.data})
    return {"sent_to": len(clients)}


@app.get("/")
def index():
    if VIZ_PAGE.exists():
        return FileResponse(VIZ_PAGE)
    return {"service": "Flyguys backend", "docs": "/docs",
            "endpoints": sorted({r.path for r in app.routes if r.path.startswith("/api") or r.path == "/ws"})}


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    DEVICE = args.device
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
