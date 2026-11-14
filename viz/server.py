"""Backend for the galaxy UI: serves the page, real results, galaxy frames, and the live fly brain.

Run from the repo root:

  pip install fastapi uvicorn
  python -m viz.server                 # http://localhost:8000
  python -m viz.server --device cuda   # if this machine has a GPU

Endpoints the page uses (all optional; the page falls back to bundled data when one is missing):

  GET  /api/health          is the backend up, which device
  GET  /api/sanity          newest results/sanity_*.json
  GET  /api/frames          galaxy/frames.json, the recorded galaxy run (see viz/frames.py)
  POST /api/decide          run real fly brains on sugar / bitter / looming levels
  POST /api/loom            push a looming level (e.g. from the UNO Q) to every open page
  WS   /ws                  live messages to the page: {"type": "loom"|"frame"|"start", ...}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "viz"
RESULTS = ROOT / "results"
FRAMES = ROOT / "galaxy" / "frames.json"

MAX_HZ = 200.0          # inputs are scaled 0-1 -> 0-200 Hz, the handbook's range
DECISION_MS = 200.0     # one decision window from rest, as in brain.sanity
OUTPUT_NAMES = {"feed": "MN9", "escape": "GF", "retreat": "MDN", "advance": "DNp09"}

app = FastAPI(title="Flyguys galaxy backend")
DEVICE = "cpu"

# ---------------------------------------------------------------- fly brain (loaded on first use)
_engine = None
_engine_lock = threading.Lock()


def engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from brain import BrainSim, Connectome, NeuronGroups
            conn = Connectome.load()
            _engine = (BrainSim(conn, device=DEVICE), NeuronGroups(conn))
        return _engine


class DecideRequest(BaseModel):
    sugar: float = Field(0.0, ge=0, le=1)
    bitter: float = Field(0.0, ge=0, le=1)
    looming: float = Field(0.0, ge=0, le=1)
    brains: int = Field(4, ge=1, le=64, description="replicate brains, differing only in random input spikes")
    silence: list[str] = Field(default_factory=list, description="outputs to silence: feed, escape, retreat, advance")
    seed: int = 0


def _decide(req: DecideRequest) -> dict:
    import torch

    sim, groups = engine()
    alive = None
    if req.silence:
        alive = torch.ones(sim.n, dtype=torch.bool)
        for name in req.silence:
            if name not in OUTPUT_NAMES:
                raise ValueError(f"unknown output {name!r}")
            alive[torch.as_tensor(groups.output(name))] = False

    rates = {"sugar": req.sugar, "bitter": req.bitter, "looming": req.looming}
    inputs = [(torch.as_tensor(groups.sense(s)), v * MAX_HZ) for s, v in rates.items() if v > 0]
    outs = list(OUTPUT_NAMES)
    mon = [groups.output(o) for o in outs]
    sizes = [len(m) for m in mon]
    monitor = torch.as_tensor([i for m in mon for i in m])

    t0 = time.perf_counter()
    state = sim.new_state(req.brains, alive=alive, seed=req.seed)
    res = sim.run(state, DECISION_MS, inputs=inputs, monitor=monitor)
    rate = res.rates_hz.cpu()                                   # (monitored neurons, brains)

    out, start = {}, 0
    for name, n in zip(outs, sizes):
        block = rate[start:start + n]
        out[OUTPUT_NAMES[name]] = {"mean_hz": round(float(block.mean()), 2), "sd_hz": round(float(block.mean(0).std()), 2) if req.brains > 1 else 0.0}
        if name == "escape":
            gf_brains = int((res.counts[start:start + n].sum(0) > 0).sum())
        start += n
    return {
        "inputs_hz": {k: v * MAX_HZ for k, v in rates.items()},
        "silenced": req.silence,
        "brains": req.brains,
        "rates": out,
        "gf_spiked": gf_brains,
        "decision_ms": DECISION_MS,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "device": DEVICE,
    }


_decide_lock = asyncio.Lock()


@app.post("/api/decide")
async def decide(req: DecideRequest):
    async with _decide_lock:            # one brain batch at a time; the engine is not thread safe
        try:
            return await asyncio.to_thread(_decide, req)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))


class CaseRequest(BaseModel):
    summary: str = Field(max_length=2000)


@app.post("/api/flycourt/analyze")
def flycourt_analyze(req: CaseRequest):
    from viz.case_analysis import analyze_case

    summary = req.summary.strip()
    if not summary:
        raise HTTPException(422, "Enter a short case summary.")
    try:
        return analyze_case(summary)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(
            502, "Case analysis unavailable. Check that Ollama is running and the configured model is installed."
        ) from None


@app.post("/api/preview")
def preview(req: DecideRequest):
    """Estimate output rates from the intact or escape-silenced cached trials."""
    import numpy as np
    from itertools import product
    from galaxy.responses import TABLE

    if req.silence not in ([], ["escape"]):
        raise HTTPException(400, "the preview supports intact or escape-silenced brains")
    if not TABLE.exists():
        raise HTTPException(404, "run python -m galaxy.responses to build the cache")
    with np.load(TABLE) as cache:
        table = cache["fearless" if req.silence else "intact"]
    coordinates = np.asarray([req.sugar, req.bitter, req.looming]) * (len(table) - 1)
    lower = coordinates.astype(int)
    fraction = coordinates - lower
    estimate = np.zeros(5)
    for corner in product((0, 1), repeat=3):
        weight = np.prod(np.where(corner, fraction, 1 - fraction))
        if weight:
            estimate += weight * table[tuple(lower + corner)].mean(axis=0)
    return {"rates": {name: float(rate) for name, rate in zip(OUTPUT_NAMES.values(), estimate)},
            "escape_probability": float(estimate[4])}


# ---------------------------------------------------------------- results and frames
@app.get("/api/health")
def health():
    return {"ok": True, "device": DEVICE, "brain_loaded": _engine is not None,
            "has_frames": FRAMES.exists(), "has_sanity": any(RESULTS.glob("sanity_intact_*.json"))}


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
    if not FRAMES.exists():
        raise HTTPException(404, "galaxy/frames.json not written yet")
    return FileResponse(FRAMES, media_type="application/json")


@app.get("/api/experiment")
def experiment():
    path = RESULTS / "galaxy_experiment.json"
    if not path.exists():
        raise HTTPException(404, "run python -m galaxy.simulate")
    return FileResponse(path, media_type="application/json")


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
            await ws.receive_text()      # the page does not send anything yet
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


class Loom(BaseModel):
    value: float = Field(..., ge=0, le=1)


@app.post("/api/loom")
async def loom(msg: Loom):
    """Called by the UNO Q bridge when a hand moves toward the sensor."""
    await broadcast({"type": "loom", "value": msg.value})
    return {"sent_to": len(clients)}


class FrameMsg(BaseModel):
    type: str          # "start" (meta + stars) or "frame" (one epoch)
    data: dict


@app.post("/api/push")
async def push(msg: FrameMsg):
    """Lets a running galaxy sim stream epochs to the page live instead of writing a file."""
    await broadcast({"type": msg.type, **msg.data})
    return {"sent_to": len(clients)}


# ---------------------------------------------------------------- the page itself (mounted last)
app.mount("/", StaticFiles(directory=VIZ, html=True), name="viz")


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    DEVICE = args.device
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
