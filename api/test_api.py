"""End-to-end check of the Phase 7 backend: every endpoint, in-process (no server needed).

  python -m api.test_api            # includes one live brain decision on the CPU (about 10 s)
  python -m api.test_api --no-live  # skip the live brain (no connectome needed)
"""
from __future__ import annotations

import argparse

from fastapi.testclient import TestClient

from .server import app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    c = TestClient(app)
    ok = lambda r: (r.status_code == 200) or (_ for _ in ()).throw(AssertionError(f"{r.request.url}: {r.status_code} {r.text[:300]}"))

    h = c.get("/api/health").json()
    assert h["ok"] and "intact" in h["tables"], h
    for path in ("/api/sanity", "/api/frames", "/api/experiment", "/api/lesions", "/api/calibration", "/api/variants", "/"):
        ok(c.get(path))
    v = c.get("/api/variants").json()
    assert "intact" in v["brains"] and {"random", "threshold"} <= set(v["baselines"])

    frames = c.get("/api/frames").json()
    from galaxy.playback import check
    assert not check(frames), check(frames)[:3]
    assert set(frames["meta"]["variants"]) >= {"intact", "threshold", "random"}

    # preview: the decision table answers like the brain did in Phase 2
    pv = lambda **kw: c.post("/api/preview", json=kw)
    r = pv(sugar=1).json()
    assert r["action_probabilities"]["feed"] > 0.9 and r["rates"]["MN9"] > 30, r
    assert pv(sugar=1, looming=1).json()["escape_probability"] > 0.9
    assert pv(sugar=1, bitter=1).json()["action_probabilities"]["idle"] > 0.9
    assert pv(obstacle=1).json()["action_probabilities"]["retreat"] > 0.9
    assert pv(target=1).json()["action_probabilities"]["advance"] > 0.9
    assert pv(looming=1, variant="threshold").json()["source"] == "baseline rule"
    for name in v["brains"]:
        ok(pv(sugar=0.5, looming=0.2, variant=name))
    assert pv(variant="nope").status_code == 404
    assert pv(sugar=2).status_code == 422

    # a fresh galaxy run on request
    g = c.post("/api/galaxy", json={"variants": ["intact", "random"], "seed": 3, "ticks": 20})
    ok(g)
    assert len(g.json()["frames"]) == 21

    # websocket fan-out
    with c.websocket_connect("/ws") as ws:
        assert c.post("/api/loom", json={"value": 0.7}).json()["sent_to"] == 1
        assert ws.receive_json() == {"type": "loom", "value": 0.7}
        c.post("/api/push", json={"type": "frame", "data": {"t": 1}})
        assert ws.receive_json() == {"type": "frame", "t": 1}

    if not args.no_live:
        d = c.post("/api/decide", json={"sugar": 1, "brains": 4})
        ok(d)
        d = d.json()
        assert d["rates"]["MN9"]["mean_hz"] > 30 and d["action_share"]["feed"] >= 0.75, d
        d = c.post("/api/decide", json={"looming": 1, "brains": 4, "silence": ["escape"]}).json()
        assert d["rates"]["GF"]["mean_hz"] == 0 and d["rates"]["MDN"]["mean_hz"] > 0, d
        assert c.post("/api/decide", json={"silence": ["wings"]}).status_code == 400
        print(f"live brain: {d['wall_seconds']} s per decision batch on {d['device']}")
    print("PASS: API checks")


if __name__ == "__main__":
    main()
