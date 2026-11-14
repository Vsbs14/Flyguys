# viz: the galaxy UI

`index.html` is the whole site: a clickable galaxy where each world is one part of the project, plus the galaxy demo player. It works in two modes.

| Mode | How to open it | What it shows |
|---|---|---|
| Hosted | The claude.ai link | Real sanity results bundled into the page, mock galaxy data, no live brain |
| Local backend | `python -m viz.server`, then http://localhost:8000 | Newest intact sanity report, `galaxy/frames.json` if it exists, the live fly brain |

A "Live backend" badge appears at the bottom left when the page is talking to the server.

## Run it

From the repo root:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py      # once, if external/ is empty
python -m viz.server              # add --device cuda on a GPU machine
```

The first "Run the fly brain" click loads the connectome (a few seconds). After that each decision runs 4 brains for 200 ms, about 3 seconds on a laptop CPU.

## Endpoints (`server.py`)

| Endpoint | Used for |
|---|---|
| `GET /api/health` | The page checks this first; if it fails the page stays in hosted mode |
| `GET /api/sanity` | Newest intact sanity report, shown on The Fly Brain |
| `GET /api/experiment` | Escape response map and paired experiment statistics |
| `POST /api/preview` | Instant cached output and escape estimate for intact or Giant Fibre silenced |
| `GET /api/frames` | `galaxy/frames.json`, played on Live Demo instead of the mock |
| `POST /api/decide` | `{"sugar":0-1, "bitter":0-1, "looming":0-1, "brains":4, "silence":["escape"]}` returns MN9, DNp09, MDN and Giant Fiber rates from the real brain |
| `POST /api/push` | `{"type":"start","data":{"meta":...,"stars":...}}` then `{"type":"frame","data":{...}}` to stream a galaxy run live |
| `WS /ws` | Where the page receives loom and frame messages |

Inputs are scaled 0–1 to 0–200 Hz, the same range as the sanity checks.

## For the galaxy sim (`frames.py`)

```python
from viz.frames import FrameWriter
fw = FrameWriter(variants=["intact", "fearless", "hungry"], seed=42, stars=stars, every=2)
fw.add(epoch, "intact", colonies=[{"id": 7, "star": 0, "pop": 0.6, "action": "settle",
                                    "inputs": {...}, "rates": {"MN9": 48, "DNp09": 12, "MDN": 3, "GF": 0}}],
       ships=[...], events=[...])
fw.save()                      # checks the file, then writes galaxy/frames.json
```

`action` is one of settle, expand, retreat, flee. `inputs` and `rates` are optional, but clicking a colony in the demo shows them. Check a file by hand with `python -m viz.frames galaxy/frames.json`.

Keep `frames.json` under about 15 MB if it will be bundled into the hosted site.
