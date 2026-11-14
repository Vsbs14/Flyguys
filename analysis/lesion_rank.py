"""Phase 4.3: rank lesioned cell types by what they do to the galaxy.

Each lesion's coarse response table (tables/lesion_screen.npz) is used as the
colonies' brain on the same galaxies. Cell types are ranked by the change in
final colonies and final living civilizations against the intact brain on the
same coarse grid ('none').

  python -m analysis.lesion_rank --seeds 10 --processes 3
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from brain.connectome import ROOT
from brain.lesion import CANDIDATES_FILE, lesion_tables
from galaxy.run import frozen_decoder
from galaxy.sim import GalaxySim
from galaxy.world import make_galaxy

_tables = None


def _job(args):
    name, seed = args
    global _tables
    if _tables is None:
        _tables = lesion_tables()
    out = GalaxySim(make_galaxy(seed), _tables[name], frozen_decoder(), seed=seed).run()
    return name, seed, out["final_colonies"], out["final_civilizations"], out["flights"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    names = list(lesion_tables())
    jobs = [(n, s) for n in names for s in range(args.seeds)]
    with Pool(args.processes) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    col = {n: np.zeros(args.seeds) for n in names}
    civ = {n: np.zeros(args.seeds) for n in names}
    fl = {n: np.zeros(args.seeds) for n in names}
    for n, s, c, v, f in res:
        col[n][s], civ[n][s], fl[n][s] = c, v, f
    info = {c["cell_type"]: c for c in json.loads(CANDIDATES_FILE.read_text(encoding="utf8"))["candidates"]}
    se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    rows = []
    for n in names:
        if n == "none":
            continue
        dc, dv = col[n] - col["none"], civ[n] - civ["none"]       # paired by seed: same galaxy
        i = info.get(n, {})
        rows.append({"cell_type": n, "super_class": i.get("super_class"), "transmitter": i.get("transmitter"),
                     "neurons": i.get("neurons"), "ids": i.get("ids", []),
                     "d_final_colonies": float(dc.mean()), "d_final_colonies_se": se(dc),
                     "d_final_civilizations": float(dv.mean()), "d_final_civilizations_se": se(dv),
                     "d_flights": float((fl[n] - fl["none"]).mean())})
    rows.sort(key=lambda r: -abs(r["d_final_colonies"]))
    doc = {"seeds": args.seeds, "lesions_screened": len(rows),
           "intact_coarse": {"final_colonies": float(col["none"].mean()), "final_civilizations": float(civ["none"].mean()),
                             "flights": float(fl["none"].mean())},
           "note": "Partial screen: only the most active candidate cell types, on the reduced network. "
                   "Re-run if galaxy parameters change.",
           "ranking": rows}
    (ROOT / "results" / "lesion_ranking.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    lines = ["# Lesion ranking (partial screen)", "",
             f"{len(rows)} of the most active candidate cell types were silenced one at a time. Each lesioned brain "
             f"ran {args.seeds} galaxies; changes are against the intact brain on the same coarse 4 x 4 x 4 grid and "
             f"the same galaxies (mean ± standard error). Intact: {doc['intact_coarse']['final_colonies']:.1f} final "
             f"colonies, {doc['intact_coarse']['final_civilizations']:.1f} living civilizations.", "",
             "| Cell type | Class | Transmitter | Neurons | Change in final colonies | Change in living civilizations | Change in flights |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['cell_type']} | {r['super_class']} | {r['transmitter']} | {r['neurons']} | "
                     f"{r['d_final_colonies']:+.1f} ± {r['d_final_colonies_se']:.1f} | "
                     f"{r['d_final_civilizations']:+.1f} ± {r['d_final_civilizations_se']:.1f} | {r['d_flights']:+.1f} |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "lesion_ranking.md").write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
