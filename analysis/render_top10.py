"""Phase 4.4: render the top 10 lesion-screen cell types in 3D. Run in Colab, not here.

Needs a FlyWire token (https://global.daf-apis.com/auth/api/v1/create_token). It is
asked for with getpass and never written to disk or to the repo.

  !pip install -q fafbseg==3.2.3 navis==1.12.0 plotly crackle-codec
  %run analysis/render_top10.py
"""
import json
from getpass import getpass
from pathlib import Path

import navis
from fafbseg import flywire

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
MAX_NEURONS_PER_TYPE = 6          # keep the download small

flywire.set_chunkedgraph_secret(getpass("FlyWire token: "), overwrite=True)
flywire.set_default_dataset("public")

ranking = json.loads((ROOT / "results" / "lesion_ranking.json").read_text(encoding="utf8"))["ranking"][:10]
out = ROOT / "analysis" / "figures"
out.mkdir(parents=True, exist_ok=True)
for rank, row in enumerate(ranking, 1):
    ids = row["ids"][:MAX_NEURONS_PER_TYPE]
    skeletons = flywire.get_skeletons(ids)
    fig = navis.plot3d(skeletons, backend="plotly", inline=False,
                       title=f"#{rank} {row['cell_type']}: {row['d_final_colonies']:+.1f} final colonies when silenced")
    fig.write_html(out / f"lesion_top{rank:02d}_{row['cell_type']}.html")
    print("wrote", rank, row["cell_type"])
