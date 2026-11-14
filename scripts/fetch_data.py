"""Download the third-party data this project needs into external/ (gitignored).

No token is required for any of these files.

  python scripts/fetch_data.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
ANNOT = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"

FILES = {
    # Shiu et al. connectome model, FlyWire v783. Signs are already applied.
    "external/Drosophila_brain_model/Connectivity_783.parquet": SHIU + "Connectivity_783.parquet",
    "external/Drosophila_brain_model/Completeness_783.csv": SHIU + "Completeness_783.csv",
    # Public FlyWire cell-type annotations (Schlegel et al. 2024), v783 root IDs.
    # Only needed to rebuild data/neuron_groups.json.
    "external/flywire_annotations/neuron_annotations.tsv": ANNOT + "Supplemental_file1_neuron_annotations.tsv",
}


def main() -> int:
    for rel, url in FILES.items():
        dest = ROOT / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have   {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch  {rel}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
