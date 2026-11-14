"""Resolve every neuron group the project uses to FlyWire v783 root IDs.

Writes data/neuron_groups.json, which is committed so nothing downstream needs
the annotation table or a FlyWire token.

Sources
-------
Every group is selected from the public FlyWire annotation table, using the
same filters as the hackathon starter notebook (fly_brain_starter.ipynb):

  sugar   cell_type == "LB3c"            sugar-sensing taste neurons, both sides
  bitter  cell_sub_class == "bitter"     bitter-sensing taste neurons, both sides
  looming cell_type in {LC4, LPLC2}      looming-detecting visual neurons

Two senses were added for Phase 2 so that every output can be driven. They
were picked by stimulating every sensory and visual projection cell type on
its own (200 Hz, 200 ms) and keeping the type that drives one output most
cleanly:

  obstacle cell_type == "LC16"   -> MDN (retreat), ~35 Hz, no other output.
                                    Matches Wu et al. 2016 (eLife): LC16
                                    activation makes flies walk backwards.
  target   cell_type == "LC31b"  -> DNp09 (advance), ~50 Hz, no other output.
                                    A connectome pick, not a published result.
                                    The three looming/taste senses never drive
                                    DNp09 above ~3 Hz.

MN9 is the pair of CB0701 motor neurons. The starter notebook reads only one of
them (720575940660219265, the ID from Shiu et al.); we average the pair because
the taste input is bilateral. DNp09 is not in the starter notebook.

  python scripts/build_neuron_groups.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / "external" / "Drosophila_brain_model"
ANNOT = ROOT / "external" / "flywire_annotations" / "neuron_annotations.tsv"
OUT = ROOT / "data" / "neuron_groups.json"

# group name -> (annotation column, value)
SELECT = {
    "sugar": ("cell_type", "LB3c"),
    "bitter": ("cell_sub_class", "bitter"),
    "LC4": ("cell_type", "LC4"),        # looming detector (visual projection neuron)
    "LPLC2": ("cell_type", "LPLC2"),    # looming detector (visual projection neuron)
    "LC16": ("cell_type", "LC16"),      # visual projection neuron, drives backward walking (obstacle)
    "LC31b": ("cell_type", "LC31b"),    # visual projection neuron, drives DNp09 (target)
    "MN9": ("cell_type", "CB0701"),     # proboscis motor neuron 9 (feed)
    "GF": ("cell_type", "DNp01"),       # Giant Fibre (escape)
    "MDN": ("cell_type", "MDN"),        # moonwalker descending neuron (walk backward)
    "DNp09": ("cell_type", "DNp09"),    # walk forward
}
SHIU_MN9 = 720575940660219265


def main() -> None:
    comp = pd.read_csv(SHIU / "Completeness_783.csv", index_col=0)
    ann = pd.read_csv(ANNOT, sep="\t", low_memory=False)
    in_model = set(comp.index)

    groups: dict[str, dict] = {}
    for group, (column, value) in SELECT.items():
        rows = ann[ann[column] == value].sort_values(["side", "root_id"])
        rows = rows[rows.root_id.isin(in_model)]
        if rows.empty:
            raise RuntimeError(f"no neurons found for {group} ({column} == {value})")
        groups[group] = {
            "ids": [int(i) for i in rows.root_id],
            "sides": rows.side.tolist(),
            "source": f"FlyWire annotations {column} == '{value}'",
        }
    assert SHIU_MN9 in groups["MN9"]["ids"], "Shiu's MN9 should be one of the CB0701 pair"

    doc = {
        "materialization": 783,
        "roles": {
            "inputs": {"sugar": ["sugar"], "bitter": ["bitter"], "looming": ["LC4", "LPLC2"],
                       "obstacle": ["LC16"], "target": ["LC31b"]},
            "outputs": {"feed": "MN9", "escape": "GF", "retreat": "MDN", "advance": "DNp09"},
        },
        "groups": groups,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    for name, g in groups.items():
        print(f"{name:7s} n={len(g['ids']):4d}  {g['source']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
