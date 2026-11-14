"""Named neuron groups (inputs and outputs), resolved to model indices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .connectome import ROOT, Connectome

GROUPS_FILE = ROOT / "data" / "neuron_groups.json"

# The three senses of the handbook rule ("tastes good / tastes bad / looms"). The response tables,
# variants, lesion screen and galaxy use exactly these. neuron_groups.json may define more senses
# (obstacle and target, for the 5-sense decision interface in brain/interface.py).
CORE_SENSES = ("sugar", "bitter", "looming")


class NeuronGroups:
    def __init__(self, conn: Connectome, path: str | Path = GROUPS_FILE):
        doc = json.loads(Path(path).read_text(encoding="utf8"))
        self.roles = doc["roles"]
        self.ids = {name: g["ids"] for name, g in doc["groups"].items()}
        self.idx = {name: conn.index_of(ids) for name, ids in self.ids.items()}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.idx[name]

    def sense(self, name: str) -> np.ndarray:
        """All input neurons for a sense: 'sugar', 'bitter' or 'looming'."""
        return np.concatenate([self.idx[g] for g in self.roles["inputs"][name]])

    def output(self, name: str) -> np.ndarray:
        """Output neurons for an action: 'feed', 'escape', 'retreat' or 'advance'."""
        return self.idx[self.roles["outputs"][name]]

    @property
    def senses(self) -> list[str]:
        return list(self.roles["inputs"])

    @property
    def outputs(self) -> list[str]:
        return list(self.roles["outputs"])

    def all_inputs(self) -> np.ndarray:
        return np.concatenate([self.sense(s) for s in self.senses])

    def core_inputs(self) -> np.ndarray:
        """Input neurons of the three core senses only."""
        return np.concatenate([self.sense(s) for s in CORE_SENSES])

    def all_outputs(self) -> np.ndarray:
        return np.concatenate([self.output(o) for o in self.outputs])
