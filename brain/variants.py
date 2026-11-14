"""Brain variants: the same fly brain with parts switched off or rewired.

A variant is a connectome plus a mask of silenced neurons. Silenced neurons are
held at rest and never spike. Input and output neurons are never removed by the
size or shuffle variants, so every variant can still sense and act.

| Variant        | What changes                                                       |
|----------------|--------------------------------------------------------------------|
| intact         | nothing (the reference)                                            |
| no_gf          | Giant Fibre silenced: no panic                                     |
| no_mdn         | MDN silenced: no caution                                           |
| no_gf_no_mdn   | both silenced                                                      |
| no_bitter      | bitter taste neurons silenced: always hungry, cannot sense hostility |
| size_50/20/05  | only 50 / 20 / 5 % of interneurons kept (nested random subsets)    |
| shuffled       | interneuron wiring shuffled, every neuron keeps its connection counts |
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome
from .neurons import NeuronGroups

VARIANT_SEED = 20261114     # fixes which interneurons the size variants keep, and the shuffle

DESCRIPTIONS = {
    "intact": "Intact brain (reference)",
    "no_gf": "Giant Fibre silenced (no panic)",
    "no_mdn": "MDN silenced (no caution)",
    "no_gf_no_mdn": "Giant Fibre and MDN silenced",
    "no_bitter": "Bitter neurons silenced (always hungry)",
    "size_50": "50% of interneurons kept",
    "size_20": "20% of interneurons kept",
    "size_05": "5% of interneurons kept",
    "shuffled": "Interneuron wiring shuffled (control)",
}
VARIANTS = list(DESCRIPTIONS)

# Sanity checks each lesion is supposed to break. None = no prediction, just record what happens.
EXPECTED_FAIL: dict[str, tuple[str, ...] | None] = {
    "intact": (),
    "no_gf": ("looming_escapes",),
    "no_mdn": (),
    "no_gf_no_mdn": ("looming_escapes",),
    "no_bitter": ("bitter_suppresses",),
    "size_50": None,
    "size_20": None,
    "size_05": None,
    "shuffled": None,
}


@dataclass
class Variant:
    name: str
    conn: Connectome          # the wiring this variant runs on
    alive: np.ndarray         # (N,) bool, False = silenced
    description: str

    @property
    def n_silenced(self) -> int:
        return int((~self.alive).sum())


def interneurons(conn: Connectome, groups: NeuronGroups) -> np.ndarray:
    """Every neuron that is not one of our input or output neurons."""
    is_io = np.zeros(conn.n, dtype=bool)
    is_io[groups.all_inputs()] = True
    is_io[groups.all_outputs()] = True
    return np.flatnonzero(~is_io)


def size_mask(conn: Connectome, groups: NeuronGroups, keep_fraction: float, seed: int = VARIANT_SEED) -> np.ndarray:
    """Keep a random `keep_fraction` of interneurons. Smaller brains are subsets of larger ones."""
    inter = interneurons(conn, groups)
    order = np.random.default_rng(seed).permutation(inter)
    alive = np.ones(conn.n, dtype=bool)
    alive[order[round(keep_fraction * len(order)):]] = False
    return alive


def shuffled_connectome(conn: Connectome, groups: NeuronGroups, seed: int = VARIANT_SEED) -> Connectome:
    """Rewire interneuron-to-interneuron connections at random.

    The postsynaptic targets of those connections are permuted among themselves.
    Every neuron keeps its number of outgoing and incoming connections and every
    connection keeps its weight, but who talks to whom is scrambled. Connections
    that touch an input or output neuron are left alone.
    """
    is_inter = np.zeros(conn.n, dtype=bool)
    is_inter[interneurons(conn, groups)] = True
    pre, post, signed = conn.edges()
    post = post.copy()
    movable = np.flatnonzero(is_inter[pre] & is_inter[post])
    post[movable] = post[np.random.default_rng(seed).permutation(movable)]
    return Connectome.from_edges(conn.ids, pre, post, signed)


def make_variant(name: str, conn: Connectome, groups: NeuronGroups) -> Variant:
    alive = np.ones(conn.n, dtype=bool)
    wiring = conn
    if name == "intact":
        pass
    elif name == "no_gf":
        alive[groups["GF"]] = False
    elif name == "no_mdn":
        alive[groups["MDN"]] = False
    elif name == "no_gf_no_mdn":
        alive[groups["GF"]] = False
        alive[groups["MDN"]] = False
    elif name == "no_bitter":
        alive[groups["bitter"]] = False
    elif name.startswith("size_"):
        alive = size_mask(conn, groups, int(name.split("_")[1]) / 100.0)
    elif name == "shuffled":
        wiring = shuffled_connectome(conn, groups)
    else:
        raise KeyError(f"unknown variant {name!r}; choose from {VARIANTS}")
    return Variant(name=name, conn=wiring, alive=alive, description=DESCRIPTIONS[name])
