"""Phase 2 walk-through: example worlds -> fly senses -> fly response -> colony decision.

Fits the decoder on the intact response table, freezes it in tables/decoder.json,
and writes results/mapping_examples.md.

The galaxy here is a stand-in (world properties drawn uniformly, supernova
warnings up to 20 epochs ahead). Phase 5 replaces it with the real star map;
the mapping code is the same.

  python -m galaxy.mapping_demo
"""
from __future__ import annotations

import numpy as np

from brain.connectome import ROOT
from brain.response import TABLE_DIR, ResponseTable

from .mapping import ColonyMemory, Decoder, SenseMap, habitability, hostility, urgency

DECODER_FILE = TABLE_DIR / "decoder.json"

# name, (water, energy, resources), (radiation, cold, distance), epochs until a threat (None = none), exposure, losses
WORLDS = [
    ("Garden world",                        (0.9, 0.8, 0.8), (0.1, 0.1, 0.2), None, 0.0, 0),
    ("Average world",                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), None, 0.0, 0),
    ("Rich but harsh world",                (0.9, 0.8, 0.8), (0.8, 0.7, 0.6), None, 0.0, 0),
    ("Barren rock",                         (0.2, 0.3, 0.2), (0.8, 0.7, 0.6), None, 0.0, 0),
    ("Garden world, supernova in 10 epochs", (0.9, 0.8, 0.8), (0.1, 0.1, 0.2), 10.0, 1.0, 0),
    ("Garden world at the blast's far edge", (0.9, 0.8, 0.8), (0.1, 0.1, 0.2), 10.0, 0.05, 0),
    ("Garden world after one lost fleet",   (0.9, 0.8, 0.8), (0.1, 0.1, 0.2), None, 0.0, 1),
    ("Garden world after three lost fleets", (0.9, 0.8, 0.8), (0.1, 0.1, 0.2), None, 0.0, 3),
]


def stand_in_sense_map(seed: int = 0) -> SenseMap:
    rng = np.random.default_rng(seed)
    n = 5000
    sugar = habitability(rng.random(n), rng.random(n), rng.random(n))
    bitter = hostility(rng.random(n), rng.random(n), rng.random(n))
    looming = urgency(rng.integers(1, 21, 2000), rng.random(2000))
    return SenseMap.fit(sugar, bitter, looming)


def frozen_decoder() -> Decoder:
    """The decoder is fitted on the intact table once and reused for every variant."""
    if DECODER_FILE.exists():
        return Decoder.load(DECODER_FILE)
    decoder = Decoder.fit(ResponseTable.load("intact"))
    decoder.save(DECODER_FILE)
    return decoder


def main() -> None:
    senses = stand_in_sense_map()
    decoder = frozen_decoder()
    tables = {name: ResponseTable.load(name) for name in ("intact", "no_gf") if (TABLE_DIR / f"{name}.npz").exists()}

    lines = [
        "# Phase 2: from a world to a decision",
        "",
        "Each example world is turned into three input rates, the cached brain answers, and the frozen decoder turns "
        "the answer into amounts between 0 and 1. Rates are the table's expected values; in the simulation each "
        "decision is drawn with its trial-to-trial noise.",
        "",
        f"Decoder (fitted on the intact brain, then frozen): MN9 full scale {decoder.feed_ref_hz:.1f} Hz, "
        f"MDN full scale {decoder.retreat_ref_hz:.1f} Hz, DNp09 full scale {decoder.advance_ref_hz:.1f} Hz, "
        f"share of growth that feeds launches {decoder.growth_to_launch}.",
    ]
    for tname, table in tables.items():
        lines += ["", f"## {'Intact brain' if tname == 'intact' else 'Giant Fibre silenced (same decoder)'}", "",
                  "| World | Sugar (Hz) | Bitter (Hz) | Looming (Hz) | MN9 (Hz) | MDN (Hz) | DNp09 (Hz) | P(flee) | Growth | Launch | Retreat |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, good, bad, t_event, exposure, losses in WORLDS:
            memory = ColonyMemory(n=1)
            for _ in range(losses):
                memory.record_loss([0])
            s = float(senses.sugar_hz(habitability(*good)))
            b = float(senses.bitter_hz(hostility(*bad)))
            threat = urgency(t_event, exposure) if t_event else 0.0
            l = float(senses.looming_hz(threat, memory.baseline_hz[0]))
            r = table.lookup(s, b, l)
            d = decoder.decode(r["mean"], gf=False)
            m = r["mean"]
            lines.append(f"| {name} | {s:.0f} | {b:.0f} | {l:.1f} | {m[0]:.1f} | {m[2]:.1f} | {m[3]:.1f} | "
                         f"{float(r['p_gf']):.2f} | {float(d['growth']):.2f} | {float(d['launch']):.2f} | "
                         f"{float(d['retreat']):.2f} |")
    lines += ["", "Growth, launch and retreat are what the colony does if the Giant Fibre stays silent. "
              "If it spikes, with the probability shown, the colony flees and does nothing else."]
    md = "\n".join(lines) + "\n"
    out = ROOT / "results" / "mapping_examples.md"
    out.write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
