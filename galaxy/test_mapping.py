"""Checks for the Phase 2 mapping. No brain or table needed.

  python -m galaxy.test_mapping
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .mapping import (ADVANCE, ESCAPE, FEED, MAX_HZ, RETREAT, ColonyMemory, Decoder, SenseMap, habitability,
                      hostility, urgency)


def test_senses() -> None:
    rng = np.random.default_rng(0)
    sugar = habitability(rng.random(5000), rng.random(5000), rng.random(5000))
    bitter = hostility(rng.random(5000), rng.random(5000), rng.random(5000))
    loom = urgency(rng.uniform(0.5, 20, 800))
    m = SenseMap.fit(sugar, bitter, loom)

    for hz in (m.sugar_hz(sugar), m.bitter_hz(bitter), m.looming_hz(loom)):
        assert hz.min() >= 0 and hz.max() <= MAX_HZ
    # the mean world sits mid-scale; 2 SD out saturates
    assert abs(float(m.sugar_hz(m.sugar_mean)) - 100) < 1e-9
    assert float(m.sugar_hz(m.sugar_mean + 2 * m.sugar_sd)) == MAX_HZ
    assert float(m.sugar_hz(m.sugar_mean + 5 * m.sugar_sd)) == MAX_HZ       # clipped, not extrapolated
    assert float(m.bitter_hz(m.bitter_mean - 2 * m.bitter_sd)) == 0.0
    # looming: nothing coming is 0 Hz, the 2 SD threat level saturates, sooner means more
    assert float(m.looming_hz(0.0)) == 0.0
    assert float(m.looming_hz(m.looming_ref)) == MAX_HZ
    assert float(m.looming_hz(urgency(2.0))) > float(m.looming_hz(urgency(10.0)))
    assert urgency(np.inf) == 0 and urgency(0.0) == 0
    # memory comes back in as extra looming input
    assert float(m.looming_hz(0.0, memory_hz=1.5)) == 1.5
    # gains scale the signal and stay inside 0-200 Hz
    g = m.with_gains(sugar=1.5, looming=0.5)
    assert float(g.sugar_hz(m.sugar_mean)) == 150.0 and float(g.sugar_hz(m.sugar_mean + 2 * m.sugar_sd)) == MAX_HZ
    assert float(g.looming_hz(m.looming_ref)) == 100.0


def test_memory() -> None:
    mem = ColonyMemory(n=3, bump_hz=0.75, half_life_epochs=20, cap_hz=3.0)
    mem.record_loss([0])
    mem.record_loss([1, 1, 1, 1, 1, 1])
    assert np.allclose(mem.baseline_hz, [0.75, 3.0, 0.0])                   # bumped, capped, untouched
    for _ in range(20):
        mem.step()
    assert np.allclose(mem.baseline_hz, [0.375, 1.5, 0.0])                  # one half-life later
    mem.resize(5)
    assert len(mem.baseline_hz) == 5 and mem.baseline_hz[4] == 0


def test_decoder() -> None:
    mean = np.zeros((2, 2, 2, 4))
    mean[..., FEED] = 70.0
    mean[..., RETREAT] = 12.0
    mean[..., ADVANCE] = 1.0
    dec = Decoder.fit(SimpleNamespace(mean=mean, name="intact"))
    assert (dec.feed_ref_hz, dec.retreat_ref_hz, dec.advance_ref_hz) == (70.0, 12.0, 10.0)   # DNp09 floor applies

    rates = np.zeros((4, 4))
    rates[0, FEED] = 35.0                                   # half-hungry, calm
    rates[1, FEED], rates[1, RETREAT] = 70.0, 6.0           # hungry but half retreating
    rates[2, FEED], rates[2, ESCAPE] = 70.0, 180.0          # Giant Fibre spiked
    rates[3, FEED] = 500.0                                  # above full scale
    d = dec.decode(rates, gf=np.array([False, False, True, False]))
    assert np.allclose(d["growth"], [0.5, 1.0, 0.0, 1.0])
    assert np.allclose(d["retreat"], [0.0, 0.5, 0.0, 0.0])
    assert np.allclose(d["launch"], [0.25, 0.25, 0.0, 0.5])
    assert d["flee"].tolist() == [False, False, True, False]
    assert Decoder.action_names(d).tolist() == ["expand", "expand", "flee", "expand"]

    # frozen: it round-trips through a file and never looks at another table
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decoder.json"
        dec.save(path)
        assert Decoder.load(path) == dec


def main() -> None:
    for test in (test_senses, test_memory, test_decoder):
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
