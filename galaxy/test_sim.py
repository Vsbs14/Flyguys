"""Checks for the galaxy simulation (idea from the phase-3-5 branch's galaxy tests).

  python -m galaxy.test_sim
"""
from __future__ import annotations

import numpy as np

from .run import frozen_decoder
from .policy import load_policy
from .sim import GalaxySim, SimConfig
from .world import GalaxyConfig, make_galaxy

SMALL = GalaxyConfig(n_stars=400, n_epochs=80, n_civs=4)


def _run(policy_name: str, seed: int) -> GalaxySim:
    sim = GalaxySim(make_galaxy(seed, SMALL), load_policy(policy_name), frozen_decoder(), SimConfig(), seed=seed)
    for t in range(SMALL.n_epochs):
        sim.step(t)
    return sim


def test_same_seed_same_run() -> None:
    a, b = _run("intact", 3).summary(), _run("intact", 3).summary()
    assert a == b, "two runs with one seed differ"


def test_same_galaxy_for_every_brain() -> None:
    g1, g2 = make_galaxy(5, SMALL), make_galaxy(5, SMALL)
    assert np.array_equal(g1.xy, g2.xy) and np.array_equal(g1.threats.epoch, g2.threats.epoch)
    assert np.array_equal(g1.origins, g2.origins)


def test_populations_stay_in_bounds() -> None:
    for name in ("intact", "no_bitter", "hand_rule"):
        sim = _run(name, 1)
        occ = sim.civ >= 0
        assert np.all(sim.pop[occ] >= 0) and np.all(sim.pop <= sim.g.capacity + 1e-9), name
        assert np.all(sim.pop[~occ] == 0), name
        assert np.all(sim.sterile_until[occ] <= SMALL.n_epochs - 1), f"{name}: colony alive on a sterile star"


def test_fearless_never_flees() -> None:
    assert _run("no_gf", 2).counts["flights"] == 0


def main() -> None:
    for test in (test_same_seed_same_run, test_same_galaxy_for_every_brain, test_populations_stay_in_bounds,
                 test_fearless_never_flees):
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
