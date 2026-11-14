"""Checks for the galaxy code. Runs in a few seconds on the CPU with synthetic tables (no brain, no GPU).

  python -m galaxy.test_galaxy
"""
from __future__ import annotations

import numpy as np

from .brains import ACTIONS, SENSES, ResponseTable
from .world import ESCAPE, FEED, IDLE, Galaxy, GalaxyConfig


def table_always(action: int, levels=(0.0, 0.5, 1.0)) -> ResponseTable:
    counts = np.zeros((len(levels) ** len(SENSES), len(ACTIONS)))
    counts[:, action] = 4
    return ResponseTable(np.array(levels), counts)


def main() -> None:
    # interpolation: exact at grid points, linear in between, probabilities sum to 1
    lv = np.array([0.0, 1.0])
    counts = np.zeros((2 ** len(SENSES), len(ACTIONS)))
    counts[:, IDLE] = 4
    sugar_on = [i for i in range(len(counts)) if np.unravel_index(i, (2,) * len(SENSES))[0] == 1]
    counts[sugar_on] = 0
    counts[sugar_on, FEED] = 4
    t = ResponseTable(lv, counts)
    assert np.allclose(t.probs([[1, 0, 0, 0, 0]]), np.eye(len(ACTIONS))[FEED])
    assert np.allclose(t.probs([[0.25, 0, 0, 0, 0]])[0, [IDLE, FEED]], [0.75, 0.25])
    assert np.allclose(t.probs(np.random.default_rng(0).random((50, len(SENSES)))).sum(1), 1)
    draws = t.sample(np.tile([0.25, 0, 0, 0, 0], (20000, 1)), np.random.default_rng(1))
    assert abs((draws == FEED).mean() - 0.25) < 0.02

    # same seed -> same run
    brains = {"a": table_always(FEED), "b": table_always(IDLE)}
    runs = []
    for _ in range(2):
        g = Galaxy(GalaxyConfig(), brains, seed=3)
        for _ in range(30):
            g.step()
        runs.append(g.census())
    assert runs[0] == runs[1]

    # the map is connected
    g = Galaxy(GalaxyConfig(), brains, seed=0)
    seen, todo = {0}, [0]
    while todo:
        for j in g.nbrs[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    assert len(seen) == g.cfg.n_systems

    # never-feeding colonies only shrink; feeding ones on clean systems can grow
    cfg = GalaxyConfig(n_raiders=0, toxic_share=0.0, clean_max=0.0)
    g = Galaxy(cfg, {"idle": table_always(IDLE), "feed": table_always(FEED)}, seed=1)
    start = g.census()
    for _ in range(20):
        g.step()
    assert g.census()["idle"]["population"] < start["idle"]["population"]
    assert g.census()["feed"]["population"] > start["feed"]["population"]

    # colonies of one variant in one system merge, population is conserved by the merge
    g = Galaxy(GalaxyConfig(n_raiders=0), {"idle": table_always(IDLE)}, seed=2)
    g.c_sys[:] = 0
    before = g.c_pop.sum()
    g._prune_and_merge()
    assert len(g.c_sys) == 1 and np.isclose(g.c_pop.sum(), before)

    # escaping colonies leave a raided system; staying colonies take the raid
    cfg = GalaxyConfig(n_raiders=1, raider_move_prob=0.0, upkeep=0.0)
    for action, should_lose in ((ESCAPE, False), (IDLE, True)):
        g = Galaxy(cfg, {"x": table_always(action)}, seed=4)
        g.c_sys[:] = g.raiders[0]
        g.c_pop[:] = 20.0
        start = 20.0 * len(g.c_pop)
        g.step()
        lost = g.census()["x"]["population"] < start * (1 - cfg.move_cost) - 1e-6
        assert lost == should_lose, (ACTIONS[action], g.census())
    print("PASS: galaxy checks")


if __name__ == "__main__":
    main()
