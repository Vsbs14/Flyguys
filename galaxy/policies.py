"""Phase 6 baselines: decision makers that are not a fly brain.

They have the same interface as `ResponseTable` (`sample(x, rng)` and
`probs(x)`), so the galaxy treats them exactly like a brain variant.

  random     every action equally likely, whatever the senses say
  threshold  a hand-written rule an engineer might write, with no brain:
             flee a raider, avoid toxic food, back off when crowded,
             eat when food is plentiful, move on when a neighbour is better
"""
from __future__ import annotations

import numpy as np

from .brains import ACTIONS

IDLE, FEED, ESCAPE, RETREAT, ADVANCE = range(len(ACTIONS))


class RandomPolicy:
    name = "random"

    def probs(self, x) -> np.ndarray:
        return np.full((len(np.atleast_2d(x)), len(ACTIONS)), 1.0 / len(ACTIONS))

    def sample(self, x, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, len(ACTIONS), len(np.atleast_2d(x)))


class ThresholdPolicy:
    """First matching rule wins. Thresholds sit where the intact brain's own dose-response turns on."""

    name = "threshold"

    def __init__(self, looming=0.1, bitter=0.5, obstacle=0.5, sugar=0.4, target=0.3):
        self.t = dict(looming=looming, bitter=bitter, obstacle=obstacle, sugar=sugar, target=target)

    def choose(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        sugar, bitter, looming, obstacle, target = x.T
        t = self.t
        return np.select(
            [looming >= t["looming"], obstacle >= t["obstacle"], (sugar >= t["sugar"]) & (bitter < t["bitter"]),
             target >= t["target"]],
            [ESCAPE, RETREAT, FEED, ADVANCE], default=IDLE)

    def probs(self, x) -> np.ndarray:
        return np.eye(len(ACTIONS))[self.choose(x)]

    def sample(self, x, rng: np.random.Generator) -> np.ndarray:
        return self.choose(x)


BASELINES = {"random": RandomPolicy, "threshold": ThresholdPolicy}
