"""Decision policies for colonies: fly brains and the baselines they are compared with.

A policy answers  sample(sugar_hz, bitter_hz, looming_hz, rng) -> {"rates": (n, 4) Hz, "gf": (n,) bool}
with rates ordered MN9, Giant Fibre, MDN, DNp09. A `ResponseTable` already has
that method, so every fly variant is a policy as it stands.

Baselines (the "does the fly matter?" ladder):
  HandRule    a few if-statements using thresholds read off the intact fly's table
  NeuralNet   a small network fitted to the intact fly's table
The shuffled-wiring control is just another response table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from brain.response import ResponseTable

from .mapping import ADVANCE, ESCAPE, FEED, MAX_HZ, RETREAT


def load_policy(name: str):
    """A fly variant by table name, or a baseline fitted to the intact table."""
    if name == "hand_rule":
        return HandRule.fit(ResponseTable.load("intact"))
    if name == "neural_net":
        return NeuralNet.fit(ResponseTable.load("intact"))
    return ResponseTable.load(name)


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """The x where y first rises through `level` (linear between grid points); the last x if it never does."""
    above = np.flatnonzero(y >= level)
    if not len(above):
        return float(x[-1])
    i = above[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))


@dataclass(frozen=True)
class HandRule:
    """The rule a person would write after looking at the fly's response table.

    flee     if looming is at least `loom_flee`
    feed     rises linearly with sugar above `sugar_on`, falls linearly to nothing as bitter
             approaches `bitter_off`, and stops when looming reaches `loom_stops_feeding`
    retreat  rises linearly with looming above `loom_retreat_on`
    advance  never
    No noise: the same input always gives the same answer.
    """
    feed_max_hz: float
    retreat_max_hz: float
    escape_max_hz: float
    sugar_on: float
    bitter_off: float
    loom_flee: float
    loom_stops_feeding: float
    loom_retreat_on: float

    @classmethod
    def fit(cls, table: ResponseTable) -> "HandRule":
        sugar, bitter, looming = table.axes
        feed = table.mean[..., FEED]
        retreat = table.mean[..., RETREAT]
        feed_max, retreat_max = float(feed.max()), float(retreat.max())
        return cls(
            feed_max_hz=feed_max,
            retreat_max_hz=retreat_max,
            escape_max_hz=float(table.mean[..., ESCAPE].max()),
            # thresholds are where the fly's own curve passes 10% (onset) or 50% (switch)
            sugar_on=_crossing(sugar, feed[:, 0, 0], 0.1 * feed_max),
            bitter_off=_crossing(bitter, -feed[-1, :, 0], -0.1 * feed_max),
            loom_flee=_crossing(looming, table.p_gf[0, 0, :], 0.5),
            loom_stops_feeding=_crossing(looming, -feed[-1, 0, :], -0.5 * feed[-1, 0, 0]),
            loom_retreat_on=_crossing(looming, retreat[0, 0, :], 0.1 * retreat_max),
        )

    def sample(self, sugar, bitter, looming, rng=None) -> dict[str, np.ndarray]:
        sugar, bitter, looming = (np.asarray(v, dtype=float) for v in (sugar, bitter, looming))
        rates = np.zeros(sugar.shape + (4,))
        appetite = np.clip((sugar - self.sugar_on) / (MAX_HZ - self.sugar_on), 0, 1)
        disgust = np.clip(1 - bitter / self.bitter_off, 0, 1)
        rates[..., FEED] = self.feed_max_hz * appetite * disgust * (looming < self.loom_stops_feeding)
        rates[..., RETREAT] = self.retreat_max_hz * np.clip(
            (looming - self.loom_retreat_on) / max(MAX_HZ - self.loom_retreat_on, 1e-9), 0, 1)
        gf = looming >= self.loom_flee
        rates[..., ESCAPE] = self.escape_max_hz * gf
        rates[..., ADVANCE] = 0.0
        return {"rates": rates, "gf": gf}

    def describe(self) -> dict:
        return asdict(self)


class NeuralNet:
    """A small multilayer perceptron fitted to the intact fly's table (means, spreads and flee probability).

    It is a smooth stand-in for the table: 3 inputs -> 32 -> 32 -> 9 outputs. If the galaxy looks
    the same with this as with the fly, then the fly's response surface is what matters, and any
    function that reproduces that surface would do.
    """

    def __init__(self, weights: list[np.ndarray], scale: np.ndarray, fit_error: dict):
        self.weights, self.scale, self.fit_error = weights, scale, fit_error

    @staticmethod
    def _features(sugar, bitter, looming) -> np.ndarray:
        # looming is compressed because the fly's response changes fastest at low rates
        return np.stack([np.asarray(sugar) / MAX_HZ, np.asarray(bitter) / MAX_HZ,
                         np.log1p(np.asarray(looming)) / np.log1p(MAX_HZ)], axis=-1)

    @classmethod
    def fit(cls, table: ResponseTable, hidden: int = 32, steps: int = 4000, seed: int = 0) -> "NeuralNet":
        import torch

        torch.manual_seed(seed)
        s, b, l = np.meshgrid(*table.axes, indexing="ij")
        x = torch.tensor(cls._features(s, b, l).reshape(-1, 3), dtype=torch.float32)
        scale = np.concatenate([np.maximum(table.mean.reshape(-1, 4).max(axis=0), 1.0)] * 2 + [[1.0]])
        target = np.concatenate([table.mean.reshape(-1, 4), table.sd.reshape(-1, 4), table.p_gf.reshape(-1, 1)], axis=1)
        y = torch.tensor(target / scale, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                  torch.nn.Tanh(), torch.nn.Linear(hidden, 9))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.mean((net(x) - y) ** 2)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            pred = net(x).numpy() * scale
        err = np.abs(pred - target)
        fit_error = {"mean_abs_error_hz": {n: float(err[:, i].mean()) for i, n in
                                           enumerate(["MN9", "Giant Fibre", "MDN", "DNp09"])},
                     "flee_probability_mean_abs_error": float(err[:, 8].mean())}
        weights = [p.detach().numpy().astype(float) for p in net.parameters()]
        return cls(weights, scale, fit_error)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = self.weights
        h = np.tanh(x @ w1.T + b1)
        h = np.tanh(h @ w2.T + b2)
        return (h @ w3.T + b3) * self.scale

    def sample(self, sugar, bitter, looming, rng: np.random.Generator) -> dict[str, np.ndarray]:
        out = self._forward(self._features(sugar, bitter, looming))
        mean, sd = np.maximum(out[..., :4], 0), np.maximum(out[..., 4:8], 0)
        p = np.clip(out[..., 8], 0, 1)
        return {"rates": np.maximum(rng.normal(mean, sd), 0), "gf": rng.random(p.shape) < p}
