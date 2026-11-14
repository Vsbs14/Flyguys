"""Phase 2: how a colony's situation becomes fly senses, and fly neurons become decisions.

The rule is the fly's own: the world tastes good, tastes bad, or something
looms, and the brain answers eat, back off, or flee.

Galaxy -> senses (`SenseMap`)
    sugar    how habitable the world is       water, energy, resources
    bitter   how hostile it is                radiation, cold, travel distance
    looming  a threat approaching             sum of exposure / (time until the event)
  Each signal is clipped at 2 standard deviations of its spread across the
  galaxy and scaled to 0-200 Hz. Smell and wind are never used.

Neurons -> decisions (`Decoder`)
    any Giant Fibre spike   abandon the system and flee; overrides everything
    otherwise, together:
      MN9 rate     growth rate (settle)
      DNp09 rate   share of spare capacity launched as colony ships (expand)
      MDN rate     pull back and consolidate (retreat)
  The scaling from rates to actions is measured once on the intact brain and
  then frozen. Every variant is decoded with the same numbers, otherwise the
  decoder would hide the differences we want to measure.

Memory (`ColonyMemory`)
    The brain restarts from rest for every decision, so the past has to come
    back in as input: losses raise a colony's looming baseline for a while.

A note on DNp09. In the intact brain it stays almost silent under all three
senses (about 1 Hz at most, and only under looming strong enough to make the
Giant Fibre fire first). Read literally, no colony would ever launch a ship. So
launching is also fed by growth: a colony that eats well builds a surplus and
sends part of it out. `growth_to_launch` sets that share; DNp09 is still
decoded and added on top, and reported as what it is.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

MAX_HZ = 200.0
CLIP_SD = 2.0

# Order of the output rates everywhere in the project: MN9, Giant Fibre, MDN, DNp09.
FEED, ESCAPE, RETREAT, ADVANCE = 0, 1, 2, 3


# --------------------------------------------------------------------------- senses

@dataclass(frozen=True)
class SenseWeights:
    """How world properties (each 0-1) combine into the two taste scores."""
    water: float = 0.4
    energy: float = 0.3
    resources: float = 0.3
    radiation: float = 0.4
    cold: float = 0.3
    distance: float = 0.3


def habitability(water, energy, resources, w: SenseWeights = SenseWeights()):
    """Sugar score: how good the world tastes."""
    return w.water * np.asarray(water) + w.energy * np.asarray(energy) + w.resources * np.asarray(resources)


def hostility(radiation, cold, distance, w: SenseWeights = SenseWeights()):
    """Bitter score: how bad the world tastes."""
    return w.radiation * np.asarray(radiation) + w.cold * np.asarray(cold) + w.distance * np.asarray(distance)


def urgency(time_until_event, exposure=1.0):
    """Looming score for one threat: exposure / (time until the event). 0 when nothing is coming."""
    t = np.asarray(time_until_event, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.asarray(exposure, dtype=float) / t
    return np.where(np.isfinite(t) & (t > 0), u, 0.0)


@dataclass(frozen=True)
class SenseMap:
    """Scores -> Poisson input rates in Hz.

    Tastes use the galaxy-wide spread: the mean world sits at 100 Hz and a world
    2 SD above (below) the mean saturates at 200 Hz (0 Hz). Looming is zero when
    nothing threatens and saturates at `looming_ref`, the level 2 SD above the
    mean of the threats colonies actually face. The gains exist for the
    sensitivity sweep; they are 1 in the main runs.
    """
    sugar_mean: float
    sugar_sd: float
    bitter_mean: float
    bitter_sd: float
    looming_ref: float
    sugar_gain: float = 1.0
    bitter_gain: float = 1.0
    looming_gain: float = 1.0

    @classmethod
    def fit(cls, sugar_scores, bitter_scores, looming_scores) -> "SenseMap":
        """Measure the spread of each signal across the galaxy."""
        loom = np.asarray(looming_scores, dtype=float)
        loom = loom[loom > 0]
        if loom.size == 0:
            raise ValueError("no threats to scale looming with")
        return cls(float(np.mean(sugar_scores)), float(np.std(sugar_scores)),
                   float(np.mean(bitter_scores)), float(np.std(bitter_scores)),
                   float(loom.mean() + CLIP_SD * loom.std()))

    def with_gains(self, sugar: float = 1.0, bitter: float = 1.0, looming: float = 1.0) -> "SenseMap":
        return replace(self, sugar_gain=sugar, bitter_gain=bitter, looming_gain=looming)

    @staticmethod
    def _taste(score, mean, sd, gain):
        z = np.clip((np.asarray(score, dtype=float) - mean) / sd, -CLIP_SD, CLIP_SD)
        return np.clip(gain * (z + CLIP_SD) / (2 * CLIP_SD) * MAX_HZ, 0.0, MAX_HZ)

    def sugar_hz(self, score):
        return self._taste(score, self.sugar_mean, self.sugar_sd, self.sugar_gain)

    def bitter_hz(self, score):
        return self._taste(score, self.bitter_mean, self.bitter_sd, self.bitter_gain)

    def looming_hz(self, score, memory_hz=0.0):
        """Threat plus the colony's remembered baseline, in Hz."""
        hz = self.looming_gain * np.clip(np.asarray(score, dtype=float) / self.looming_ref, 0.0, 1.0) * MAX_HZ
        return np.clip(hz + np.asarray(memory_hz, dtype=float), 0.0, MAX_HZ)


# --------------------------------------------------------------------------- memory

@dataclass
class ColonyMemory:
    """A looming baseline per colony that jumps on a loss and fades over time.

    The fly's Giant Fibre goes from never firing to almost always firing between
    about 1 and 3 Hz of looming input. One loss (`bump_hz`) leaves a colony just
    under that edge; repeated losses push it over, so a battered colony becomes
    jumpy for a few half-lives.
    """
    n: int
    bump_hz: float = 0.75
    half_life_epochs: float = 20.0
    cap_hz: float = 3.0
    baseline_hz: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.baseline_hz is None:
            self.baseline_hz = np.zeros(self.n)

    def step(self) -> None:
        """One epoch passes."""
        self.baseline_hz *= 0.5 ** (1.0 / self.half_life_epochs)

    def record_loss(self, colonies, weight=1.0) -> None:
        """A colony lost a fleet, a neighbour, or had to flee."""
        np.add.at(self.baseline_hz, np.asarray(colonies, dtype=int), self.bump_hz * np.asarray(weight, dtype=float))
        np.minimum(self.baseline_hz, self.cap_hz, out=self.baseline_hz)

    def resize(self, n: int) -> None:
        if n > len(self.baseline_hz):
            self.baseline_hz = np.concatenate([self.baseline_hz, np.zeros(n - len(self.baseline_hz))])
        self.n = len(self.baseline_hz)


# --------------------------------------------------------------------------- decisions

ADVANCE_FLOOR_HZ = 10.0     # DNp09 full scale is never set below this, so one stray spike is not "launch everything"


@dataclass(frozen=True)
class Decoder:
    """Firing rates -> amounts in 0-1. Fitted once on the intact brain, then frozen."""
    feed_ref_hz: float
    retreat_ref_hz: float
    advance_ref_hz: float
    growth_to_launch: float = 0.5
    fitted_on: str = "intact"

    @classmethod
    def fit(cls, intact_table, growth_to_launch: float = 0.5) -> "Decoder":
        """Full scale of each output = its largest mean rate anywhere on the intact response table."""
        m = intact_table.mean
        return cls(
            feed_ref_hz=float(m[..., FEED].max()),
            retreat_ref_hz=float(m[..., RETREAT].max()),
            advance_ref_hz=float(max(m[..., ADVANCE].max(), ADVANCE_FLOOR_HZ)),
            growth_to_launch=growth_to_launch,
            fitted_on=intact_table.name,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path: str | Path) -> "Decoder":
        return cls(**json.loads(Path(path).read_text(encoding="utf8")))

    def decode(self, rates, gf) -> dict[str, np.ndarray]:
        """rates (..., 4) in Hz and gf (...) bool -> flee, growth, advance, retreat, launch."""
        rates = np.asarray(rates, dtype=float)
        flee = np.asarray(gf, dtype=bool)
        calm = ~flee                                     # a fleeing colony does nothing else
        growth = np.clip(rates[..., FEED] / self.feed_ref_hz, 0.0, 1.0) * calm
        advance = np.clip(rates[..., ADVANCE] / self.advance_ref_hz, 0.0, 1.0) * calm
        retreat = np.clip(rates[..., RETREAT] / self.retreat_ref_hz, 0.0, 1.0) * calm
        launch = np.clip(advance + self.growth_to_launch * growth, 0.0, 1.0) * (1.0 - retreat)
        return {"flee": flee, "growth": growth, "advance": advance, "retreat": retreat, "launch": launch}

    @staticmethod
    def action_names(d: dict[str, np.ndarray]) -> np.ndarray:
        """One label per colony for display: flee, retreat, expand or settle."""
        out = np.full(np.shape(d["flee"]), "settle", dtype=object)
        out[d["launch"] > 0.05] = "expand"
        out[d["retreat"] > 0.5] = "retreat"
        out[d["flee"]] = "flee"
        return out
