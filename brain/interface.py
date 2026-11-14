"""Phase 2: what a colony senses goes in, one decision comes out.

The world side of the project speaks in *intensities*: one number in [0, 1]
per sense, per colony. 0 means the sense is not stimulated, 1 means its
neurons fire at the maximum input rate (200 Hz, the top of the range the
starter notebook and the benchmark use).

  sense     neurons        world meaning (suggested)
  sugar     LB3c           food here is good
  bitter    bitter GRNs    food here is toxic
  looming   LC4 + LPLC2    something is rushing at us (a threat)
  obstacle  LC16           something blocks the way ahead
  target    LC31b          something worth moving towards

The brain answers with the firing rate of four descending/motor outputs. A
`DecisionRule` turns those rates into one of five actions:

  idle     no output is driven strongly enough
  feed     MN9   (proboscis extension)
  escape   Giant Fibre (DNp01, jump)
  retreat  MDN   (walk backwards)
  advance  DNp09 (walk forwards)

Each output's rate is divided by its *reference rate*: what that output fires
at when its own sense is fully on and nothing else is (calibrated by
`python -m brain.calibrate`). The action is the output with the highest
normalised score, or `idle` if no score reaches `threshold`. Normalising
matters because the outputs live on very different scales: the Giant Fibre
reaches 200 Hz, MDN about 35 Hz.

  brain = FlyBrain.load(device="cuda")
  d = brain.decide([[1, 0, 0, 0, 0],      # colony 0: sugar only
                    [1, 0, 1, 0, 0]])     # colony 1: sugar while a threat looms
  d.actions                               # ['feed', 'escape']
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .connectome import ROOT, Connectome
from .neurons import NeuronGroups
from .sim import BrainSim

SENSES = ("sugar", "bitter", "looming", "obstacle", "target")
OUTPUTS = ("feed", "escape", "retreat", "advance")
ACTIONS = ("idle",) + OUTPUTS
MAX_HZ = 200.0
DECISION_MS = 200.0
RULE_FILE = ROOT / "data" / "decision_rule.json"

# Which sense calibrates which output: the output's reference rate is measured
# with this sense at full intensity and every other sense off.
DRIVER = {"feed": "sugar", "escape": "looming", "retreat": "obstacle", "advance": "target"}


def intensity_to_hz(intensity) -> torch.Tensor:
    """(B, n_senses) intensities in [0, 1] -> (n_senses, B) Poisson rates in Hz.

    Values outside [0, 1] are clipped; the mapping is linear.
    """
    x = torch.as_tensor(np.asarray(intensity, dtype=np.float32))
    if x.dim() == 1:
        x = x[None, :]
    if x.shape[1] != len(SENSES):
        raise ValueError(f"expected {len(SENSES)} intensities per brain {SENSES}, got shape {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("intensities must be finite")
    return (x.clamp(0.0, 1.0) * MAX_HZ).T.contiguous()


@dataclass
class DecisionRule:
    """Output rates (Hz) -> action. Stored in data/decision_rule.json."""

    ref_hz: dict                     # output -> reference rate in Hz
    threshold: float = 0.25          # minimum normalised score for any action
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = [o for o in OUTPUTS if not self.ref_hz.get(o, 0) > 0]
        if missing:
            raise ValueError(f"reference rate must be > 0 for every output; bad: {missing}")

    def scores(self, rates_hz) -> torch.Tensor:
        """(B, n_outputs) rates in OUTPUTS order -> normalised scores."""
        r = torch.as_tensor(rates_hz, dtype=torch.float32)
        ref = torch.tensor([self.ref_hz[o] for o in OUTPUTS], dtype=torch.float32, device=r.device)
        return r / ref

    def choose(self, rates_hz) -> torch.Tensor:
        """(B, n_outputs) rates -> (B,) action index into ACTIONS (0 = idle).

        Ties go to the earlier output in OUTPUTS (feed, escape, retreat, advance).
        """
        s = self.scores(rates_hz)
        best, arg = s.max(dim=1)
        return torch.where(best >= self.threshold, arg + 1, torch.zeros_like(arg))

    def save(self, path: str | Path = RULE_FILE) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=1) + "\n", encoding="utf8")
        return path

    @classmethod
    def load(cls, path: str | Path = RULE_FILE) -> "DecisionRule":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m brain.calibrate` first")
        return cls(**json.loads(path.read_text(encoding="utf8")))


@dataclass
class Decisions:
    actions: list[str]               # (B,) action names
    action_idx: torch.Tensor         # (B,) index into ACTIONS
    rates_hz: torch.Tensor           # (B, n_outputs) mean rate over each output type's neurons
    scores: torch.Tensor             # (B, n_outputs) rates / reference rates


class FlyBrain:
    """A batch of fly brains that turn sense intensities into actions."""

    def __init__(self, sim: BrainSim, groups: NeuronGroups, rule: DecisionRule | None = None,
                 decision_ms: float = DECISION_MS):
        if tuple(groups.senses) != SENSES or tuple(groups.outputs) != OUTPUTS:
            raise ValueError(f"neuron_groups.json must define senses {SENSES} and outputs {OUTPUTS}; "
                             f"it has {groups.senses} and {groups.outputs}")
        self.sim, self.groups, self.rule, self.decision_ms = sim, groups, rule, decision_ms
        self._monitor = torch.from_numpy(groups.all_outputs())
        sizes = [len(groups.output(o)) for o in OUTPUTS]
        self._slices = np.cumsum([0] + sizes)

    @classmethod
    def load(cls, device: str = "cpu", rule_path: str | Path = RULE_FILE) -> "FlyBrain":
        conn = Connectome.load()
        return cls(BrainSim(conn, device=device), NeuronGroups(conn), DecisionRule.load(rule_path))

    def output_rates(self, intensity, alive=None, seed: int = 0) -> torch.Tensor:
        """Run one decision window from rest. Returns (B, n_outputs) rates in Hz, on the CPU."""
        hz = intensity_to_hz(intensity)
        state = self.sim.new_state(hz.shape[1], alive=alive, seed=seed)
        res = self.sim.run(state, self.decision_ms, monitor=self._monitor,
                           inputs=[(self.groups.sense(s), hz[i]) for i, s in enumerate(SENSES)])
        r = res.rates_hz.cpu()
        return torch.stack([r[a:b].mean(0) for a, b in zip(self._slices[:-1], self._slices[1:])], dim=1)

    def decide(self, intensity, alive=None, seed: int = 0) -> Decisions:
        """(B, n_senses) intensities in [0, 1] -> one decision per brain.

        alive : optional (N,) or (N, B) bool mask; False silences a neuron.
        Decisions are stochastic (Poisson input); use different seeds for replicates.
        """
        if self.rule is None:
            raise RuntimeError("no DecisionRule; run `python -m brain.calibrate` or pass rule=")
        rates = self.output_rates(intensity, alive=alive, seed=seed)
        idx = self.rule.choose(rates)
        return Decisions([ACTIONS[i] for i in idx.tolist()], idx, rates, self.rule.scores(rates))
