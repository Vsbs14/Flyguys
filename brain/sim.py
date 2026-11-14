"""Batched leaky integrate-and-fire simulator for the whole fly brain.

A faithful PyTorch port of Shiu et al.'s Brian2 model (`model.py`):

    dv/dt = (v_rest - v + g) / tau_m        (unless refractory)
    dg/dt = -g / tau_syn                    (unless refractory)
    spike when v > v_thresh  ->  v = v_reset, g = 0
    a presynaptic spike adds  w = count x sign x w_syn  to g, t_delay later
    input neurons get Poisson kicks of w_syn x 250 mV straight into v and
    have no refractory period, exactly as in Shiu's `poi()`

A refractory neuron sits at (v_reset, g = 0) and ignores everything: Brian2
does not integrate its equations, and it also discards synaptic input that
arrives during the refractory period, because `g` carries the
`unless refractory` flag. `brain.test_equivalence` checks all of this against
Brian2 spike for spike.

State is a matrix of shape (neurons x brains): every column is one independent
brain with its own inputs, its own random stream and its own mask.

Two things make it fast enough to run thousands of brains:

* Integration is exact (Brian2's `linear` method), three dense passes a step.
* Spikes are rare, so synaptic transmission is event driven. W is stored
  sorted by presynaptic neuron; each spike gathers that neuron's outgoing
  connections and scatter-adds them into its own brain's column. Cost scales
  with the number of spikes, not with (connections x brains).

Each step follows Brian2's schedule: integrate -> threshold -> synapses
(delayed spikes arrive, Poisson kicks) -> reset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .connectome import Connectome
from .params import SHIU, LIFParams


@dataclass
class BrainState:
    """Everything that defines a batch of brains mid-simulation."""

    u: torch.Tensor                    # (N, B) membrane potential minus v_rest, mV
    g: torch.Tensor                    # (N, B) synaptic drive, mV
    alive: torch.Tensor | None         # (N, B) or (N, 1) bool; False = silenced
    exempt: torch.Tensor               # (N,) bool; True = no refractory period (input neurons)
    gen: torch.Generator
    step: int = 0
    # one entry per past step, newest last: (neuron idx, brain idx, flat idx of refractory spikes)
    hist: list = field(default_factory=list)

    @property
    def n_brains(self) -> int:
        return self.u.shape[1]


@dataclass
class RunResult:
    duration_ms: float
    counts: torch.Tensor               # (n_monitor, B) spikes per monitored neuron
    total_spikes: torch.Tensor         # (B,) spikes across the whole brain
    first_spike_ms: torch.Tensor       # (n_monitor, B) time of first spike, NaN if none
    all_counts: torch.Tensor | None    # (N, B) if record_all

    @property
    def rates_hz(self) -> torch.Tensor:
        return self.counts.float() * (1000.0 / self.duration_ms)


class BrainSim:
    def __init__(self, conn: Connectome, device: str | torch.device = "cpu", params: LIFParams = SHIU,
                 dtype: torch.dtype = torch.float32):
        self.p = params
        self.dtype = dtype
        self.device = torch.device(device)
        self.n = conn.n
        # W, sorted by presynaptic neuron (the CSR arrays of W transposed).
        self.ptr = torch.from_numpy(conn.ptr).to(self.device)
        self.post = torch.from_numpy(conn.post).to(self.device)
        self.w = (torch.from_numpy(conn.signed.astype(np.float64)) * params.w_syn).to(self.device, dtype)

        p = params
        self.decay_m = math.exp(-p.dt / p.tau_m)
        self.decay_s = math.exp(-p.dt / p.tau_syn)
        # exact solution of the coupled v/g equations over one step
        self.g_to_u = p.tau_syn / (p.tau_syn - p.tau_m) * (self.decay_s - self.decay_m)
        self.u_thresh = p.v_thresh - p.v_rest
        self.u_reset = p.v_reset - p.v_rest
        self.kick = p.w_syn * p.poisson_gain
        self.n_delay = round(p.t_delay / p.dt)
        # A neuron that spikes at step s is held for steps s+1 .. s+n_hold and is free again at
        # step s + t_refrac/dt, which is when Brian2's `not_refractory` turns true.
        self.n_hold = max(round(p.t_refrac / p.dt) - 1, 0)

    def weight_matrix(self) -> torch.Tensor:
        """W as one sparse tensor on the device, W[post, pre] in mV."""
        pre = torch.repeat_interleave(torch.arange(self.n, device=self.device), self.ptr[1:] - self.ptr[:-1])
        idx = torch.stack([self.post.long(), pre])
        return torch.sparse_coo_tensor(idx, self.w, (self.n, self.n))

    def new_state(self, n_brains: int, alive: torch.Tensor | None = None, seed: int = 0) -> BrainState:
        """A batch of brains at rest. `alive` False entries are silenced neurons."""
        if alive is not None:
            alive = torch.as_tensor(alive, dtype=torch.bool, device=self.device)
            if alive.dim() == 1:
                alive = alive[:, None]
            if alive.shape[0] != self.n or alive.shape[1] not in (1, n_brains):
                raise ValueError(f"alive must be ({self.n}, {n_brains}) or ({self.n}, 1), got {tuple(alive.shape)}")
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        return BrainState(
            u=torch.zeros(self.n, n_brains, device=self.device, dtype=self.dtype),
            g=torch.zeros(self.n, n_brains, device=self.device, dtype=self.dtype),
            alive=alive,
            exempt=torch.zeros(self.n, dtype=torch.bool, device=self.device),
            gen=gen,
        )

    def _deliver(self, g_flat: torch.Tensor, n_idx: torch.Tensor, b_idx: torch.Tensor, n_brains: int) -> None:
        """Add the outgoing weights of every (neuron, brain) spike into g."""
        start = self.ptr[n_idx]
        cnt = self.ptr[n_idx + 1] - start
        total = int(cnt.sum())
        if total == 0:
            return
        rep = torch.repeat_interleave(torch.arange(n_idx.numel(), device=self.device), cnt)
        base = start - (torch.cumsum(cnt, 0) - cnt)
        pos = base[rep] + torch.arange(total, device=self.device)
        flat = self.post[pos].long() * n_brains + b_idx[rep]
        g_flat.index_add_(0, flat, self.w[pos])

    @torch.no_grad()
    def run(
        self,
        state: BrainState,
        duration_ms: float,
        inputs: list[tuple[torch.Tensor, torch.Tensor | float]] = (),
        monitor: torch.Tensor | None = None,
        record_all: bool = False,
    ) -> RunResult:
        """Advance every brain by `duration_ms`. Can be called again to continue.

        inputs  : list of (neuron indices, rate). Rate is Hz, a float or a (B,)
                  tensor with one value per brain. Each listed neuron gets its
                  own independent Poisson spike train.
        monitor : neuron indices whose spikes are counted per brain.
        """
        dev, B, N = self.device, state.n_brains, self.n
        n_steps = round(duration_ms / self.p.dt)
        u, g = state.u, state.g
        u_flat, g_flat = u.view(-1), g.view(-1)
        alive = state.alive

        # ---- inputs -------------------------------------------------------
        if len(inputs):
            in_idx = torch.cat([torch.as_tensor(i, dtype=torch.long, device=dev).reshape(-1) for i, _ in inputs])
            if in_idx.unique().numel() != in_idx.numel():
                raise ValueError("a neuron appears in more than one input group")
            probs = []
            for idx, rate in inputs:
                r = torch.as_tensor(rate, dtype=torch.float32, device=dev).reshape(-1)
                if r.numel() not in (1, B):
                    raise ValueError(f"rate must be a scalar or have one value per brain ({B})")
                probs.append((r * (self.p.dt * 1e-3)).expand(B).repeat(len(torch.as_tensor(idx).reshape(-1)), 1))
            in_prob = torch.cat(probs)                                  # (n_in, B)
            if alive is not None:
                in_prob = in_prob * alive[in_idx].float()               # silenced inputs never fire
            state.exempt[in_idx] = True
            in_flat = in_idx[:, None] * B + torch.arange(B, device=dev)[None, :]   # (n_in, B)
            any_input = bool((in_prob > 0).any())
        else:
            any_input = False

        # ---- recording ----------------------------------------------------
        if monitor is None:
            monitor = torch.zeros(0, dtype=torch.long)
        monitor = torch.as_tensor(monitor, dtype=torch.long, device=dev).reshape(-1)
        n_mon = monitor.numel()
        mon_pos = torch.full((N,), -1, dtype=torch.long, device=dev)
        mon_pos[monitor] = torch.arange(n_mon, device=dev)
        counts = torch.zeros(n_mon * B, dtype=torch.float32, device=dev)
        first = torch.full((n_mon * B,), float("inf"), device=dev)
        total = torch.zeros(B, dtype=torch.float32, device=dev)
        all_counts = torch.zeros(N * B, dtype=torch.int32, device=dev) if record_all else None

        keep = max(self.n_delay, self.n_hold)
        hist = state.hist

        for _ in range(n_steps):
            state.step += 1

            # 1. integrate. Refractory neurons are held at (u_reset, g = 0).
            ref = [h[2] for h in hist[-self.n_hold:] if h is not None and h[2].numel()] if self.n_hold else []
            if ref:
                ref_flat = torch.cat(ref) if len(ref) > 1 else ref[0]
            u.mul_(self.decay_m).add_(g, alpha=self.g_to_u)
            g.mul_(self.decay_s)
            if ref:
                u_flat[ref_flat] = self.u_reset

            # 2. threshold
            crossed = (u > self.u_thresh).nonzero()
            n_x, b_x = crossed[:, 0], crossed[:, 1]

            # 3. synapses: spikes from n_delay steps ago arrive, then Poisson kicks
            if len(hist) >= self.n_delay and hist[-self.n_delay] is not None:
                h = hist[-self.n_delay]
                self._deliver(g_flat, h[0], h[1], B)
                if ref:
                    g_flat[ref_flat] = 0.0      # input to a refractory neuron is discarded
            if any_input:
                ev = torch.rand(in_prob.shape, generator=state.gen, device=dev) < in_prob
                u_flat.index_add_(0, in_flat[ev], torch.full((int(ev.sum()),), self.kick, device=dev, dtype=u.dtype))

            # 4. reset, and record the spikes that were actually emitted
            if n_x.numel():
                flat_x = n_x * B + b_x
                u_flat[flat_x] = self.u_reset
                g_flat[flat_x] = 0.0
                if alive is not None:
                    # A silenced neuron never emits a spike; it is just put back to rest.
                    ok = alive[n_x, b_x if alive.shape[1] > 1 else torch.zeros_like(b_x)]
                    n_x, b_x, flat_x = n_x[ok], b_x[ok], flat_x[ok]
            if n_x.numel():
                hist.append((n_x, b_x, flat_x[~state.exempt[n_x]]))
                total.index_add_(0, b_x, torch.ones(b_x.numel(), device=dev))
                if all_counts is not None:
                    all_counts.index_add_(0, flat_x, torch.ones(flat_x.numel(), dtype=torch.int32, device=dev))
                pos = mon_pos[n_x]
                sel = pos >= 0
                if sel.any():
                    mflat = pos[sel] * B + b_x[sel]
                    counts.index_add_(0, mflat, torch.ones(mflat.numel(), device=dev))
                    t = torch.full((mflat.numel(),), state.step * self.p.dt, device=dev)
                    first.scatter_reduce_(0, mflat, t, reduce="amin")
            else:
                hist.append(None)
            if len(hist) > keep:
                del hist[: len(hist) - keep]

        first[torch.isinf(first)] = float("nan")
        return RunResult(
            duration_ms=n_steps * self.p.dt,
            counts=counts.view(n_mon, B),
            total_spikes=total,
            first_spike_ms=first.view(n_mon, B),
            all_counts=all_counts.view(N, B) if all_counts is not None else None,
        )
