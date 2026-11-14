"""The galaxy: star systems, jump lanes, resources, raiders, and colonies run by fly brains.

One tick:
  1. Every colony senses its system (see `senses`):
       sugar    = how full the system's resources are (resource / capacity)
       bitter   = the system's toxicity
       looming  = a raider fleet is in the system (1), or next door (`looming_adjacent`)
       obstacle = crowding: everyone's population here / the system's capacity
       target   = how much richer the best neighbouring system is
  2. Its brain variant picks an action from the response tables.
  3. Actions happen:
       feed     harvest resources; grow, but toxic resources poison instead
       escape   jump to a random neighbour, preferring one without raiders
       retreat  move to the least crowded neighbour
       advance  send settlers to the richest neighbour (the whole colony if small)
       idle     nothing
  4. Raiders strike every colony still in their system, then wander, drawn to
     populated systems.
  5. Upkeep, resource regrowth, deaths, and colonies of the same variant in the
     same system merge.

Colonies are coupled through shared, depletable resources, crowding, raiders
that follow population, and settlers moving between systems. All numbers live
in `GalaxyConfig`; the sense and action mappings are the first version of the
Phase 5 design and are meant to be retuned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .brains import ACTIONS, SENSES, ResponseTable

IDLE, FEED, ESCAPE, RETREAT, ADVANCE = range(len(ACTIONS))


@dataclass
class GalaxyConfig:
    # map
    n_systems: int = 60
    n_arms: int = 3
    lanes_per_system: int = 3          # nearest neighbours linked by jump lanes (plus a spanning tree)
    # resources
    capacity_min: float = 60.0
    capacity_max: float = 160.0
    regrowth: float = 0.06             # fraction of the missing resource regrown per tick
    toxic_share: float = 0.25          # share of systems with toxic resources
    toxic_min: float = 0.6             # toxicity of toxic systems, uniform in [toxic_min, 1]
    clean_max: float = 0.15            # toxicity of clean systems, uniform in [0, clean_max]
    # colonies
    colonies_per_variant: int = 8
    start_population: float = 20.0
    harvest_rate: float = 0.3          # resource a feeding colony can take, per unit population
    growth: float = 0.5                # population gained per unit of clean resource eaten
    poison: float = 1.0                # population lost per unit of toxic resource eaten
    upkeep: float = 0.03               # population lost per tick by everyone
    move_cost: float = 0.05            # population lost by escape / retreat
    settle_share: float = 0.4          # share of population sent out by advance
    min_split: float = 10.0            # below twice this, advance moves the whole colony
    death_below: float = 1.0
    max_colonies: int = 3000
    # raiders
    n_raiders: int = 4
    raid_damage: float = 0.5           # share of population lost to a raid
    raider_move_prob: float = 0.5
    # sense mapping
    looming_adjacent: float = 0.0      # looming felt from a raider in a neighbouring system
    target_gain: float = 2.0           # target = gain * (best neighbour's fullness - own fullness)

    def as_dict(self) -> dict:
        return asdict(self)


def make_map(cfg: GalaxyConfig, rng: np.random.Generator):
    """Spiral-armed disk of systems; lanes = k nearest neighbours + a spanning tree (always connected)."""
    n = cfg.n_systems
    r = np.sqrt(rng.uniform(0.04, 1.0, n))
    arm = rng.integers(0, cfg.n_arms, n)
    theta = arm * 2 * np.pi / cfg.n_arms + 2.2 * r + rng.normal(0, 0.25, n)
    pos = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    edges = set()
    for i in range(n):
        for j in np.argsort(d[i])[1:cfg.lanes_per_system + 1]:
            edges.add((min(i, j), max(i, j)))
    # Prim's spanning tree so the lane graph is connected
    inside = {0}
    while len(inside) < n:
        best = min(((d[i, j], i, j) for i in inside for j in range(n) if j not in inside))
        edges.add((min(best[1], best[2]), max(best[1], best[2])))
        inside.add(best[2])
    nbrs = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    return pos, sorted(edges), [np.array(sorted(x)) for x in nbrs]


class Galaxy:
    def __init__(self, cfg: GalaxyConfig, brains: dict[str, ResponseTable], seed: int = 0):
        self.cfg, self.brains, self.rng = cfg, brains, np.random.default_rng(seed)
        self.variants = list(brains)
        rng, n = self.rng, cfg.n_systems
        self.pos, self.lanes, self.nbrs = make_map(cfg, rng)
        self.capacity = rng.uniform(cfg.capacity_min, cfg.capacity_max, n)
        self.resource = self.capacity.copy()
        toxic = rng.random(n) < cfg.toxic_share
        self.toxicity = np.where(toxic, rng.uniform(cfg.toxic_min, 1.0, n), rng.uniform(0, cfg.clean_max, n))
        self.raiders = rng.integers(0, n, cfg.n_raiders)
        # colonies as parallel arrays
        nv = len(self.variants)
        k = cfg.colonies_per_variant * nv
        self.c_sys = rng.integers(0, n, k)
        self.c_pop = np.full(k, cfg.start_population)
        self.c_var = np.repeat(np.arange(nv), cfg.colonies_per_variant)
        self.c_id = np.arange(k)
        self.next_id = k
        self.c_act = np.zeros(k, dtype=int)
        self.tick = 0

    # ---- sensing --------------------------------------------------------
    def system_population(self) -> np.ndarray:
        return np.bincount(self.c_sys, weights=self.c_pop, minlength=self.cfg.n_systems)

    def senses(self) -> np.ndarray:
        cfg = self.cfg
        full = self.resource / self.capacity
        crowd = self.system_population() / self.capacity
        raid_here = np.zeros(cfg.n_systems)
        raid_here[self.raiders] = 1.0
        raid_near = np.array([raid_here[nb].max() if len(nb) else 0.0 for nb in self.nbrs])
        best_nb = np.array([full[nb].max() if len(nb) else 0.0 for nb in self.nbrs])
        s = self.c_sys
        x = np.stack([
            full[s],
            self.toxicity[s],
            np.maximum(raid_here[s], cfg.looming_adjacent * raid_near[s]),
            crowd[s],
            cfg.target_gain * (best_nb[s] - full[s]),
        ], 1)
        return np.clip(x, 0.0, 1.0)

    # ---- one tick -------------------------------------------------------
    def step(self) -> dict:
        cfg, rng = self.cfg, self.rng
        x = self.senses()
        act = np.zeros(len(self.c_sys), dtype=int)
        for v, name in enumerate(self.variants):
            m = self.c_var == v
            if m.any():
                act[m] = self.brains[name].sample(x[m], rng)
        self.c_act = act
        act_by_variant = {name: np.bincount(act[self.c_var == v], minlength=len(ACTIONS))
                          for v, name in enumerate(self.variants)}
        start_sys = self.c_sys.copy()
        var_at_decision = self.c_var.copy()

        # feed: colonies feeding in the same system share its resource in proportion to demand
        feed = act == FEED
        demand = np.where(feed, cfg.harvest_rate * self.c_pop, 0.0)
        tot = np.bincount(self.c_sys, weights=demand, minlength=cfg.n_systems)
        share = np.divide(np.minimum(self.resource, tot), tot, out=np.zeros_like(tot), where=tot > 0)
        eaten = demand * share[self.c_sys]
        self.resource -= np.bincount(self.c_sys, weights=eaten, minlength=cfg.n_systems)
        tox = self.toxicity[self.c_sys]
        self.c_pop += eaten * (cfg.growth * (1 - tox) - cfg.poison * tox)

        # moves, decided on the state at the start of the tick
        raid_here = np.zeros(cfg.n_systems, dtype=bool)
        raid_here[self.raiders] = True
        crowd = self.system_population() / self.capacity
        full = self.resource / self.capacity
        new_cols = []
        moves = []                              # (from, to, action, variant) for playback
        for i in np.flatnonzero(act >= ESCAPE):
            nb = self.nbrs[start_sys[i]]
            if not len(nb):
                continue
            if act[i] == ESCAPE:
                safe = nb[~raid_here[nb]]
                to = rng.choice(safe if len(safe) else nb)
                self.c_sys[i] = to
                self.c_pop[i] *= 1 - cfg.move_cost
            elif act[i] == RETREAT:
                to = nb[np.argmin(crowd[nb])]
                self.c_sys[i] = to
                self.c_pop[i] *= 1 - cfg.move_cost
            else:  # ADVANCE: settlers go to the richest neighbour (the whole colony if it is small)
                to = nb[np.argmax(full[nb])]
                if self.c_pop[i] >= 2 * cfg.min_split and len(self.c_sys) + len(new_cols) < cfg.max_colonies:
                    out = self.c_pop[i] * cfg.settle_share
                    self.c_pop[i] -= out
                    new_cols.append((to, out, self.c_var[i]))
                else:
                    self.c_sys[i] = to
            moves.append((int(start_sys[i]), int(to), ACTIONS[act[i]], int(self.c_var[i])))
        if new_cols:
            d, p, v = map(np.array, zip(*new_cols))
            self.c_sys = np.concatenate([self.c_sys, d])
            self.c_pop = np.concatenate([self.c_pop, p])
            self.c_var = np.concatenate([self.c_var, v])
            self.c_act = np.concatenate([self.c_act, np.full(len(d), ADVANCE)])
            self.c_id = np.concatenate([self.c_id, np.arange(self.next_id, self.next_id + len(d))])
            self.next_id += len(d)

        # raids hit whoever is in a raider's system now
        raided = np.isin(self.c_sys, self.raiders)
        raid_loss = self.c_pop[raided] * cfg.raid_damage
        self.c_pop[raided] -= raid_loss
        self.raiders_before_move = self.raiders.copy()
        # raiders wander, drawn to people
        pop = self.system_population()
        for k in range(len(self.raiders)):
            if rng.random() < cfg.raider_move_prob:
                nb = self.nbrs[self.raiders[k]]
                w = pop[nb] + 5.0
                self.raiders[k] = rng.choice(nb, p=w / w.sum())

        # upkeep, regrowth, deaths, merges
        self.c_pop *= 1 - cfg.upkeep
        self.resource += cfg.regrowth * (self.capacity - self.resource)
        self.last_moves = [m for m in moves if m[0] != m[1]]
        self.last_raided_systems = sorted({int(r) for r in self.raiders_before_move})
        self._prune_and_merge()
        self.tick += 1
        return {"actions": np.bincount(act, minlength=len(ACTIONS)), "raided": int(raided.sum()),
                "raid_loss": float(raid_loss.sum()), "senses_mean": x.mean(0) if len(x) else np.zeros(len(SENSES)),
                "act_by_variant": act_by_variant, "x": x, "act": act, "var": var_at_decision}

    def _prune_and_merge(self):
        alive = self.c_pop >= self.cfg.death_below
        for a in ("c_sys", "c_pop", "c_var", "c_id", "c_act"):
            setattr(self, a, getattr(self, a)[alive])
        key = self.c_sys * len(self.variants) + self.c_var
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        if len(uniq) < len(key):
            pop = np.bincount(inv, weights=self.c_pop)
            keep = np.sort(first)                   # first row of each (system, variant) group survives
            self.c_sys, self.c_var, self.c_id, self.c_act = (self.c_sys[keep], self.c_var[keep],
                                                             self.c_id[keep], self.c_act[keep])
            self.c_pop = pop[inv[keep]]

    # ---- reporting --------------------------------------------------------
    def census(self) -> dict:
        out = {}
        for v, name in enumerate(self.variants):
            m = self.c_var == v
            out[name] = {"population": float(self.c_pop[m].sum()), "colonies": int(m.sum()),
                         "systems": int(len(np.unique(self.c_sys[m])))}
        return out

    def frame(self) -> dict:
        """Compact snapshot for playback (Phase 7)."""
        return {"tick": self.tick, "raiders": self.raiders.tolist(),
                "resource": np.round(self.resource / self.capacity, 3).tolist(),
                "colonies": [[int(i), int(s), round(float(p), 1), int(v), int(a)]
                             for i, s, p, v, a in zip(self.c_id, self.c_sys, self.c_pop, self.c_var, self.c_act)]}
