"""The colony loop: thousands of colonies, each deciding with a (cached) fly brain.

Every epoch, for every colony:
  1. work out how its world tastes and what looms (galaxy -> senses, in Hz)
  2. ask its brain (a response-table lookup, or a baseline rule)
  3. apply the decision: grow, launch ships, pull back, or flee
then ships move, new colonies are founded, threats strike, and colonies of
different civilizations that come within range meet for the first time.

A "policy" is anything with  sample(sugar_hz, bitter_hz, looming_hz, rng)
returning {"rates": (n, 4) Hz, "gf": (n,) bool}. Fly variants and baselines all
go through the same decoder and the same galaxy rules, so the brain is the only
thing that differs between runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .mapping import MAX_HZ, ColonyMemory, Decoder
from .world import KINDS, Galaxy

OUTCOMES = ["coexist", "one_retreats", "mutual_flight", "escalation"]
DEATHS = ["destroyed", "starved", "lost_fleeing", "conflict"]


@dataclass(frozen=True)
class SimConfig:
    # growth
    growth_rate: float = 0.25          # logistic growth per epoch at full appetite
    starve_rate: float = 0.01          # shrinkage per epoch with no appetite at all
    min_pop: float = 0.02              # a colony below this dies out
    home_pop: float = 0.5              # a home world starts at this share of its capacity
    # ships
    launch_above: float = 0.5          # spare capacity is population above this share of capacity
    min_ship: float = 0.03
    target_quantile: float = 0.5       # ships only go to worlds better than this share of the galaxy
    launch_cooldown: int = 3
    transit_survival: float = 0.9
    # fleeing and retreating
    flee_survival: float = 0.6         # share of a colony that makes it onto the ships
    retreat_evacuation: float = 0.5    # share of population sent home at full retreat
    retreat_pause: int = 10            # epochs without launches after backing off from a neighbour
    # first contact
    contact_rounds: int = 3
    contact_gain_hz: float = 6.0       # looming a neighbour causes per unit of its launch share
    trade_gain_hz: float = 20.0        # sugar a neighbour offers per unit of its growth
    conflict_loss: float = 0.3
    trade_bonus: float = 0.05
    ship_exposure: float = 0.01        # looming exposure of a foreign ship heading into the neighbourhood
    # the observer at the Sun
    detect_range: float = 0.2
    detect_pop: float = 0.3

    def as_dict(self) -> dict:
        return asdict(self)


class GalaxySim:
    def __init__(self, galaxy: Galaxy, policy, decoder: Decoder, cfg: SimConfig = SimConfig(), seed: int = 0,
                 senses=None, frames=None, label: str = ""):
        self.g, self.policy, self.decoder, self.cfg = galaxy, policy, decoder, cfg
        self.senses = senses or galaxy.senses
        self.rng = np.random.default_rng([seed, 7919])
        self.frames, self.label = frames, label
        n = galaxy.n
        self.civ = np.full(n, -1)
        self.pop = np.zeros(n)
        self.uid = np.full(n, -1)                 # unique id of the colony on each star
        self.travel = np.zeros(n)                 # length of the jump that founded each colony, as a share of max_jump
        self.cooldown = np.zeros(n, dtype=int)
        self.sterile_until = np.zeros(n, dtype=int)
        self.last_growth = np.zeros(n)
        self.last_launch = np.zeros(n)
        self.memory = ColonyMemory(n=n)
        self.ships: list[dict] = []
        self.next_uid = 0
        self.born: dict[int, int] = {}            # colony uid -> epoch founded
        self.colonisers: set[int] = set()         # colony uids that launched at least one colony ship
        self.civ_started = np.zeros(len(galaxy.origins), dtype=bool)

        # pairs of stars close enough for contact (and for a jump)
        i, j = np.nonzero(np.triu(galaxy.dist <= galaxy.cfg.max_jump, k=1))
        self.edge_i, self.edge_j = i, j
        self.met_i = np.full(len(i), -1)
        self.met_j = np.full(len(i), -1)
        self.adjacent = galaxy.dist <= galaxy.cfg.max_jump
        np.fill_diagonal(self.adjacent, False)
        self.quality = galaxy.sugar_score(np.arange(n)) - galaxy.bitter_score(np.arange(n), 0.0)
        self.worth_settling = self.quality >= np.quantile(self.quality, cfg.target_quantile)
        self.near_sun = galaxy.dist[galaxy.sun] <= cfg.detect_range

        self.series = {k: [] for k in ("colonies", "occupied_fraction", "civilizations", "population",
                                       "detectable_civilizations", "detectable_colonies")}
        self.counts = {"founded": 0, "launches": 0, "flights": 0, "flights_survived": 0,
                       "deaths": dict.fromkeys(DEATHS, 0), "encounters": dict.fromkeys(OUTCOMES, 0),
                       "ships_lost": 0}
        self.events: list[dict] = []

    # ------------------------------------------------------------------ helpers

    def _found(self, star: int, civ: int, pop: float, t: int, memory_hz: float = 0.0, travel: float = 0.0) -> None:
        self.civ[star], self.pop[star] = civ, min(pop, self.g.capacity[star])
        self.travel[star] = travel
        self.uid[star] = self.next_uid
        self.born[self.next_uid] = t
        self.next_uid += 1
        self.cooldown[star] = 0
        self.last_growth[star] = self.last_launch[star] = 0.0
        self.memory.baseline_hz[star] = memory_hz
        self.counts["founded"] += 1

    def _remove(self, star: int, cause: str | None) -> None:
        if cause:
            self.counts["deaths"][cause] += 1
        self.civ[star], self.pop[star], self.uid[star] = -1, 0.0, -1
        self.memory.baseline_hz[star] = 0.0
        self.last_growth[star] = self.last_launch[star] = 0.0

    def _ship(self, origin: int, target: int, payload: float, t: int, kind: str, memory_hz: float = 0.0) -> None:
        eta = max(1, int(np.ceil(self.g.dist[origin, target] / self.g.cfg.ship_speed)))
        self.ships.append({"from": origin, "to": target, "civ": int(self.civ[origin]), "uid": int(self.uid[origin]),
                           "payload": payload, "depart": t, "arrive": t + eta, "kind": kind, "memory": memory_hz})

    def _inputs(self, stars: np.ndarray, threat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sugar, bitter and looming rates in Hz for the colonies on `stars`."""
        crowding = self.pop[stars] / self.g.capacity[stars]
        sugar = self.senses.sugar_hz(self.g.sugar_score(stars, crowding))
        # travel distance: how far the founders had to come. A home world has none.
        bitter = self.senses.bitter_hz(self.g.bitter_score(stars, self.travel[stars]))
        looming = self.senses.looming_hz(threat[stars], self.memory.baseline_hz[stars])
        return sugar, bitter, looming

    def _threat_scores(self, t: int) -> np.ndarray:
        """Looming score per star: scheduled events plus foreign ships heading into the neighbourhood."""
        score = self.g.looming_scores(t)
        for s in self.ships:
            if s["kind"] != "colony":
                continue
            near = self.adjacent[s["to"]].copy()
            near[s["to"]] = True
            watchers = np.flatnonzero(near & (self.civ >= 0) & (self.civ != s["civ"]))
            if len(watchers):
                score[watchers] += self.cfg.ship_exposure / max(s["arrive"] - t, 1)
        return score

    def _flee(self, star: int, t: int, threat: np.ndarray) -> None:
        """Abandon the system. Survivors head for the best safe star in range, or are lost."""
        self.counts["flights"] += 1
        self.memory.record_loss([star])
        near = self.g.neighbours[star]
        ok = near[(self.sterile_until[near] <= t) & (threat[near] == 0)
                  & ((self.civ[near] == -1) | (self.civ[near] == self.civ[star]))]
        if len(ok):
            own = self.civ[ok] == self.civ[star]
            score = self.quality[ok] + 0.3 * own - 0.3 * self.g.dist[star, ok] / self.g.cfg.max_jump
            self._ship(star, int(ok[np.argmax(score)]), self.pop[star] * self.cfg.flee_survival, t, "refugee",
                       self.memory.baseline_hz[star])
            self.counts["flights_survived"] += 1
            self._remove(star, None)
        else:
            self._remove(star, "lost_fleeing")

    # ------------------------------------------------------------------ one epoch

    def step(self, t: int) -> None:
        g, cfg = self.g, self.cfg
        frame_events: list[dict] = []
        actions = np.full(g.n, "settle", dtype=object)
        shown = {}

        # civilizations arise
        for c in np.flatnonzero(g.origin_epoch == t):
            home = int(g.origins[c])
            if self.civ[home] == -1 and self.sterile_until[home] <= t:
                self._found(home, int(c), cfg.home_pop * g.capacity[home], t)
                self.civ_started[c] = True

        threat = self._threat_scores(t)
        occ = np.flatnonzero(self.civ >= 0)
        if len(occ):
            sugar, bitter, looming = self._inputs(occ, threat)
            resp = self.policy.sample(sugar, bitter, looming, self.rng)
            d = self.decoder.decode(resp["rates"], resp["gf"])
            actions[occ] = Decoder.action_names(d)
            self.last_growth[occ], self.last_launch[occ] = d["growth"], d["launch"]
            if self.frames is not None:
                shown = {"stars": occ, "sugar": sugar, "bitter": bitter, "looming": looming,
                         "rates": resp["rates"], "gf": resp["gf"]}

            # flee: overrides everything else
            for star in occ[d["flee"]]:
                self._flee(int(star), t, threat)

            stay = ~d["flee"]
            stars, growth, launch, retreat = occ[stay], d["growth"][stay], d["launch"][stay], d["retreat"][stay]

            # grow or starve
            rate = cfg.growth_rate * growth - cfg.starve_rate * (1 - growth)
            room = np.where(rate > 0, 1 - self.pop[stars] / g.capacity[stars], 1.0)
            self.pop[stars] = np.clip(self.pop[stars] * (1 + rate * room), 0, g.capacity[stars])
            for star in stars[self.pop[stars] < cfg.min_pop]:
                self._remove(int(star), "starved")

            # retreat: send part of the population back to the nearest friendly colony
            for k in np.flatnonzero((retreat > 0.1) & (self.civ[stars] >= 0)):
                star = int(stars[k])
                near = g.neighbours[star]
                friends = near[(self.civ[near] == self.civ[star]) & (threat[near] == 0)]
                moved = retreat[k] * cfg.retreat_evacuation * self.pop[star]
                if len(friends) and moved >= cfg.min_ship:
                    self._ship(star, int(friends[0]), moved, t, "refugee", self.memory.baseline_hz[star])
                    self.pop[star] -= moved

            # launch colony ships from spare capacity
            spare = self.pop[stars] - cfg.launch_above * g.capacity[stars]
            payload = launch * spare
            free = (self.civ == -1) & (self.sterile_until <= t) & self.worth_settling
            can = ((payload >= cfg.min_ship) & (self.cooldown[stars] == 0) & (self.civ[stars] >= 0)
                   & (self.adjacent[stars] @ free > 0))
            if can.any():
                targeted = {(s["civ"], s["to"]) for s in self.ships}
                for k in np.flatnonzero(can):
                    star = int(stars[k])
                    near = g.neighbours[star]
                    options = near[free[near]]
                    options = options[[(int(self.civ[star]), int(o)) not in targeted for o in options]]
                    if not len(options):
                        continue
                    # the best world in range, a little nearer preferred, as judged from here
                    score = (self.quality[options] - 0.4 * g.dist[star, options] / g.cfg.max_jump
                             + self.rng.normal(0, 0.02, len(options)))
                    target = int(options[np.argmax(score)])
                    self._ship(star, target, payload[k], t, "colony")
                    targeted.add((int(self.civ[star]), target))
                    self.pop[star] -= payload[k]
                    self.cooldown[star] = cfg.launch_cooldown
                    self.colonisers.add(int(self.uid[star]))
                    self.counts["launches"] += 1
                    actions[star] = "expand"

        # ships arrive
        still_flying = []
        for s in self.ships:
            if s["arrive"] > t:
                still_flying.append(s)
                continue
            x = s["to"]
            if self.sterile_until[x] > t or (self.civ[x] >= 0 and self.civ[x] != s["civ"]):
                self.counts["ships_lost"] += 1                      # dead system, or someone else got there first
                origin = s["from"]
                if self.uid[origin] == s["uid"]:
                    self.memory.record_loss([origin])               # the colony that sent it remembers
            elif self.civ[x] == -1:
                self._found(x, s["civ"], s["payload"] * cfg.transit_survival, t, memory_hz=0.5 * s["memory"],
                            travel=float(np.clip(g.dist[s["from"], x] / g.cfg.max_jump, 0, 1)))
            else:
                self.pop[x] = min(self.pop[x] + s["payload"] * cfg.transit_survival, g.capacity[x])
        self.ships = still_flying

        # threats strike
        th = g.threats
        for e in np.flatnonzero(th.epoch == t):
            hit = th.stars[e][th.exposure[e] >= 1.0]
            seen = th.stars[e][th.exposure[e] < 1.0]
            frame_events.append({"type": KINDS[th.kind[e]], "star": int(th.centre[e])})
            for star in hit[self.civ[hit] >= 0]:
                self.pop[star] *= 1 - th.kill[e]
                if self.pop[star] < cfg.min_pop:
                    self._remove(int(star), "destroyed")
                else:
                    self.memory.record_loss([star])
            if th.sterilises[e]:
                self.sterile_until[hit] = t + g.cfg.sterile_epochs
            self.memory.record_loss(seen[self.civ[seen] >= 0])       # watching a neighbour die leaves a mark

        frame_events += self._first_contacts(t, threat)

        self.memory.step()
        np.maximum(self.cooldown - 1, 0, out=self.cooldown)
        self._record(t)
        if self.frames is not None:
            self._write_frame(t, actions, shown, frame_events)

    # ------------------------------------------------------------------ first contact

    def _first_contacts(self, t: int, threat: np.ndarray) -> list[dict]:
        """Colonies of different civilizations meeting for the first time: up to 3 rounds of two brains reacting."""
        cfg = self.cfg
        ei, ej = self.edge_i, self.edge_j
        new = ((self.civ[ei] >= 0) & (self.civ[ej] >= 0) & (self.civ[ei] != self.civ[ej])
               & ((self.uid[ei] != self.met_i) | (self.uid[ej] != self.met_j)))
        idx = np.flatnonzero(new)
        if not len(idx):
            return []
        self.met_i[idx], self.met_j[idx] = self.uid[ei[idx]], self.uid[ej[idx]]
        a, b = ei[idx], ej[idx]
        both = np.concatenate([a, b])
        sugar0, bitter0, loom0 = self._inputs(both, threat)
        n = len(a)
        adv = self.last_launch[both].copy()        # what each side was doing before they met
        gro = self.last_growth[both].copy()
        fled = np.zeros(2 * n, dtype=bool)
        backed = np.zeros(2 * n, dtype=bool)
        memory = np.zeros(2 * n)
        other = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
        first_loom = last_loom = None
        for _ in range(cfg.contact_rounds):
            # the neighbour's advance looms; its settling tastes mildly sweet (a trade opportunity)
            memory += self.memory.bump_hz * adv[other]
            looming = np.clip(loom0 + cfg.contact_gain_hz * adv[other] + memory, 0, MAX_HZ)
            sugar = np.clip(sugar0 + cfg.trade_gain_hz * gro[other], 0, MAX_HZ)
            resp = self.policy.sample(sugar, bitter0, looming, self.rng)
            d = self.decoder.decode(resp["rates"], resp["gf"])
            live = ~fled
            fled |= d["flee"] & live
            backed |= (d["retreat"] > 0.5) & live
            adv = np.where(fled, 0.0, d["launch"])
            gro = np.where(fled, 0.0, d["growth"])
            first_loom = looming if first_loom is None else first_loom
            last_loom = looming
        rising = last_loom > first_loom + 1e-9

        events = []
        for k in range(n):
            ia, ib = k, k + n
            sa, sb = int(a[k]), int(b[k])
            if self.civ[sa] < 0 or self.civ[sb] < 0:
                continue                                   # one of them already left this epoch
            ida, idb = int(self.uid[sa]), int(self.uid[sb])
            if (fled[ia] and fled[ib]) or (backed[ia] and backed[ib] and not fled[ia] and not fled[ib]):
                outcome = "mutual_flight"
            elif fled[ia] or fled[ib] or backed[ia] or backed[ib]:
                outcome = "one_retreats"
            elif rising[ia] and rising[ib] and adv[ia] >= 0.2 and adv[ib] >= 0.2:
                outcome = "escalation"
            else:
                outcome = "coexist"
            self.counts["encounters"][outcome] += 1
            events.append({"type": "encounter", "a": ida, "b": idb, "outcome": outcome,
                           "star": sa, "star_a": sa, "star_b": sb})
            for i, star in ((ia, sa), (ib, sb)):
                if fled[i]:
                    self._flee(star, t, threat)
                elif backed[i]:
                    self.cooldown[star] = max(self.cooldown[star], cfg.retreat_pause)
                    self.memory.record_loss([star])
                elif outcome == "escalation":
                    self.pop[star] *= 1 - cfg.conflict_loss
                    self.memory.record_loss([star])
                    if self.pop[star] < cfg.min_pop:
                        self._remove(star, "conflict")
                elif outcome == "coexist":
                    self.pop[star] = min(self.pop[star] * (1 + cfg.trade_bonus), self.g.capacity[star])
        return events

    # ------------------------------------------------------------------ measurements

    def _record(self, t: int) -> None:
        occ = self.civ >= 0
        flying = {s["civ"] for s in self.ships}
        loud = occ & self.near_sun & (self.pop >= self.cfg.detect_pop)
        s = self.series
        s["colonies"].append(int(occ.sum()))
        s["occupied_fraction"].append(float(occ.mean()))
        s["civilizations"].append(len(set(self.civ[occ].tolist()) | flying))
        s["population"].append(float(self.pop.sum()))
        s["detectable_civilizations"].append(len(set(self.civ[loud].tolist())))
        s["detectable_colonies"].append(int(loud.sum()))

    def _write_frame(self, t: int, actions, shown: dict, events: list[dict]) -> None:
        look = {int(star): i for i, star in enumerate(shown.get("stars", []))}
        colonies = []
        for star in np.flatnonzero(self.civ >= 0):
            c = {"id": int(self.uid[star]), "star": int(star), "pop": float(self.pop[star]),
                 "action": str(actions[star]), "civ": int(self.civ[star])}
            i = look.get(int(star))
            if i is not None:
                r = shown["rates"][i]
                c["inputs"] = {"sugar": float(shown["sugar"][i] / MAX_HZ), "bitter": float(shown["bitter"][i] / MAX_HZ),
                               "looming": float(shown["looming"][i] / MAX_HZ)}
                c["rates"] = {"MN9": float(r[0]), "DNp09": float(r[3]), "MDN": float(r[2]),
                              "GF": int(shown["gf"][i])}
            colonies.append(c)
        ships = [{"from": s["from"], "to": s["to"],
                  "progress": float((t - s["depart"]) / max(s["arrive"] - s["depart"], 1))} for s in self.ships]
        self.frames.add(t, self.label, colonies=colonies, ships=ships, events=events)

    def run(self) -> dict:
        for t in range(self.g.cfg.n_epochs):
            self.step(t)
        return self.summary()

    def summary(self) -> dict:
        cfg_g = self.g.cfg
        settled_early = [u for u, t0 in self.born.items() if t0 <= cfg_g.n_epochs - 50]
        s = self.series
        tail = slice(-50, None)
        return {
            "final_colonies": s["colonies"][-1],
            "final_civilizations": s["civilizations"][-1],
            "final_occupied_fraction": s["occupied_fraction"][-1],
            "peak_colonies": max(s["colonies"]),
            "mean_colonies_last_50": float(np.mean(s["colonies"][tail])),
            "mean_detectable_civilizations_last_50": float(np.mean(s["detectable_civilizations"][tail])),
            "epochs_with_anyone_detectable": float(np.mean(np.array(s["detectable_civilizations"]) > 0)),
            "civilizations_started": int(self.civ_started.sum()),
            "civilizations_extinct": int(self.civ_started.sum()) - s["civilizations"][-1],
            "colonies_founded": self.counts["founded"],
            "colony_deaths": dict(self.counts["deaths"]),
            "colony_deaths_total": int(sum(self.counts["deaths"].values())),
            "flights": self.counts["flights"],
            "flights_survived": self.counts["flights_survived"],
            "launches": self.counts["launches"],
            "ships_lost": self.counts["ships_lost"],
            "encounters": dict(self.counts["encounters"]),
            # Landis: the share of colonies that go on to found colonies of their own
            "effective_p": (len(self.colonisers & set(settled_early)) / len(settled_early)) if settled_early else 0.0,
            "series": {k: list(v) for k, v in s.items()},
        }
