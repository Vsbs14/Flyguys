"""The galaxy every brain variant plays in: stars, worlds, threats and home worlds.

Built once per seed, following Forgan's Monte Carlo approach: draw a synthetic
galaxy from distributions, then let events play out on it. Nothing in here
depends on the brain, so every variant gets the same star map, the same threat
schedule and the same starting civilizations. The only thing that differs
between runs is how colonies decide.

Units: the disk has radius 1 (about 50,000 light-years). One epoch is 0.5 Myr.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .mapping import CLIP_SD, SenseMap, SenseWeights, habitability, hostility


@dataclass(frozen=True)
class GalaxyConfig:
    n_stars: int = 1500
    disk_scale: float = 0.28           # exponential scale length of the stellar disk
    n_epochs: int = 400
    epoch_myr: float = 0.5
    sun_radius: float = 0.52           # the Sun sits about 26,000 ly from the centre

    # travel
    ship_speed: float = 0.03           # disk radii per epoch (about 0.003 c)
    max_jump: float = 0.12             # farthest a colony ship can go (about 6,000 ly)

    # civilizations
    n_civs: int = 8
    last_origin_epoch: int = 40
    origin_spacing: float = 0.25

    # supernovae: any given star is sterilised about 1.5 times per Gyr (the published rate of
    # supernovae within 8 pc). The kill radius is scaled to our much sparser star map and the
    # event rate is then set so each star still faces that hazard.
    sterilisations_per_gyr: float = 1.5
    sn_radius: float = 0.05
    sn_warning: int = 20               # epochs a doomed star is visible in advance
    sterile_epochs: int = 60           # how long a sterilised system stays dead

    # black holes: rare, slow, wide
    bh_per_epoch: float = 0.01
    bh_radius: float = 0.08
    bh_warning: int = 40

    # asteroid swarms: common, local, survivable
    swarm_per_epoch: float = 0.3
    swarm_kill: float = 0.5
    swarm_warning: int = 5

    def as_dict(self) -> dict:
        return asdict(self)


KINDS = ["supernova", "black_hole", "asteroid_swarm"]


@dataclass
class Threats:
    """Scheduled events. `stars`/`exposure` list every star an event touches.

    exposure 1 = inside the kill radius. Between 1 and 2 radii the event is
    visible and frightening but harmless, with exposure falling to 0.
    """
    epoch: np.ndarray          # (E,) when it happens
    kind: np.ndarray           # (E,) index into KINDS
    centre: np.ndarray         # (E,) star at the centre
    warning: np.ndarray        # (E,) epochs of advance notice
    kill: np.ndarray           # (E,) fraction of a colony killed inside the kill radius
    sterilises: np.ndarray     # (E,) bool
    stars: list                # E arrays of affected star indices
    exposure: list             # E arrays, same shapes


@dataclass
class Galaxy:
    cfg: GalaxyConfig
    seed: int
    xy: np.ndarray             # (N, 2)
    props: dict                # per-star world properties of the best planet, each (N,) in 0-1
    n_planets: np.ndarray      # (N,)
    capacity: np.ndarray       # (N,) carrying capacity, 0.3-1
    dist: np.ndarray           # (N, N) float32
    neighbours: list           # N arrays: stars within max_jump, nearest first
    threats: Threats
    origins: np.ndarray        # (C,) home star of each civilization
    origin_epoch: np.ndarray   # (C,)
    sun: int                   # star nearest the Sun's position
    senses: SenseMap
    weights: SenseWeights

    @property
    def n(self) -> int:
        return len(self.xy)

    def sugar_score(self, stars, crowding=0.0):
        """Habitability. A full world has used up a fifth of its resources."""
        p = self.props
        return habitability(p["water"][stars], p["energy"][stars], p["resources"][stars] * (1 - 0.2 * crowding),
                            self.weights)

    def bitter_score(self, stars, travel):
        """Hostility. `travel` is the length of the jump that reached the world, as a share of a full jump."""
        p = self.props
        return hostility(p["radiation"][stars], p["cold"][stars], travel, self.weights)

    def looming_scores(self, epoch: int) -> np.ndarray:
        """(N,) summed exposure / (epochs until the event) over every threat currently visible."""
        t = self.threats
        score = np.zeros(self.n)
        ahead = t.epoch - epoch
        for e in np.flatnonzero((ahead > 0) & (ahead <= t.warning)):
            score[t.stars[e]] += t.exposure[e] / ahead[e]
        return score


def _disk(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """Stars thinning out exponentially from the centre, inside radius 1."""
    r = np.empty(0)
    while len(r) < n:
        draw = rng.gamma(2.0, scale, size=2 * n)        # radial density r * exp(-r / scale)
        r = np.concatenate([r, draw[draw < 1.0]])
    r = r[:n]
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _worlds(rng: np.random.Generator, xy: np.ndarray, w: SenseWeights) -> tuple[dict, np.ndarray]:
    """Give each star 1-5 planets and keep the best one as its world."""
    n = len(xy)
    n_planets = 1 + np.minimum(rng.poisson(1.5, n), 4)
    exists = np.arange(5)[None, :] < n_planets[:, None]
    r = np.hypot(xy[:, 0], xy[:, 1])[:, None]
    planets = {
        "water": rng.beta(2, 2, (n, 5)),
        "energy": rng.beta(2, 2, (n, 5)),
        "resources": rng.beta(2, 2, (n, 5)),
        # radiation is worse near the crowded galactic centre
        "radiation": np.clip(rng.beta(2, 3, (n, 5)) + 0.45 * np.exp(-r / 0.15), 0, 1),
        "cold": rng.beta(2, 2, (n, 5)),
    }
    quality = (habitability(planets["water"], planets["energy"], planets["resources"], w)
               - hostility(planets["radiation"], planets["cold"], 0.0, w))
    best = np.argmax(np.where(exists, quality, -np.inf), axis=1)
    return {k: v[np.arange(n), best] for k, v in planets.items()}, n_planets


def _threats(rng: np.random.Generator, cfg: GalaxyConfig, dist: np.ndarray) -> Threats:
    n = len(dist)
    # supernova rate that gives each star the published sterilisation hazard
    hazard_per_epoch = cfg.sterilisations_per_gyr * cfg.epoch_myr / 1000.0
    mean_in_zone = float((dist <= cfg.sn_radius).sum(axis=1).mean())
    sn_per_epoch = hazard_per_epoch * n / mean_in_zone

    rows = []
    for kind, rate, radius, warning, kill, sterilises in (
        (0, sn_per_epoch, cfg.sn_radius, cfg.sn_warning, 1.0, True),
        (1, cfg.bh_per_epoch, cfg.bh_radius, cfg.bh_warning, 1.0, True),
        (2, cfg.swarm_per_epoch, 0.0, cfg.swarm_warning, cfg.swarm_kill, False),
    ):
        count = rng.poisson(rate * cfg.n_epochs)
        for epoch, centre in zip(rng.integers(1, cfg.n_epochs, count), rng.integers(0, n, count)):
            d = dist[centre]
            if radius > 0:
                stars = np.flatnonzero(d <= 2 * radius)
                exposure = np.clip(2.0 - d[stars] / radius, 0.0, 1.0)
            else:
                stars, exposure = np.array([centre]), np.array([1.0])
            rows.append((int(epoch), kind, int(centre), warning, kill, sterilises, stars, exposure))
    rows.sort(key=lambda r: r[0])
    col = lambda i, dtype: np.array([r[i] for r in rows], dtype=dtype)
    return Threats(col(0, int), col(1, int), col(2, int), col(3, int), col(4, float), col(5, bool),
                   [r[6] for r in rows], [r[7] for r in rows])


def _origins(rng: np.random.Generator, cfg: GalaxyConfig, xy: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Home worlds: good planets, away from the core and the rim, spread apart."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    good = np.flatnonzero((quality >= np.quantile(quality, 0.75)) & (r > 0.15) & (r < 0.9))
    chosen: list[int] = []
    for star in rng.permutation(good):
        if all(np.hypot(*(xy[star] - xy[c])) >= cfg.origin_spacing for c in chosen):
            chosen.append(int(star))
        if len(chosen) == cfg.n_civs:
            break
    return np.array(chosen)


def make_galaxy(seed: int, cfg: GalaxyConfig = GalaxyConfig(), weights: SenseWeights = SenseWeights()) -> Galaxy:
    rng = np.random.default_rng(seed)
    xy = _disk(rng, cfg.n_stars, cfg.disk_scale)
    props, n_planets = _worlds(rng, xy, weights)
    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]).astype(np.float32)

    neighbours = []
    for i in range(cfg.n_stars):
        near = np.flatnonzero((dist[i] <= cfg.max_jump) & (np.arange(cfg.n_stars) != i))
        neighbours.append(near[np.argsort(dist[i, near])])

    hab = habitability(props["water"], props["energy"], props["resources"], weights)
    capacity = 0.3 + 0.7 * (hab - hab.min()) / (hab.max() - hab.min())
    threats = _threats(rng, cfg, dist)

    # Spread of each signal across the galaxy, for the 2 SD clip. Travel distance is taken as the
    # jump from each star's nearest neighbour, the shortest trip that could have settled it.
    nearest = np.array([dist[i, nb[0]] if len(nb) else cfg.max_jump for i, nb in enumerate(neighbours)])
    travel = np.clip(nearest / cfg.max_jump, 0, 1)
    bitter = hostility(props["radiation"], props["cold"], travel, weights)
    urgencies = [exp[exp > 0][:, None] / np.arange(1, w + 1)[None, :]
                 for exp, w in zip(threats.exposure, threats.warning)]
    senses = SenseMap.fit(hab, bitter, np.concatenate([u.ravel() for u in urgencies]))

    origins = _origins(rng, cfg, xy, hab - bitter)
    origin_epoch = np.sort(rng.integers(0, cfg.last_origin_epoch + 1, len(origins)))
    sun = int(np.argmin(np.hypot(xy[:, 0] - cfg.sun_radius, xy[:, 1])))
    return Galaxy(cfg, seed, xy, props, n_planets, capacity, dist, neighbours, threats, origins, origin_epoch, sun,
                  senses, weights)


def describe(g: Galaxy) -> dict:
    """Numbers worth knowing about a galaxy before running anything on it."""
    t, cfg = g.threats, g.cfg
    killed = np.zeros(g.n)
    for stars, exposure, sterilises in zip(t.stars, t.exposure, t.sterilises):
        if sterilises:
            killed[stars[exposure >= 1.0]] += 1
    gyr = cfg.n_epochs * cfg.epoch_myr / 1000.0
    return {
        "stars": g.n,
        "mean_stars_within_a_jump": float(np.mean([len(nb) for nb in g.neighbours])),
        "stars_with_no_neighbour_in_range": int(sum(len(nb) == 0 for nb in g.neighbours)),
        "events": {k: int((t.kind == i).sum()) for i, k in enumerate(KINDS)},
        "sterilisations_per_star_per_gyr": float(killed.mean() / gyr),
        "stars_sterilised_at_least_once": float((killed > 0).mean()),
        "civilizations": int(len(g.origins)),
        "looming_ref": g.senses.looming_ref,
        "clip_sd": CLIP_SD,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(make_galaxy(0)), indent=1))
