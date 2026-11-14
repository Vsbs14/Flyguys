"""Replay paired galaxy experiments using cached fly responses and a threshold rule."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from galaxy.responses import LESIONS, OUTPUTS, SENSES, TABLE
from viz.frames import FrameWriter

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (*LESIONS, "threshold")


@dataclass
class Colony:
    population: float = 0.35
    alarm: float = 0.0


def world(seed, count, epochs):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, count)
    radius = np.sqrt(rng.random(count))
    xy = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    positions = [{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(xy)]
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    habitat = rng.uniform(0.35, 1, count)
    hostility = rng.uniform(0, 0.65, count)
    explosions = np.full(count, epochs + 100)
    supernovae = [[] for _ in range(epochs)]
    scheduled = [(int(rng.integers(5, epochs)), int(star))
                 for star in rng.choice(count, max(1, count // 4), replace=False)]
    for epoch, star in sorted(scheduled):
        if explosions[star] < epoch:
            continue
        supernovae[epoch].append(star)
        within = distance[star] <= 0.16
        explosions[within] = np.minimum(explosions[within], epoch)
    observer = np.linalg.norm(xy - [0.5, 0], axis=1) <= 0.45
    draws = rng.random((epochs, count, 4))
    return positions, distance, habitat, hostility, explosions, supernovae, observer, draws


def response(table, signals, draws, scale):
    if table is None:
        sugar, bitter, looming = signals
        return (float(sugar >= 0.5 and bitter <= 0.5), float(looming >= 0.6),
                float(bitter >= 0.5), float(sugar >= 0.5 and bitter <= 0.5)), None
    continuous = signals * (table.shape[0] - 1)
    lower = continuous.astype(int)
    index = lower + (draws[:3] < continuous - lower)
    replicate = int(draws[3] * table.shape[3])
    raw = table[tuple(index) + (replicate,)]
    drive = np.clip(raw[:4] / scale, 0, 1)
    drive[1] = raw[4]
    return drive, dict(zip(OUTPUTS.values(), map(float, raw[:4])))


def run(variant, scenario, table, scale, epochs, gains, writer=None):
    _, distance, habitat, hostility, explosions, supernovae, observer, draws = scenario
    home = int(np.argmax(habitat - hostility - (explosions < epochs)))
    colonies = {home: Colony()}
    fleets = []
    settled = np.zeros(epochs)
    extinction = None
    for t in range(epochs):
        events = [{"type": "supernova", "star": int(i)} for i in supernovae[t]]
        colonies = {i: c for i, c in colonies.items() if explosions[i] > t}
        traveling = []
        for source, target, start, arrival, passengers in fleets:
            if explosions[target] <= t:
                if source in colonies:
                    colonies[source].alarm = max(colonies[source].alarm, 0.5)
            elif t == arrival:
                colonies.setdefault(target, Colony(passengers))
            else:
                traveling.append((source, target, start, arrival, passengers))
        fleets = traveling
        snapshots = []
        abandoned = []
        occupied = set(colonies) | {f[1] for f in fleets}
        for star, colony in sorted(colonies.items()):
            colony.population -= 0.12 * hostility[star] ** 2
            if colony.population <= 0:
                abandoned.append(star)
                continue
            looming = (min(1, 3 / (explosions[star] - t)) if explosions[star] < epochs else 0)
            signals = np.clip(np.array([habitat[star], hostility[star], looming + colony.alarm]) * gains, 0, 1)
            colony.alarm *= 0.7
            (feed, escape, retreat, advance), rates = response(table, signals, draws[t, star], scale)
            action = "settle"
            if escape:
                action = "flee"
                destinations = []
                for target, separation in enumerate(distance[star]):
                    arrival = t + max(2, int(np.ceil(separation * 12)))
                    if (target not in occupied and separation <= 0.38
                            and explosions[target] >= epochs and hostility[target] <= 0.4):
                        destinations.append((separation, target, arrival))
                if destinations:
                    _, target, arrival = min(destinations)
                    fleets.append((star, target, t, arrival, colony.population))
                    occupied.add(target)
                    abandoned.append(star)
                else:
                    action = "retreat"
            else:
                colony.population = min(1, colony.population + 0.22 * feed * (1 - retreat))
                candidates = [int(i) for i in np.argsort(distance[star])
                              if i not in occupied and explosions[i] > t and distance[star, i] <= 0.38]
                if retreat >= 0.5:
                    action = "retreat"
                elif colony.population * (0.45 * feed + 0.55 * advance) >= 0.22 and candidates:
                    target = max(candidates[:5], key=lambda i: habitat[i] - hostility[i])
                    arrival = t + max(2, int(np.ceil(distance[star, target] * 12)))
                    fleets.append((star, target, t, arrival, 0.35))
                    occupied.add(target)
                    colony.population *= 0.7
                    action = "expand"
            if writer:
                snapshot = {"id": star, "star": star, "pop": colony.population, "action": action,
                            "inputs": dict(zip(SENSES, map(float, signals)))}
                if rates is not None:
                    snapshot["rates"] = rates
                snapshots.append(snapshot)
        for star in abandoned:
            del colonies[star]
        settled[t] = len(colonies) / len(habitat)
        if not colonies and not fleets and extinction is None:
            extinction = t
        if writer:
            ships = [{"from": a, "to": b, "progress": (t - depart) / (arrive - depart)}
                     for a, b, depart, arrive, _ in fleets]
            writer.add(t, variant, snapshots, ships, events)
    return {"settled_fraction": settled, "survivors": len(colonies),
            "extinction_epoch": extinction,
            "detectable_colonies": sum(bool(observer[i]) and c.population >= 0.5 for i, c in colonies.items())}


def summary(values):
    data = np.asarray(values)
    return {"mean": data.mean(axis=0).tolist(),
            "se": (data.std(axis=0, ddof=1) / np.sqrt(len(data))).tolist() if len(data) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--stars", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--output", type=Path, default=ROOT / "galaxy" / "frames.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "galaxy_experiment.json")
    args = parser.parse_args()
    if args.seed < 0 or args.seeds < 1 or args.stars < 4 or args.epochs < 6:
        parser.error("require nonnegative seed, at least one replicate, four stars and six epochs")
    with np.load(args.table) as archive:
        tables = {variant: archive[variant] for variant in LESIONS}
        table_seed = int(archive["seed"])
    shape = tables["intact"].shape
    if (len(shape) != 5 or shape[:3] != (shape[0],) * 3
            or shape[0] < 2 or shape[3] < 1 or shape[4] != 5
            or any(table.shape != shape or not np.isfinite(table).all() for table in tables.values())):
        parser.error("response cache must contain complete, equally sized variant tables")
    scale = np.maximum(tables["intact"][..., :4].max(axis=(0, 1, 2, 3)), 1)
    scenarios = [world(seed, args.stars, args.epochs) for seed in range(args.seed, args.seed + args.seeds)]
    writer = FrameWriter(list(VARIANTS), args.seed, scenarios[0][0])
    writer.meta.update(source="fly-response-table", levels=tables["intact"].shape[0])
    report = {}
    paired = {}
    gains = (0.8, 1.0, 1.2)
    for gain in gains:
        results = {}
        survivors = {}
        for variant in VARIANTS:
            runs = [run(variant, scenario, tables.get(variant), scale, args.epochs, gain,
                        writer if gain == 1 and i == 0 else None)
                    for i, scenario in enumerate(scenarios)]
            survivors[variant] = np.array([r["survivors"] for r in runs])
            extinctions = [r["extinction_epoch"] for r in runs if r["extinction_epoch"] is not None]
            results[variant] = {k: summary([r[k] for r in runs])
                                for k in ("settled_fraction", "survivors", "detectable_colonies")}
            results[variant]["extinction_fraction"] = summary([r["extinction_epoch"] is not None for r in runs])
            results[variant]["extinction_epoch_if_extinct"] = summary(extinctions) if extinctions else None
        report[str(gain)] = results
        paired[str(gain)] = {v: summary(values - survivors["intact"])
                             for v, values in survivors.items() if v != "intact"}
    writer.save(args.output)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({
        "config": {"seed": args.seed, "seeds": args.seeds, "stars": args.stars, "epochs": args.epochs,
                   "response_levels": tables["intact"].shape[0], "response_replicates": tables["intact"].shape[3],
                   "response_seed": table_seed, "rate_scale": dict(zip(OUTPUTS.values(), map(float, scale)))},
        "assumptions": {
            "supernova_damage_radius": 0.16,
            "hostility_loss_per_epoch": "0.12 * hostility squared",
            "evacuation": "Nearest unoccupied star within 0.38 units with hostility at most 0.4 and no scheduled destruction; hold if none",
            "survivors": "Settled colonies at the final epoch; fleets in transit are excluded",
            "observer_position": [0.5, 0],
            "observer_radius": 0.45,
            "observer_minimum_population": 0.5,
            "threshold_rule": {"sugar_at_least": 0.5, "bitter_at_most": 0.5, "looming_escape_at_least": 0.6},
        },
        "gains": report,
        "paired_survivor_difference_from_intact": paired,
        "escape_response_at_zero_bitter": tables["intact"][:, 0, :, :, 4].mean(axis=2).tolist(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
