"""Phase 7 figures for the pitch, drawn from the saved results. No GPU needed.

  python -m analysis.figures          # writes analysis/*.png

  fig_dose_response.png   how strongly each sense has to fire before the fly acts (Phase 2)
  fig_lesion_screen.png   which silenced cell types change behaviour most (Phase 4)
  fig_response_map.png    the intact brain's choice across sugar x looming and sugar x bitter (Phase 3)
  fig_experiments.png     how well each decision maker keeps its colonies alive (Phase 6)
  fig_galaxy.png          the last frame of the recorded intact galaxy (Phase 5/7)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
RESULTS = ROOT / "results"

# Reference palette (dataviz skill), light mode. Categorical slots in fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#b9b8b2"
ACTIONS = ("idle", "feed", "escape", "retreat", "advance")
ACTION_COLOR = {"idle": NEUTRAL, "feed": SERIES[0], "escape": SERIES[1], "retreat": SERIES[2], "advance": SERIES[3]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def dose_response():
    doc = _load("decision_calibration.json")
    if not doc:
        return None
    driver = {"sugar": "feed", "looming": "escape", "obstacle": "retreat", "target": "advance"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = doc["result"]["dose_response"]
    for sense, out in driver.items():
        pts = [(r["dose"], r["action_fraction"][out]) for r in rows if r["sense"] == sense]
        x, y = zip(*pts)
        ax.plot(x, y, color=ACTION_COLOR[out], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("sense intensity (1 = 200 Hz on every neuron of the sense)")
    ax.set_ylabel("share of brains choosing the action")
    ax.set_title("How strong a sense must be before the fly acts")
    ax.legend(handles=[Patch(color=ACTION_COLOR[o], label=f"{s} → {o}") for s, o in driver.items()],
              loc="lower right", fontsize=8)
    return fig


def lesion_screen():
    doc = _load("lesion_screen.json")
    if not doc:
        return None
    top = doc["lesions"][:20][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(top))
    picked = set(doc.get("picked", []))
    slug = lambda ct: "no-" + "".join(ch if ch.isalnum() else "-" for ch in ct).strip("-").lower()
    colors = [SERIES[1] if slug(r["cell_type"]) in picked else SERIES[0] for r in top]
    ax.barh(y, [r["score"] for r in top], color=colors, height=0.7)
    ax.axvline(doc["cutoff"], color=INK2, lw=1, ls="--")
    ax.annotate("noise ceiling (best unlesioned run)", (doc["cutoff"], len(top) - 0.4), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_yticks(y, [f"{r['cell_type']} ({r['n_neurons']})" for r in top], fontsize=8)
    ax.set_xlabel("behaviour change when silenced (0 = none, 1 = completely different)")
    ax.set_title("Lesion screen: the 20 cell types that matter most")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=SERIES[1], label="picked as a galaxy variant"),
                       Patch(color=SERIES[0], label="other lesion")], loc="lower right", fontsize=8)
    return fig


def response_map():
    p = ROOT / "tables" / "intact.npz"
    if not p.exists():
        return None
    z = np.load(p)
    L = len(z["levels"])
    counts = z["counts"].reshape((L,) * 5 + (len(ACTIONS),))
    cmap = ListedColormap([ACTION_COLOR[a] for a in ACTIONS])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    panels = [("looming", (slice(None), 0, slice(None), 0, 0)), ("bitter", (slice(None), slice(None), 0, 0, 0))]
    for ax, (other, sl) in zip(axes, panels):
        m = counts[sl].argmax(-1).T                       # rows: other sense, cols: sugar
        ax.imshow(m, origin="lower", cmap=cmap, vmin=-0.5, vmax=len(ACTIONS) - 0.5,
                  extent=(-0.125, 1.125, -0.125, 1.125))
        ax.set_xlabel("sugar")
        ax.set_ylabel(other)
        ax.grid(False)
        ax.set_title(f"sugar × {other}", fontsize=10)
        lv = z["levels"]
        for i, xv in enumerate(lv):
            for j, yv in enumerate(lv):
                c = counts[sl][i, j]
                share = c.max() / c.sum()
                if share < 1:
                    ax.text(xv, yv, f"{share:.0%}", ha="center", va="center", fontsize=7, color=INK)
    fig.suptitle("What the intact fly brain chooses (other senses off)", x=0.02, ha="left", fontweight="bold")
    fig.legend(handles=[Patch(color=ACTION_COLOR[a], label=a) for a in ACTIONS], loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.02, 0.005, "Percent = share of replicate brains agreeing, where not unanimous.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def experiments():
    doc = _load("experiments.json")
    if not doc:
        return None
    hazards = list(doc["solo"])
    names = sorted(doc["solo"]["normal"], key=lambda n: -doc["solo"]["normal"][n]["mean_population"]["mean"])
    fig, axes = plt.subplots(1, len(hazards), figsize=(11, 4.6), sharey=True)
    for ax, hz in zip(axes, hazards):
        res = doc["solo"][hz]
        y = np.arange(len(names))[::-1]
        m = [res[n]["mean_population"]["mean"] for n in names]
        se = [res[n]["mean_population"]["se"] or 0 for n in names]
        colors = [SERIES[1] if n == "intact" else (NEUTRAL if n in ("random", "threshold", "shuffled") else SERIES[0])
                  for n in names]
        ax.barh(y, m, xerr=se, color=colors, height=0.68, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        ax.set_title(f"{hz} ({doc['run']['hazards'][hz]['n_raiders']} raiders)", fontsize=10)
        ax.set_yticks(y, names, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean population")
    fig.suptitle("Which decision maker keeps its colonies alive? (mean ± SE over paired seeds)",
                 x=0.02, ha="left", fontweight="bold")
    fig.legend(handles=[Patch(color=SERIES[1], label="intact fly brain"), Patch(color=SERIES[0], label="lesioned brain"),
                        Patch(color=NEUTRAL, label="control / no brain")], loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return fig


def galaxy():
    p = ROOT / "galaxy" / "frames.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf8"))
    stars = {s["id"]: s for s in doc["stars"]}
    frame = doc["frames"][-1]["galaxies"].get("intact")
    if frame is None:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_facecolor("#14141a")
    fig.set_facecolor("#14141a")
    for a, b in doc.get("lanes", []):
        ax.plot([stars[a]["x"], stars[b]["x"]], [stars[a]["y"], stars[b]["y"]], color="#3a3a46", lw=0.7, zorder=1)
    xs = [s["x"] for s in doc["stars"]]
    ys = [s["y"] for s in doc["stars"]]
    tox = [s.get("toxicity", 0) >= 0.5 for s in doc["stars"]]
    ax.scatter(xs, ys, s=18, c=["#7b5a8c" if t else "#9a9aa8" for t in tox], zorder=2, lw=0)
    for c in frame["colonies"]:
        s = stars[c["star"]]
        ax.scatter(s["x"], s["y"], s=30 + 4 * c.get("population", 10), color=ACTION_COLOR[c["brain_action"]],
                   edgecolor="#14141a", lw=1.5, zorder=3, alpha=0.95)
    raiders = [e["star"] for e in frame["events"] if e["type"] == "raider"]
    ax.scatter([stars[r]["x"] for r in raiders], [stars[r]["y"] for r in raiders], marker="x", s=70,
               color=SERIES[7], lw=2, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Intact fly galaxy at tick {doc['frames'][-1]['t']}", color="#f0efec", loc="left")
    handles = [Patch(color=ACTION_COLOR[a], label=f"colony: {a}") for a in ACTIONS] + [
        Patch(color="#7b5a8c", label="toxic star"), Patch(color=SERIES[7], label="raider")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8, labelcolor="#f0efec")
    return fig


FIGURES = {"fig_dose_response.png": dose_response, "fig_lesion_screen.png": lesion_screen,
           "fig_response_map.png": response_map, "fig_experiments.png": experiments, "fig_galaxy.png": galaxy}


def main() -> None:
    for name, fn in FIGURES.items():
        fig = fn()
        if fig is None:
            print(f"skip   {name} (its result file is missing)")
            continue
        fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote  analysis/{name}")


if __name__ == "__main__":
    main()
