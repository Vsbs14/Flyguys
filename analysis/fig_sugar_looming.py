"""Phase 3 figures: the "habitable world next to a dying star" dilemma.

Two figures, each intact vs Giant Fibre silenced, over sugar x looming with no
bitter input:

  analysis/figures/flee_probability.png   probability that the Giant Fibre spikes (the colony flees)
  analysis/figures/feeding_rate.png       MN9 firing rate (the colony keeps eating)

The plotted numbers are also saved as CSV next to each image.

  python -m analysis.fig_sugar_looming
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from brain.response import ResponseTable

OUT = Path(__file__).resolve().parent / "figures"

# One-hue sequential ramp, light -> dark; the lightest step recedes toward the surface.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE)

PANELS = [("intact", "Intact brain"), ("no_gf", "Giant Fibre silenced")]


def _fmt(hz: float) -> str:
    return f"{hz:g}"


def heatmaps(values: dict[str, np.ndarray], sugar: np.ndarray, looming: np.ndarray, title: str, subtitle: str,
             cbar_label: str, vmax: float, path: Path, note_if_flat: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.89, top=0.78, bottom=0.14, wspace=0.07)
    mesh = None
    for ax, (key, label) in zip(axes, PANELS):
        v = values[key]
        ax.set_facecolor(SURFACE)
        # 2 px surface-coloured gaps between cells
        mesh = ax.pcolormesh(np.arange(len(looming) + 1), np.arange(len(sugar) + 1), v, cmap=CMAP, vmin=0, vmax=vmax,
                             edgecolors=SURFACE, linewidth=1.5)
        ax.set_title(label, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xticks(np.arange(len(looming)) + 0.5, [_fmt(x) for x in looming])
        ax.set_yticks(np.arange(len(sugar)) + 0.5, [f"{y:.0f}" for y in sugar])
        ax.set_xlabel("Looming input (Hz), uneven steps", color=INK_2)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
        if np.allclose(v, v.flat[0]):
            ax.text(len(looming) / 2, len(sugar) / 2, note_if_flat.format(value=v.flat[0]), ha="center", va="center",
                    color=INK_2, fontsize=11)
    axes[0].set_ylabel("Sugar input (Hz)", color=INK_2)
    cax = fig.add_axes([0.905, 0.14, 0.014, 0.64])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0, labelsize=9)
    cb.set_label(cbar_label, color=INK_2)
    fig.text(0.075, 0.935, title, fontsize=13.5, color=INK, weight="bold")
    fig.text(0.075, 0.875, subtitle, fontsize=10, color=INK_2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "sugar_hz", "looming_hz", cbar_label])
        for key, _ in PANELS:
            for i, s in enumerate(sugar):
                for k, l in enumerate(looming):
                    w.writerow([key, f"{s:.1f}", _fmt(l), f"{values[key][i, k]:.4f}"])
    print("wrote", path)


def main() -> None:
    tables = {key: ResponseTable.load(key) for key, _ in PANELS}
    sugar, _, looming = tables["intact"].axes
    seeds = 5

    p_flee = {k: t.p_gf[:, 0, :] for k, t in tables.items()}                 # bitter = 0 slice
    edge = looming[np.argmax(p_flee["intact"].mean(axis=0) >= 0.5)]
    heatmaps(p_flee, sugar, looming,
             title=f"The intact fly flees once looming passes about {_fmt(edge)} Hz, however sweet the world",
             subtitle=f"Probability that the Giant Fibre spikes within a 200 ms decision. Bitter input 0 Hz, {seeds} runs per cell.",
             cbar_label="P(Giant Fibre spikes)", vmax=1.0, path=OUT / "flee_probability.png",
             note_if_flat="{value:.0f} in every cell: it never flees")

    mn9 = {k: t.mean[:, 0, :, 0] for k, t in tables.items()}
    vmax = float(np.ceil(max(v.max() for v in mn9.values()) / 10) * 10)
    heatmaps(mn9, sugar, looming,
             title="Looming switches feeding off, with or without the Giant Fibre",
             subtitle=f"Mean MN9 firing rate over a 200 ms decision. Bitter input 0 Hz, {seeds} runs per cell.",
             cbar_label="MN9 rate (Hz)", vmax=vmax, path=OUT / "feeding_rate.png",
             note_if_flat="{value:.0f} Hz in every cell")


if __name__ == "__main__":
    main()
