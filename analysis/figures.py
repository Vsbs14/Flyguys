"""Phase 7 figures, made from the saved result files.

  analysis/figures/sanity_table.png        the six sanity checks on the T4
  analysis/figures/encounter_outcomes.png  how first contacts end, per brain variant
  analysis/figures/baseline_ladder.png     fly vs hand rule vs neural net vs shuffled wiring
  analysis/figures/survival_curves.png     living colonies over time, per variant
  analysis/figures/effective_p.png         share of colonies that colonise, vs Landis's threshold
  analysis/figures/lesion_ranking.png      which cell types change the galaxy most

Each figure is skipped with a note if its input file does not exist yet. The
plotted numbers are saved as CSV next to each image.

  python -m analysis.figures
  python -m analysis.figures --results some/other/results --out some/dir
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical palette, fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD = "#006300", "#c92a2a"
EPOCH_MYR = 0.5
SERIES_STEP = 4
LANDIS_P = 0.311

# colour follows the entity: each policy keeps its slot in every figure
POLICY_ORDER = ["intact", "no_gf", "no_mdn", "no_gf_no_mdn", "no_bitter", "size_50", "size_20", "size_05"]
POLICY_LABEL = {
    "intact": "Intact", "no_gf": "No Giant Fibre", "no_mdn": "No MDN", "no_gf_no_mdn": "No GF, no MDN",
    "no_bitter": "No bitter (hungry)", "size_50": "50% of interneurons", "size_20": "20% of interneurons",
    "size_05": "5% of interneurons", "hand_rule": "Hand-written rule", "neural_net": "Small neural net",
    "shuffled": "Shuffled wiring", "fearless": "No Giant Fibre", "hungry": "No bitter (hungry)",
}
POLICY_COLOR = {p: CAT[i] for i, p in enumerate(POLICY_ORDER)}
OUTCOMES = ["coexist", "one_retreats", "mutual_flight", "escalation"]
OUTCOME_LABEL = {"coexist": "Coexist", "one_retreats": "One side retreats", "mutual_flight": "Mutual flight (dark forest)",
                 "escalation": "Escalation"}
OUTCOME_COLOR = dict(zip(OUTCOMES, CAT[:4]))


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED})


def _axes(ax, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig, title: str, subtitle: str, x: float = 0.06) -> None:
    fig.text(x, 0.945, title, fontsize=13, color=INK, weight="bold", va="top")
    fig.text(x, 0.875, subtitle, fontsize=9.5, color=INK_2, va="top")


def _save(fig, path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)


def _load(path: Path):
    if not path.exists():
        print(f"skip: {path} does not exist yet")
        return None
    return json.loads(path.read_text(encoding="utf8"))


def _base_rows(doc: dict) -> list[dict]:
    """Rows at the default mapping gains."""
    return [r for r in doc["rows"] if all(abs(g - 1.0) < 1e-9 for g in r.get("gains", [1, 1, 1]))]


def _policies(rows: list[dict], order: list[str] | None = None) -> list[str]:
    present = list(dict.fromkeys(r["policy"] for r in rows))
    order = order or POLICY_ORDER
    return [p for p in order if p in present] + [p for p in present if p not in order]


def _mean_se(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _label(p: str) -> str:
    return POLICY_LABEL.get(p, p)


# ----------------------------------------------------------------------------- 1. sanity table

def sanity_table(results: Path, out: Path) -> None:
    doc = _load(results / "sanity_intact_cuda.json")
    if doc is None:
        return
    checks, run = doc["result"]["checks"], doc["run"]
    _style()
    fig = plt.figure(figsize=(13, 4.6), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_pass = sum(c["passed"] for c in checks)
    _title(fig, f"The fly brain passes {n_pass} of {len(checks)} sanity checks",
           f"Intact brain, {doc['result']['replicates']} replicate brains per condition on a {run['device_name']}. "
           "Taste input 200 Hz, looming 150 Hz, 200 ms from rest.", x=0.04)
    cols = [(0.04, "Check"), (0.36, "Pass criterion"), (0.70, "Measured"), (0.915, "Result")]
    top, step = 0.74, 0.105
    for x, head in cols:
        ax.text(x, top, head, fontsize=9.5, color=MUTED, va="center")
    ax.plot([0.04, 0.97], [top - 0.045, top - 0.045], color=AXIS, linewidth=1)
    rows = []
    for i, c in enumerate(checks):
        y = top - step * (i + 1) + 0.02
        name = c["name"].replace(" -> ", "  →  ")
        ax.text(cols[0][0], y, name, fontsize=10.5, color=INK, va="center")
        ax.text(cols[1][0], y, c["criterion"], fontsize=9, color=INK_2, va="center")
        ax.text(cols[2][0], y, c["measured"].split(",")[0], fontsize=10, color=INK, va="center")
        ok = c["passed"]
        ax.text(cols[3][0], y, ("✓ Pass" if ok else "✗ Fail"), fontsize=10.5, weight="bold",
                color=GOOD if ok else BAD, va="center")
        if i < len(checks) - 1:
            ax.plot([0.04, 0.97], [y - step / 2, y - step / 2], color=GRID, linewidth=0.8)
        rows.append([c["name"], c["criterion"], c["measured"], "pass" if ok else "fail"])
    _save(fig, out / "sanity_table.png", ["check", "criterion", "measured", "result"], rows)


# ----------------------------------------------------------------------------- 2. encounter outcomes

def encounter_outcomes(results: Path, out: Path) -> None:
    doc = _load(results / "experiments" / "variants.json")
    if doc is None:
        return
    rows = _base_rows(doc)
    policies = _policies(rows)
    totals = {p: np.array([sum(r["encounters"].get(o, 0) for r in rows if r["policy"] == p) for o in OUTCOMES], float)
              for p in policies}
    per_run = {p: totals[p].sum() / max(sum(r["policy"] == p for r in rows), 1) for p in policies}
    n_seeds = len({r["seed"] for r in rows})
    _style()
    fig, ax = plt.subplots(figsize=(10, 0.52 * len(policies) + 2.3), facecolor=SURFACE)
    fig.subplots_adjust(left=0.2, right=0.86, top=1 - 1.25 / fig.get_figheight(), bottom=0.95 / fig.get_figheight())
    _axes(ax, grid="x")
    csv_rows = []
    for i, p in enumerate(policies):
        tot = totals[p].sum()
        share = totals[p] / tot if tot > 0 else np.zeros(len(OUTCOMES))
        left = 0.0
        for o, s in zip(OUTCOMES, share):
            ax.barh(i, s, left=left, height=0.62, color=OUTCOME_COLOR[o], edgecolor=SURFACE, linewidth=2)
            if s >= 0.08:
                ax.text(left + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if o == "escalation" else "#ffffff")
            left += s
            csv_rows.append([p, o, int(totals[p][OUTCOMES.index(o)]), f"{s:.4f}"])
        note = f"{per_run[p]:.1f} per run" if tot > 0 else "no encounters"
        ax.text(1.015, i, note, va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(policies)), [_label(p) for p in policies])
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of first contacts")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOR[o]) for o in OUTCOMES]
    fig.legend(handles, [OUTCOME_LABEL[o] for o in OUTCOMES], loc="lower left", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.06, 0.0), labelcolor=INK_2, handlelength=1.2, columnspacing=1.6)
    lead = {p: OUTCOMES[int(np.argmax(totals[p]))] for p in policies if totals[p].sum() > 0}
    if "intact" in lead:
        title = f"First contact between intact flies most often ends in: {OUTCOME_LABEL[lead['intact']].lower()}"
    else:
        title = "How first contacts end, by brain variant"
    fig.text(0.06, 1 - 0.28 / fig.get_figheight(), title, fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.06, 1 - 0.62 / fig.get_figheight(),
             f"Outcome of each first meeting of two civilizations (up to 3 rounds), pooled over {n_seeds} galaxy seeds.",
             fontsize=9.5, color=INK_2, va="top")
    _save(fig, out / "encounter_outcomes.png", ["policy", "outcome", "encounters", "share"], csv_rows)


# ----------------------------------------------------------------------------- 3. baseline ladder

def baseline_ladder(results: Path, out: Path) -> None:
    doc = _load(results / "experiments" / "ladder.json")
    if doc is None:
        return
    rows = _base_rows(doc)
    policies = _policies(rows, ["intact", "hand_rule", "neural_net", "shuffled"])
    n_seeds = len({r["seed"] for r in rows})
    panels = [("final_colonies", "Living colonies at the end"), ("final_civilizations", "Civilizations alive at the end")]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.13, wspace=0.22)
    csv_rows = []
    stats = {}
    for ax, (key, label) in zip(axes, panels):
        _axes(ax)
        for i, p in enumerate(policies):
            m, se = _mean_se([r[key] for r in rows if r["policy"] == p])
            stats[(key, p)] = (m, se)
            ax.bar(i, m, width=0.56, color=CAT[0] if p == "intact" else AXIS, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(i, m, yerr=se, color=INK_2, linewidth=1.2, capsize=3)
            ax.text(i, m + se + 0.02 * max(m, 1e-9), f"{m:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
            csv_rows.append([key, p, f"{m:.4f}", f"{se:.4f}", n_seeds])
        ax.set_xticks(range(len(policies)), [_label(p).replace(" ", "\n", 1) for p in policies])
        ax.tick_params(axis="x", labelcolor=INK_2)
        ax.set_title(label, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_ylim(0, None)
        ax.margins(y=0.15)
    title = "Does the fly matter? The fly against three simpler stand-ins"
    if ("final_colonies", "intact") in stats and ("final_colonies", "hand_rule") in stats:
        (mi, si), (mh, sh) = stats[("final_colonies", "intact")], stats[("final_colonies", "hand_rule")]
        if abs(mi - mh) <= 2 * np.hypot(si, sh):
            title = "The fly and a hand-written rule end with about the same number of colonies"
        else:
            title = f"The fly ends with {'more' if mi > mh else 'fewer'} colonies than a hand-written rule"
    _title(fig, title, f"Same galaxies for every policy. Mean ± standard error over {n_seeds} galaxy seeds. "
                       "The fly is highlighted.")
    _save(fig, out / "baseline_ladder.png", ["measure", "policy", "mean", "standard_error", "seeds"], csv_rows)


# ----------------------------------------------------------------------------- 4. survival curves

def survival_curves(results: Path, out: Path) -> None:
    doc = _load(results / "experiments" / "variants.json")
    if doc is None:
        return
    rows = _base_rows(doc)
    policies = _policies(rows)
    n_seeds = len({r["seed"] for r in rows})
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.74, top=0.78, bottom=0.12)
    _axes(ax)
    csv_rows, ends = [], []
    for p in policies:
        runs = np.array([r["series"]["colonies"] for r in rows if r["policy"] == p], dtype=float)
        t = np.arange(runs.shape[1]) * SERIES_STEP * EPOCH_MYR
        m = runs.mean(axis=0)
        se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(m)
        c = POLICY_COLOR.get(p, MUTED)
        ax.fill_between(t, m - se, m + se, color=c, alpha=0.16, linewidth=0)
        ax.plot(t, m, color=c, linewidth=2, label=_label(p))
        ends.append((m[-1], p, c))
        csv_rows += [[p, f"{ti:.1f}", f"{mi:.3f}", f"{si:.3f}"] for ti, mi, si in zip(t, m, se)]
    ax.set_xlabel("Time (million years)")
    ax.set_ylabel("Living colonies")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, labelcolor=INK_2)
    ends.sort(reverse=True)
    best, worst = ends[0], ends[-1]
    title = f"Most colonies at the end: {_label(best[1])}. Fewest: {_label(worst[1])}"
    _title(fig, title, f"Mean number of living colonies, band = ± standard error over {n_seeds} galaxy seeds. "
                       "Same star map and threats for every variant.", x=0.08)
    _save(fig, out / "survival_curves.png", ["policy", "time_myr", "mean_colonies", "standard_error"], csv_rows)


# ----------------------------------------------------------------------------- 5. effective P

def effective_p(results: Path, out: Path) -> None:
    doc = _load(results / "experiments" / "variants.json")
    if doc is None:
        return
    rows = _base_rows(doc)
    policies = _policies(rows)
    n_seeds = len({r["seed"] for r in rows})
    _style()
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(policies) + 2.2), facecolor=SURFACE)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.2, right=0.95, top=1 - 1.2 / h, bottom=0.6 / h)
    _axes(ax, grid="x")
    csv_rows, stats = [], {}
    for i, p in enumerate(policies):
        m, se = _mean_se([r["effective_p"] for r in rows if r["policy"] == p])
        stats[p] = m
        ax.plot([0, m], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.errorbar(m, i, xerr=se, color=INK_2, linewidth=1.2, capsize=3, zorder=2)
        ax.scatter(m, i, s=70, color=POLICY_COLOR.get(p, MUTED), edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m + se + 0.012, i, f"{m:.2f}", va="center", fontsize=9, color=INK)
        csv_rows.append([p, f"{m:.4f}", f"{se:.4f}", n_seeds])
    ax.axvline(LANDIS_P, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(LANDIS_P + 0.006, -0.72, f"Landis's percolation threshold, {LANDIS_P}", ha="left", va="center",
            fontsize=9, color=INK_2)
    ax.set_yticks(range(len(policies)), [_label(p) for p in policies])
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.set_ylim(len(policies) - 0.4, -1.0)
    ax.set_xlim(0, max(0.6, max(stats.values()) * 1.2 + 0.05))
    ax.set_xlabel("Effective P: share of colonies that go on to found a colony")
    above = [p for p in policies if stats[p] > LANDIS_P]
    if len(above) == len(policies):
        title = "Every brain variant colonises above Landis's percolation threshold"
    elif not above:
        title = "No brain variant reaches Landis's percolation threshold"
    else:
        title = f"{len(above)} of {len(policies)} brain variants colonise above Landis's percolation threshold"
    fig.text(0.06, 1 - 0.28 / h, title, fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.06, 1 - 0.62 / h, f"Mean ± standard error over {n_seeds} galaxy seeds. Landis's threshold is for a "
             "cubic lattice; our disk has more neighbours in range.", fontsize=9.5, color=INK_2, va="top")
    _save(fig, out / "effective_p.png", ["policy", "effective_p_mean", "standard_error", "seeds"], csv_rows)


# ----------------------------------------------------------------------------- 6. lesion ranking

def _lesion_rows(doc) -> list[tuple[str, float, float | None]]:
    """(cell type, change in final colonies, standard error or None) from whatever shape the ranking file has."""
    items = doc
    if isinstance(doc, dict):
        items = next((doc[k] for k in ("ranking", "lesions", "rows", "results") if isinstance(doc.get(k), list)), None)
        if items is None:
            items = [{"cell_type": k, **v} for k, v in doc.items() if isinstance(v, dict)]
    out = []
    for r in items:
        name = r.get("cell_type") or r.get("name") or r.get("lesion")
        key = next((k for k in r if ("delta" in k or "change" in k) and "colon" in k and "se" not in k.split("_")), None)
        key = key or next((k for k in r if "delta" in k or "change" in k), None)
        if name is None or key is None or name == "none":
            continue
        se_key = next((k for k in r if k != key and ("se" in k.split("_") or "error" in k) and
                       ("colon" in k or "delta" in k or "change" in k)), None)
        out.append((str(name), float(r[key]), float(r[se_key]) if se_key else None))
    return out


def lesion_ranking(results: Path, out: Path, top: int = 15) -> None:
    doc = _load(results / "lesion_ranking.json")
    if doc is None:
        return
    rows = _lesion_rows(doc)
    if not rows:
        print("skip: could not find cell types and changes in lesion_ranking.json")
        return
    rows.sort(key=lambda r: -abs(r[1]))
    shown = rows[:top]
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 0.38 * len(shown) + 2.1), facecolor=SURFACE)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.2, right=0.94, top=1 - 1.2 / h, bottom=0.6 / h)
    _axes(ax, grid="x")
    for i, (name, d, se) in enumerate(shown):
        ax.barh(i, d, height=0.62, color=CAT[0] if d < 0 else CAT[1], edgecolor=SURFACE, linewidth=2)
        if se:
            ax.errorbar(d, i, xerr=se, color=INK_2, linewidth=1.1, capsize=2.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(shown)), [r[0] for r in shown])
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.invert_yaxis()
    ax.set_xlabel("Change in living colonies at the end, against the intact brain")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT[0]), plt.Rectangle((0, 0), 1, 1, color=CAT[1])]
    ax.legend(handles, ["Fewer colonies", "More colonies"], loc="lower right", frameon=False, fontsize=9,
              labelcolor=INK_2)
    name, d, _ = shown[0]
    fig.text(0.06, 1 - 0.28 / h, f"Silencing {name} changes the galaxy most ({d:+.1f} colonies)", fontsize=13,
             color=INK, weight="bold", va="top")
    fig.text(0.06, 1 - 0.62 / h, f"One cell type silenced at a time. The {len(shown)} largest effects of "
             f"{len(rows)} cell types screened.", fontsize=9.5, color=INK_2, va="top")
    _save(fig, out / "lesion_ranking.png", ["cell_type", "change_in_final_colonies", "standard_error"],
          [[n, f"{d:.4f}", "" if se is None else f"{se:.4f}"] for n, d, se in rows])


FIGURES = [sanity_table, encounter_outcomes, baseline_ladder, survival_curves, effective_p, lesion_ranking]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args()
    for make in FIGURES:
        try:
            make(Path(args.results), Path(args.out))
        except Exception as e:                          # one broken input should not stop the other figures
            print(f"FAILED {make.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
