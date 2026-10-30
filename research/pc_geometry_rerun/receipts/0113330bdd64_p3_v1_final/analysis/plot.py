"""Scientific figures from audited saved data; all article points remain visible."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent.parent
DATA = json.loads((ROOT/"analysis/aggregates.json").read_text())
OLD = json.loads((RESEARCH/"exploratory/p2_lexical_v1/article_results.json").read_text())
OUT = ROOT/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.facecolor": "white"})
COLORS = ["#23758b", "#cb763b"]


def save(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=190, bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"), bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
titles = ["Ordinary: PC gain over ID", "Ordinary: advantage over random",
          "Within token ID: PC gain over ID", "Within token ID: advantage over random"]
for ax, key, threshold, title in zip(axes.flat, "ABCD", [.02, .01, .01, .005], titles):
    e = DATA["primary_endpoints"][key]
    med, lo, hi = e["median"], e["lower"], e["upper"]
    ax.errorbar(med, 0, xerr=[[med-lo], [hi-med]], fmt="o", ms=7,
                color=COLORS[0], lw=2.6, capsize=5)
    ax.axvline(threshold, color=COLORS[1], ls="--", lw=1.4)
    ax.text(threshold, .34, f"Criterion {threshold:g}", ha="center", fontsize=9)
    ax.text(.5, .11, f"{med:.4f}  [{lo:.4f}, {hi:.4f}]",
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.set(xlim=(0, hi*1.30), ylim=(-.55, .55), yticks=[], title=f"{key}   {title}",
           xlabel="Article-median normalized error reduction")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.15)
fig.suptitle("P3 primary result: H1 supported; H2 supported", fontsize=15)
fig.tight_layout(rect=(0, .065, 1, .94))
fig.text(.5, .015, "block06 / C1; 64 fresh articles; frozen readouts. 10,000 article resamples; "
         "98.75% interval per endpoint.", ha="center", fontsize=9)
save(fig, "01_primary_intervals")


def distribution(ax, values, labels, colors):
    positions = np.arange(len(values))
    box = ax.boxplot(values, positions=positions, widths=.42, patch_artist=True,
                     showfliers=False, medianprops={"color": "#182c3c", "linewidth": 1.7})
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=.18)
    rng = np.random.default_rng(20261020)
    for x, v, color in zip(positions, values, colors):
        ax.scatter(x+rng.uniform(-.14, .14, len(v)), v, s=14,
                   color=color, alpha=.6, linewidths=0, zorder=3)
    ax.axhline(0, color="#77828d", lw=.8)
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=.15)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
for row, (mode, mode_title) in enumerate([("ordinary", "Ordinary"),
                                       ("within_id", "Within article × token ID")]):
    for col, (metric, label) in enumerate([("gain", "PC gain over ID"),
                                          ("random_advantage", "PC advantage over random")]):
        values = [[r[mode][metric] for r in DATA["article_metrics"] if r["point"] == point]
                  for point in ["block06", "final_ln"]]
        distribution(axes[row, col], values, ["block06\nPrimary", "final_ln\nSecondary"], COLORS)
        axes[row, col].set(title=f"{mode_title}: {label}", ylabel="Normalized error reduction")
fig.suptitle("Within-ID specificity differs by extraction point", fontsize=15)
fig.tight_layout(rect=(0, .07, 1, .94))
fig.text(.5, .018, "All 64 article points per group. Boxes show median and IQR, not confidence "
         "intervals. Negative differences are retained.", ha="center", fontsize=9)
save(fig, "02_article_distributions")


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for col, point in enumerate(["block06", "final_ln"]):
    for row, (old_key, new_key, label) in enumerate([
        ("pc_gain", "gain", "PC gain over baseline"),
        ("random_advantage", "random_advantage", "PC advantage over random")]):
        values = [[r[old_key] for r in OLD if r["point"] == point and
                   r["condition"] == "C1" and r["family"] == family] for family in ["original", "ID"]]
        values.append([r["ordinary"][new_key] for r in DATA["article_metrics"] if r["point"] == point])
        ax = axes[row, col]
        distribution(ax, values, ["P2 original\nKnown articles", "P2 ID (EDA)\nKnown articles",
                                  "P3 fixed ID\nFresh articles"], ["#82929c", "#508ca0", "#228768"])
        ax.set(title=f"{point}: {label}", ylabel="Ordinary normalized error reduction")
        ax.axvline(1.5, color="#c4cbd0", ls=":", lw=1)
fig.suptitle("Frozen ID-controlled gains recur on new articles", fontsize=15)
fig.tight_layout(rect=(0, .07, 1, .94))
fig.text(.5, .018, "C1 only; 64 articles per group. P2 and P3 use different evaluation articles. "
         "Comparisons across cohorts are descriptive.", ha="center", fontsize=9)
save(fig, "03_previous_and_fresh_articles")
print("Rendered 3 PNG and 3 SVG figures.")
