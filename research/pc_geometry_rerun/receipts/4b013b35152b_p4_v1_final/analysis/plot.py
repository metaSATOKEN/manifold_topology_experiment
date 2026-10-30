"""Static publication-friendly figures from audited article summaries only."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pcgeom_p4_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "analysis/aggregates.json").read_text())["scientific_results"]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
KEYS = ["gpt2", "opt", "qwen"]
NAMES = {"gpt2": "GPT-2", "opt": "OPT-1.3B", "qwen": "Qwen2.5-1.5B"}
MID = {"gpt2": "block 06", "opt": "block 12", "qwen": "block 14"}
COLORS = {"gpt2": "#285080", "opt": "#bc5a19", "qwen": "#188879"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#25364a", "text.color": "#25364a",
                     "svg.fonttype": "none", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=190)
    fig.savefig(FIG / (name + ".svg"))
    plt.close(fig)


def interval(ax, x, s, color, horizontal=False, **kw):
    error = [[s["median"] - s["lo"]], [s["hi"] - s["median"]]]
    if horizontal:
        return ax.errorbar(s["median"], x, xerr=error, fmt="o", color=color,
                           capsize=3, markersize=6, **kw)
    return ax.errorbar(x, s["median"], yerr=error, fmt="o", color=color,
                       capsize=3, markersize=6, **kw)


fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
pairs = [(k, p) for k in KEYS for p in ["middle", "final_norm"]]
labels = [f"{NAMES[k]}  |  {MID[k] if p == 'middle' else 'final norm'}" for k, p in pairs]
for ax, metric, threshold, title in zip(
        axes, ["pc_gain", "advantage"], [.01, .005],
        ["PC gain over baseline", "PC gain minus mean random gain"]):
    for i, (key, point) in enumerate(pairs):
        v = D[key][point]
        s = v["metrics"]["within"]["W"][metric]
        interval(ax, i, s, COLORS[key], horizontal=True,
                 markerfacecolor=COLORS[key] if v["decision"] == "SUPPORTS_MINIMUM_CRITERION" else "white")
    ax.axvline(0, color="#7b8790", lw=.8)
    ax.axvline(threshold, color="#a83838", linestyle="--", lw=1.2,
               label=f"Minimum criterion = {threshold:g}")
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("Article median (R-squared difference)")
    ax.grid(axis="x", alpha=.16)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_yticks(range(6), labels)
axes[0].invert_yaxis()
fig.suptitle("P4 primary results: within-ID training (W)", fontsize=16, y=.97)
fig.text(.5, .025, "64 held-out articles per condition | 99.5833% bootstrap intervals, 12 primary endpoints\n"
         "Filled: both minimum criteria supported. Open: inconclusive. No condition is excluded.",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, .11, 1, .91], w_pad=2.0)
save(fig, "01_primary_intervals")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for row, metric in enumerate(["R1", "V1"]):
    for col, point in enumerate(["middle", "final_norm"]):
        ax = axes[row, col]
        for i, key in enumerate(KEYS):
            a = D[key][point]["geometry"]["C0"][metric]
            b = D[key][point]["geometry"]["C1"][metric]
            ax.plot([i - .14, i + .14], [a["median"], b["median"]], color=COLORS[key], lw=2)
            interval(ax, i - .14, a, COLORS[key], markerfacecolor="white")
            interval(ax, i + .14, b, COLORS[key])
        ax.set_xticks(range(3), [NAMES[k] for k in KEYS], fontsize=10)
        ax.set_ylim(-.04, 1.06)
        ax.grid(axis="y", alpha=.16)
        ax.set_ylabel("|corr(PC1 score, norm)|" if metric == "R1" else "PC1 variance fraction")
        ax.set_title(("Middle block" if point == "middle" else "After final normalization")
                     + (" — norm association" if row == 0 else " — variance concentration"), fontsize=12)
fig.suptitle("Per-article geometry and first-token sensitivity", fontsize=16, y=.98)
fig.text(.5, .025, "Open/left: C0, all tokens. Filled/right: C1, first token removed.\n"
         "Article PCA is recomputed for each condition. Descriptive 95% intervals, n = 64.",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, .095, 1, .94], h_pad=2.2)
save(fig, "02_geometry_first_token")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.9), sharey=True)
for ax, point in zip(axes, ["middle", "final_norm"]):
    for j, key in enumerate(KEYS):
        xs = np.arange(3) + (j - 1) * .08
        ss = [D[key][point]["metrics"]["within"][name]["advantage"] if name != "W"
              else D[key][point]["within_W_advantage_95_descriptive"] for name in ["O", "R", "W"]]
        ax.plot(xs, [s["median"] for s in ss], color=COLORS[key], label=NAMES[key])
        for x, s in zip(xs, ss):
            interval(ax, x, s, COLORS[key])
    ax.axhline(0, color="#667786", lw=.8)
    ax.set_xticks(range(3), ["O\nAll-token training", "R\nRepeated-ID training", "W\nWithin-ID training"], fontsize=10)
    ax.set_title("Middle block" if point == "middle" else "After final normalization", pad=14)
    ax.grid(axis="y", alpha=.16)
axes[0].set_ylabel("PC gain minus mean random gain\nWithin-article x ID evaluation")
axes[1].legend(fontsize=10)
fig.suptitle("Readout design changes the PC–random comparison", fontsize=15, y=.98)
fig.text(.5, .025, "All intervals here are descriptive 95% intervals; primary decisions use Figure 1.\n"
         "R keeps O penalties; W changes the objective and selects its own penalties. This is not a causal ablation.",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, .13, 1, .92], w_pad=2)
save(fig, "03_readout_sensitivity")

fig, ax = plt.subplots(figsize=(10, 6.3))
for i, (key, point) in enumerate(pairs):
    for off, q, marker in [(-.12, "top8_total_fraction", "o"), (.12, "top8_within_fraction", "s")]:
        s = {k: v * 100 if k in ("median", "lo", "hi") else v
             for k, v in D[key][point]["shared_top8_capture"][q].items()}
        ax.errorbar(s["median"], i + off,
                    xerr=[[s["median"] - s["lo"]], [s["hi"] - s["median"]]],
                    fmt=marker, color=COLORS[key], capsize=3, markersize=6,
                    markerfacecolor="white" if off < 0 else COLORS[key])
ax.set_yticks(range(6), labels)
ax.invert_yaxis()
ax.set_xlim(0, 104)
ax.set_xlabel("Variance captured by the same training top 8 PCs (%)")
ax.grid(axis="x", alpha=.16)
ax.plot([], [], "o", markerfacecolor="white", color="#46596c", label="Total article variation")
ax.plot([], [], "s", color="#46596c", label="Within article x ID variation")
ax.legend(loc="lower right", fontsize=10)
fig.suptitle("Shared basis transfer: variance capture depends on model and layer", fontsize=14, y=.97)
fig.text(.5, .025, "C1 only; bases are fixed using training articles. Test articles do not refit PCA.\n"
         "Medians and descriptive 95% bootstrap intervals over 64 held-out articles.",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, .11, 1, .92])
save(fig, "04_shared_variance")
