"""Static scientific figures from saved JSON only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
d = json.loads((HERE/"results.json").read_text())["points"]
OUT = HERE/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180})
colors = ["#2166ac", "#b35806"]


def finish(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(OUT/f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
for ax, (point, r), color in zip(axes, d.items(), colors):
    keys = ["O", "R", "W_fixed", "W_CV"]
    stats = [r["models"][k]["advantage"] for k in keys]
    y = np.array([s["median"] for s in stats])
    e = np.array([[s["median"]-s["lower"], s["upper"]-s["median"]] for s in stats]).T
    ax.axhline(0, color="#555555", lw=1)
    ax.errorbar(np.arange(4), y, yerr=e, color=color, marker="o", capsize=5, lw=2)
    for i, s in enumerate(stats):
        ax.annotate(f"{s['median']:+.4f}", (i, s["upper"]+.002), ha="center", fontsize=10)
    ax.set_xticks(range(4), ["Original\nfrozen", "Repeated-only\nordinary", "Within-ID\nfixed penalty", "Within-ID\nCV penalty"])
    ax.set_title(point)
    ax.set_ylim(-.027, .042)
    ax.grid(axis="y", alpha=.18)
axes[0].set_ylabel("PC advantage over mean of 20 random readouts\n(within-ID R² difference)")
fig.suptitle("The final-layer reversal changes with readout training", fontsize=15)
fig.supxlabel("Medians and descriptive 95% article-bootstrap intervals; 64 previously seen P3 articles", fontsize=10)
finish(fig, "01_readout_reversal")

fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
keys = ["top8_total_capture", "top8_between_capture", "top8_within_capture"]
for j, (point, r) in enumerate(d.items()):
    s = [r["geometry"][k] for k in keys]
    values = np.array([z["median"]*100 for z in s])
    xpos = np.arange(3)+(j-.5)*.32
    e = np.array([[z["median"]-z["lower"], z["upper"]-z["median"]] for z in s]).T*100
    ax.bar(xpos, values, width=.30, color=colors[j], label=point, alpha=.88)
    ax.errorbar(xpos, values, yerr=e, fmt="none", color="#222222", capsize=4)
    for x, y in zip(xpos, values):
        ax.text(x, y+2, f"{y:.2f}%", ha="center", fontsize=10)
ax.set_xticks(range(3), ["All variation", "Between IDs", "Within the same ID"])
ax.set_ylim(0, 110)
ax.set_ylabel("Share of squared state variation captured by top 8 PCs (%)")
ax.set_title("Final-layer PCs retain within-ID state variation", fontsize=15)
ax.legend(loc="center left")
ax.grid(axis="y", alpha=.18)
fig.supxlabel("Same repeated-token population for all decompositions; this is variance, not semantic information", fontsize=10)
finish(fig, "02_variance_capture")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, layout="constrained")
for ax, (point, r), color in zip(axes, d.items(), colors):
    s = [r["models"]["W_CV"]["gain"], r["models"]["complement"]["mean_gain"]]
    y = np.array([v["median"] for v in s])
    e = np.array([[v["median"]-v["lower"], v["upper"]-v["median"]] for v in s]).T
    ax.bar(range(2), y, color=[color, "#8d9da8"], width=.55)
    ax.errorbar(range(2), y, yerr=e, fmt="none", color="#222222", capsize=5)
    for i, v in enumerate(s):
        ax.text(i, v["upper"]+.004, f"{v['median']:.4f}", ha="center")
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks([0, 1], ["Top 8 PCs", "8 directions outside top 8\n(mean of 20 readouts)"])
    ax.set_title(point)
    ax.set_ylim(0, .16)
    ax.grid(axis="y", alpha=.18)
axes[0].set_ylabel("Within-ID R² gain over the same control baseline")
fig.suptitle("Some readable association remains outside the top 8 PCs", fontsize=15)
fig.supxlabel("All readouts trained on old P2 training articles; descriptive 95% intervals on known P3 articles", fontsize=10)
finish(fig, "03_complement")
print("Saved 3 PNG/SVG figures.")
