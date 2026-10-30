"""Scientific figures from committed summary values only."""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/pcgeom_eda3_mpl"
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
S = json.loads((OUT / "summary.json").read_text())["results"]
R = json.loads((OUT / "report_data.json").read_text())["results"]
POINTS = [(m, p) for m in ["gpt2", "opt", "qwen"] for p in ["middle", "final_norm"]]
LABELS = ["GPT-2 middle", "GPT-2 final LN", "OPT middle", "OPT final LN", "Qwen middle", "Qwen final RMSNorm"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.facecolor": "white"})
BLUE, ORANGE, GREEN = "#167096", "#ce7020", "#33866b"


def forest(ax, series, title):
    offsets = np.linspace(-.22, .22, len(series)) if len(series) > 1 else [0]
    for offset, (label, name, color, column) in zip(offsets, series):
        stats = [S[m][p][column][name] for m, p in POINTS]
        mid = np.array([d["estimate"] for d in stats])
        lo, hi = np.array([d["lo"] for d in stats]), np.array([d["hi"] for d in stats])
        ax.errorbar(mid, np.arange(6) + offset, xerr=[mid - lo, hi - mid],
                    fmt="o", ms=4.5, capsize=2, color=color, label=label, lw=1.3)
    ax.axvline(0, lw=.8, c="#777777")
    ax.set_yticks(np.arange(6), LABELS)
    ax.set_ylim(5.6, -.65)
    ax.set_title(title, fontsize=12, loc="left", pad=14)
    ax.set_xlabel("Median article ΔR² (descriptive 95% interval)")
    ax.grid(axis="x", alpha=.15)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=.95)


def save(fig, name, note):
    fig.text(.02, .02, note, fontsize=9, color="#444444")
    fig.savefig(FIG / (name + ".png"), dpi=175, bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=.13, right=.99, wspace=.5, bottom=.15, top=.85)
fig.suptitle("EDA-0003 | PC1 and PC2–8 after position and log-norm controls", fontsize=15, y=.97)
forest(axes[0], [("PC1 (1D)", "A1_gain", ORANGE, "metrics"),
                 ("PC2–8 (7D)", "A7_gain", BLUE, "metrics"),
                 ("PC1–8 (8D)", "A8_gain", GREEN, "metrics")], "A. Increment beyond baseline B")
forest(axes[1], [("Add PC2–8 to PC1", "A8_minus_A1", BLUE, "metrics"),
                 ("Add PC1 to PC2–8", "A8_minus_A7", ORANGE, "metrics")], "B. Paired full-minus-reduced readouts")
save(fig, "01_components", "64 articles per condition; known P4 evaluation data. Intervals are unadjusted; no new confirmatory decisions.")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=.13, right=.99, wspace=.5, bottom=.15, top=.85)
fig.suptitle("EDA-0003 | Length controls and direction-related association", fontsize=15, y=.97)
forest(axes[0], [("Linear log-norm: B − P", "B_minus_P", ORANGE, "metrics"),
                 ("Five norm hinges: N − B", "N_minus_B", BLUE, "metrics")], "A. Increment from modeling length")
forest(axes[1], [("Raw PC8 beyond N", "NR_gain", ORANGE, "metrics"),
                 ("Unit-state PC8 beyond N", "NU_gain", BLUE, "metrics"),
                 ("Unit-state PC8 − random8", "NU_adv", GREEN, "metrics")], "B. After flexible length controls")
save(fig, "02_radius_direction", "N = position + log-norm + five fixed-knot hinges. Unit-state PCA is a different basis; differences are not causal length effects.")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=.13, right=.99, wspace=.5, bottom=.15, top=.85)
fig.suptitle("EDA-0003 | Does repeated-token weighting explain the median gains?", fontsize=15, y=.97)
forest(axes[0], [("Equal token weights", "A8_gain", BLUE, "metrics"),
                 ("Equal ID-group weights", "A8_gain", ORANGE, "equal_ID_metrics")], "A. Original raw PC8 beyond B")
forest(axes[1], [("Equal token weights", "NU_gain", BLUE, "metrics"),
                 ("Equal ID-group weights", "NU_gain", ORANGE, "equal_ID_metrics")], "B. Unit-state PC8 beyond N")
save(fig, "03_equal_id", "Same predictions, different scoring weights; no refitting. Positive medians do not rule out severe failures in individual articles.")

rows = json.loads((OUT / "results/qwen/middle/articles.json").read_text())
raw = np.array([r["metrics"]["A8_gain"] for r in rows])
unit = np.array([r["metrics"]["NU_gain"] for r in rows])
bad = int(raw.argmin())
fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), gridspec_kw={"width_ratios": [1, 1, 1.15]})
fig.subplots_adjust(left=.065, right=.99, wspace=.37, bottom=.22, top=.80)
fig.suptitle("EDA-0003 | Qwen middle: a positive median can conceal a large failure", fontsize=14, y=.97)
for ax in axes[:2]:
    ax.scatter(np.arange(64), raw, s=16, color=ORANGE, alpha=.8, label="Raw PC8 beyond B")
    ax.scatter(np.arange(64), unit, s=16, color=BLUE, alpha=.8, label="Unit-state PC8 beyond N")
    ax.axhline(0, color="#777777", lw=.8)
    ax.set_xlabel("Article index (all 64 retained)")
    ax.set_ylabel("Article ΔR²")
axes[0].set_title("A. Full range: no exclusions", loc="left")
axes[0].annotate("wt2_train_27981\nraw gain = −24.225", (bad, raw[bad]),
                 xytext=(2, -15), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.04))
axes[1].set_title("B. Zoom near zero", loc="left")
axes[1].set_ylim(-.32, .23)
axes[1].text(.02, .03, "1 raw point below range\n(shown in panel A)", transform=axes[1].transAxes, fontsize=8.5)
strata = S["qwen"]["middle"]["bias"]["type"]
x = np.arange(len(strata))
for shift, name, color, label in [(-.18, "B_to_A8", ORANGE, "Raw8 / B"), (.18, "N_to_NU", BLUE, "Unit8 / N")]:
    axes[2].bar(x + shift, [t["contributions"][name]["estimate"] for t in strata],
                width=.34, color=color, label=label)
axes[2].axhline(0, color="#777777", lw=.8)
axes[2].set_xticks(x, ["space", "number", "letters", "punct.", "other"], rotation=25)
axes[2].set_title("C. Token-type contributions", loc="left")
axes[2].set_ylabel("Mean article contribution (additive)")
axes[2].legend(fontsize=8)
save(fig, "04_article_bias", "Raw8: median +0.07077; article mean −0.31019 [−1.07462, +0.07582]. Exploratory 95% interval; original P4 median decision unchanged.")
print("4 figures written in PNG and SVG.")
