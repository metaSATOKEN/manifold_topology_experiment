"""Render verified P2 receipt summaries; no model execution or raw-data writes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis/aggregates.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
POINTS = ["block06", "final_ln"]
CS = ["C0", "C1", "C4"]
COLORS = ["#62758d", "#216b82", "#c4763b"]


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(11, 6.3))
settings = [
    ("A", .5, (.40, .85), "Transferred position correlation"),
    ("B", .1, (0, .5), "Position advantage over random"),
    ("C", .02, (0, .14), "Surprisal gain over controls"),
    ("D", .01, (0, .10), "Surprisal advantage over random")]
for ax, (key, threshold, limits, title) in zip(axes.flat, settings):
    e = DATA["primary_endpoints"][key]
    med = e["median"]
    ax.errorbar(med, 0, xerr=[[med-e["lower"]], [e["upper"]-med]],
                fmt="o", ms=7, lw=2.5, color="#216b82", capsize=5)
    ax.axvline(threshold, ls="--", lw=1.4, color="#b56c3b")
    ax.text(threshold, .36, f"Threshold {threshold:g}", ha="center", fontsize=9)
    ax.text(med, -.27, f'{med:.4f}  [{e["lower"]:.4f}, {e["upper"]:.4f}]',
            ha="center", fontsize=9)
    ax.set(xlim=limits, ylim=(-.6, .6), yticks=[], title=f"{key}: {title}")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.15)
fig.suptitle("P2 preregistered results: H1 supported; H2 supported", fontsize=15)
fig.tight_layout(rect=(0, .06, 1, .94))
fig.text(.5, .015, "block06 / C1. 64 held-out articles; 10,000 article resamples; "
         "98.75% interval per endpoint; training fit fixed.", ha="center", fontsize=9)
save(fig, "01_preregistered_intervals")

fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
for col, point in enumerate(POINTS):
    for row, names, labels in [
        (0, ["position_r", "random_position_r"], ["Selected PC", "Random (mean of 20)"]),
        (1, ["surprisal_baseline_r2", "surprisal_pca_r2", "surprisal_random_r2"],
         ["Controls", "Controls + 8 PCs", "Controls + 8 random"])]:
        ax = axes[row, col]
        for j, (name, label) in enumerate(zip(names, labels)):
            values = [DATA["summaries"][point][c][name] for c in CS]
            med = np.asarray([v["median"] for v in values])
            lo = np.asarray([v["q1"] for v in values])
            hi = np.asarray([v["q3"] for v in values])
            x = np.arange(3) + (j - (len(names)-1)/2) * .13
            color = ["#216b82", "#c4763b"][j] if row == 0 else COLORS[j]
            ax.errorbar(x, med, yerr=[med-lo, hi-med], fmt="o-", lw=1.5,
                        ms=5, capsize=3, label=label, color=color)
        ax.axhline(0, lw=.7, color="#adb3bc")
        ax.set(xticks=range(3), xticklabels=["C0\nAll", "C1\nDrop first", "C4\nUnit norm"],
               xlim=(-.35, 2.35), title=point,
               ylabel="Signed position r" if row == 0 else "Held-out surprisal R²",
               ylim=(-.3, .9) if row == 0 else (-.08, .43))
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=.15)
fig.suptitle("Transfer and prediction differ by extraction point", fontsize=15)
fig.tight_layout(rect=(0, .07, 1, .94))
fig.text(.5, .015, "Medians and article IQRs (not confidence intervals). "
         "Random models averaged within each article. Only block06/C1 is primary.",
         ha="center", fontsize=9)
save(fig, "02_transfer_and_prediction")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, point in zip(axes, POINTS):
    for j, (key, label, color) in enumerate([
        ("pc1_variance_fraction", "First frozen PC", "#216b82"),
        ("top8_variance_fraction", "All 8 frozen PCs", "#9ab7bd")]):
        med = np.asarray([DATA["summaries"][point][c][key]["median"] for c in CS]) * 100
        ax.bar(np.arange(3) + (j-.5)*.34, med, width=.33, color=color, label=label)
        for i, v in enumerate(med):
            ax.text(i+(j-.5)*.34, v+1.3, f"{v:.1f}", ha="center", fontsize=9)
    ax.set(xticks=range(3), xticklabels=["C0\nAll", "C1\nDrop first", "C4\nUnit norm"],
           ylim=(0, 108), title=point, ylabel="Median held-out variance share (%)")
    ax.grid(axis="y", alpha=.13)
fig.suptitle("Variance concentration survives transfer, but depends on layer and condition",
             fontsize=13)
fig.tight_layout(rect=(0, .14, 1, .94))
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
           bbox_to_anchor=(.5, .05), ncol=2, fontsize=9, frameon=False)
fig.text(.5, .015, "Directions fixed from 64 training articles; shares measured within each "
         "held-out article. No test PCA refitting.", ha="center", fontsize=9)
save(fig, "03_transferred_variance")

fig, axes = plt.subplots(3, 1, figsize=(11, 8.4))
features = ["position", "token_chars", "surprisal", "norm"]
cmap = plt.colormaps["YlGnBu"].copy()
cmap.set_bad("#eceff2")
for ax, c in zip(axes, CS):
    mat = np.array([[DATA["features"]["block06"][c][f][str(pc)]["absolute"]["median"]
                     if DATA["features"]["block06"][c][f][str(pc)]["absolute"]["median"]
                     is not None else np.nan for pc in range(1, 9)] for f in features])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(4):
        for j in range(8):
            v = mat[i, j]
            ax.text(j, i, "N/A" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="white" if v > .57 else "#182d43", fontsize=9)
    ax.set(xticks=range(8), xticklabels=[f"PC{k}" for k in range(1, 9)],
           yticks=range(4), yticklabels=["Position", "Decoded characters", "Surprisal", "Norm"],
           title=f"block06 / {c}")
    ax.tick_params(length=0)
fig.suptitle("Frozen PCs carry overlapping feature associations in held-out articles",
             fontsize=14)
fig.subplots_adjust(hspace=.65, top=.92, bottom=.09, left=.17, right=.86)
cax = fig.add_axes([.89, .19, .018, .62])
fig.colorbar(im, cax=cax, label="Median |Pearson r|")
fig.text(.5, .018, "Secondary description across 64 articles. C4 norm correlation is undefined. "
         "Correlation does not identify independent functions.", ha="center", fontsize=9)
save(fig, "04_features")

lines = ["# P2 詳細集計表", "",
         "全64評価記事の記述集計。区間は四分位範囲（IQR）で、信頼区間ではない。"
         "ランダム対照は記事内で20組を平均した後に集約。対応差の中央値は中央値同士の差と異なる。", "",
         "## 全条件の指標", "",
         "| 点 | 条件 | 指標 | 有効記事 | 中央値 | Q1 | Q3 | 最小 | 最大 | 正 / 負 / 0 |",
         "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
for point in POINTS:
    for c in CS:
        for k, s in DATA["summaries"][point][c].items():
            lines.append(f'| {point} | {c} | {k} | {s["n"]} | {s["median"]:.6f} | '
                         f'{s["q1"]:.6f} | {s["q3"]:.6f} | {s["min"]:.6f} | {s["max"]:.6f} | '
                         f'{s["positive"]} / {s["negative"]} / {s["zero"]} |')
lines += ["", "## 訓練で固定した位置PCと分散", "",
          "| 点 | 条件 | 選択PC | 符号 | 訓練位置r中央値 | 訓練PC1分散比 | 訓練8PC合計分散比 | 数値rank |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for point in POINTS:
    for c in CS:
        f = DATA["fits"][point][c]
        s = f["selected_pc"]
        lines.append(f'| {point} | {c} | {s["component"]} | {s["sign"]} | '
                     f'{s["training_median"]:.6f} | {f["training_pc_variance_fractions"][0]:.6f} | '
                     f'{f["training_top8_variance_fraction"]:.6f} | {f["rank"]} |')
lines += ["", "## 全PC・全特徴の相関", "",
          "各セルは記事中央値。符号付き相関は訓練共分散の固有ベクトルの固定符号による。"
          "位置プローブの訓練符号合わせとは別。C4のnormは定数なのでNA。", ""]
for point in POINTS:
    for c in CS:
        lines += [f"### {point} / {c}", "",
                  "| 特徴 | 集計 | " + " | ".join(f"PC{k}" for k in range(1, 9)) + " |",
                  "| --- | --- | " + " | ".join(["---:"]*8) + " |"]
        for f in features:
            for kind in ["signed", "absolute"]:
                vals = [DATA["features"][point][c][f][str(pc)][kind]["median"] for pc in range(1, 9)]
                lines.append(f"| {f} | {kind} | " +
                             " | ".join("NA" if v is None else f"{v:.6f}" for v in vals) + " |")
        lines.append("")
(ROOT / "analysis/DETAIL_TABLES.md").write_text("\n".join(lines) + "\n")
print("Wrote four PNG/SVG figures and complete descriptive tables.")
