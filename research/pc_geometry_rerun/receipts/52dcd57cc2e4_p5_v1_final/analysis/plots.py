"""All preselected texts, fixed PCA coordinates, no new model fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from intake import OUT, RELEASE, write
from verify import RAW, load

FIG = OUT/"figures"
TEXTS = ["narrative", "technical", "repetitive"]
POINTS = ["block06", "block07", "block12_preln", "final_ln"]
CONDITIONS = ["C0", "C1", "C4"]
LABEL = {"block06": "Block 6", "block07": "Block 7 (primary)",
         "block12_preln": "Before final LayerNorm", "final_ln": "After final LayerNorm"}
rows = json.loads((OUT/"recomputed_metrics.json").read_text())
INDEX = {(r["device"], r["text"], r["point"], r["condition"]): r["metrics"] for r in rows}
inventory = []
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def data(device, text, point, condition):
    arrays = load(RAW/"analysis"/f"{device}_{text}_{point}_{condition}.npz")
    return arrays, INDEX[device, text, point, condition]


def oriented(arrays):
    scores = arrays["scores"].copy()
    # Allowed display-only sign flip to match the historical start-at-right panel.
    if 0 in arrays["positions"] and scores[0, 0] < 0:
        scores[:, 0] *= -1
    return scores


def plot2d(ax, arrays, metric, title, line=False, zoom=False, legacy_bounds=False):
    s = oriented(arrays)
    pos = arrays["positions"]
    if line:
        ax.plot(s[:, 0], s[:, 1], lw=.7, alpha=.45, color="#607d8b", zorder=1)
    ax.scatter(s[:, 0], s[:, 1], c=pos, cmap="viridis", s=15, zorder=2)
    if 0 in pos:
        ax.scatter(s[0, 0], s[0, 1], c="#cb3a3e", marker="^", s=72, zorder=4)
    ax.scatter(s[-1, 0], s[-1, 1], c="#2270b5", marker="v", s=55, zorder=4)
    excluded = 0
    if zoom:
        lo, hi = np.quantile(s[:, 0], [.05, .95])
        inside = (s[:, 0] >= lo) & (s[:, 0] <= hi)
        ylo, yhi = s[inside, 1].min(), s[inside, 1].max()
        pad = max(yhi-ylo, 1e-6)*.08
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo-pad, yhi+pad)
        excluded = int((~inside).sum())
        title += f"\nPC1 central 90% display; {excluded}/{len(s)} outside"
        ax.set_aspect("equal", adjustable="box")
    elif legacy_bounds:
        ax.set_xlim(-200, 3150)
        ax.set_ylim(-1675, 1675)
        ax.set_aspect("equal", adjustable="box")
    else:
        center = (s[:, :2].min(0)+s[:, :2].max(0))/2
        span = max(np.ptp(s[:, :2], axis=0))*1.16
        ax.set_xlim(center[0]-span/2, center[0]+span/2)
        ax.set_ylim(center[1]-span/2, center[1]+span/2)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"PC1 ({100*metric['V1']:.2f}%)")
    ax.set_ylabel(f"PC2 ({100*metric['pc2_variance']:.2f}%)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=.12)
    return excluded


def save(fig, name, description, conditions):
    path = FIG/name
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    inventory.append(dict(path="figures/"+name, description=description, conditions=conditions,
                          equal_units=True, source="raw/analysis PCA arrays verified by analysis/verify.py"))


def main():
    FIG.mkdir(exist_ok=True)
    # Historical figure is rendered as an unmodified image, never raster-edited.
    old = plt.imread(RELEASE/"historical_pointclouds.png")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=.25)
    top = fig.add_subplot(grid[0, :])
    top.imshow(old)
    top.axis("off")
    top.set_title("Archived figure (unchanged)", fontsize=14)
    for j, text in enumerate(TEXTS):
        a, m = data("cpu", text, "block07", "C0")
        ax = fig.add_subplot(grid[1, j])
        plot2d(ax, a, m, text+" — reproduced, CPU / block 7 / C0", line=True, legacy_bounds=True)
    fig.suptitle("The historical variance concentration and connected-line appearance reproduce",
                 fontsize=15, y=.99)
    save(fig, "01_historical_comparison.png", "旧図原画像と同じ３文章の再現。下段の線は隣接token間の補間。",
         ["cpu/block07/C0/"+t for t in TEXTS])
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), layout="constrained")
    excluded = {}
    for i, text in enumerate(TEXTS):
        a, m = data("cpu", text, "block07", "C0")
        plot2d(axes[i, 0], a, m, text+" — full range, points only", legacy_bounds=True)
        excluded[text] = plot2d(axes[i, 1], a, m, text+" — same PCA, magnified", zoom=True)
        b, n = data("cpu", text, "block07", "C1")
        plot2d(axes[i, 2], b, n, text+" — first token removed, PCA refit")
    fig.suptitle("A remote first token and a structured cloud\nZoom changes the display; C1 recomputes the PCA",
                 fontsize=15)
    save(fig, "02_points_zoom_and_C1.png", "線なし全点、同一PCAの中央90%拡大、先頭除外後の再PCA。外側点は図中に計数。",
         ["cpu/block07/"+c+"/"+t for t in TEXTS for c in ["C0", "C1"]])
    # Complete atlases: all 72 conditions in 2D and 3D; device is never selected post hoc.
    for device in ["cpu", "cuda"]:
        for point in POINTS:
            for dimension in [2, 3]:
                fig = plt.figure(figsize=(16, 14) if dimension == 3 else (14, 12), layout="constrained")
                for i, condition in enumerate(CONDITIONS):
                    for j, text in enumerate(TEXTS):
                        a, m = data(device, text, point, condition)
                        if dimension == 2:
                            ax = fig.add_subplot(3, 3, i*3+j+1)
                            plot2d(ax, a, m, text+" / "+condition)
                        else:
                            ax = fig.add_subplot(3, 3, i*3+j+1, projection="3d")
                            s = oriented(a)
                            ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=a["positions"], cmap="viridis", s=13)
                            if 0 in a["positions"]:
                                ax.scatter(*s[0, :3], c="#cb3a3e", marker="^", s=65)
                            center = (s[:, :3].min(0)+s[:, :3].max(0))/2
                            span = max(np.ptp(s[:, :3], axis=0))*1.1
                            for axis, setlim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                                setlim(center[axis]-span/2, center[axis]+span/2)
                            ax.set_box_aspect([1, 1, 1], zoom=.82)
                            ax.view_init(20, -60)
                            ax.set_xlabel("PC1", labelpad=12, fontsize=9)
                            ax.set_ylabel("PC2", labelpad=12, fontsize=9)
                            ax.set_zlabel("PC3", labelpad=12, fontsize=9)
                            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                                axis.set_major_locator(MaxNLocator(3))
                                axis.set_tick_params(labelsize=8)
                            ax.set_title(f"{text} / {condition} — V1 {100*m['V1']:.2f}%", fontsize=10)
                fig.suptitle(f"{device.upper()} / {LABEL[point]} / PCA-{dimension}D\n"
                             "C0: all tokens   C1: first token removed   C4: unit length", fontsize=14)
                save(fig, f"atlas_{device}_{point}_{dimension}d.png",
                     f"{device} {point}の全３文章×３処理、{dimension}D等縮尺。",
                     [device+"/"+point+"/"+c+"/"+t for c in CONDITIONS for t in TEXTS])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for j, text in enumerate(TEXTS):
        m = INDEX["cpu", text, "block07", "C0"]
        names = ["position", "log_norm", "log_chars", "surprisal"]
        values = np.array([m["thickness_correlations"][n] for n in names])
        image = axes[j].imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axes[j].set_xticks(range(7), [f"PC{k}" for k in range(2, 9)])
        axes[j].set_yticks(range(4), names)
        axes[j].set_title(text)
        for y in range(4):
            for x in range(7):
                axes[j].text(x, y, f"{values[y,x]:+.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(values[y,x]) > .6 else "black")
    fig.colorbar(image, ax=axes, shrink=.7, label="Pearson r")
    fig.suptitle("Residual PC2–8 after subtracting the fitted quadratic mean\n"
                 "CPU / block 7 / C0 — descriptive associations, all axes shown", fontsize=13)
    save(fig, "03_thickness_all_axes.png", "主条件の残差PC2〜8×全４特徴×３文章。軸を選別しない。",
         ["cpu/block07/C0/"+t for t in TEXTS])
    write(OUT/"figure_manifest.json", dict(figures=inventory, figure_count=len(inventory),
        atlas_conditions_2d=72, atlas_conditions_3d=72, primary_zoom_outside=excluded,
        display_sign_rule="Flip PC1 only if position zero has negative score; visualization only.",
        original_figure_unchanged=True))
    lines = ["# P5 全図", "", "全３文章を表示。各処理でPCAを再計算しており、異なる図のPC軸は同一ではない。",
             "線付き図の線は補間。色はtoken順（紫→黄）、赤△は先頭。2Dの青▽は末尾。",
             "3Dは３軸とも同じ単位・同じ幅。CPUとGPUの全条件を掲載する。", ""]
    for item in inventory:
        lines += ["## "+Path(item["path"]).stem, "", item["description"], "",
                  "!["+item["description"]+"]("+item["path"]+")", ""]
    (OUT/"GALLERY.md").write_text("\n".join(lines))
    print(json.dumps(dict(figures=len(inventory), primary_zoom_outside=excluded)))


if __name__ == "__main__":
    main()
