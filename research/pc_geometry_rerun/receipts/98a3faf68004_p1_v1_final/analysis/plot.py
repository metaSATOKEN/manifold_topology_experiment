"""Plot receipt aggregates without reading models or modifying observations."""
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/"analysis"/"aggregates.json").read_text())
OUT = ROOT/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False, "svg.fonttype":"none",
                     "figure.facecolor":"white", "savefig.facecolor":"white"})
COLORS = ["#4361a3","#de7b43","#aa5273","#539782","#7769a5"]
LABELS = ["C0\nAll 512","C1\nDrop first","C2\nDrop top 6","C3\nRandom 6\n(mean of 20)","C4\nUnit norm"]


def save(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"), bbox_inches="tight")
    plt.close(fig)


def values(point, condition, metric):
    return [r[metric] for r in DATA["metrics"] if r["point"]==point
            and r["condition"]==condition and r[metric] is not None]


fig, axes = plt.subplots(1,2,figsize=(11.4,5.3))
for ax, metric in zip(axes,["R1","V1"]):
    cs = ["C0","C1","C2","C3","C4"]
    for i,c in enumerate(cs):
        v = np.asarray(values("block06",c,metric))
        if not len(v):
            ax.text(i,.46,"N/A\n(constant norm)",ha="center",color="#777777")
            continue
        # Same article ordering in each condition; deterministic jitter for readability.
        jitter = np.linspace(-.15,.15,len(v))
        ax.scatter(i+jitter,v,s=13,color=COLORS[i],alpha=.48,zorder=3)
        q1,med,q3 = np.quantile(v,[.25,.5,.75])
        ax.plot([i,i],[q1,q3],lw=6,color=COLORS[i],alpha=.9,zorder=4)
        ax.plot([i-.17,i+.17],[med,med],color="#182537",lw=2.3,zorder=5)
        ax.text(i,min(1.055,max(v)+.045),f"{med:.3f}",ha="center",fontsize=10)
    # Pair C0/C1 observations. Other conditions are shown without extra connecting lines.
    a=np.asarray(values("block06","C0",metric))
    b=np.asarray(values("block06","C1",metric))
    for x,y in zip(a,b):
        ax.plot([0,1],[x,y],color="#a9adb6",alpha=.20,lw=.6,zorder=1)
    ax.set(xticks=range(5),xticklabels=LABELS,ylim=(-.025,1.1),xlim=(-.5,4.5),
           ylabel="Absolute Pearson correlation" if metric=="R1" else "Share of all centered variance",
           title=f"block06: {metric}")
    ax.grid(axis="y",alpha=.17)
fig.suptitle("One first-token observation dominates the middle-layer PC1",fontsize=15,y=1.02)
fig.text(.5,-.015,"64 articles. Dots: articles; thick bars: IQR; dark ticks: medians. These bars are not confidence intervals.",
         ha="center",fontsize=9)
fig.tight_layout()
save(fig,"01_primary_conditions")

fig, axes=plt.subplots(1,3,figsize=(11.4,3.6))
settings=[("A",.8,(-.03,1.03),"R1 after first-token removal"),
          ("B",-.1,(-1.03,.07),"Paired change in R1 (C1 - C0)"),
          ("D",-.05,(-1.03,.07),"Paired change in V1 (C2 - C3 mean)")]
for ax,(k,threshold,lims,title) in zip(axes,settings):
    e=DATA["primary_endpoints"][k]
    med=e["median"]
    ax.errorbar(med,0,xerr=[[med-e["lower"]],[e["upper"]-med]],
                fmt="o",color="#233b60",capsize=5,ms=7,lw=2)
    ax.axvline(threshold,ls="--",color="#bd6351",lw=1.5)
    ax.text(threshold,.38,f"Threshold {threshold:g}",ha="center",fontsize=9)
    ax.text(med,-.24,f'{med:.4f}\n[{e["lower"]:.4f}, {e["upper"]:.4f}]',
            ha="center",fontsize=9)
    ax.set(xlim=lims,ylim=(-.65,.7),yticks=[],title=f"{k}: {title}")
    ax.grid(axis="x",alpha=.15)
    ax.spines["left"].set_visible(False)
fig.suptitle("Preregistered endpoints: H1 refuted; H2 supported",fontsize=15,y=1.04)
fig.text(.5,-.03,"Article bootstrap, 10,000 resamples; 98.333333% interval per endpoint. Dashed lines: fixed decision thresholds.",
         ha="center",fontsize=9)
fig.tight_layout()
save(fig,"02_preregistered_intervals")

points=["block01","block06","block12_preln","final_ln"]
fig,axes=plt.subplots(1,2,figsize=(11.4,4.4))
for ax,metric in zip(axes,["R1","V1"]):
    mat=np.array([[DATA["summaries"][p][c][metric]["median"]
                  if DATA["summaries"][p][c][metric]["median"] is not None else np.nan
                  for c in ["C0","C1","C2","C3","C4"]] for p in points])
    cmap=plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#eeeeee")
    im=ax.imshow(mat,vmin=0,vmax=1,cmap=cmap,aspect="auto")
    for i in range(4):
        for j in range(5):
            v=mat[i,j]
            ax.text(j,i,"N/A" if np.isnan(v) else f"{v:.3f}",ha="center",va="center",
                    color="white" if v>.57 else "#172b41")
    ax.set(xticks=range(5),xticklabels=["C0","C1","C2","C3 mean","C4"],
           yticks=range(4),yticklabels=points,title=f"Median {metric} across 64 articles")
    ax.tick_params(length=0)
fig.suptitle("The pattern differs substantially across extraction points",fontsize=15,y=1.02)
fig.subplots_adjust(wspace=.55,left=.12,right=.86,bottom=.13)
cax=fig.add_axes([.90,.19,.018,.62])
fig.colorbar(im,cax=cax,label="0 to 1")
save(fig,"03_layers")

fig,axes=plt.subplots(3,1,figsize=(11.4,7.8))
features=["position","token_chars","surprisal"]
for ax,c in zip(axes,["C0","C1","C4"]):
    mat=np.array([[s["median"] for s in DATA["features"]["block06"][c][f]["by_pc"]]
                  for f in features])
    im=ax.imshow(mat,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    for i in range(3):
        for j in range(8):
            ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",
                    color="white" if mat[i,j]>.57 else "#172b41")
    ax.set(xticks=range(8),xticklabels=[f"PC{x}" for x in range(1,9)],
           yticks=range(3),yticklabels=["Position","Token characters","Surprisal"],
           title={"C0":"C0: all tokens","C1":"C1: first token removed, PCA refitted",
                  "C4":"C4: unit-normalized states, PCA refitted"}[c])
    ax.tick_params(length=0)
fig.suptitle("block06: feature association changes PC rank after refitting",fontsize=15,y=.99)
fig.subplots_adjust(hspace=.63,top=.91,bottom=.09,left=.14,right=.87)
fig.colorbar(im,ax=list(axes),fraction=.025,pad=.025,label="Median |Pearson r|")
fig.text(.5,.018,"Exploratory within-article associations; each article and condition has its own PCA basis. No held-out prediction.",
         ha="center",fontsize=9)
save(fig,"04_features")

lines=["# 詳細集計表", "",
       "64記事。角括弧は第1–第3四分位（信頼区間ではない）。C3は記事内20反復の平均を記事間で集約。", "",
       "| 抽出点 | 条件 | R1 中央値 [Q1, Q3] | V1 中央値 [Q1, Q3] |",
       "| --- | --- | --- | --- |"]
for p in points:
    for c in ["C0","C1","C2","C3","C4"]:
        cells=[]
        for metric in ["R1","V1"]:
            s=DATA["summaries"][p][c][metric]
            cells.append("NA（単位ノルム）" if s["median"] is None else
                         f'{s["median"]:.6f} [{s["q1"]:.6f}, {s["q3"]:.6f}]')
        lines.append(f"| {p} | {c} | {' | '.join(cells)} |")
lines += ["", "## 特徴相関", "", "各記事の符号不定性を避け、|r|を記事間で集約。数値は中央値。", ""]
for p in points:
    lines += [f"### {p}", "",
              "| 条件 | 特徴 | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in ["C0","C1","C2","C4"]:
        for f in features:
            vals=" | ".join(f'{s["median"]:.4f}' for s in DATA["features"][p][c][f]["by_pc"])
            lines.append(f"| {c} | {f} | {vals} |")
    lines.append("")
(ROOT/"analysis"/"DETAIL_TABLES.md").write_text("\n".join(lines)+"\n")
(ROOT/"analysis"/"plot_environment.json").write_text(json.dumps(
    dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
         note="Plotting only; frozen NumPy 2.2.6 was used for numerical receipt verification and aggregates."),
    indent=2)+"\n")
print("Saved four PNG/SVG figures and detailed tables.")
