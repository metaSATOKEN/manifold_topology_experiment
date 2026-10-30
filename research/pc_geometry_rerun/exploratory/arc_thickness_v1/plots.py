import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/pcgeom-eda4-mpl")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import OUT

spec=json.loads((OUT/"spec.json").read_text())
summary=json.loads((OUT/"summary.json").read_text())
FIG=OUT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})


def load(m,p,c,d):
    return dict(np.load(OUT/"results"/m/p/c/(d+".npz"),allow_pickle=False))


def save(fig,name):
    fig.savefig(FIG/(name+".png"),dpi=140,bbox_inches="tight")
    plt.close(fig)


def scatter(ax,s,pos,title,connect=False):
    if connect:
        ax.plot(s[:,0],s[:,1],color=".55",alpha=.45,lw=.65,zorder=1)
    im=ax.scatter(s[:,0],s[:,1],c=pos,cmap="viridis",s=7,edgecolors="none",zorder=3)
    if np.any(pos==0):
        ax.scatter(*s[pos==0][0,:2],c="red",marker="^",s=35,zorder=4)
    ax.set_aspect("equal",adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.ticklabel_format(style="sci",axis="both",scilimits=(-2,3))
    return im


labels=list(summary)
fields=["V1","Q_curve_2d","Q_curve_full","max_gap","occupancy"]
fig,axes=plt.subplots(1,5,figsize=(14,8.7),sharey=True,layout="constrained")
for j,(ax,f) in enumerate(zip(axes,fields)):
    vals=np.array([summary[k][f]["median"] for k in labels])
    # Display Q below zero at zero color only; literal text retains negatives.
    im=ax.imshow(np.clip(vals[:,None],0,1),vmin=0,vmax=1,cmap="Blues",aspect="auto")
    for i,v in enumerate(vals):
        ax.text(0,i,f"{v:.3f}",ha="center",va="center",color="white" if v>.55 else "black")
    ax.set_xticks([]);ax.set_title(f)
    ax.set_yticks(np.arange(len(labels)),labels if j==0 else [])
axes[0].set_yticklabels(labels)
fig.suptitle("Article medians | 64 articles per row | Q: blocked curve approximation\n2D projection and full-state geometry are evaluated separately",fontsize=13)
save(fig,"01_geometry_overview")

# Representative pages: two fixed IDs, same pages for every model/extraction point.
for m in spec["models"]:
    for p in spec["points"]:
        for ex,d in enumerate(spec["examples"]):
            fig=plt.figure(figsize=(14.5,10),layout="constrained")
            for i,c in enumerate(spec["conditions"]):
                a=load(m,p,c,d);s=a["scores"];pos=a["positions"]
                ev=np.cumsum(a["eigenvalues"])/float(a["total"])
                ax=fig.add_subplot(3,4,i*4+1)
                im=scatter(ax,s,pos,f"{c}: points only | V2={ev[1]:.1%}")
                ax=fig.add_subplot(3,4,i*4+2)
                scatter(ax,s,pos,"Token-order segments (not observations)",connect=True)
                ax=fig.add_subplot(3,4,i*4+3)
                scatter(ax,s,pos,"Central 90% display window")
                low,high=np.quantile(s[:,:2],[.05,.95],axis=0)
                outside=np.any((s[:,:2]<low)|(s[:,:2]>high),axis=1).sum()
                ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
                ax.set_aspect("equal",adjustable="box")
                ax.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax.tick_params(axis="x",labelsize=7)
                ax.set_title(f"Zoom only | {outside}/{len(s)} outside")
                ax=fig.add_subplot(3,4,i*4+4,projection="3d")
                rad=np.log1p(a["radius"])
                sc=ax.scatter(s[:,0],s[:,1],s[:,2],c=rad,cmap="magma",s=5)
                if np.any(pos==0):
                    ax.scatter(*s[pos==0][0,:3],c="red",marker="^",s=32)
                low3,high3=s[:,:3].min(0),s[:,:3].max(0)
                center3=(low3+high3)/2
                span3=max(np.max(high3-low3)*1.1,1e-12)
                for dim,setter in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):
                    setter(center3[dim]-span3/2,center3[dim]+span3/2)
                ax.set_box_aspect((1,1,1))
                ax.set_title(f"3D: original log-norm | V3={ev[2]:.1%}")
                ax.set_xlabel("PC1",labelpad=0);ax.set_ylabel("PC2",labelpad=0);ax.set_zlabel("PC3",labelpad=0)
                for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
                    axis.set_major_locator(plt.MaxNLocator(3))
                ax.tick_params(labelsize=6,pad=0)
                fig.colorbar(sc,ax=ax,shrink=.5,pad=.13)
            fig.suptitle(f"{m} / {p} / {d} (fixed example {ex+1})\n2D color: original token position; red triangle: first token. Axes use equal data units; PCA refitted per row.",fontsize=12)
            save(fig,f"example_{ex+1}_{m}_{p}")

        # All 64 articles, no selection based on result.
        for c in spec["conditions"]:
            fig,axes=plt.subplots(8,8,figsize=(16,14),layout="constrained")
            for ax,d in zip(axes.ravel(),spec["article_ids"]):
                a=load(m,p,c,d);s=a["scores"];pos=a["positions"]
                ax.scatter(s[:,0],s[:,1],c=pos,cmap="viridis",s=1.5,edgecolors="none")
                if np.any(pos==0):
                    ax.scatter(*s[pos==0][0,:2],c="red",marker="^",s=8)
                ax.set_aspect("equal",adjustable="datalim")
                ax.set_title(d.replace("wt2_train_","")+" | "+f"V1 {a['eigenvalues'][0]/a['total']:.0%}",fontsize=6)
                ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f"All 64 articles | {m} / {p} / {c}\nPoints only; equal data units; separate article PCA/scale; first token in red",fontsize=13)
            save(fig,f"atlas_{m}_{p}_{c}")
        print("plots",m,p,flush=True)

# Holding the C0 basis fixed versus refitting C1.
fig,axes=plt.subplots(6,3,figsize=(12,16),layout="constrained")
d=spec["examples"][0]
for i,(m,p) in enumerate((m,p) for m in spec["models"] for p in spec["points"]):
    c0=load(m,p,"C0",d);c1=load(m,p,"C1",d)
    scatter(axes[i,0],c0["scores"],c0["positions"],f"{m}/{p}: C0")
    scatter(axes[i,1],c0["scores"][1:],c0["positions"][1:],"C0 axes; first point hidden; display rescaled")
    scatter(axes[i,2],c1["scores"],c1["positions"],"C1: PCA refitted")
fig.suptitle(f"Display change versus basis change | {d}\nNo connecting lines. The middle column retains the original C0 coordinates.",fontsize=13)
save(fig,"02_fixed_basis_vs_refit")

feature_names=["position","norm","chars","surprisal","controlled_surprisal"]
fig,axes=plt.subplots(1,5,figsize=(16,9),sharey=True,layout="constrained")
for j,(ax,f) in enumerate(zip(axes,feature_names)):
    for i,k in enumerate(labels):
        row=summary[k]["thickness_gain_"+f]
        lo,hi=row["ci95"];v=row["median"]
        ax.plot([lo,hi],[i,i],color="#175B85",lw=2)
        ax.plot(v,i,"o",color="#175B85",ms=4)
    ax.axvline(0,color=".5",lw=.8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title(f);ax.set_xlabel("Delta R2")
    ax.set_yticks(np.arange(18),labels if j==0 else [])
axes[0].set_yticklabels(labels);axes[0].invert_yaxis()
fig.suptitle("Conditional within-article thickness readout\nMedian and descriptive 95% article-bootstrap interval; fixed 5 contiguous folds; all-PCA coordinates",fontsize=13)
save(fig,"03_thickness_readout")
