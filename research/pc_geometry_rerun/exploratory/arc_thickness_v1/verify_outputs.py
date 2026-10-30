"""Check derived geometry decisions and report transcriptions."""
import json
import re
import hashlib
from pathlib import Path
import numpy as np
from analyze import OUT, BASE, PROJECT, write, stamp

rows=json.loads((OUT/"article_results.json").read_text())
summary=json.loads((OUT/"summary.json").read_text())
spec=json.loads((OUT/"spec.json").read_text())
freeze=json.loads((OUT/"FREEZE.json").read_text())
start=json.loads((OUT/"execution_start.json").read_text())
assert freeze["at"] < start["at"]
for p,h in freeze["files"].items():
    assert hashlib.sha256((PROJECT/p).read_bytes()).hexdigest()==h
for p,h in start["source_sha256"].items():
    assert hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h
seen=set()
for row in rows:
    m,p,c,d=[row[k] for k in ["model","point","condition","article_id"]]
    key=(m,p,c,d)
    assert key not in seen
    seen.add(key)
    a=dict(np.load(OUT/"results"/m/p/c/(d+".npz"),allow_pickle=False))
    s=a["scores"][:,0]
    x=np.sort(s);span=x[-1]-x[0]
    gap=float(np.diff(x).max()/span)
    bins=np.minimum(19,np.floor((x-x[0])/span*20).astype(int))
    occ=len(np.unique(bins))/20
    share=float(np.max(s*s)/np.sum(s*s))
    low,high=np.percentile(s,[5,95]);x90=np.sort(s[(s>=low)&(s<=high)])
    span90=np.ptp(x90);b90=np.minimum(19,np.floor((x90-x90[0])/span90*20).astype(int))
    q=1-a["curve_fold_sse"].sum(1)/np.array([a["total"],np.sum(a["scores"][:,:2]**2),np.sum(a["scores"][:,:3]**2)])
    for name,value in [("max_gap",gap),("occupancy",occ),("max_pc1_share",share),
                       ("max_gap90",float(np.diff(x90).max()/span90)),("occupancy90",len(np.unique(b90))/20)]:
        np.testing.assert_allclose(row["metrics"][name],value,atol=1e-12)
    for j,name in enumerate(["full","2d","3d"]):
        np.testing.assert_allclose(row["metrics"]["Q_curve_"+name],q[1,j],rtol=1e-12,atol=1e-10)
    for j,name in [(0,"full"),(1,"2d")]:
        expected=bool(occ>=.8 and gap<=.1 and share<=.5 and q[1,j]>=.9)
        assert row["metrics"]["graph_"+name]==expected
        assert row["metrics"]["curved_graph_"+name]==bool(expected and q[1,j]-q[0,j]>=.01)
assert len(seen)==1152

failure_diagnostics={}
for c in ["C0","C1"]:
    a=dict(np.load(OUT/"results/qwen/middle"/c/"wt2_train_27981.npz",allow_pickle=False))
    pos=a["feature_positions"]
    e=(a["feature_predictions"][1,:,4]-a["feature_targets"][:,4])**2
    i=e.argmax();where=np.flatnonzero(a["positions"]==pos[i])[0]
    fold=a["folds"][where];tr=a["folds"]!=fold
    failure_diagnostics[c]=dict(article="wt2_train_27981",worst_position=int(pos[i]),
        squared_error_share=float(e[i]/e.sum()),original_norm=float(a["radius"][where]),
        training_fold_norm_min=float(a["radius"][tr].min()),training_fold_norm_max=float(a["radius"][tr].max()),
        status="descriptive inspection of the saved worst-article predictions; no new fit")
    assert pos[i]==1 and e[i]/e.sum()>.999
write(OUT/"failure_diagnostics.json",failure_diagnostics)

report=(OUT/"ANALYSIS_REPORT_JA.md").read_text()
tables=[]
block=[]
for line in report.splitlines()+[""]:
    if line.startswith("|"):
        block.append([v.strip() for v in line.split("|")[1:-1]])
    elif block:
        tables.append(block);block=[]
assert len(tables)==5
cells=0
def fmt(v):
    return f"{v:.3g}" if abs(v)>=100 else f"{v:.4f}"
for key,g,row1,row2,row3 in zip(summary,summary.values(),tables[0][2:],tables[1][2:],tables[2][2:]):
    expected1=[f"{g[f]['median']:.1%}" for f in ["V1","V3","max_pc1_share","max_gap"]]+[f"{g['occupancy']['median']:.0%}"]
    assert row1[1:]==expected1
    expected2=[fmt(g[f]["median"]) for f in ["Q_curve_2d","Q_curve_full","curve_gain_2d"]]+[f"{g[f]['count']}/64" for f in ["graph_2d","curved_graph_2d","graph_full"]]
    assert row2[1:]==expected2
    r=g["thickness_gain_controlled_surprisal"]
    expected3=[f"{fmt(r['median'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]",
               fmt(r["mean"]),f"{r['positive']}/64",fmt(r["min"])]
    assert row3[1:]==expected3
    cells+=15
for metric in ["norm","chars","controlled_surprisal"]:
    assert all(g["thickness_gain_"+metric]["ci95"][0]>0 for g in summary.values())
assert [summary["gpt2/final_norm/"+c]["graph_2d"]["count"] for c in ["C0","C1","C4"]]==[39,39,2]
assert [summary["gpt2/final_norm/"+c]["curved_graph_2d"]["count"] for c in ["C0","C1","C4"]]==[19,21,2]
assert all(g["graph_2d"]["count"]==0 for k,g in summary.items() if not k.startswith("gpt2/final_norm/"))
figures=list((OUT/"figures").glob("*.png"))
assert len(figures)==33
write(OUT/"output_checks.json",dict(status="PASS",at=stamp(),conditions_checked=len(seen),
     report_numeric_cells=cells,figure_count=len(figures),failure_diagnostics_recorded=True,
     freeze_before_execution=True,analysis_source_unchanged=True,
     note="Final preservation, artifact links and operational-record contracts are checked in final anchor."))
print("Output checks PASS:",len(seen),"conditions,",cells,"report numeric cells,",len(figures),"figures")
