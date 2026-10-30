"""Separate reconstruction from original states and saved coefficients/coordinates."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from analyze import OUT, BASE, write, stamp

start = time.monotonic()
raw = BASE/"receipts/4b013b35152b_p4_v1_final/raw"
spec = json.loads((OUT/"spec.json").read_text())
rows = json.loads((OUT/"article_results.json").read_text())
assert len(rows) == 1152
assert json.loads((OUT/"failures.json").read_text()) == []
max_errors = dict(svd=0., old_v1=0., curve_sse_relative=0., feature_prediction=0., bootstrap=0.)
counts = dict(conditions=0, curve_fold_sse=0, feature_predictions=0, direct_svd=0)


def augmented_prediction(x, y, tr, te):
    sd = np.std(x[tr],axis=0)
    mask = sd > 1e-12
    scale = np.where(mask, sd, 1.)
    mu = np.mean(x[tr],axis=0)
    design = np.c_[np.ones(tr.sum()), (x[tr]-mu)/scale*mask]
    test = np.c_[np.ones(te.sum()), (x[te]-mu)/scale*mask]
    reg = np.diag(np.r_[0., np.repeat(np.sqrt(.01*tr.sum()),x.shape[1])])
    coef = np.linalg.lstsq(np.r_[design,reg], np.r_[y[tr],np.zeros((len(reg),y.shape[1]))],rcond=None)[0]
    return test @ coef


for ri,row in enumerate(rows):
    model,point,condition,doc = [row[k] for k in ["model","point","condition","article_id"]]
    original = dict(np.load(raw/"docs"/(model+"__"+doc)/"raw.npz",allow_pickle=False))
    a = dict(np.load(OUT/"results"/model/point/condition/(doc+".npz"),allow_pickle=False))
    x = original["h_"+point].astype(float)[a["positions"]]
    if condition == "C4":
        x /= np.sqrt(np.sum(x*x,axis=1))[:,None]
    y = x-x.mean(0)
    s = y @ a["components"].T
    np.testing.assert_allclose(s,a["scores"],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(np.sum(y*y),a["total"],rtol=1e-10)
    np.testing.assert_allclose(a["components"] @ a["components"].T,np.eye(8),atol=1e-7)
    if doc == spec["examples"][0]:
        _,sing,v = svd(y,full_matrices=False,lapack_driver="gesdd")
        e = float(np.max(abs(sing[:8]**2-a["eigenvalues"]))/a["total"])
        max_errors["svd"] = max(max_errors["svd"],e)
        assert e < 1e-8
        np.testing.assert_allclose(abs(v[:8] @ a["components"].T),np.eye(8),atol=1e-6)
        counts["direct_svd"] += 1
    if condition != "C4":
        old=json.loads((raw/"evaluation"/model/point/doc/"metrics.json").read_text())["geometry"][condition]
        error=abs(row["metrics"]["V1"]-old["V1"])
        max_errors["old_v1"]=max(max_errors["old_v1"],error)
        assert error < 1e-8
        np.testing.assert_allclose(row["metrics"]["V8"],old["V8"],atol=1e-8)
    for mi,degree in enumerate([1,2]):
        design=np.vander(a["z"],degree+1,increasing=True)
        for f in range(5):
            tr,te=a["folds"]!=f,a["folds"]==f
            q,r=np.linalg.qr(design[tr],mode="reduced")
            coef=np.linalg.solve(r,q.T @ y[tr])
            pred=design[te] @ coef
            err=y[te]-pred
            ep=err @ a["components"][:3].T
            ss=np.array([np.einsum("ij,ij->",err,err),np.einsum("ij,ij->",ep[:,:2],ep[:,:2]),np.einsum("ij,ij->",ep,ep)])
            old=a["curve_fold_sse"][mi,f]
            rel=np.max(abs(ss-old)/np.maximum(1,abs(old)))
            max_errors["curve_sse_relative"]=max(max_errors["curve_sse_relative"],float(rel))
            np.testing.assert_allclose(ss,old,atol=1e-7,rtol=1e-6)
            counts["curve_fold_sse"]+=3
    d=np.c_[np.ones(len(a["z"])),a["z"],a["z"]**2]
    residual=a["scores"][:,1:8]-d @ a["curve_coef"] @ a["components"][1:8].T
    np.testing.assert_allclose(residual,a["thickness"],rtol=1e-9,atol=1e-7)
    take=a["positions"]>0
    pos=a["positions"][take]
    rel=pos/(len(original["position"])-1)
    norm=np.log1p(a["radius"][take])
    chars=np.log1p(original["token_chars"][pos])
    frequency=np.load(raw/"fit"/model/"frequency.npz",allow_pickle=False)["counts"]
    freq=np.log1p(frequency[original["token_id"][pos]])
    arc=np.c_[a["z"][take],a["z"][take]**2]
    cov=np.c_[rel,rel**2,norm,norm**2,chars,freq]
    targets=np.c_[rel,norm,chars,original["surprisal"][pos],original["surprisal"][pos]]
    np.testing.assert_allclose(targets,a["feature_targets"],atol=1e-12)
    folds=np.minimum(4,pos*5//len(original["position"]))
    for cols,base in [(slice(0,4),arc),(slice(4,5),np.c_[arc,cov])]:
        for mi,design in enumerate([base,np.c_[base,residual[take]]]):
            for f in range(5):
                tr,te=folds!=f,folds==f
                pred=augmented_prediction(design,targets[:,cols],tr,te)
                expected=a["feature_predictions"][mi,te,cols]
                error=float(np.max(abs(pred-expected)))
                max_errors["feature_prediction"]=max(max_errors["feature_prediction"],error)
                np.testing.assert_allclose(pred,expected,atol=1e-7,rtol=1e-6)
                counts["feature_predictions"]+=pred.size
    r2=1-np.sum((a["feature_predictions"]-targets[None,:,:])**2,axis=1)/np.sum((targets-targets.mean(0))**2,axis=0)
    np.testing.assert_allclose(r2,a["feature_r2"],rtol=1e-10,atol=1e-9)
    counts["conditions"]+=1
    if (ri+1)%192==0:
        print("verified",ri+1,"/1152",round(time.monotonic()-start,1),flush=True)

summary=json.loads((OUT/"summary.json").read_text())
for key,group in summary.items():
    selected=[r for r in rows if "/".join([r["model"],r["point"],r["condition"]])==key]
    rng=np.random.default_rng(spec["seed"])
    ix=rng.integers(0,len(selected),size=(spec["bootstrap"],len(selected)))
    for field,out in group.items():
        v=[r["metrics"][field] for r in selected if r["metrics"][field] is not None]
        assert len(v)==out["n"]
        if not v:continue
        if isinstance(v[0],bool):
            assert sum(v)==out["count"]
            continue
        arr=np.asarray(v)
        ordered=np.sort(arr[ix],axis=1)
        boot=(ordered[:,31]+ordered[:,32])/2
        ci=np.percentile(boot,[2.5,97.5])
        error=float(np.max(abs(ci-out["ci95"])))
        max_errors["bootstrap"]=max(max_errors["bootstrap"],error)
        np.testing.assert_allclose(ci,out["ci95"],rtol=1e-10,atol=1e-8)
        assert np.median(arr)==out["median"]
        assert min(arr)==out["min"] and max(arr)==out["max"]

write(OUT/"verification.json",dict(status="PASS",at=stamp(),seconds=time.monotonic()-start,
                                  counts=counts,max_errors=max_errors,
                                  reviewer="same-agent separate reconstruction; not independent review"))
print("Verification PASS",counts,max_errors,flush=True)
