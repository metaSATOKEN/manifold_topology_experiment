"""Frozen-fit evaluation and article-level confirmatory summary; no fitting."""
import numpy as np
import p2_io as disk
import p2_numeric as num
import p2_fit as fit


def regression_metrics(y, predictions):
    ok=np.isfinite(y)
    y=y[ok]
    if len(y)<3 or np.ptp(y)==0:
        return dict(valid=False,reason="constant_or_insufficient_test_target",n=int(len(y)))
    p=predictions[:,ok]
    disk.require(np.isfinite(p).all(),"Nonfinite prediction")
    sst=float(np.square(y-y.mean()).sum())
    sse=np.square(p-y).sum(1)
    r2=1-sse/sst
    return dict(valid=True,n=len(y),sst=sst,sse=sse.tolist(),r2=r2.tolist(),
                delta_r2=float((sse[0]-sse[1])/sst),
                random_advantage=float((sse[2:].mean()-sse[1])/sst))


def evaluate(raw, doc, point, condition, arrays, meta, counts):
    if not meta["valid"]:
        return {},dict(document_id=doc,point=point,condition=condition,valid=False,
                       reason="invalid_training_fit:"+meta["reason"])
    x,index,norms,reason=fit.prepare(raw,point,condition)
    if reason:
        return {},dict(document_id=doc,point=point,condition=condition,valid=False,reason=reason)
    bases=np.concatenate([arrays["components"][None],arrays["random_components"]])
    z=((x-arrays["mean"])@bases.reshape(-1,x.shape[1]).T).reshape(len(x),len(bases),-1)
    position_scores=np.full((len(x),len(bases)),np.nan)
    position_r=[]
    for g,sel in enumerate(meta["position_selection"]):
        if sel["component"] is None:
            position_r.append(None)
            continue
        position_scores[:,g]=z[:,g,sel["component"]-1]*sel["sign"]
        position_r.append(num.correlation(position_scores[:,g],raw["position"][index]))
    scores=z[:,0,:]
    centered=x-x.mean(0)
    total=float(np.square(centered).sum()/(len(x)-1))
    features=dict(position=raw["position"][index],token_chars=raw["token_chars"][index],
                  surprisal=raw["surprisal"][index],norm=norms)
    correlations={name:[None if name=="norm" and condition=="C4" else num.correlation(scores[:,i],f)
                        for i in range(scores.shape[1])] for name,f in features.items()}
    output=dict(kept_positions=index.astype(np.int64),pc_scores=scores,position_scores=position_scores)
    result=dict(document_id=doc,point=point,condition=condition,valid=True,n_tokens=len(x),
                position_r=position_r,
                position_random_advantage=None if any(r is None for r in position_r) else
                    float(position_r[0]-np.mean(position_r[1:])),
                direction_variance_fractions=None if total<=0 else
                    (np.var(scores,axis=0,ddof=1)/total).tolist(),
                subspace_variance_fraction=None if total<=0 else
                    float(np.var(scores,axis=0,ddof=1).sum()/total),
                total_test_variance=total,correlations=correlations,
                correlation_denominators={k:int(np.isfinite(f).sum()) for k,f in features.items()},
                heads={})
    for head in fit.HEADS:
        b,y=fit.controls(raw,index,norms,counts,head)
        output[head+"__target"]=y.astype(float)
        if not meta["heads"][head]["valid"]:
            result["heads"][head]=dict(valid=False,reason=meta["heads"][head]["reason"],n=int(np.isfinite(y).sum()))
            continue
        predictions=[]
        for model in range(len(bases)+1):
            pc=np.zeros_like(z[:,0,:]) if model==0 else z[:,model-1,:]
            xx=np.column_stack([b,pc])
            predictions.append(fit.predict(xx,*[arrays[head+"__"+k][model]
                for k in ("x_mean","x_scale","active","coef","y_mean")]))
        output[head+"__predictions"]=np.stack(predictions)
        result["heads"][head]=regression_metrics(y,output[head+"__predictions"])
    return output,result


def interval(values):
    cfg=disk.spec()
    v=np.asarray(values,float)
    if not len(v):
        return dict(n=0,planned=cfg["test_documents"],median=None,lower=None,upper=None)
    rng=np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    b=np.median(v[rng.integers(0,len(v),(cfg["bootstrap_repetitions"],len(v)))],axis=1)
    q=.05/(2*cfg["bootstrap_endpoints"])
    lo,hi=np.quantile(b,[q,1-q])
    return dict(n=len(v),planned=cfg["test_documents"],median=float(np.median(v)),
                lower=float(lo),upper=float(hi))


def judge_pair(left,right,t1,t2):
    minimum=disk.spec()["minimum_valid_documents"]
    if min(left["n"],right["n"])<minimum:
        return "INSUFFICIENT_DATA"
    if left["lower"]>=t1 and right["lower"]>=t2:
        return "SUPPORTED"
    if left["upper"]<t1 or right["upper"]<t2:
        return "REFUTED"
    return "INCONCLUSIVE"


def summarize(rows,complete):
    cfg=disk.spec()
    measures={k:[] for k in "ABCD"}
    perdoc=[];missing=[]
    for r in rows:
        if r["point"]!=cfg["primary_point"] or r["condition"]!=cfg["primary_condition"]:
            continue
        vals=dict.fromkeys("ABCD")
        if r["valid"]:
            vals["A"]=r["position_r"][0]
            vals["B"]=r["position_random_advantage"]
            head=r["heads"]["surprisal"]
            if head["valid"]:
                vals.update(C=head["delta_r2"],D=head["random_advantage"])
        perdoc.append(dict(document_id=r["document_id"],**vals))
        for k,v in vals.items():
            if v is None:
                missing.append(dict(document_id=r["document_id"],endpoint=k,
                    reason=r.get("reason","undefined_required_position_or_surprisal_statistic")))
            else:
                measures[k].append(v)
    endpoints={k:interval(v) for k,v in measures.items()}
    adjudication=dict(H1="PENDING_COMPLETE_RUN",H2="PENDING_COMPLETE_RUN")
    if complete:
        adjudication=dict(
            H1=judge_pair(endpoints["A"],endpoints["B"],cfg["position_min_correlation"],cfg["position_min_advantage"]),
            H2=judge_pair(endpoints["C"],endpoints["D"],cfg["surprisal_min_gain"],cfg["surprisal_min_random_advantage"]))
    return dict(primary_point=cfg["primary_point"],primary_condition=cfg["primary_condition"],
                endpoints=endpoints,scientific_adjudication=adjudication,missing=missing,
                document_endpoints=perdoc,
                uncertainty="Test-article percentile bootstrap, 10000 replicates, 98.75% each; training fit fixed")
