"""All fixed evaluation endpoints, no training/tuning and no model imports."""
from pathlib import Path
import sys
import numpy as np
import p4_io as disk
import p4_numeric as num
import p4_fit as fit


def metrics(y,pred):
    ss=float(np.square(y-y.mean()).sum())
    if len(y)<3 or ss<=1e-12:return dict(valid=False,reason='constant_or_insufficient_target',tokens=len(y))
    r2=1-np.square(pred-y[:,None]).sum(0)/ss;gain=r2[1:]-r2[0]
    return dict(valid=True,tokens=len(y),target_ss=ss,r2=r2.tolist(),pc_gain=float(gain[0]),random_mean_gain=float(gain[1:].mean()),advantage=float(gain[0]-gain[1:].mean()))


def geometry(h):
    x=h.astype(float);norm=np.linalg.norm(x,axis=1);x-=x.mean(0)
    gram=x@x.T;total=float(np.trace(gram))
    if total<=1e-20:return dict(valid=False,reason='zero_state_variance',tokens=len(x))
    vals,vec=np.linalg.eigh(gram);vals=vals[::-1];score=vec[:,-1]*np.sqrt(max(vals[0],0.))
    r=num.correlation(score,norm)
    return dict(valid=True,tokens=len(x),V1=float(vals[0]/total),V8=float(vals[:8].sum()/total),R1=None if r is None or (vals[0]-vals[1])/max(vals[0],1e-30)<=1e-8 else abs(r),relative_gap12=float((vals[0]-vals[1])/max(vals[0],1e-30)),norm_sd=float(norm.std()),norm_mean=float(norm.mean()))


def variance_capture(x,ids,components):
    total=x-x.mean(0);within,repeat,groups=fit.center(ids,x)
    st=total@components.T;sw=within@components.T
    a=float(np.square(total).sum());b=float(np.square(within).sum())
    return dict(total_state_ss=a,within_state_ss=b,between_state_ss=a-b,top8_total_fraction=None if a<=1e-20 else float(np.square(st).sum()/a),top8_within_fraction=None if b<=1e-20 else float(np.square(sw).sum()/b),within_groups=groups,repeated_tokens=int(repeat.sum()))


def read_fits(root,key,point):
    folder=Path(root)/'fit'/key/point
    common=num.load_npz(folder/'common.npz');models={name:num.load_npz(folder/(name+'.npz')) for name in ['O','R','W']}
    counts=num.load_npz(Path(root)/'fit'/key/'frequency.npz')['counts']
    return common,models,counts


def evaluate_one(raw,key,point,common,models,counts):
    x,b,ids,y=fit.features(raw,point,counts)
    bases=np.concatenate([common['components'],common['random_components'].reshape(-1,x.shape[1])])
    z=fit.transform(b,common,'feature');w=fit.transform((x-common['mean'])@bases.T,common,'score')
    cy,repeat,groups=fit.center(ids,y)
    zc=fit.center(ids,z[:,[0,1,2,3,7]])[0];wc=fit.center(ids,w)[0]
    pred={name:fit.predict(z,w,models[name],ids) for name in ['O','R']}
    pred['W']=fit.predict(zc,wc,models['W'])
    within={};ordinary={name:metrics(y,pred[name]) for name in ['O','R']}
    eligible=int(repeat.sum())>=disk.spec()['within_min_tokens'] and groups>=disk.spec()['within_min_groups']
    for name,v in pred.items():
        within[name]=metrics(cy[repeat],fit.center(ids,v)[0][repeat]) if eligible else dict(valid=False,reason='within_coverage_below_32_tokens_or_8_groups',tokens=int(repeat.sum()))
        within[name]['groups']=groups
    geo={c:geometry(raw['h_'+point][start:]) for c,start in [('C0',0),('C1',1)]}
    coverage=dict(tokens=len(y),repeated_tokens=int(repeat.sum()),repeated_groups=groups,
        training_ID_known_fraction=float(np.isin(ids,models['O']['vocab']).mean()))
    row=dict(model=key,point=point,ordinary=ordinary,within=within,geometry=geo,coverage=coverage,shared_top8_capture=variance_capture(x,ids,common['components']))
    arrays={**pred,'within_mask':repeat.astype(np.int64),'within_target':cy}
    return row,arrays


def evaluate(root,key):
    root=Path(root);disk.check_fit(root);done=disk.committed(root)
    for point in disk.spec()['points']:
        common,models,counts=read_fits(root,key,point)
        for f in disk.fixtures():
            if f['split']!='test' or disk.doc_key(key,f['id']) not in done:continue
            folder=root/'evaluation'/key/point/f['id']
            if (folder/'COMMIT.json').exists():
                for n,h in disk.read(folder/'COMMIT.json').items():disk.require(disk.digest(folder/n)==h,'Changed evaluation')
                continue
            raw,_=num.validate_raw(root,key,f['id']);row,arrays=evaluate_one(raw,key,point,common,models,counts)
            row['document_id']=f['id'];disk.write(folder/'metrics.json',row,immutable=True);num.save_npz(folder/'predictions.npz',arrays)
            disk.write(folder/'COMMIT.json',{n:disk.digest(folder/n) for n in ['metrics.json','predictions.npz']},immutable=True)
        print(f'P4 evaluated {key}/{point}.',flush=True)


def describe(values,primary=False):
    a=np.array([np.nan if v is None else v for v in values],float);valid=np.isfinite(a);a=a[valid]
    if not len(a):return dict(n=0,missing=len(values),median=None,lo=None,hi=None,positive=0)
    # Same article resampling when endpoints share the same validity mask.
    rng=np.random.Generator(np.random.PCG64(disk.spec()['bootstrap_seed']))
    sample=rng.integers(0,len(a),size=(disk.spec()['bootstrap_repetitions'],len(a)))
    med=np.median(a[sample],axis=1);tail=.05/(2*disk.spec()['primary_endpoints']) if primary else .025
    lo,hi=np.quantile(med,[tail,1-tail])
    return dict(n=len(a),missing=len(values)-len(a),median=float(np.median(a)),lo=float(lo),hi=float(hi),positive=int((a>0).sum()),confidence=1-2*tail)


def judge(gain,advantage,complete):
    if not complete or min(gain['n'],advantage['n'])<disk.spec()['minimum_valid_documents']:return 'NOT_ADJUDICABLE'
    t1=disk.spec()['within_min_gain'];t2=disk.spec()['within_min_random_advantage']
    if gain['lo']>t1 and advantage['lo']>t2:return 'SUPPORTS_MINIMUM_CRITERION'
    if gain['hi']<=t1 or advantage['hi']<=t2:return 'EVIDENCE_BELOW_MINIMUM_CRITERION'
    return 'INCONCLUSIVE'


def summarize(root):
    root=Path(root);done=disk.committed(root);complete=len(done)==len(disk.fixtures())*len(disk.spec()['models']) and all((root/'evaluation'/k/p/f['id']/'COMMIT.json').exists() for k in disk.spec()['models'] for p in disk.spec()['points'] for f in disk.fixtures() if f['split']=='test')
    results={};article_count=0
    for key in disk.spec()['models']:
        results[key]={}
        for point in disk.spec()['points']:
            rows=[]
            for f in disk.fixtures():
                if f['split']!='test':continue
                p=root/'evaluation'/key/point/f['id']/'metrics.json'
                rows.append(disk.read(p) if p.exists() else None)
            article_count+=sum(r is not None for r in rows)
            table={mode:{name:{metric:describe([r[mode][name].get(metric) if r and r[mode][name]['valid'] else None for r in rows],primary=mode=='within' and name=='W') for metric in ['pc_gain','advantage','random_mean_gain']} for name in (['O','R'] if mode=='ordinary' else ['O','R','W'])} for mode in ['ordinary','within']}
            # Only the two preregistered W endpoints receive the primary family label.
            table['within']['W']['random_mean_gain']=describe([r['within']['W'].get('random_mean_gain') if r and r['within']['W']['valid'] else None for r in rows])
            geo={c:{q:describe([r['geometry'][c].get(q) if r and r['geometry'][c]['valid'] else None for r in rows]) for q in ['R1','V1','V8','relative_gap12']} for c in ['C0','C1']}
            delta={q:describe([r['geometry']['C1'][q]-r['geometry']['C0'][q] if r and r['geometry']['C0'].get(q) is not None and r['geometry']['C1'].get(q) is not None else None for r in rows]) for q in ['R1','V1','V8']}
            diff={a+'_to_'+b:describe([r['within'][b]['advantage']-r['within'][a]['advantage'] if r and r['within'][a]['valid'] and r['within'][b]['valid'] else None for r in rows]) for a,b in [('O','R'),('R','W'),('O','W')]}
            primary=table['within']['W'];decision=judge(primary['pc_gain'],primary['advantage'],complete)
            results[key][point]=dict(metrics=table,geometry=geo,geometry_C1_minus_C0=delta,within_advantage_changes=diff,decision=decision,
                shared_top8_capture={q:describe([r['shared_top8_capture'][q] if r else None for r in rows]) for q in ['top8_total_fraction','top8_within_fraction']})
    ev=disk.events(root)
    return dict(run_id=disk.check_run(root)['run_id'],planned_model_documents=len(disk.fixtures())*len(disk.spec()["models"]),committed_model_documents=len(done),planned_evaluation_article_points=disk.spec()["test_documents"]*len(disk.spec()["models"])*len(disk.spec()["points"]),analyzed_evaluation_article_points=article_count,
        forward_attempts=sum(e['kind']=='FORWARD_STARTED' for e in ev),worker_failures=sum(e['kind']=='WORKER_FAILURE' for e in ev),
        incomplete_model_documents=len(disk.fixtures())*len(disk.spec()["models"])-len(done),scientific_results=results,complete=complete and article_count==disk.spec()["test_documents"]*len(disk.spec()["models"])*len(disk.spec()["points"]),
        stopping_rule='This is the final new comparison; no automatic model/layer/intervention extension',causal_mechanism='UNRESOLVED')


if __name__=='__main__':evaluate(Path(sys.argv[1]),sys.argv[2])
