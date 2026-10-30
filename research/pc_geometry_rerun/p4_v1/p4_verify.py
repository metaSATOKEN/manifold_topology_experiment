"""Saved-observation audit. No LM imports or forward calls."""
from pathlib import Path
import sys
import numpy as np
import p4_io as disk
import p4_numeric as num
import p4_fit as fit
import p4_analysis as analysis


def near(a,b):
    if isinstance(a,dict):
        disk.require(set(a)==set(b),'Audit inventory')
        for k in a:near(a[k],b[k])
    elif isinstance(a,list):
        disk.require(len(a)==len(b),'Audit length')
        for x,y in zip(a,b):near(x,y)
    elif isinstance(a,(str,bool)) or a is None:disk.require(a==b,'Audit label')
    else:np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-7)


def audit_equations(z,w,y,ids,m):
    """Direct gradients in the original joint system, not the Schur solver."""
    ordinary=ids is not None;k=disk.spec()['components'];q=z.shape[1]
    pred=fit.predict(z,w,m,ids);base=np.column_stack([np.ones(len(z)),z]) if ordinary else z
    if ordinary:
        inv=np.searchsorted(m['vocab'],ids);count=len(m['vocab'])
    maximum=0.
    for j in range(pred.shape[1]):
        residual=pred[:,j]-y;pen=np.repeat(float(m['alpha']),base.shape[1]);
        if ordinary:pen[0]=0.
        grad=base.T@residual+len(y)*pen*m['beta'][j]
        denom=max(1.,float(np.max(abs(base.T@y))))
        maximum=max(maximum,float(np.max(abs(grad)))/denom)
        if j:
            ww=w[:,(j-1)*k:j*k];g=ww.T@residual+len(y)*m['gammas'][j-1]*m['theta'][j-1]
            maximum=max(maximum,float(np.max(abs(g)))/max(1.,float(np.max(abs(ww.T@y)))))
        if ordinary:
            g=np.bincount(inv,weights=residual,minlength=count)+m['tau']*m['token_effect'][j]
            maximum=max(maximum,float(np.max(abs(g)))/max(1.,float(np.max(abs(y)))))
        else:
            # Direct augmented QR/SVD solution is independent of normal equations.
            xx=base if j==0 else np.column_stack([base,w[:,(j-1)*k:j*k]])
            pp=pen if j==0 else np.r_[pen,np.repeat(m['gammas'][j-1],k)]
            coef=np.linalg.lstsq(np.vstack([xx,np.diag(np.sqrt(len(y)*pp))]),np.r_[y,np.zeros(len(pp))],rcond=None)[0]
            want=m['beta'][j] if j==0 else np.r_[m['beta'][j],m['theta'][j-1]]
            np.testing.assert_allclose(coef,want,atol=1e-7,rtol=1e-6)
    disk.require(maximum<1e-6,'Joint normal equation residual')
    return maximum


def direct_predictions(raw,point,common,m,counts,within=False):
    x,b,ids,_=fit.features(raw,point,counts);k=disk.spec()['components']
    basis=np.concatenate([common['components'],common['random_components'].reshape(-1,x.shape[1])])
    idx=np.array([0,1,2,3,7]) if within else np.arange(8)
    beta=m['beta'] if within else m['beta'][:,1:]
    B=beta*common['feature_active'][idx]/common['feature_scale'][idx]
    state=np.zeros((len(beta),x.shape[1]));offset=np.zeros(len(beta)) if within else m['beta'][:,0].copy()
    if not within:offset-=B@common['feature_mean'][idx]
    for j,c in enumerate(m['theta'],1):
        sl=slice((j-1)*k,j*k);coef=c*common['score_active'][sl]/common['score_scale'][sl]
        state[j]=coef@basis[sl]
        if not within:offset[j]-=coef@common['score_mean'][sl]+state[j]@common['mean']
    pred=b[:,idx]@B.T+x@state.T+offset
    if within:return fit.center(ids,pred)[0]
    ix=np.searchsorted(m['vocab'],ids);known=ix<len(m['vocab']);ix=np.minimum(ix,len(m['vocab'])-1);known&=m['vocab'][ix]==ids
    return pred+m['token_effect'][:,ix].T*known[:,None]


def audit_geometry(h,row):
    for c,start in [('C0',0),('C1',1)]:
        x=h[start:].astype(float);norm=np.linalg.norm(x,axis=1);x-=x.mean(0)
        # SVD of the original rectangular state matrix, independent of Gram PCA.
        u,s,_=np.linalg.svd(x,full_matrices=False)
        total=np.square(s).sum()
        if total<=1e-20:disk.require(not row[c]['valid'],'Geometry zero variance');continue
        r=num.correlation(u[:,0]*s[0],norm)
        expected=dict(valid=True,tokens=len(x),V1=float(s[0]**2/total),V8=float(np.square(s[:8]).sum()/total),
            R1=None if r is None or (s[0]**2-s[1]**2)/max(s[0]**2,1e-30)<=1e-8 else abs(r),relative_gap12=float((s[0]**2-s[1]**2)/max(s[0]**2,1e-30)),norm_sd=float(norm.std()),norm_mean=float(norm.mean()))
        near(row[c],expected)


def verify(root,output):
    root=Path(root);disk.check_fit(root);done=disk.committed(root)
    disk.require(set(done)==set(disk.split_keys('train')+disk.split_keys('test')),'Incomplete extraction')
    ev=disk.events(root);seal=disk.read(root/'fit/COMMIT.json')['sealed_at'];fit_sha=disk.digest(root/'fit/COMMIT.json')
    for key in disk.spec()['models']:
        for f in disk.fixtures():
            commits=[e for e in ev if e['kind']=='DOCUMENT_COMMITTED' and e.get('model_key')==key and e.get('document_id')==f['id']]
            disk.require(len(commits)==1,'Missing or duplicate document commit event')
            meta=disk.read(root/'docs'/disk.doc_key(key,f['id'])/'doc.json')
            starts=[e for e in ev if e['kind']=='FORWARD_STARTED' and e.get('model_key')==key and e.get('document_id')==f['id'] and e['attempt']==meta['attempt']]
            disk.require(len(starts)==1 and starts[0]['at']<=commits[0]['at'] and commits[0]['attempt']==meta['attempt'],'Missing/misaligned committed forward event')

    for e in ev:
        if e['kind']=='FORWARD_STARTED' and e['split']=='test':
            disk.require(e['at']>=seal and e['fit_commit_sha256']==fit_sha,'Test before fit seal')
    checked=0;max_gradient=0.;max_pred=0.;trainfixtures=[f for f in disk.fixtures() if f['split']=='train']
    for key in disk.spec()['models']:
        raws=[num.validate_raw(root,key,f['id'])[0] for f in trainfixtures]
        counts=num.load_npz(root/'fit'/key/'frequency.npz')['counts']
        np.testing.assert_array_equal(counts,np.bincount(np.concatenate([r['token_id'] for r in raws]),minlength=disk.model(key)['vocab']))
        for point in disk.spec()['points']:
            common,models,_=analysis.read_fits(root,key,point)
            # Rebuild all training-derived quantities and CV choices from raw data.
            c,ms,meta=fit.fit_point(raws,trainfixtures,key,point,counts,audit=common)
            for n in common:np.testing.assert_allclose(common[n],c[n],atol=1e-8,rtol=1e-7)
            for name,m in models.items():
                for n in m:np.testing.assert_allclose(m[n],ms[name][n],atol=1e-8,rtol=1e-7)
            near(meta,disk.read(root/'fit'/key/point/'cv.json'))
            # Direct gradients and QR for the within fit, avoiding Schur algebra.
            blocks=[fit.features(r,point,counts) for r in raws]
            basis=np.concatenate([common['components'],common['random_components'].reshape(-1,disk.model(key)['hidden'])])
            z=fit.transform(np.concatenate([b[1] for b in blocks]),common,'feature')
            w=fit.transform((np.concatenate([b[0] for b in blocks])-common['mean'])@basis.T,common,'score')
            ids=np.concatenate([b[2] for b in blocks]);y=np.concatenate([b[3] for b in blocks]);lengths=[len(b[3]) for b in blocks]
            zc,wc,yc,mask=fit.within_arrays(ids,z,w,y,lengths)
            for name,zz,ww,yy,jj in [('O',z,w,y,ids),('R',z[mask],w[mask],y[mask],ids[mask]),('W',zc[mask],wc[mask],yc[mask],None)]:
                max_gradient=max(max_gradient,audit_equations(zz,ww,yy,jj,models[name]))
            print(f'P4 audit fits {key}/{point}: PCA, all CV choices, joint gradients, within QR passed.',flush=True)
            for f in disk.fixtures():
                if f['split']!='test':continue
                raw,_=num.validate_raw(root,key,f['id']);folder=root/'evaluation'/key/point/f['id']
                for n,h in disk.read(folder/'COMMIT.json').items():disk.require(disk.digest(folder/n)==h,'Evaluation checksum')
                row=disk.read(folder/'metrics.json');saved=num.load_npz(folder/'predictions.npz')
                expected,arrays=analysis.evaluate_one(raw,key,point,common,models,counts);expected['document_id']=f['id'];near(row,expected)
                for n in saved:np.testing.assert_allclose(saved[n],arrays[n],atol=1e-8,rtol=1e-7)
                for name in ['O','R','W']:
                    direct=direct_predictions(raw,point,common,models[name],counts,name=='W')
                    max_pred=max(max_pred,float(np.max(abs(direct-saved[name]))))
                    np.testing.assert_allclose(direct,saved[name],atol=1e-7,rtol=1e-6)
                    checked+=direct.size
                audit_geometry(raw['h_'+point],row['geometry'])
            print(f'P4 audit evaluation {key}/{point}: saved predictions and rectangular SVD passed.',flush=True)
        del raws
    result=analysis.summarize(root)
    disk.require(result['complete'],'Incomplete analysis')
    result.update(audit=dict(status='PASS',predictions_reconstructed=checked,max_prediction_difference=max_pred,max_joint_gradient_relative=max_gradient,
        training_CV_recomputed=True,geometry_gram_vs_rectangular_svd=True,same_agent_self_audit=True,independent_reviewer=False),local_receipt_verified=False)
    disk.write(output,result)
    return result


if __name__=='__main__':verify(sys.argv[1],sys.argv[2])
