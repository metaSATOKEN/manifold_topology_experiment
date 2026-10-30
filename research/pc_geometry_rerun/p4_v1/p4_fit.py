"""Train-only shared PCA and joint/within-ID readouts; NumPy, no LM imports."""
from pathlib import Path
import os,sys,uuid
import numpy as np
import p4_io as disk
import p4_numeric as num


def center(ids,values):
    _,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    v=np.asarray(values,float);one=v.ndim==1
    if one:v=v[:,None]
    sums=np.zeros((len(count),v.shape[1]));np.add.at(sums,inv,v)
    out=v-(sums/count[:,None])[inv]
    return (out[:,0] if one else out),count[inv]>=2,int((count>=2).sum())


def scale(x):
    mu=x.mean(0);sd=x.std(0);active=sd>1e-12
    return mu,np.where(active,sd,1.),active.astype(np.int64)


def transform(x,m,prefix):
    return (x-m[prefix+'_mean'])/m[prefix+'_scale']*m[prefix+'_active']


def features(raw,point,counts):
    t=raw['position'][1:].astype(float);n=len(raw['position']);p=t/(n-1)
    x=raw['h_'+point][1:].astype(float);ids=raw['token_id'][1:]
    b=np.column_stack([p,p*p,p**3,np.log1p(t)/np.log(n),np.log1p(raw['token_chars'][1:]),
        np.log1p(raw['token_utf8_bytes'][1:]),-np.log((counts[ids]+1)/(counts.sum()+len(counts))),
        np.log1p(np.linalg.norm(x,axis=1))])
    return x,b,ids,raw['surprisal'][1:]


def basis_seed(key,point,rep):
    return int(disk.sha(f"{disk.spec()['random_seed']}:{key}:{point}:{rep}".encode())[:16],16)


def make_common(raws,key,point,counts,audit=None):
    blocks=[features(r,point,counts) for r in raws]
    x=np.concatenate([b[0] for b in blocks]);b=np.concatenate([b[1] for b in blocks])
    mean=x.mean(0);x-=mean;cov=x.T@x/(len(x)-1);total=float(np.trace(cov));k=disk.spec()['components']
    disk.require(total>0,'Zero training variance')
    if audit is None:
        eig,v=np.linalg.eigh(cov);eig=eig[::-1].copy();pc=v[:,::-1][:,:k].T.copy()
        for row in pc:
            if row[np.argmax(abs(row))]<0:row*=-1
    else:
        eig=np.linalg.eigvalsh(cov)[::-1];pc=audit['components']
        np.testing.assert_allclose(audit['mean'],mean,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(audit['eigenvalues'],eig,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(pc@pc.T,np.eye(k),atol=1e-8,rtol=0)
        disk.require(np.max(abs(cov@pc.T-pc.T*eig[:k]))/total<1e-8,'PCA eigen residual')
        disk.require(all(v[np.argmax(abs(v))]>=0 for v in pc),'PCA sign')
    disk.require(eig[k-1]>total*1e-12,'Training rank < 8')
    random=[]
    for rep in range(disk.spec()['random_subspaces']):
        q,_=np.linalg.qr(np.random.Generator(np.random.PCG64(basis_seed(key,point,rep))).normal(size=(x.shape[1],k)),mode='reduced')
        r=q.T.copy()
        for v in r:
            if v[np.argmax(abs(v))]<0:v*=-1
        random.append(r)
    random=np.stack(random);basis=np.concatenate([pc,random.reshape(-1,x.shape[1])])
    scores=x@basis.T
    common=dict(mean=mean,components=pc,random_components=random,eigenvalues=eig)
    for prefix,a in [('feature',b),('score',scores)]:
        mu,sd,active=scale(a);common.update({prefix+'_mean':mu,prefix+'_scale':sd,prefix+'_active':active})
    z=transform(b,common,'feature');w=transform(scores,common,'score')
    ids=np.concatenate([r[2] for r in blocks]);y=np.concatenate([r[3] for r in blocks])
    return common,z,w,ids,y,[len(r[3]) for r in blocks]


class Context:
    """Sufficient statistics; exact Schur elimination of penalized ID intercepts."""
    def __init__(self,z,w,y,ids=None):
        self.ordinary=ids is not None
        if self.ordinary:z=np.column_stack([np.ones(len(z)),z])
        self.q=z.shape[1];self.k=disk.spec()['components'];self.n=len(y)
        self.x=np.column_stack([z,w]);self.xx=self.x.T@self.x;self.xy=self.x.T@y
        self.cache={}
        if ids is not None:
            self.vocab,inv,self.count=np.unique(ids,return_inverse=True,return_counts=True)
            self.sums=np.zeros((len(self.vocab),self.x.shape[1]));self.sy=np.zeros(len(self.vocab))
            np.add.at(self.sums,inv,self.x);np.add.at(self.sy,inv,y)
    def matrices(self,tau):
        if not self.ordinary:return self.xx,self.xy
        if tau not in self.cache:
            inv=1/(self.count+tau)
            gram=self.xx-self.sums.T@(self.sums*inv[:,None]);rhs=self.xy-self.sums.T@(self.sy*inv)
            self.cache[tau]=(gram,rhs)
        return self.cache[tau]
    def solve(self,alpha,gammas,tau=1.):
        gram,rhs=self.matrices(float(tau));betas=[];theta=[];effects=[];errors=[]
        for j in range(len(gammas)+1):
            idx=np.arange(self.q) if j==0 else np.r_[np.arange(self.q),np.arange(self.q+(j-1)*self.k,self.q+j*self.k)]
            pen=np.repeat(float(alpha),self.q)
            if self.ordinary:pen[0]=0.
            if j:pen=np.r_[pen,np.repeat(gammas[j-1],self.k)]
            mat=gram[np.ix_(idx,idx)]+self.n*np.diag(pen);coef=np.linalg.solve(mat,rhs[idx])
            errors.append(float(np.max(abs(mat@coef-rhs[idx]))/max(1.,float(np.max(abs(rhs[idx]))))))
            betas.append(coef[:self.q])
            if j:theta.append(coef[self.q:])
            if self.ordinary:effects.append((self.sy-self.sums[:,idx]@coef)/(self.count+tau))
        m=dict(beta=np.array(betas),theta=np.array(theta).reshape(-1,self.k),alpha=np.array(alpha),gammas=np.array(gammas),normal_residual=np.array(errors))
        if self.ordinary:m.update(vocab=self.vocab,token_effect=np.array(effects),tau=np.array(tau))
        return m


def predict(z,w,m,ids=None):
    if ids is not None:z=np.column_stack([np.ones(len(z)),z])
    pred=z@m['beta'].T
    if ids is not None:
        ix=np.searchsorted(m['vocab'],ids);known=ix<len(m['vocab']);ix=np.minimum(ix,len(m['vocab'])-1);known&=m['vocab'][ix]==ids
        pred+=m['token_effect'][:,ix].T*known[:,None]
    k=disk.spec()['components']
    for j,c in enumerate(m['theta']):pred[:,j+1]+=w[:,j*k:(j+1)*k]@c
    return pred


def cv_fit(z,w,y,folds,ids=None):
    cfg=disk.spec();grid=cfg['alphas'];ng=w.shape[1]//cfg['components'];data=[]
    for fold in range(4):
        tr,va=folds!=fold,folds==fold
        disk.require(tr.any() and va.any(),'Empty CV fold')
        data.append((Context(z[tr],w[tr],y[tr],None if ids is None else ids[tr]),va))
    base=[]
    for alpha in grid:
        for tau in (cfg['taus'] if ids is not None else [1.]):
            sse=0.
            for ctx,va in data:
                m=ctx.solve(alpha,[],tau);pred=predict(z[va],w[va],m,None if ids is None else ids[va])[:,0]
                sse+=float(np.square(y[va]-pred).sum())
            base.append(dict(alpha=alpha,tau=tau,mse=sse/len(y)))
    selected=min(base,key=lambda r:r['mse']);alpha=selected['alpha'];tau=selected['tau']
    losses=[]
    for gamma in grid:
        sse=np.zeros(ng)
        for ctx,va in data:
            m=ctx.solve(alpha,np.repeat(gamma,ng),tau);pred=predict(z[va],w[va],m,None if ids is None else ids[va])
            sse+=np.square(pred[:,1:]-y[va,None]).sum(0)
        losses.append(sse/len(y))
    losses=np.array(losses);gammas=np.array(grid)[np.argmin(losses,axis=0)]
    model=Context(z,w,y,ids).solve(alpha,gammas,tau)
    return model,dict(base_candidates=base,selected_base=selected,gamma_grid=grid,gamma_mse=losses.tolist(),selected_gammas=gammas.tolist(),fold_tokens=[int(v.sum()) for _,v in data],cv_condition='Full training unsupervised PCA/scales/frequency held fixed; no evaluation articles')


def within_arrays(ids,z,w,y,lengths):
    zc=[];wc=[];yc=[];mask=[];start=0
    for n in lengths:
        sl=slice(start,start+n);j=ids[sl]
        c,m,_=center(j,z[sl][:,[0,1,2,3,7]]);zc.append(c);wc.append(center(j,w[sl])[0]);yc.append(center(j,y[sl])[0]);mask.append(m);start+=n
    return np.concatenate(zc),np.concatenate(wc),np.concatenate(yc),np.concatenate(mask)


def fit_point(raws,fixtures,key,point,counts,audit=None):
    common,z,w,ids,y,lengths=make_common(raws,key,point,counts,audit)
    folds=np.concatenate([np.repeat(f['fold'],n) for f,n in zip(fixtures,lengths)])
    ordinary,ocv=cv_fit(z,w,y,folds,ids)
    zc,wc,yc,repeat=within_arrays(ids,z,w,y,lengths)
    disk.require(repeat.sum()>=128,'Insufficient repeated training tokens')
    repeated=Context(z[repeat],w[repeat],y[repeat],ids[repeat]).solve(float(ordinary['alpha']),ordinary['gammas'],float(ordinary['tau']))
    within,wcv=cv_fit(zc[repeat],wc[repeat],yc[repeat],folds[repeat])
    meta=dict(model=key,point=point,train_tokens=len(y),repeated_train_tokens=int(repeat.sum()),O=ocv,W=wcv,
        R='Uses O alpha/tau/gammas; same full-training features and bases; repeated rows only')
    return common,{'O':ordinary,'R':repeated,'W':within},meta


def seal(root):
    root=Path(root);done=disk.committed(root)
    if (root/'fit/COMMIT.json').exists():disk.check_fit(root);return
    disk.require(set(disk.split_keys('train'))<=set(done),'All three models training required')
    disk.require(not(set(done)&set(disk.split_keys('test'))),'Fit after test is forbidden')
    folder=root/'fit'
    if folder.exists():
        orphan=root/'attempts'/('fit_'+uuid.uuid4().hex[:12])/'incomplete_fit'
        orphan.parent.mkdir(parents=True);os.replace(folder,orphan)
    folder.mkdir();fixtures=[f for f in disk.fixtures() if f['split']=='train']
    for key in disk.spec()['models']:
        raws=[num.validate_raw(root,key,f['id'])[0] for f in fixtures]
        counts=np.bincount(np.concatenate([r['token_id'] for r in raws]),minlength=disk.model(key)['vocab']).astype(np.int64)
        num.save_npz(folder/key/'frequency.npz',{'counts':counts})
        for point in disk.spec()['points']:
            print(f'P4 fit {key}/{point}: training-only PCA and O/R/W CV.',flush=True)
            common,models,meta=fit_point(raws,fixtures,key,point,counts)
            dest=folder/key/point;num.save_npz(dest/'common.npz',common)
            for name,m in models.items():num.save_npz(dest/(name+'.npz'),m)
            disk.write(dest/'cv.json',meta,immutable=True)
        del raws
    disk.write(folder/'COMMIT.json',dict(sealed_at=disk.stamp(),capsule=disk.capsule(),
        train_commits={k:done[k] for k in disk.split_keys('train')},
        files={str(p.relative_to(folder)):disk.digest(p) for p in disk.files(folder)}),immutable=True)
    disk.check_fit(root)
    print('P4: all three-model fits sealed BEFORE any evaluation forward.',flush=True)


if __name__=='__main__':seal(sys.argv[1])
