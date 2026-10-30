"""Meaningful numerical, preservation, archive and adapter tests; synthetic data only."""
from pathlib import Path
import json,io,zipfile,types
import numpy as np
import pytest
import p4_io as disk
import p4_fit as fit
import p4_numeric as num
import p4_analysis as ana
import p4_verify as ver
import p4_worker as worker


@pytest.mark.parametrize('ordinary',[True,False])
@pytest.mark.parametrize('alpha',[.0001,.1,1.])
def test_joint_ridge_against_augmented_lstsq(ordinary,alpha):
    rng=np.random.default_rng(86);n=160;z=rng.normal(size=(n,8 if ordinary else 5));w=rng.normal(size=(n,16));y=rng.normal(size=n);ids=np.arange(n)%13 if ordinary else None
    tau=2.;gammas=np.array([.01,.5]);m=fit.Context(z,w,y,ids).solve(alpha,gammas,tau)
    pred=fit.predict(z,w,m,ids)
    for j in range(3):
        x=np.column_stack([np.ones(n),z]) if ordinary else z.copy();pen=np.r_[0.,np.repeat(alpha,8)] if ordinary else np.repeat(alpha,5)
        if j:x=np.column_stack([x,w[:,(j-1)*8:j*8]]);pen=np.r_[pen,np.repeat(gammas[j-1],8)]
        pen=pen*n
        if ordinary:
            x=np.column_stack([x,np.eye(13)[ids]]);pen=np.r_[pen,np.repeat(tau,13)]
        coef=np.linalg.lstsq(np.vstack([x,np.diag(np.sqrt(pen))]),np.r_[y,np.zeros(len(pen))],rcond=None)[0]
        np.testing.assert_allclose(x@coef,pred[:,j],atol=1e-10,rtol=1e-9)
    assert ver.audit_equations(z,w,y,ids,m)<1e-10


def test_center_singletons_and_article_groups():
    ids=np.array([1,1,2,3,3]);y=np.array([2.,4,9,10,14]);c,m,g=fit.center(ids,y)
    np.testing.assert_array_equal(m,[1,1,0,1,1]);np.testing.assert_allclose(c,[-1,1,0,-2,2]);assert g==2
    # Same ID in different articles must not share an article mean.
    z=np.tile(y[:,None],(1,8));w=np.tile(y[:,None],(1,16))
    zz,ww,yy,mask=fit.within_arrays(np.r_[ids,ids],np.vstack([z,z+100]),np.vstack([w,w+100]),np.r_[y,y+100],[5,5])
    np.testing.assert_allclose(yy,np.r_[c,c]);assert mask.sum()==8


def test_unknown_ID_zero_effect():
    rng=np.random.default_rng(2);z=rng.normal(size=(60,8));w=rng.normal(size=(60,8));ids=np.arange(60)%7
    m=fit.Context(z,w,rng.normal(size=60),ids).solve(.1,[.1],1.)
    unknown=np.repeat(999,60);p=fit.predict(z,w,m,unknown);expected=np.column_stack([np.ones(60),z])@m['beta'].T;expected[:,1]+=w@m['theta'][0]
    np.testing.assert_allclose(p,expected)


def synthetic_raw(rng,n=48,d=16,vocab=20):
    ids=np.arange(n)%vocab;y=np.r_[np.nan,rng.uniform(1,8,n-1)]
    return dict(token_id=ids.astype(np.int64),position=np.arange(n,dtype=np.int64),token_chars=(ids%4+1).astype(np.int64),token_utf8_bytes=(ids%4+1).astype(np.int64),logit_source_index=np.arange(-1,n-1,dtype=np.int64),surprisal=y,target_logit=np.r_[np.nan,np.zeros(n-1)],logsumexp=y.copy(),h_middle=rng.normal(size=(n,d)).astype(np.float32),h_final_norm=rng.normal(size=(n,d)).astype(np.float32))


def test_fit_evaluate_and_raw_coordinate_predictions(monkeypatch):
    cfg=disk.spec();cfg.update(random_subspaces=2,bootstrap_repetitions=200,minimum_valid_documents=3,within_min_tokens=10,within_min_groups=3)
    monkeypatch.setattr(disk,'spec',lambda:cfg)
    rng=np.random.default_rng(819);raws=[synthetic_raw(rng,n=48+i%3) for i in range(8)];fixtures=[dict(fold=i%4) for i in range(8)]
    counts=np.bincount(np.concatenate([r['token_id'] for r in raws]),minlength=20)
    common,models,meta=fit.fit_point(raws,fixtures,'gpt2','middle',counts)
    raw=synthetic_raw(rng,n=51);row,arrays=ana.evaluate_one(raw,'gpt2','middle',common,models,counts)
    for name in ['O','R','W']:
        expected=ver.direct_predictions(raw,'middle',common,models[name],counts,name=='W')
        np.testing.assert_allclose(expected,arrays[name],atol=1e-10,rtol=1e-9)
    ver.audit_geometry(raw['h_middle'],row['geometry'])
    # Audit PCA using eigvalsh/eigen equation, then reconstruct all CV and fits.
    c,ms,meta2=fit.fit_point(raws,fixtures,'gpt2','middle',counts,audit=common)
    ver.near(meta,meta2)
    for n in models:
        for k in models[n]:np.testing.assert_allclose(models[n][k],ms[n][k])
    assert row['within']['W']['valid']
    assert meta['O']['selected_base']==min(meta['O']['base_candidates'],key=lambda x:x['mse'])


def test_bootstrap_and_decision_not_missing_as_negative(monkeypatch):
    cfg=disk.spec();cfg['bootstrap_repetitions']=200;monkeypatch.setattr(disk,'spec',lambda:cfg)
    a=ana.describe([.02]*64,True);b=ana.describe([.006]*64,True)
    assert ana.judge(a,b,True)=='SUPPORTS_MINIMUM_CRITERION'
    assert ana.judge(a,b,False)=='NOT_ADJUDICABLE'
    assert ana.judge(ana.describe([None]*64,True),b,True)=='NOT_ADJUDICABLE'
    assert ana.judge(a,ana.describe([-.01]*64,True),True)=='EVIDENCE_BELOW_MINIMUM_CRITERION'
    c=dict(b,lo=.001,hi=.02);assert ana.judge(a,c,True)=='INCONCLUSIVE'
    assert abs(a['confidence']-(1-.05/12))<1e-15


def test_environment_JSON_roundtrip_and_real_change():
    env=dict(gpu_name='A100',compute_capability=(8,0),python='3.12',torch='2.8',cuda_build='12.8',nvidia_smi='test',driver_query_exit=0,packages=[('numpy','2.2.6')],tf32=False,dtype='float32',attention='eager')
    previous=json.loads(json.dumps(env));worker.require_compatible(previous,env)
    with pytest.raises(ValueError,match='gpu_name'):worker.require_compatible(previous,dict(env,gpu_name='Blackwell'))


@pytest.mark.parametrize('name',['../escape','/absolute','a\\b','a/../b','a//b','./x',''])
def test_archive_paths_rejected(name):
    with pytest.raises(ValueError):disk.safe_name(name)


def test_part_set_restore_missing_mixed_corruption(tmp_path,monkeypatch):
    # Production partitioning/restore, reduced cap for a real multipart fixture.
    package=tmp_path/'package';package.mkdir();(package/'spec.json').write_text('{}');(package/'fixtures.json').write_text('[]')
    monkeypatch.setattr(disk,'HERE',package);monkeypatch.setattr(disk,'SOURCE_NAMES',('spec.json','fixtures.json'))
    monkeypatch.setattr(disk,'spec',lambda:{'models':['gpt2']});monkeypatch.setattr(disk,'fixtures',lambda:[])
    monkeypatch.setattr(disk,'PART_TARGET',10000)
    root=tmp_path/'run';disk.initialize(root)
    for j in range(6):disk.atomic(root/'observations'/f'{j}.dat',np.random.default_rng(j).bytes(7000))
    paths=disk.snapshot(root,tmp_path/'archives','FINAL');assert len(paths)>3
    bundle,_=disk.inspect_bundle(paths);restored=tmp_path/'restored';disk.restore(paths,restored)
    assert {str(p.relative_to(root)):disk.digest(p) for p in disk.files(root)}=={str(p.relative_to(restored)):disk.digest(p) for p in disk.files(restored)}
    with pytest.raises(ValueError,match='Missing PART'):disk.inspect_bundle(paths[1:])
    other=disk.snapshot(root,tmp_path/'archives','CHECKPOINT')
    with pytest.raises(ValueError,match='Mixed snapshot'):disk.inspect_bundle(paths+other)
    with pytest.raises(ValueError,match='INDEX'):disk.inspect_bundle(paths[:-1])
    with pytest.raises(ValueError,match='live run'):disk.restore(paths,restored)
    # A self-consistent replacement part still fails the global expected checksum.
    altered=tmp_path/'altered.zip'
    with zipfile.ZipFile(paths[0]) as src,zipfile.ZipFile(altered,'w') as dst:
        for n in src.namelist():dst.writestr(n,src.read(n))
    with pytest.raises(ValueError,match='part'):disk.inspect_bundle([altered,*paths[1:]])


def test_NPZ_pickle_and_large_header_rejected(tmp_path):
    p=tmp_path/'unsafe.npz';np.savez(p,x=np.array([{}],object))
    with pytest.raises(ValueError):num.load_npz(p)
    p=tmp_path/'safe.npz';a={'x':np.ones((5,3),np.float64)};num.save_npz(p,a);num.save_npz(p,a)
    with pytest.raises(ValueError):num.save_npz(p,{'x':np.zeros((5,3))})


def test_adapter_routing():
    middle=lambda n:list(range(n))
    m=types.SimpleNamespace(transformer=types.SimpleNamespace(h=middle(12),ln_f='ln'),model=types.SimpleNamespace(decoder=types.SimpleNamespace(layers=middle(24),final_layer_norm='lnopt',project_out=None),layers=middle(28),norm='rms'))
    for k,n,norm in [('gpt2',12,'ln'),('opt',24,'lnopt'),('qwen',28,'rms')]:
        layers,normal=worker.modules(m,k);assert len(layers)==n and normal==norm
    m.model.decoder.project_out='unexpected'
    with pytest.raises(ValueError):worker.modules(m,'opt')


def test_frozen_common_text_and_no_old_article():
    fs=disk.fixtures();old=set(disk.read(disk.HERE/'input_manifest.json')['old_article_ids'])
    assert len(fs)==128 and len({f['id'] for f in fs})==128
    assert not old&{f['id'] for f in fs}
    assert sum(f['split']=='train' for f in fs)==sum(f['split']=='test' for f in fs)==64
    for f in fs:
        assert disk.sha(f['text'].encode())==f['text_sha256']
        assert f['max_jaccard_against_previous_and_selected']<.5
        for k in disk.spec()['models']:
            ids=f['token_ids'][k];assert 256<=len(ids)<=512
            assert disk.sha(np.array(ids,dtype='<i8').tobytes())==f['input_ids_sha256'][k]
    assert [sum(f['fold']==i for f in fs) for i in range(4)]==[16]*4


def test_synthetic_end_to_end_fit_seal_test_verify_noop_restore(tmp_path,monkeypatch):
    cfg=disk.spec();cfg.update(train_documents=8,test_documents=4,documents=12,random_subspaces=2,bootstrap_repetitions=200,minimum_valid_documents=3,within_min_tokens=10,within_min_groups=3)
    pkg=tmp_path/'package';pkg.mkdir();rng=np.random.default_rng(673);fixtures=[];data={}
    for i in range(12):
        f=dict(id=f'test_article_{i:02d}',text=f'synthetic fixture {i}',split='train' if i<8 else 'test',fold=i%4 if i<8 else None,token_ids={},input_ids_sha256={})
        f['text_sha256']=disk.sha(f['text'].encode())
        for key in cfg['models']:
            raw=synthetic_raw(rng,n=48+i%3);data[key,f['id']]=raw
            f['token_ids'][key]=raw['token_id'].tolist();f['input_ids_sha256'][key]=disk.sha(raw['token_id'].astype('<i8').tobytes())
        fixtures.append(f)
    models={k:dict(hidden=16,vocab=20,repo='SYNTHETIC/'+k,revision='test-only') for k in cfg['models']}
    for n,v in [('spec.json',cfg),('fixtures.json',fixtures),('models.json',models)]:disk.write(pkg/n,v)
    monkeypatch.setattr(disk,'HERE',pkg);monkeypatch.setattr(disk,'SOURCE_NAMES',('spec.json','fixtures.json','models.json'))
    root=tmp_path/'run';disk.initialize(root)
    def commit(split):
        run=disk.check_run(root);seal=None if split=='train' else disk.digest(root/'fit/COMMIT.json')
        for key in cfg['models']:
            for f in fixtures:
                if f['split']!=split:continue
                attempt='synthetic_'+key+'_'+f['id'];r=data[key,f['id']];dest=root/'docs'/disk.doc_key(key,f['id'])
                decoded=['a'*(int(j)%4+1) for j in r['token_id']]
                disk.event(root,attempt,'FORWARD_STARTED',model_key=key,document_id=f['id'],split=split,fit_commit_sha256=seal,synthetic_test=True)
                num.save_npz(dest/'raw.npz',r)
                meta=dict(document_id=f['id'],model_key=key,run_id=run['run_id'],attempt=attempt,split=split,text=f['text'],text_sha256=f['text_sha256'],model=models[key]['repo'],revision='test-only',specification_sha256=run['specification_sha256'],fit_commit_sha256=seal,input_ids_sha256=f['input_ids_sha256'][key],tokens=[str(j) for j in r['token_id']],decoded_tokens=decoded,extraction_checks=dict(middle_matches=True,final_matches=True,normalization_max_relative_error=0.,surprisal_ce_max_abs=0.))
                disk.write(dest/'doc.json',meta);num.validate_raw(root,key,f['id']);disk.seal_doc(dest)
                disk.event(root,attempt,'DOCUMENT_COMMITTED',model_key=key,document_id=f['id'],split=split,synthetic_test=True)
    # Cannot fit from missing training, nor extract evaluation before seal.
    with pytest.raises(ValueError,match='training'):fit.seal(root)
    with pytest.raises(FileNotFoundError):worker.main(root,'gpt2','test')
    commit('train');fit.seal(root);sealed=disk.digest(root/'fit/COMMIT.json');fit.seal(root);assert sealed==disk.digest(root/'fit/COMMIT.json')
    commit('test')
    for key in cfg['models']:ana.evaluate(root,key)
    result=ver.verify(root,tmp_path/'verified.json');assert result['complete'] and result['audit']['status']=='PASS'
    count=sum(e['kind']=='FORWARD_STARTED' for e in disk.events(root))
    for key in cfg['models']:assert worker.main(root,key,'test')==0
    assert sum(e['kind']=='FORWARD_STARTED' for e in disk.events(root))==count
    monkeypatch.setattr(disk,'PART_TARGET',200000)
    paths=disk.snapshot(root,tmp_path/'archives','FINAL');disk.restore(paths,tmp_path/'restored')
    assert disk.digest(tmp_path/'restored/fit/COMMIT.json')==sealed
    # Fit corruption cannot be silently accepted on resume.
    target=root/'fit/gpt2/middle/cv.json';target.write_text('{}')
    with pytest.raises(ValueError,match='Changed fit'):disk.check_fit(root)


def test_budget_persists_and_does_not_reset(tmp_path,monkeypatch):
    import p4_launch as launch
    now=[100.];monkeypatch.setattr(launch.time,'monotonic',lambda:now[0])
    path=tmp_path/'budget.json';b=launch.Budget(path);now[0]+=10;b.tick(force=True)
    assert disk.read(path)['seconds']==10
    # Unclosed previous process gets a 60-second debit, not a fresh allowance.
    c=launch.Budget(path);assert c.seconds()==70 and c.sessions==2
    now[0]+=8;c.close();d=launch.Budget(path);assert d.seconds()==78 and d.sessions==3
    d.close()
    with pytest.raises(ValueError,match='resume limit'):launch.Budget(path)


def test_child_process_timeout_terminates_owned_process(tmp_path):
    import p4_launch as launch,sys,time,os
    log=tmp_path/'child.log'
    with pytest.raises(TimeoutError):
        launch.process([sys.executable,'-u','-c','import time; print("started", flush=True); time.sleep(10)'],time.monotonic()+.25,log,os.environ.copy())
    assert 'started' in log.read_text()


def test_delivery_disabled_records_no_claim_of_disk_save(tmp_path):
    from p4_download import Delivery
    p=tmp_path/'small.zip';p.write_bytes(b'zip test')
    d=Delivery(tmp_path/'delivery',enabled=False);d.enqueue(p);d.enqueue(p)
    records=d.status()['items'];assert len(records)==1
    assert records[0]['state']=='disabled' and not records[0]['user_disk_confirmed'] and not records[0]['browser_received']
