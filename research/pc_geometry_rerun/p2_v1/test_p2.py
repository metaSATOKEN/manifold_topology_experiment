"""Synthetic, model-free tests. These are not P2 study outcomes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
import pytest
import p2_io as disk
import p2_numeric as num
import p2_fit as fit
import p2_analysis as analysis
import p2_verify as verify
import p2_worker as worker
import p2_launch as launch


@pytest.fixture
def tiny(monkeypatch,tmp_path):
    cfg=disk.spec().copy()
    cfg.update(documents=7,train_documents=4,test_documents=3,tokens=32,hidden=12,
               components=3,random_subspaces=4,minimum_valid_documents=3,
               bootstrap_repetitions=500,vocabulary_size=50,points=["block06","final_ln"])
    fixtures=[dict(id=f"synthetic_{i}",text=f"Synthetic fixture {i}",
                   token_ids=((np.arange(32)+i)%50).tolist(),split="train" if i<4 else "test")
              for i in range(7)]
    monkeypatch.setattr(disk,"spec",lambda:cfg)
    monkeypatch.setattr(disk,"fixtures",lambda:fixtures)
    root=tmp_path/"synthetic_run"
    disk.initialize(root)
    return root,cfg,fixtures


def raw_sample(i,cfg,fixture):
    rng=np.random.default_rng(1200+i)
    n,d=cfg["tokens"],cfg["hidden"]
    h=rng.normal(scale=.15,size=(n,d))
    t=np.linspace(-1,1,n)
    latent=rng.uniform(-1,1,n)
    h[:,0]+=t*5;h[:,1]+=latent*4;h[0,2]+=30
    h=h.astype(np.float32)
    y=5+latent*2
    y[0]=np.nan
    chars=np.arange(n,dtype=np.int64)%7+1
    raw={f"h_{p}":h.copy() for p in cfg["points"]}
    raw.update(token_id=np.asarray(fixture["token_ids"],np.int64),
               position=np.arange(n,dtype=np.int64),mask=np.ones(n,np.int64),
               token_chars=chars,token_utf8_bytes=chars.copy(),
               logit_source_index=np.arange(-1,n-1,dtype=np.int64),
               surprisal=y.astype(float),target_logit=(-y).astype(float),
               logsumexp=np.r_[np.nan,np.zeros(n-1)])
    return raw


def commit(root,cfg,fixture,i,raw=None):
    raw=raw_sample(i,cfg,fixture) if raw is None else raw
    run=disk.read(root/"run.json")
    sha=disk.digest(root/"fit"/"COMMIT.json") if fixture["split"]=="test" else None
    metadata=dict(synthetic=True,document_id=fixture["id"],split=fixture["split"],
        fit_commit_sha256=sha,run_id=run["run_id"],text=fixture["text"],
        text_sha256=disk.sha(fixture["text"].encode()),specification_sha256=run["specification_sha256"],
        model=cfg["model"],revision=cfg["revision"],
        tokens=[str(x) for x in fixture["token_ids"]],
        decoded_tokens=["a"*int(x) for x in raw["token_chars"]],
        input_ids_sha256=disk.sha(raw["token_id"].tobytes()),
        extraction_checks=dict(block01_matches=True,block06_matches=True,final_ln_matches=True,
                               preln_to_final_max_abs=0.,surprisal_ce_max_abs=0.))
    dest=root/"docs"/fixture["id"]
    num.save_npz(dest/"raw.npz",raw);disk.write(dest/"doc.json",metadata)
    num.validate_raw(root,fixture["id"]);disk.seal_doc(dest)
    disk.event(root,"synthetic","FORWARD_STARTED",document_id=fixture["id"],
               split=fixture["split"],fit_commit_sha256=sha)
    disk.event(root,"synthetic","DOCUMENT_COMMITTED",document_id=fixture["id"])
    return raw


def train(tiny):
    root,cfg,fixtures=tiny
    raws=[commit(root,cfg,f,i) for i,f in enumerate(fixtures[:4])]
    fit.seal_fit(root)
    return raws


def full(tiny):
    train(tiny)
    root,cfg,fixtures=tiny
    for i in range(4,7):
        commit(root,cfg,fixtures[i],i)
    verify.evaluate_saved(root)
    return verify.verify(root)


def test_training_seal_required_before_test(tiny):
    root,_,_=tiny
    with pytest.raises(ValueError,match="sealed"):
        fit.check_frozen_fit(root)
    with pytest.raises(ValueError,match="training"):
        fit.seal_fit(root)


def test_full_numeric_audit_and_zip_roundtrip(tiny,tmp_path):
    result=full(tiny)
    root,_,_=tiny
    assert result["test_documents"]==3 and result["evaluated_test_point_conditions"]==18
    assert result["endpoints"]["C"]["median"]>.3
    result["status"]="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT"
    disk.write(root/"summary.json",result)
    z=disk.snapshot(root,tmp_path/"archives","FINAL")
    received=verify.receipt(z)
    assert received["status"]=="LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    assert received["endpoints"]=={k:result["endpoints"][k] for k in "ABCD"}


def test_replay_does_not_change_fit_or_predictions(tiny):
    full(tiny)
    root,_,_=tiny
    before={str(p.relative_to(root)):disk.digest(p) for p in disk.files(root)}
    fit.seal_fit(root);verify.evaluate_saved(root)
    assert before=={str(p.relative_to(root)):disk.digest(p) for p in disk.files(root)}


def test_test_labels_do_not_affect_fit_selection_or_features(tiny):
    train(tiny)
    root,cfg,fixtures=tiny
    models,counts=verify.frozen_models(root)
    a,m=models[("block06","C1")]
    raw=raw_sample(4,cfg,fixtures[4]);altered=copy.deepcopy(raw)
    altered["surprisal"][1:]+=100
    before=disk.digest(root/"fit"/"COMMIT.json")
    out,r=analysis.evaluate(raw,"one","block06","C1",a,m,counts)
    other,q=analysis.evaluate(altered,"one","block06","C1",a,m,counts)
    np.testing.assert_array_equal(out["surprisal__predictions"],other["surprisal__predictions"])
    assert r["position_r"]==q["position_r"]
    assert r["heads"]["surprisal"]["sse"]!=q["heads"]["surprisal"]["sse"]
    assert disk.digest(root/"fit"/"COMMIT.json")==before


def test_shifted_test_position_is_not_resigned(tiny):
    train(tiny)
    root,cfg,fixtures=tiny
    models,counts=verify.frozen_models(root);a,m=models[("block06","C1")]
    raw=raw_sample(4,cfg,fixtures[4])
    _,r=analysis.evaluate(raw,"one","block06","C1",a,m,counts)
    raw["position"]=raw["position"][::-1].copy()
    _,q=analysis.evaluate(raw,"one","block06","C1",a,m,counts)
    assert q["position_r"][0]==pytest.approx(-r["position_r"][0])


def test_test_does_not_recenter_or_recompute_basis(tiny):
    train(tiny)
    root,cfg,fixtures=tiny
    models,counts=verify.frozen_models(root);a,m=models[("block06","C1")]
    raw=raw_sample(4,cfg,fixtures[4]);delta=a["components"][0]*10
    raw["h_block06"]=(raw["h_block06"].astype(float)+delta).astype(np.float32)
    out,_=analysis.evaluate(raw,"one","block06","C1",a,m,counts)
    np.testing.assert_allclose(out["pc_scores"],(raw["h_block06"][1:].astype(float)-a["mean"])@a["components"].T)
    assert abs(out["pc_scores"][:,0].mean())>5


def test_transferred_subspace_variance_is_sum_of_fixed_direction_fractions(tiny):
    train(tiny);root,cfg,fixtures=tiny
    models,counts=verify.frozen_models(root)
    out,result=analysis.evaluate(raw_sample(4,cfg,fixtures[4]),"one","block06","C1",
                                *models[("block06","C1")],counts)
    ratio=result["subspace_variance_fraction"]
    assert 0<=ratio<=1
    assert ratio==pytest.approx(sum(result["direction_variance_fractions"]))
    assert ratio==pytest.approx(np.var(out["pc_scores"],axis=0,ddof=1).sum()/
                               result["total_test_variance"])


def test_frequency_uses_training_only_and_smooths_unseen(tiny):
    raws=train(tiny)
    root,_,_=tiny
    counts=num.load_npz(root/"fit"/"frequency.npz")["counts"]
    assert counts.sum()==4*32
    np.testing.assert_array_equal(counts,np.bincount(np.concatenate([r["token_id"] for r in raws]),minlength=50))
    assert counts[-1]==0
    raw=copy.deepcopy(raws[0]);raw["token_id"][:]=49
    x,idx,norms,_=fit.prepare(raw,"block06","C1")
    b,y=fit.controls(raw,idx,norms,counts,"surprisal")
    assert np.isfinite(b).all()
    assert b[0,-2]==pytest.approx(np.log(4*32+50))


def test_control_baseline_identical_across_conditions(tiny):
    train(tiny)
    root,_,_=tiny;models,_=verify.frozen_models(root)
    for key in ("x_mean","x_scale","coef","active","y_mean"):
        xs=[models[("block06",c)][0]["surprisal__"+key][0] for c in ("C0","C1","C4")]
        for x in xs[1:]:np.testing.assert_allclose(x,xs[0])


def test_ridge_matches_augmented_least_squares(tiny):
    _,cfg,_=tiny
    rng=np.random.default_rng(89);x=rng.normal(size=(100,4));y=rng.normal(size=100)
    mu,sd,active,coef,ym=fit.ridge(x,y)
    z=(x-mu)/sd*active
    aug=np.vstack([z,np.sqrt(len(y)*cfg["ridge_alpha"])*np.eye(4)])
    expected=np.linalg.lstsq(aug,np.r_[y-ym,np.zeros(4)],rcond=None)[0]
    np.testing.assert_allclose(coef,expected,atol=1e-12)


def test_constant_training_column_stays_disabled(tiny):
    x=np.column_stack([np.arange(20),np.ones(20)]);y=np.arange(20)*2
    fitted=fit.ridge(x,y)
    altered=x.copy();altered[:,1]=1e6
    np.testing.assert_array_equal(fit.predict(x,*fitted),fit.predict(altered,*fitted))


def test_random_directions_are_orthonormal_repeatable_and_distinct(tiny):
    a=fit.random_bases(12,3,"block06","C1")
    np.testing.assert_array_equal(a,fit.random_bases(12,3,"block06","C1"))
    assert not np.array_equal(a,fit.random_bases(12,3,"block06","C0"))
    for group in a:np.testing.assert_allclose(group@group.T,np.eye(3),atol=1e-12)


def test_pca_covariance_matches_direct_svd(tiny):
    root,cfg,fixtures=tiny
    raws=[raw_sample(i,cfg,f) for i,f in enumerate(fixtures[:4])]
    a,m=fit.fit_condition(raws,fit.frequency(raws),"block06","C1")
    x=np.concatenate([r["h_block06"][1:].astype(float) for r in raws]);x-=x.mean(0)
    _,s,vt=np.linalg.svd(x,full_matrices=False)
    np.testing.assert_allclose(a["eigenvalues"],s*s/(len(x)-1),atol=1e-10)
    np.testing.assert_allclose(abs(a["components"]@vt[:3].T),np.eye(3),atol=1e-10)


@pytest.mark.parametrize("what",["basis","coef","frequency","test_prediction","raw","split","fit_reference"])
def test_corruption_is_rejected(tiny,what):
    full(tiny);root,cfg,fixtures=tiny
    if what=="basis" or what=="coef":
        path=root/"fit"/"block06__C1"/"arrays.npz"
        a=num.load_npz(path)
        a["components" if what=="basis" else "surprisal__coef"].flat[0]+=1
        np.savez(path,**a)
    elif what=="frequency":
        path=root/"fit"/"frequency.npz";a=num.load_npz(path);a["counts"][0]+=1;np.savez(path,**a)
    elif what=="test_prediction":
        path=verify.folder_for(root,fixtures[4]["id"],"block06","C1")/"arrays.npz"
        a=num.load_npz(path);a["surprisal__predictions"][1,3]+=1;np.savez(path,**a)
    else:
        path=root/"docs"/fixtures[4]["id"]
        if what=="raw":
            a=num.load_npz(path/"raw.npz");a["h_block06"][1,1]+=1;np.savez(path/"raw.npz",**a)
        else:
            m=disk.read(path/"doc.json")
            m["split" if what=="split" else "fit_commit_sha256"]="wrong"
            disk.write(path/"doc.json",m)
    with pytest.raises((ValueError,AssertionError)):
        verify.verify(root)


def test_fit_cannot_be_rebuilt_after_test(tiny):
    full(tiny);root,_,_=tiny
    (root/"fit"/"COMMIT.json").rename(root/"fit"/"previous_commit.json")
    with pytest.raises(ValueError,match="after test"):fit.seal_fit(root)


@pytest.mark.parametrize("field",["components","surprisal__coef"])
def test_numeric_fit_audit_rejects_corruption_even_with_updated_hash(tiny,field):
    train(tiny);root,_,_=tiny
    folder=root/"fit"
    path=folder/"block06__C1"/"arrays.npz"
    arrays=num.load_npz(path);arrays[field].flat[0]+=.1
    np.savez(path,**arrays)
    commit=disk.read(folder/"COMMIT.json")
    commit["files"]["block06__C1/arrays.npz"]=disk.digest(path)
    disk.write(folder/"COMMIT.json",commit)
    fit.check_frozen_fit(root)
    with pytest.raises((ValueError,AssertionError)):
        fit.audit_fit(root)


def test_numeric_evaluation_audit_rejects_rehashed_wrong_prediction(tiny):
    full(tiny);root,_,fixtures=tiny
    folder=verify.folder_for(root,fixtures[4]["id"],"block06","C1")
    path=folder/"arrays.npz"
    arrays=num.load_npz(path);arrays["surprisal__predictions"][1,3]+=.1
    np.savez(path,**arrays)
    commit=disk.read(folder/"COMMIT.json");commit["arrays.npz"]=disk.digest(path)
    disk.write(folder/"COMMIT.json",commit)
    verify.point_commit(folder)
    with pytest.raises(AssertionError):
        verify.verify(root)


def test_incomplete_fit_is_retained_before_retry(tiny):
    root,cfg,fixtures=tiny
    for i,f in enumerate(fixtures[:4]):commit(root,cfg,f,i)
    path=root/"fit"/"partial.txt";disk.atomic(path,b"interrupted fit evidence")
    fit.seal_fit(root)
    preserved=list((root/"attempts").glob("fit_*/incomplete_fit_before_retry/partial.txt"))
    assert len(preserved)==1 and preserved[0].read_bytes()==b"interrupted fit evidence"
    fit.audit_fit(root)


def test_incomplete_evaluation_is_retained_before_retry(tiny):
    train(tiny);root,cfg,fixtures=tiny
    commit(root,cfg,fixtures[4],4)
    folder=verify.folder_for(root,fixtures[4]["id"],"block06","C1")
    disk.atomic(folder/"partial.txt",b"interrupted evaluation evidence")
    before=disk.digest(root/"fit"/"COMMIT.json")
    verify.evaluate_saved(root)
    preserved=list((root/"attempts").glob("eval_*/incomplete_before_retry/*/block06__C1/partial.txt"))
    assert len(preserved)==1 and preserved[0].read_bytes()==b"interrupted evaluation evidence"
    assert before==disk.digest(root/"fit"/"COMMIT.json")
    assert verify.verify(root,partial=True)["test_documents"]==1


def test_rank_and_zero_norm_are_explicit(tiny):
    _,cfg,fixtures=tiny
    raws=[raw_sample(i,cfg,f) for i,f in enumerate(fixtures[:4])]
    for raw in raws:raw["h_block06"][:]=0
    for condition,reason in [("C0","zero_training_variance"),("C4","zero_norm_unit_conversion")]:
        a,m=fit.fit_condition(raws,fit.frequency(raws),"block06",condition)
        assert not m["valid"] and m["reason"]==reason and a=={}
    for raw in raws:
        raw["h_block06"][:]=np.outer(np.arange(32),np.ones(12))
    _,m=fit.fit_condition(raws,fit.frequency(raws),"block06","C0")
    assert m["reason"]=="training_rank_below_components"


def test_zero_test_norm_and_constant_target_are_na(tiny):
    train(tiny);root,cfg,fixtures=tiny
    models,counts=verify.frozen_models(root);raw=raw_sample(4,cfg,fixtures[4])
    raw["h_block06"][3]=0
    _,m=analysis.evaluate(raw,"one","block06","C4",*models[("block06","C4")],counts)
    assert not m["valid"] and m["reason"]=="zero_norm_unit_conversion"
    raw["surprisal"][1:]=4.
    _,m=analysis.evaluate(raw,"one","block06","C1",*models[("block06","C1")],counts)
    assert not m["heads"]["surprisal"]["valid"]


def test_character_control_excludes_direct_length(tiny):
    _,cfg,fixtures=tiny;raw=raw_sample(0,cfg,fixtures[0])
    _,index,norms,_=fit.prepare(raw,"block06","C1");counts=fit.frequency([raw])
    b,y=fit.controls(raw,index,norms,counts,"log_chars")
    raw["token_chars"]*=100;raw["token_utf8_bytes"]*=100
    bb,yy=fit.controls(raw,index,norms,counts,"log_chars")
    np.testing.assert_array_equal(b,bb);assert not np.array_equal(y,yy)


def test_negative_improvement_not_clamped(tiny):
    y=np.arange(10,dtype=float)
    p=np.stack([y,y+5,y+2,y+3])
    m=analysis.regression_metrics(y,p)
    assert m["delta_r2"]<0 and m["random_advantage"]<0


@pytest.mark.parametrize("l1,u1,l2,u2,n,expected",[
    (.5,.7,.1,.2,64,"SUPPORTED"),(.1,.49,.2,.3,64,"REFUTED"),
    (.6,.7,.01,.09,64,"REFUTED"),(.4,.6,.05,.15,64,"INCONCLUSIVE"),
    (.6,.7,.2,.3,57,"INSUFFICIENT_DATA")])
def test_preregistered_boundaries(l1,u1,l2,u2,n,expected):
    one=dict(n=n,lower=l1,upper=u1);two=dict(n=n,lower=l2,upper=u2)
    assert analysis.judge_pair(one,two,.5,.1)==expected


def test_bootstrap_uses_article_count_and_fixed_level(tiny):
    _,cfg,_=tiny
    values=[.1,.2,.3];got=analysis.interval(values)
    rng=np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    draws=np.median(np.array(values)[rng.integers(0,3,(500,3))],axis=1)
    lo,hi=np.quantile(draws,[.00625,.99375])
    assert got["n"]==3 and got["lower"]==lo and got["upper"]==hi


def test_partial_checkpoint_never_adjudicates(tiny):
    root,cfg,fixtures=tiny
    commit(root,cfg,fixtures[0],0)
    result=verify.verify(root,partial=True)
    assert set(result["scientific_adjudication"].values())=={"PENDING_COMPLETE_RUN"}


@pytest.mark.parametrize("path",["../escape","/absolute","a/../x","a\\x","./x"])
def test_unsafe_archive_paths_rejected(path):
    with pytest.raises(ValueError):disk.safe_name(path)


def test_duplicate_and_foreign_zip_rejected(tmp_path):
    p=tmp_path/"bad.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("a","1");z.writestr("a","2")
    with pytest.raises(ValueError):disk.inspect_zip(p)
    with zipfile.ZipFile(p,"w") as z:
        z.writestr("ARCHIVE_INDEX.json",json.dumps(dict(experiment="pcgeom_p1_v1",capsule="foreign")))
    with pytest.raises(ValueError):disk.inspect_zip(p)


def test_environment_json_roundtrip_and_real_change():
    env={k:None for k in ("gpu_name","compute_capability","python","torch","cuda_build","nvidia_smi",
                          "driver_query_exit","packages","tf32","dtype","attention")}
    env.update(gpu_name="A100",packages=[("numpy","2.2.6")],compute_capability=(8,0))
    worker.require_compatible(json.loads(json.dumps(env)),env)
    changed=copy.deepcopy(env);changed["gpu_name"]="different"
    with pytest.raises(ValueError):worker.require_compatible(env,changed)


def test_subprocess_deadline_preserves_log(tmp_path):
    log=tmp_path/"timeout.log"
    with pytest.raises(TimeoutError):
        launch.process([sys.executable,"-u","-c","import time; print('started'); time.sleep(10)"],
                       time.monotonic()+.3,log,launch.child_environment(tmp_path))
    assert log.exists() and "started" in log.read_text()
