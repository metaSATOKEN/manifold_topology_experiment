"""Model-free correctness and recovery tests; synthetic results are not study data."""
import copy
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest
import p1_io as disk
import p1_numeric as num
import p1_analysis as analysis
import p1_verify as verify
import p1_worker as worker


def raw_sample(n=32,d=12):
    rng=np.random.default_rng(43)
    h=rng.normal(size=(n,d)).astype(np.float32)
    h[0]*=30
    return {"h_"+p:h.copy() for p in disk.spec()["points"]} | dict(
        position=np.arange(n,dtype=np.int64),token_chars=np.arange(n,dtype=np.int64)%7+1,
        surprisal=np.r_[np.nan,rng.uniform(1,5,n-1)])


def test_exclusions_and_random_stream():
    raw=raw_sample(512,12)
    x=raw["h_block06"]
    x[4]=x[0]
    _,top,indices,random=analysis.plans(x,"doc","block06")
    assert list(top[:2])==[0,4]
    assert len(top)==6 and len(indices["C1"])==511 and len(indices["C2"])==506
    assert len(random)==20 and all(len(set(r["removed"]))==6 for r in random)
    assert random==analysis.plans(x,"doc","block06")[3]
    assert random!=analysis.plans(x,"other","block06")[3]
    assert indices["C1"][0]==1


@pytest.mark.parametrize("scale",[1.,10.,1000.])
def test_svd_and_gram_agree(scale):
    raw=raw_sample()
    x=raw["h_block06"].astype(float)*scale
    a,m=num.pca(x,{k:raw[k] for k in ("position","token_chars","surprisal")})
    g=analysis.gram_metrics(x)
    assert g["R1"]==pytest.approx(m["R1"],abs=1e-10)
    assert g["V1"]==pytest.approx(m["V1"],abs=1e-10)
    num.verify_pca(x,a,m,{k:raw[k] for k in ("position","token_chars","surprisal")})


def test_all_conditions_and_unit_norm():
    raw=raw_sample()
    arrays,meta=analysis.analyze_point(raw,"test","block06")
    analysis.verify_point(raw,"test","block06",arrays,meta)
    assert meta["conditions"]["C4"]["metrics"]["R1"] is None
    assert meta["conditions"]["C4"]["metrics"]["signed_R1"] is None
    assert meta["conditions"]["C0"]["metrics"]["correlation_denominators"]["surprisal"]==31
    assert meta["conditions"]["C1"]["metrics"]["correlation_denominators"]["surprisal"]==31
    assert set(meta["fixed_C0_basis"])=={"C1","C2"}


@pytest.mark.parametrize("bad",["scores","R1","indices","random","diagnostics"])
def test_corruption_is_rejected(bad):
    raw=raw_sample()
    a,m=analysis.analyze_point(raw,"test","block06")
    if bad=="scores": a["C0__scores"][1,0]+=1
    if bad=="R1": m["conditions"]["C0"]["metrics"]["R1"]=0.
    if bad=="indices": m["conditions"]["C1"]["kept_positions"][0]=0
    if bad=="random": m["random_controls"][0]["removed"]=[0,0]
    if bad=="diagnostics": m["diagnostics"]["position0_is_max"]=False
    with pytest.raises((ValueError,AssertionError)):
        analysis.verify_point(raw,"test","block06",a,m)


def test_zero_norm_is_explicit_and_not_replaced():
    raw=raw_sample()
    raw["h_block06"][3]=0
    a,m=analysis.analyze_point(raw,"test","block06")
    assert m["conditions"]["C4"]["reason"]=="zero_norm_unit_conversion"
    assert m["conditions"]["C0"]["valid"]
    analysis.verify_point(raw,"test","block06",a,m)


def test_zero_variance_and_constant_feature():
    raw=raw_sample()
    raw["h_block06"][:]=0
    a,m=analysis.analyze_point(raw,"test","block06")
    assert not a
    assert all(not c["valid"] for c in m["conditions"].values())
    assert all(not c["metrics"]["valid"] for c in m["random_controls"])
    analysis.verify_point(raw,"test","block06",a,m)
    assert num.correlation(np.ones(20),np.arange(20)) is None


def test_rank_deficiency_does_not_invent_feature_correlations():
    x=np.outer(np.arange(32,dtype=float),np.arange(1,13,dtype=float))
    features={"position":np.arange(32)}
    arrays,metrics=num.pca(x,features)
    assert metrics["effective_rank"]==1
    assert metrics["correlations"]["position"][1:]==[None]*7
    assert metrics["V1"]==pytest.approx(1.)
    num.verify_pca(x,arrays,metrics,features)


def endpoint(value,n=64):
    return dict(n=n,planned=64,lower=value,upper=value,median=value)


@pytest.mark.parametrize("a,b,d,h1,h2",[
    (.9,-.02,-.1,"SUPPORTED","SUPPORTED"),
    (.7,-.02,0.,"REFUTED","REFUTED"),
    (.9,-.2,-.1,"REFUTED","SUPPORTED"),
    (.8,-.1,-.05,"SUPPORTED","SUPPORTED")])
def test_hypothesis_thresholds(a,b,d,h1,h2):
    assert analysis.adjudicate(endpoint(a),endpoint(b),endpoint(d))==dict(H1=h1,H2=h2)


def test_uncertainty_and_insufficient_data():
    a=endpoint(.8);a.update(lower=.7,upper=.9)
    d=endpoint(-.05);d.update(lower=-.1,upper=0.)
    assert analysis.adjudicate(a,endpoint(-.02),d)==dict(H1="INCONCLUSIVE",H2="INCONCLUSIVE")
    assert analysis.adjudicate(endpoint(.99,57),endpoint(0),endpoint(-.9,57))==dict(
        H1="INSUFFICIENT_DATA",H2="INSUFFICIENT_DATA")
    assert analysis.interval([.5]*64,64)["lower"]==.5
    assert analysis.interval([],64)["median"] is None


def test_no_random_pseudoreplication():
    _,row=analysis.analyze_point(raw_sample(),"test","block06")
    summary=analysis.summarize(None,[row])
    assert summary["endpoints"]["D"]["n"]==1
    assert summary["scientific_adjudication"]["H1"]=="PENDING_COMPLETE_RUN"


def test_actual_p0_environment_roundtrip():
    path=Path(__file__).resolve().parent.parent/"receipts/7ec0f9a0ff0c_v2_final/raw/environment_contract.json"
    recorded=disk.read(path)
    current=copy.deepcopy(recorded)
    current["packages"]=[tuple(x) for x in recorded["packages"]]
    worker.require_compatible(recorded,current)
    current["gpu_name"]="different GPU"
    with pytest.raises(ValueError,match="gpu_name"):
        worker.require_compatible(recorded,current)


@pytest.fixture
def tiny_run(tmp_path,monkeypatch):
    cfg=disk.spec() | dict(tokens=32,hidden=12,documents=2,minimum_valid_documents=2)
    fixtures=copy.deepcopy(disk.fixtures()[:2])
    for f in fixtures:
        f["token_ids"]=f["token_ids"][:32]
    monkeypatch.setattr(disk,"spec",lambda:cfg)
    monkeypatch.setattr(disk,"fixtures",lambda:fixtures)
    root=tmp_path/"run"
    disk.initialize(root)
    for i,f in enumerate(fixtures):
        raw=raw_sample()
        raw.update(token_id=np.array(f["token_ids"],np.int64),mask=np.ones(32,np.int64),
            token_chars=np.ones(32,np.int64),token_utf8_bytes=np.ones(32,np.int64),
            logit_source_index=np.arange(-1,31,dtype=np.int64),
            target_logit=np.r_[np.nan,np.zeros(31)],logsumexp=raw["surprisal"].copy())
        meta=dict(document_id=f["id"],run_id=disk.read(root/"run.json")["run_id"],text=f["text"],
            text_sha256=disk.sha(f["text"].encode()),specification_sha256=disk.digest(disk.HERE/"spec.json"),
            model=cfg["model"],revision=cfg["revision"],tokens=["x"]*32,decoded_tokens=["x"]*32,
            input_ids_sha256=disk.sha(raw["token_id"].tobytes()),
            extraction_checks=dict(block01_matches=True,block06_matches=True,final_ln_matches=True,
                preln_to_final_max_abs=0.,surprisal_ce_max_abs=0.))
        dest=root/"docs"/f["id"]
        num.save_npz(dest/"raw.npz",raw)
        disk.write(dest/"doc.json",meta)
        disk.seal_doc(dest)
        disk.event(root,"synthetic","FORWARD_STARTED",document_id=f["id"])
        disk.event(root,"synthetic","DOCUMENT_COMMITTED",document_id=f["id"])
    return root


def test_saved_run_analysis_receipt_and_idempotency(tiny_run,tmp_path):
    root=tiny_run
    before=disk.committed(root)
    verify.analyze(root)
    hashes={str(p):disk.digest(p) for p in disk.files(root/"analysis")}
    verify.analyze(root)
    assert hashes=={str(p):disk.digest(p) for p in disk.files(root/"analysis")}
    result=verify.verify(root)
    assert result["analyzed_document_points"]==8
    assert result["endpoints"]["D"]["n"]==2
    result["status"]="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT"
    disk.write(root/"summary.json",result)
    archive=disk.snapshot(root,tmp_path/"archives","FINAL")
    check=verify.receipt(archive)
    assert check["status"]=="LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    assert disk.committed(root)==before


def test_partial_run_preserves_missing_analysis(tiny_run):
    result=verify.verify(tiny_run,partial=True)
    assert result["analyzed_document_points"]==0
    assert result["scientific_adjudication"]["H1"]=="PENDING_COMPLETE_RUN"
    with pytest.raises(FileNotFoundError):
        verify.verify(tiny_run)


@pytest.mark.parametrize("failure",["missing","corrupt","wrong_spec"])
def test_raw_failures(tiny_run,failure):
    doc=disk.fixtures()[0]["id"]
    p=tiny_run/"docs"/doc/"raw.npz"
    if failure=="missing": p.unlink()
    elif failure=="corrupt": p.write_bytes(p.read_bytes()+b"corrupt")
    else:
        run=disk.read(tiny_run/"run.json")
        run["specification_sha256"]="0"*64
        disk.write(tiny_run/"run.json",run)
    with pytest.raises(ValueError):
        disk.committed(tiny_run)


@pytest.mark.parametrize("name",["../escape","/absolute","a/../x","a\\x"])
def test_unsafe_names(name):
    with pytest.raises(ValueError): disk.safe_name(name)


def test_duplicate_zip_member(tiny_run,tmp_path):
    archive=disk.snapshot(tiny_run,tmp_path/"archives","CHECKPOINT")
    with zipfile.ZipFile(archive,"a") as z:
        with pytest.warns(UserWarning):
            z.writestr("run.json",b"{}")
    with pytest.raises(ValueError,match="Duplicate"):
        disk.inspect_zip(archive)


def test_noop_never_imports_torch(tiny_run,monkeypatch):
    class Content:
        def is_dir(self): return True
    monkeypatch.setattr(worker,"Path",lambda p:Content() if str(p)=="/content" else Path(p))
    monkeypatch.setattr(worker.sys,"platform","linux")
    before=disk.committed(tiny_run)
    assert worker.main(tiny_run,2)==0
    assert disk.committed(tiny_run)==before


def test_input_fixture_manifest_consistency():
    fixtures=disk.fixtures()
    manifest=disk.read(disk.HERE/"input_manifest.json")
    assert len(fixtures)==64 and len({x["id"] for x in fixtures})==64
    assert len({x["input_ids_sha256"] for x in fixtures})==64
    assert manifest["eligible_articles"]==602
    for f in fixtures:
        assert len(f["token_ids"])==512
        assert disk.sha(np.asarray(f["token_ids"],dtype="<i8").tobytes())==f["input_ids_sha256"]
        assert disk.sha(f["text"].encode())==f["text_sha256"]


def test_npz_objects_rejected(tmp_path):
    p=tmp_path/"object.npz"
    np.savez(p,x=np.array([object()],dtype=object))
    with pytest.raises(ValueError):
        num.load_npz(p)


def test_resume_and_explicit_new_preserve_old_run(tiny_run,tmp_path):
    import p1_launch as launch
    archive=disk.snapshot(tiny_run,tmp_path/"archives","CHECKPOINT")
    base=tmp_path/"colab"
    base.mkdir()
    resumed=launch.resolve(base,str(archive))
    assert disk.committed(resumed)==disk.committed(tiny_run)
    assert launch.resolve(base,"")==resumed
    with pytest.raises(ValueError):
        launch.resolve(base,str(archive))
    fresh=launch.resolve(base,"NEW")
    assert fresh!=resumed and not disk.committed(fresh)
    assert len(disk.committed(resumed))==2


def test_process_deadline_retains_log(tmp_path):
    import os
    import time
    import p1_launch as launch
    path=tmp_path/"timeout.log"
    with pytest.raises(TimeoutError):
        launch.process([sys.executable,"-u","-c","import time; print('began'); time.sleep(30)"],
                       time.monotonic()+.25,path,os.environ.copy())
    assert path.exists() and "began" in path.read_text()
