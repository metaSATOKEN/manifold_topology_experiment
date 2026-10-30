"""Model-free scientific and transport tests; synthetic observations are not P3 data."""
import copy
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import pytest
import p3_io as disk
import p3_numeric as num
import p3_frozen as fit
import p3_analysis as analysis
import p3_verify as verify
import p3_worker as worker
import p3_launch as launch


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    cfg = disk.spec().copy()
    cfg.update(documents=3, test_documents=3, tokens=32,
               minimum_valid_documents=3, bootstrap_repetitions=500,
               within_min_tokens=8, within_min_groups=3)
    fixtures = [dict(id=f"synthetic_{i}", text=f"Synthetic only {i}",
                    token_ids=((np.arange(32)+i)%10).tolist(), split="test") for i in range(3)]
    monkeypatch.setattr(disk, "spec", lambda: cfg)
    monkeypatch.setattr(disk, "fixtures", lambda: fixtures)
    root = tmp_path/"run"
    disk.initialize(root)
    return root, cfg, fixtures


def raw_sample(cfg, fixture, i=0):
    rng = np.random.default_rng(990+i)
    n = cfg["tokens"]
    h = rng.normal(size=(n, 768)).astype(np.float32)
    y = np.r_[np.nan, rng.uniform(1, 10, n-1)]
    return dict(h_block06=h, h_final_ln=h*.8,
        token_id=np.array(fixture["token_ids"], np.int64), position=np.arange(n, dtype=np.int64),
        mask=np.ones(n, np.int64), token_chars=np.ones(n, np.int64),
        token_utf8_bytes=np.ones(n, np.int64),
        logit_source_index=np.arange(-1, n-1, dtype=np.int64),
        surprisal=y, target_logit=-y, logsumexp=np.r_[np.nan, np.zeros(n-1)])


def commit(tiny, i=0):
    root, cfg, fixtures = tiny
    f = fixtures[i]
    r = raw_sample(cfg, f, i)
    run = disk.read(root/"run.json")
    sha = disk.digest(root/"fit/COMMIT.json")
    disk.event(root, "synthetic", "FORWARD_STARTED", document_id=f["id"], split="test", fit_commit_sha256=sha)
    meta = dict(synthetic=True, document_id=f["id"], split="test", fit_commit_sha256=sha,
        run_id=run["run_id"], text=f["text"], text_sha256=disk.sha(f["text"].encode()),
        specification_sha256=run["specification_sha256"], model=cfg["model"], revision=cfg["revision"],
        tokens=[str(x) for x in f["token_ids"]], decoded_tokens=["a"]*cfg["tokens"],
        input_ids_sha256=disk.sha(r["token_id"].tobytes()),
        extraction_checks=dict(block01_matches=True, block06_matches=True, final_ln_matches=True,
                               preln_to_final_max_abs=0., surprisal_ce_max_abs=0.))
    folder = root/"docs"/f["id"]
    num.save_npz(folder/"raw.npz", r)
    disk.write(folder/"doc.json", meta)
    num.validate_raw(root, f["id"])
    disk.seal_doc(folder)
    disk.event(root, "synthetic", "DOCUMENT_COMMITTED", document_id=f["id"])
    return r


def full(tiny):
    root, cfg, fixtures = tiny
    for i in range(len(fixtures)):
        commit(tiny, i)
    disk.write(root/"model_files.json", disk.read(disk.HERE/"expected_model_files.json"))
    verify.evaluate_saved(root)
    return verify.verify(root)


def test_complete_roundtrip_receipt(tiny, tmp_path):
    result = full(tiny)
    root, _, _ = tiny
    assert result["evaluated_test_point_conditions"] == 6
    result["status"] = "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT"
    disk.write(root/"summary.json", result)
    path = disk.snapshot(root, tmp_path/"archives", "FINAL")
    got = verify.receipt(path)
    assert got["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    assert got["endpoints"] == result["endpoints"]


def test_noop_evaluation_preserves_all_evidence(tiny):
    full(tiny)
    root, _, _ = tiny
    before = {str(p.relative_to(root)): disk.digest(p) for p in disk.files(root)}
    verify.evaluate_saved(root)
    assert before == {str(p.relative_to(root)): disk.digest(p) for p in disk.files(root)}


@pytest.mark.parametrize("field", ["secondary_descriptive", "forward_attempts"])
def test_receipt_rejects_incorrect_summary(tiny, tmp_path, field):
    result = full(tiny)
    root, _, _ = tiny
    result["status"] = "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT"
    if field == "forward_attempts":
        result[field] += 1
    else:
        result[field]["final_ln"]["A"]["median"] += 1
    disk.write(root/"summary.json", result)
    path = disk.snapshot(root, tmp_path/"archives", "FINAL")
    with pytest.raises((ValueError, AssertionError)):
        verify.receipt(path)


def test_test_targets_do_not_change_predictions(tiny):
    root, cfg, fs = tiny
    models, counts = fit.models(root)
    raw = raw_sample(cfg, fs[0])
    changed = copy.deepcopy(raw)
    changed["surprisal"][1:] += np.arange(cfg["tokens"]-1)*100
    p, r = analysis.evaluate(raw, "one", "block06", "C1", models["block06", "C1"], counts)
    q, s = analysis.evaluate(changed, "one", "block06", "C1", models["block06", "C1"], counts)
    np.testing.assert_array_equal(p["predictions"], q["predictions"])
    np.testing.assert_array_equal(p["within_predictions"], q["within_predictions"])
    assert r["ordinary"]["sse"] != s["ordinary"]["sse"]


def test_pca_mean_is_not_reestimated(tiny):
    root, cfg, fs = tiny
    models, counts = fit.models(root)
    m = models["block06", "C1"]
    raw = raw_sample(cfg, fs[0])
    raw["h_block06"] += 10*m["components"][0]
    _, score, _ = fit.predict(raw, "block06", m, counts)
    np.testing.assert_allclose(score, (raw["h_block06"][1:].astype(float)-m["mean"])@m["components"].T)


def test_unknown_ids_have_zero_identity_effect(tiny):
    root, cfg, fs = tiny
    models, counts = fit.models(root)
    m = models["block06", "C1"]
    unseen = np.setdiff1d(np.arange(50257), m["vocab"])[0]
    raw = raw_sample(cfg, fs[0]); raw["token_id"][:] = unseen
    p, _, known = fit.predict(raw, "block06", m, counts)
    altered = copy.deepcopy(m); altered["token_effect"] += 10000
    q, _, _ = fit.predict(raw, "block06", altered, counts)
    np.testing.assert_array_equal(p, q)
    assert not known.any() and np.isfinite(p).all()


def test_within_matches_explicit_pair_differences_and_cancels_id_constants():
    token = np.array([5, 5, 7, 7, 9])
    y = np.array([1., 3., 10., 14., 99.])
    pred = np.stack([y*0, y*.5, y*.2]+[y*.3]*19)
    yc, pc, keep, groups = analysis.within_center(token, y, pred)
    np.testing.assert_array_equal(yc[keep], [-1, 1, -2, 2])
    assert not keep[-1] and groups == 2
    shifted = y+np.array([100, 100, -50, -50, 8])
    shifted_pred = pred+np.array([9, 9, -7, -7, 1])
    y2, p2, _, _ = analysis.within_center(token, shifted, shifted_pred)
    np.testing.assert_allclose(yc, y2, equal_nan=True)
    np.testing.assert_allclose(pc, p2, equal_nan=True)
    m = analysis.errors(yc[keep], pc[:, keep], centered=True)
    assert m["sst"] == 10
    assert m["gain"] == pytest.approx(.75)


def test_within_negative_gain_retained():
    y = np.array([-1., 1., -2., 2.])
    p = np.stack([y, -y, y*.5]+[y*.5]*19)
    m = analysis.errors(y, p, centered=True)
    assert m["gain"] == -4 and m["random_advantage"] < 0


def test_singleton_and_constant_within_targets_are_na(tiny):
    root, cfg, fs = tiny
    models, counts = fit.models(root)
    raw = raw_sample(cfg, fs[0])
    raw["token_id"] = np.arange(cfg["tokens"], dtype=np.int64)
    _, m = analysis.evaluate(raw, "x", "block06", "C1", models["block06", "C1"], counts)
    assert m["within_id"]["reason"] == "insufficient_repeated_tokens_or_groups"
    raw["token_id"] %= 5
    raw["surprisal"][1:] = raw["token_id"][1:]+1.
    _, m = analysis.evaluate(raw, "x", "block06", "C1", models["block06", "C1"], counts)
    assert m["within_id"]["reason"] == "zero_target_variance"
    assert m["ordinary"]["valid"]


@pytest.mark.parametrize("l1,u1,l2,u2,n,expected", [
    (.02,.05,.01,.03,64,"SUPPORTED"),(.001,.019,.02,.03,64,"REFUTED"),
    (.03,.05,0,.009,64,"REFUTED"),(.01,.03,.005,.02,64,"INCONCLUSIVE"),
    (.03,.05,.02,.03,57,"INSUFFICIENT_DATA")])
def test_fixed_thresholds(l1,u1,l2,u2,n,expected):
    ep = dict(A=dict(n=n,lower=l1,upper=u1),B=dict(n=n,lower=l2,upper=u2))
    assert analysis.adjudicate(ep, ["A","B"], [.02,.01], True) == expected
    assert analysis.adjudicate(ep, ["A","B"], [.02,.01], False) == "PENDING"


def test_bootstrap_article_units_and_missing(tiny):
    _, cfg, _ = tiny
    values = [-1., 0., 2., None]
    m = analysis.interval(values)
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    x = np.array(values[:3])
    expected = np.quantile(np.median(x[rng.integers(0,3,(500,3))], axis=1), [.00625,.99375])
    assert m["n"] == 3 and m["negative"] == 1
    np.testing.assert_array_equal([m["lower"],m["upper"]], expected)


def test_partial_never_adjudicates(tiny):
    root, _, _ = tiny
    commit(tiny)
    verify.evaluate_saved(root)
    m = verify.verify(root, partial=True)
    assert set(m["scientific_adjudication"].values()) == {"PENDING"}


@pytest.mark.parametrize("kind", ["fixed_source", "fit_commit", "prediction", "raw", "split", "reference"])
def test_corrupt_evidence_is_rejected(tiny, kind):
    full(tiny)
    root, _, fs = tiny
    folder = root/"docs"/fs[0]["id"]
    if kind == "fixed_source":
        path = root/"sources/frozen_block06_C1.npz"
        path.write_bytes(path.read_bytes()[:-1]+b"x")
    elif kind == "fit_commit":
        c = disk.read(root/"fit/COMMIT.json");c["new_training_performed"] = True
        disk.write(root/"fit/COMMIT.json", c)
    elif kind == "prediction":
        path = verify.folder_for(root, fs[0]["id"], "block06", "C1")/"arrays.npz"
        m = num.load_npz(path);m["predictions"][1, 2] += .1;np.savez(path, **m)
        c = disk.read(path.parent/"COMMIT.json");c["arrays.npz"] = disk.digest(path)
        disk.write(path.parent/"COMMIT.json", c)  # hashing alone must not validate wrong math
    elif kind == "raw":
        path = folder/"raw.npz";m = num.load_npz(path);m["h_block06"][1, 1] += 1;np.savez(path, **m)
    else:
        m = disk.read(folder/"doc.json")
        m["split" if kind=="split" else "fit_commit_sha256"] = "wrong"
        disk.write(folder/"doc.json", m)
    with pytest.raises((ValueError, AssertionError)):
        verify.verify(root)


def test_incomplete_evaluation_retained(tiny):
    root, _, fs = tiny
    commit(tiny)
    folder = verify.folder_for(root, fs[0]["id"], "block06", "C1")
    disk.atomic(folder/"partial.txt", b"interrupted evidence")
    verify.evaluate_saved(root)
    paths = list((root/"attempts").glob("eval_*/incomplete_before_retry/*/block06__C1/partial.txt"))
    assert len(paths) == 1 and paths[0].read_bytes() == b"interrupted evidence"


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../x", "a\\x", "./x"])
def test_unsafe_paths(path):
    with pytest.raises(ValueError):
        disk.safe_name(path)


def test_foreign_and_duplicate_zip_rejected(tmp_path):
    path = tmp_path/"bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ARCHIVE_INDEX.json", json.dumps(dict(experiment="pcgeom_p2_v1", capsule="foreign")))
    with pytest.raises(ValueError):
        disk.inspect_zip(path)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a", "1"); z.writestr("a", "2")
    with pytest.raises(ValueError):
        disk.inspect_zip(path)


def test_environment_serialization_and_changed_hardware():
    keys = ["gpu_name","compute_capability","python","torch","cuda_build","nvidia_smi",
            "driver_query_exit","packages","tf32","dtype","attention"]
    env = dict.fromkeys(keys)
    env.update(gpu_name="A100", packages=[("numpy","2.2.6")], compute_capability=(8,0))
    worker.require_compatible(json.loads(json.dumps(env)), env)
    other = copy.deepcopy(env); other["gpu_name"] = "changed"
    with pytest.raises(ValueError):
        worker.require_compatible(env, other)


def test_subprocess_timeout_keeps_log(tmp_path):
    log = tmp_path/"timeout.log"
    with pytest.raises(TimeoutError):
        launch.process([sys.executable,"-u","-c","import time;print('started');time.sleep(10)"],
                       time.monotonic()+.3, log, launch.child_environment(tmp_path))
    assert "started" in log.read_text()
