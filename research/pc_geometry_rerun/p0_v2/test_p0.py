"""Synthetic local engineering tests. No pretrained model, tokenizer or network."""
import ast
import base64
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
import numpy as np
import pytest
import p0_io as disk
import p0_numeric as num
import p0_verify as verifier
import p0_launch as launcher


def make_doc(root, fixture, seed):
    rng = np.random.default_rng(seed)
    n, d = 256, 768
    ids = np.arange(n, dtype=np.int64)
    a = {key: v for key, v in (
        ("token_id", ids), ("position", ids), ("mask", np.ones(n, np.int64)),
        ("token_chars", np.ones(n, np.int64)), ("token_utf8_bytes", np.ones(n, np.int64)),
        ("logit_source_index", np.arange(-1, n-1, dtype=np.int64)))}
    a.update(surprisal=np.r_[np.nan, np.full(n-1, 2.)],
             target_logit=np.r_[np.nan, np.zeros(n-1)], logsumexp=np.r_[np.nan, np.full(n-1, 2.)])
    for p in disk.spec()["points"]:
        a["h_"+p] = (rng.normal(size=(n, d)) + np.arange(n)[:, None]*.01).astype(np.float32)
    m = disk.read(root/"run.json")
    folder = root/"docs"/fixture["id"]
    num.save_npz(folder/"raw.npz", a)
    disk.write(folder/"doc.json", dict(document_id=fixture["id"], run_id=m["run_id"],
        text=fixture["text"], text_sha256=disk.sha(fixture["text"].encode()),
        specification_sha256=m["specification_sha256"], model=disk.spec()["model"],
        revision=disk.spec()["revision"], tokens=["x"]*n, decoded_tokens=["x"]*n,
        input_ids_sha256=disk.sha(ids.tobytes()), extraction_checks=dict(block01_matches=True,
        block06_matches=True, final_ln_matches=True, preln_to_final_max_abs=0., surprisal_ce_max_abs=0.),
        synthetic_test=True))
    disk.seal_doc(folder)
    disk.event(root, "synthetic", "FORWARD_STARTED", document_id=fixture["id"])
    disk.event(root, "synthetic", "DOCUMENT_COMMITTED", document_id=fixture["id"])


@pytest.fixture(scope="module")
def complete_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p0-synthetic")/"run"
    disk.initialize(root)
    for i, fixture in enumerate(disk.fixtures()):
        make_doc(root, fixture, i+40)
    verifier.analyze(root)
    return root


def test_known_pca_spectrum_and_denominator():
    diagonal = np.diag(np.arange(10, 0, -1, dtype=np.float64))
    x = np.concatenate([diagonal, -diagonal])
    arrays, metrics = num.pca(x, {"position": np.arange(20)}, 8)
    expected = 2*np.square(np.arange(10, 0, -1))/(len(x)-1)
    np.testing.assert_allclose(arrays["eigenvalues"], expected)
    assert metrics["V1"] == pytest.approx(100/sum(np.square(np.arange(1, 11))))
    assert metrics["R1"] == pytest.approx(0., abs=1e-14)
    num.verify_pca(x, arrays, metrics, {"position": np.arange(20)})


def test_zero_variance_rejected():
    with pytest.raises(ValueError, match="variance"):
        num.pca(np.ones((20, 10)), {}, 8)


def test_undefined_features_and_first_token():
    x = np.random.default_rng(9).normal(size=(20, 10))
    _, m = num.pca(x, {"constant": np.ones(20), "surprisal": np.r_[np.nan, np.arange(19)]})
    assert m["correlations"]["constant"] == [None]*8
    assert m["correlation_denominators"]["surprisal"] == 19


def test_full_synthetic_recompute(complete_root):
    r = verifier.verify(complete_root)
    assert (r["committed_documents"], r["analyzed_document_points"], r["unique_committed_tokens"]) == (4, 16, 1024)
    assert r["forward_attempts"] == r["commit_events"] == 4


def test_reanalysis_idempotent(complete_root):
    before = {str(p): disk.digest(p) for p in disk.files(complete_root/"analysis")}
    verifier.analyze(complete_root)
    assert before == {str(p): disk.digest(p) for p in disk.files(complete_root/"analysis")}


def test_wrong_summary_metric_rejected(complete_root, tmp_path):
    root = tmp_path/"copy"
    shutil.copytree(complete_root, root)
    summary = verifier.verify(root)
    summary["committed_documents"] = 3
    disk.write(root/"summary.json", summary)
    with pytest.raises(ValueError, match="Summary"):
        verifier.verify(root)


def test_surprisal_alignment_rejected(complete_root, tmp_path):
    root = tmp_path/"copy"
    shutil.copytree(complete_root, root)
    doc = disk.fixtures()[0]["id"]
    path = root/"docs"/doc/"raw.npz"
    a = num.load_npz(path)
    a["logit_source_index"][2] = 2
    b = io.BytesIO()
    np.savez(b, **a)
    path.write_bytes(b.getvalue())
    with pytest.raises(ValueError, match="off-by-one"):
        num.validate_raw(root, doc)


def test_bad_first_token_na_rejected(complete_root, tmp_path):
    root = tmp_path/"copy"
    shutil.copytree(complete_root, root)
    doc = disk.fixtures()[0]["id"]
    path = root/"docs"/doc/"raw.npz"
    a = num.load_npz(path)
    a["surprisal"][0] = 0.
    b = io.BytesIO()
    np.savez(b, **a)
    path.write_bytes(b.getvalue())
    with pytest.raises(ValueError, match="NA"):
        num.validate_raw(root, doc)


def test_bad_variance_and_rotated_basis_rejected(complete_root):
    doc, point = disk.fixtures()[0]["id"], disk.spec()["points"][0]
    raw, _ = num.validate_raw(complete_root, doc)
    base = complete_root/"analysis"/doc/point
    arrays = num.load_npz(base.with_suffix(".npz"))
    metrics = disk.read(base.with_suffix(".json"))
    metrics["V1"] = .999
    with pytest.raises(ValueError, match="Metric"):
        num.verify_pca(raw["h_"+point], arrays, metrics, verifier.features(raw))
    arrays["scores"][:, 0] *= 1.01
    with pytest.raises(AssertionError):
        num.verify_pca(raw["h_"+point], arrays, disk.read(base.with_suffix(".json")), verifier.features(raw))


def test_archive_restore_and_receipt(complete_root, tmp_path):
    root = tmp_path/"source"
    shutil.copytree(complete_root, root)
    checks = verifier.fault_checks(root, tmp_path)
    disk.write(root/"protocol_checks.json", dict(archive_roundtrip={"passed": True, "restored_documents": 2},
        no_op_replay={"passed": True, "forward_calls_added": 0, "documents_unchanged": 4}, fault_checks=checks))
    summary = verifier.verify(root)
    summary["status"] = "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT"
    disk.write(root/"summary.json", summary)
    disk.write(root/"status.json", {"status": summary["status"]})
    archive = disk.snapshot(root, tmp_path/"archives", "FINAL")
    dest = tmp_path/"restored"
    disk.restore(archive, dest)
    assert disk.committed(root) == disk.committed(dest)
    r = verifier.receipt(archive, tmp_path/"receipt.json")
    assert r["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    assert r["archive_sha256"] == disk.digest(archive)


def test_controlled_partial_restore(complete_root, tmp_path):
    root = tmp_path/"partial"
    disk.initialize(root)
    for i, f in enumerate(disk.fixtures()[:2]):
        make_doc(root, f, i)
    assert verifier.verify(root, complete=False)["committed_documents"] == 2
    a = disk.snapshot(root, tmp_path/"archives", "CHECKPOINT")
    restored = tmp_path/"restored"
    disk.restore(a, restored)
    before = disk.committed(restored)
    for i, f in enumerate(disk.fixtures()[2:]):
        make_doc(restored, f, i+2)
    assert all(disk.committed(restored)[k] == v for k, v in before.items())
    assert len(disk.committed(restored)) == 4


def test_empty_fault_report_cannot_pass_receipt(complete_root, tmp_path):
    root = tmp_path/"copy"
    shutil.copytree(complete_root, root)
    disk.write(root/"protocol_checks.json", dict(
        archive_roundtrip={"passed": True, "restored_documents": 2},
        no_op_replay={"passed": True, "forward_calls_added": 0, "documents_unchanged": 4},
        fault_checks={}))
    archive = disk.snapshot(root, tmp_path/"archives", "FINAL")
    with pytest.raises(ValueError, match="protocol checks"):
        verifier.receipt(archive, tmp_path/"receipt.json")


def test_partial_archive_cannot_pass_final_receipt(complete_root, tmp_path):
    archive = disk.snapshot(complete_root, tmp_path/"archives", "CHECKPOINT")
    with pytest.raises(ValueError, match="Not a final"):
        verifier.receipt(archive, tmp_path/"receipt.json")


def test_partial_does_not_count_uncommitted(tmp_path):
    root = tmp_path/"run"
    disk.initialize(root)
    disk.atomic(root/"docs"/disk.fixtures()[0]["id"]/"raw.npz", b"incomplete")
    assert disk.committed(root) == {}
    with pytest.raises(ValueError, match="Missing committed"):
        verifier.verify(root)


def test_fault_copy_rejection_preserves_original(complete_root, tmp_path):
    before = {str(p): disk.digest(p) for p in disk.files(complete_root)}
    result = verifier.fault_checks(complete_root, tmp_path)
    assert len(result) == 4 and all(v["rejected"] for v in result.values())
    assert before == {str(p): disk.digest(p) for p in disk.files(complete_root)}


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", "dir/../../escaped", "dir\\escaped", "a//b", ".", ""])
def test_archive_paths_rejected(name):
    with pytest.raises(ValueError):
        disk.safe_name(name)


def test_unexpected_document_rejected(complete_root, tmp_path):
    root = tmp_path/"copy"
    shutil.copytree(complete_root, root)
    (root/"docs"/"extra_doc").mkdir()
    with pytest.raises(ValueError, match="document"):
        disk.committed(root)


def test_object_npz_rejected(tmp_path):
    p = tmp_path/"object.npz"
    np.savez(p, values=np.array([{"bad": 1}], dtype=object))
    with pytest.raises(ValueError, match="type/shape"):
        num.load_npz(p)


def test_child_process_timeout_keeps_log(tmp_path):
    with pytest.raises(TimeoutError):
        launcher.process([sys.executable, "-u", "-c", "import time; print('fixture'); time.sleep(10)"],
                         time.monotonic()+.5, tmp_path/"timeout.log", {})
    assert "fixture" in (tmp_path/"timeout.log").read_text()


def test_onecell_payload_matches_readable_sources():
    source = (disk.HERE/"pcgeom_p0_v2_onecell.py").read_text()
    tree = ast.parse(source)
    values = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ("_PAYLOAD", "_PAYLOAD_SHA256"):
                    values[t.id] = ast.literal_eval(n.value)
    raw = zlib.decompress(base64.b64decode(values["_PAYLOAD"], validate=True))
    assert disk.sha(raw) == values["_PAYLOAD_SHA256"]
    sources = json.loads(raw)
    assert set(sources) == set(disk.SOURCE_NAMES)
    assert all(body == (disk.HERE/n).read_text() for n, body in sources.items())
    for name, body in sources.items():
        if name.endswith(".py"):
            compile(body, name, "exec")
