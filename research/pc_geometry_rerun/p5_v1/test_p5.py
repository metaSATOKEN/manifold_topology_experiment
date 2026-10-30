"""Local meaningful tests: synthetic geometry, untrained CPU hooks, transport faults."""
import ast
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
import p5_io as disk
import p5_analysis as analysis
from p5_numeric import analyze


def must_reject(fn):
    try:
        fn()
    except (ValueError, KeyError, zipfile.BadZipFile):
        return
    raise AssertionError("Expected validation rejection")


def synthetic():
    rng = np.random.default_rng(31)
    n, d = 200, 16
    z = np.linspace(-1, 1, n)
    h = rng.normal(0, .001, (n, d))
    h[:, 0] = z
    h[:, 1] = .6*z*z
    raw = dict(h=h.astype(np.float32), position=np.arange(n), token_chars=np.arange(n)%8,
               surprisal=np.r_[np.nan, np.ones(n-1)])
    metrics, arrays = analyze(raw, "C0", disk.spec())
    assert metrics["curved_graph_2d"] and metrics["curved_graph_full"]
    assert metrics["curve_gain_full"] > .01
    # A single far point yields excellent V1 without an occupied arc.
    cloud = rng.normal(0, .01, (n, d))
    cloud[0, 0] = 100
    outlier, _ = analyze(dict(raw, h=cloud.astype(np.float32)), "C0", disk.spec())
    assert outlier["V1"] > .99 and outlier["max_gap"] > .9
    assert not outlier["graph_2d"] and not outlier["graph_full"]
    # Independent Gram eigendecomposition and held-out full-space regression.
    y = h-h.mean(0)
    eigen, vectors = np.linalg.eigh(y@y.T)
    assert abs(metrics["V1"]-eigen[-1]/np.sum(y*y)) < 1e-7
    np.testing.assert_allclose(arrays["components"]@arrays["components"].T, np.eye(8), atol=1e-10)
    for degree in [1, 2]:
        design = np.vander(arrays["z"], N=degree+1, increasing=True)
        for fold in range(5):
            test = arrays["folds"] == fold
            train = ~test
            # Pseudoinverse differs from the production least-squares solve.
            beta = np.linalg.pinv(design[train], rcond=1e-12)@y[train]
            error = y[test]-design[test]@beta
            np.testing.assert_allclose(np.sum(error*error), arrays["curve_fold_sse"][degree-1, fold, 0],
                                       atol=2e-7, rtol=1e-6)
    unit = analyze(raw, "C4", disk.spec())[0]
    without = analyze(raw, "C1", disk.spec())[1]
    assert len(without["positions"]) == n-1 and without["positions"][0] == 1
    return dict(curved_fixture=True, isolated_point_rejected=True, independent_gram=True,
                independent_fold_sse=True, C1_and_C4=True)


def adapter():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from p5_worker import collect, compatibility
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=2, n_embd=24, vocab_size=50257,
        n_positions=1024, output_hidden_states=True, use_cache=True, attn_implementation="eager")).eval()
    tokenizer = AutoTokenizer.from_pretrained(disk.HERE.parent/"design_inputs/p4_v1/gpt2",
                                             local_files_only=True)
    entry = disk.entries()[0]
    ids = tokenizer.encode(entry["text"], return_tensors="pt")
    assert ids[0].tolist() == entry["token_ids"]
    arrays, checks = collect(model, ids, tokenizer)
    analysis.verify_unit(arrays, entry, width=24)
    assert checks["hooks"] == {p: 1 for p in disk.spec()["points"]}
    assert not np.array_equal(arrays["block06"], arrays["block07"])
    assert compatibility(dict(compute_capability=(8, 0))) == compatibility(dict(compute_capability=[8, 0]))
    # Corrupt a valid normalization observation, preserving all array dimensions.
    broken = {**arrays, "final_ln": arrays["final_ln"]+1}
    try:
        analysis.verify_unit(broken, entry, width=24)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted normalization was accepted")
    return arrays, checks, dict(random_untrained_GPT2_CPU=True, hidden_indices_6_7_final=True,
        separate_numpy_normalization=True, corrupt_norm_rejected=True, tuple_list_regression=True,
        pretrained_weights_downloaded=False, production_gpu_tested=False)


def transport(arrays, checks):
    with tempfile.TemporaryDirectory(prefix="pcgeom_p5_test_") as t:
        folder = Path(t)
        root = folder/"run"
        disk.initialize(root)
        entry = disk.entries()[0]
        disk.save_unit(root, "cpu_narrative", arrays, dict(device="cpu", text_id="narrative",
                       text_sha256=entry["utf8_sha256"], checks=checks))
        assert len(disk.committed(root)) == 1
        must_reject(lambda: disk.save_unit(root, "cpu_narrative", arrays, {}))
        path = disk.archive(root, folder/"archives", "MOCK")
        destination = folder/"restored"
        disk.restore(path, destination)
        assert len(disk.committed(destination)) == 1
        assert disk.digest(root/"units/cpu_narrative/raw.npz") == disk.digest(destination/"units/cpu_narrative/raw.npz")
        def corrupt_archive(name, transform):
            target = folder/name
            with zipfile.ZipFile(path) as src, zipfile.ZipFile(target, "w") as dst:
                for item in src.infolist():
                    data = src.read(item.filename)
                    altered = transform(item.filename, data)
                    if altered is not None:
                        dst.writestr(item.filename, altered)
            must_reject(lambda: disk.inspect_zip(target))
        corrupt_archive("corrupt.zip", lambda n, b: b+b"x" if n.endswith("raw.npz") else b)
        corrupt_archive("missing.zip", lambda n, b: None if n.endswith("raw.npz") else b)
        corrupt_archive("wrong_capsule.zip", lambda n, b: json.dumps(dict(json.loads(b), capsule="other")).encode()
                        if n == "ARCHIVE.json" else b)
        malicious = folder/"traversal.zip"
        with zipfile.ZipFile(malicious, "w") as z:
            z.writestr("../outside", "never")
            z.writestr("ARCHIVE.json", "{}")
        must_reject(lambda: disk.inspect_zip(malicious))
        duplicate = folder/"duplicate.zip"
        with zipfile.ZipFile(duplicate, "w") as z:
            z.writestr("ARCHIVE.json", "{}")
            z.writestr("ARCHIVE.json", "{}")
        must_reject(lambda: disk.inspect_zip(duplicate))
        rawfile = destination/"units/cpu_narrative/raw.npz"
        rawfile.write_bytes(rawfile.read_bytes()+b"x")
        must_reject(lambda: disk.committed(destination))
        # Real worker no-op check must return before model/hardware/download operations.
        import p5_worker
        for e in disk.entries()[1:]:
            disk.save_unit(root, "cpu_"+e["id"], arrays, dict(device="cpu", text_id=e["id"], checks=checks))
        before = {p: disk.digest(p) for p in (root/"units").rglob("*") if p.is_file()}
        p5_worker.main(root, "cpu")
        assert before == {p: disk.digest(p) for p in (root/"units").rglob("*") if p.is_file()}
        return dict(zip_roundtrip=True, duplicate_commit_rejected=True, corrupt_zip_rejected=True,
            missing_member_rejected=True, different_capsule_rejected=True, path_traversal_rejected=True,
            duplicate_zip_member_rejected=True, corrupt_committed_unit_rejected=True,
            completed_device_worker_noop=True, fixture_zip_bytes=path.stat().st_size)


def pipeline():
    """Production-dimensional synthetic arrays through all 72 analysis cells."""
    rng = np.random.default_rng(617)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pcgeom_p5_pipeline_") as t:
        folder = Path(t)
        root = folder/"run"
        disk.initialize(root)
        for entry in disk.entries():
            n, width = entry["tokens"], 768
            raw = {p: rng.normal(size=(n, width)).astype(np.float32) for p in disk.spec()["points"]}
            pre = raw["block12_preln"].astype(float)
            centered = pre-pre.mean(1, keepdims=True)
            raw["final_ln"] = (centered/np.sqrt(np.mean(centered**2, 1, keepdims=True)+1e-5)).astype(np.float32)
            pieces = np.array(["x"]*n)
            raw.update(token_id=np.array(entry["token_ids"]), position=np.arange(n),
                       logit_source_index=np.arange(n)-1, token_text=pieces,
                       token_chars=np.ones(n, dtype=int), token_bytes=np.ones(n, dtype=int),
                       surprisal=np.r_[np.nan, np.repeat(2., n-1)],
                       logsumexp=np.r_[np.nan, np.repeat(3., n-1)],
                       target_logit=np.r_[np.nan, np.repeat(1., n-1)],
                       norm_weight=np.ones(width, dtype=np.float32),
                       norm_bias=np.zeros(width, dtype=np.float32), norm_eps=np.array(1e-5))
            for device in disk.spec()["devices"]:
                checks = {k: True for k in ["hidden_state_alignment", "norm_reconstruction",
                                           "logit_reconstruction", "cross_entropy_alignment"]}
                disk.save_unit(root, device+"_"+entry["id"], raw, dict(device=device, text_id=entry["id"],
                    checks=checks, fixture_only=True))
        result = analysis.main(root, full=True)
        assert result["analyzed_conditions"] == 72 and not result["analysis_failures"]
        assert len(result["cpu_gpu_comparisons"]) == 12
        assert all(x["within_tolerance"] for x in result["cpu_gpu_comparisons"])
        path = disk.archive(root, folder/"archive", "SYNTHETIC_NOT_EXPERIMENT")
        disk.restore(path, folder/"restore")
        from p5_launch import process, child_environment, resolve
        import sys
        log = folder/"timeout.log"
        try:
            process([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(10)"],
                    time.monotonic()+.3, log, child_environment(folder))
        except TimeoutError:
            pass
        else:
            raise AssertionError("Child timeout was not enforced")
        assert log.exists()
        base = folder/"freshbase"
        base.mkdir()
        live = resolve(base, "")
        assert resolve(base, "") == live
        must_reject(lambda: resolve(base, str(path)))
        return dict(complete_synthetic_conditions=72, cpu_cuda_mock_pairs=12,
                    synthetic_zip_bytes=path.stat().st_size, seconds=time.monotonic()-start,
                    child_timeout_and_log=True, same_VM_pointer=True, mixed_restore_rejected=True,
                    real_GPU_comparison=False)


def main():
    import importlib.metadata as md
    start = time.monotonic()
    for path in disk.HERE.glob("*.py"):
        ast.parse(path.read_text())
    numeric = synthetic()
    arrays, checks, adapters = adapter()
    movement = transport(arrays, checks)
    complete = pipeline()
    record = dict(status="PASS", at=disk.stamp(), seconds=time.monotonic()-start,
        numerical=numeric, adapter=adapters, transport=movement, pipeline=complete,
        local_versions={n: md.version(n) for n in ["torch", "transformers", "numpy", "tokenizers"]},
        limitation="Local versions differ from pinned Colab runtime. No pretrained model or GPU execution.",
        source_sha256=disk.source_hashes())
    disk.write(disk.HERE/"test_results.json", record)
    print(disk.canonical(record))


if __name__ == "__main__":
    main()
