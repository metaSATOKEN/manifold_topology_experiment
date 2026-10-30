"""Offline raw verification and deterministic descriptive summaries."""
import io
from pathlib import Path
import sys
import numpy as np
import p5_io as disk
from p5_numeric import analyze


def load_unit(root, key):
    with np.load(Path(root)/"units"/key/"raw.npz", allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def verify_unit(a, entry, width=768):
    n = entry["tokens"]
    np.testing.assert_array_equal(a["token_id"], entry["token_ids"])
    np.testing.assert_array_equal(a["position"], np.arange(n))
    np.testing.assert_array_equal(a["logit_source_index"], np.arange(n)-1)
    for point in disk.spec()["points"]:
        disk.require(a[point].shape == (n, width) and a[point].dtype == np.float32 and
                     np.isfinite(a[point]).all(), "Invalid hidden state array")
    for k in ["surprisal", "logsumexp", "target_logit"]:
        disk.require(a[k].shape == (n,) and np.isnan(a[k][0]) and np.isfinite(a[k][1:]).all(),
                     "Invalid next-token alignment")
    np.testing.assert_allclose(a["surprisal"][1:], a["logsumexp"][1:]-a["target_logit"][1:], atol=1e-10, rtol=1e-10)
    disk.require(np.all(a["surprisal"][1:] >= -1e-10), "Negative surprisal")
    np.testing.assert_array_equal(a["token_chars"], [len(t) for t in a["token_text"]])
    np.testing.assert_array_equal(a["token_bytes"], [len(t.encode()) for t in a["token_text"]])
    # NumPy double-precision LayerNorm reconstruction is separate from the torch check.
    pre = a["block12_preln"].astype(np.float64)
    centered = pre-pre.mean(1, keepdims=True)
    reconstructed = centered/np.sqrt(np.mean(centered**2, axis=1, keepdims=True)+float(a["norm_eps"]))
    reconstructed = reconstructed*a["norm_weight"]+a["norm_bias"]
    np.testing.assert_allclose(reconstructed, a["final_ln"], atol=3e-5, rtol=3e-5)


def main(root, full=False):
    root = Path(root)
    cfg = disk.spec()
    commits = disk.committed(root)
    entries = {e["id"]: e for e in disk.entries()}
    rows, failures = [], []
    for key, meta in commits.items():
        a = load_unit(root, key)
        verify_unit(a, entries[meta["text_id"]])
        disk.require(all(meta["checks"][k] is True for k in [
            "hidden_state_alignment", "norm_reconstruction", "logit_reconstruction", "cross_entropy_alignment"]),
            "Extraction checks missing")
        for point in cfg["points"]:
            raw = dict(a, h=a[point])
            for condition in cfg["conditions"]:
                identity = dict(key=key, device=meta["device"], text=meta["text_id"], point=point, condition=condition)
                try:
                    metrics, arrays = analyze(raw, condition, cfg)
                    if meta["device"] == "cpu" and point == "block07" and condition == "C0":
                        metrics["historical_pc1_difference_pp"] = 100*(metrics["V1"]-cfg["reference_pc1"][meta["text_id"]])
                        metrics["historical_pc2_difference_pp"] = 100*(metrics["pc2_variance"]-cfg["reference_pc2"][meta["text_id"]])
                        metrics["historical_variance_agreement"] = abs(metrics["historical_pc1_difference_pp"]) <= 100*cfg["reference_abs_tolerance"]
                    rows.append(dict(identity, status="OK", metrics=metrics))
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **arrays)
                    disk.atomic(root/"analysis"/(key+"_"+point+"_"+condition+".npz"), buf.getvalue())
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    failures.append(dict(identity, error=type(exc).__name__+": "+str(exc)))
    comparisons = []
    for e in entries:
        if "cpu_"+e not in commits or "cuda_"+e not in commits:
            continue
        a, b = load_unit(root, "cpu_"+e), load_unit(root, "cuda_"+e)
        for point in cfg["points"]:
            x, y = a[point].astype(float), b[point].astype(float)
            comparisons.append(dict(text=e, point=point, max_abs=float(np.max(abs(x-y))),
                relative_frobenius=float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-12)),
                within_tolerance=bool(np.allclose(x, y, atol=cfg["cpu_gpu_atol"], rtol=cfg["cpu_gpu_rtol"]))))
    if full:
        disk.require(set(commits) == set(disk.keys()), "Full verification requires six units")
        disk.require(len(rows)+len(failures) == 72, "Analysis inventory incomplete")
        disk.require(len(comparisons) == 12, "CPU/GPU comparison inventory incomplete")
    result = dict(status="RAW_VERIFIED_COMPLETE" if len(commits) == 6 else "PARTIAL_RAW_VERIFIED",
        capsule=disk.capsule(), committed_units=len(commits), planned_units=6,
        analyzed_conditions=len(rows), planned_conditions=72, analysis_failures=failures,
        rows=rows, cpu_gpu_comparisons=comparisons,
        scientific_interpretation="Pending receipt verification; geometry criteria do not establish intrinsic dimension.",
        at=disk.stamp())
    disk.write(root/"summary.json", result)
    print(disk.canonical({k: result[k] for k in ["status", "committed_units", "analyzed_conditions", "analysis_failures"]}))
    return result


if __name__ == "__main__":
    main(sys.argv[1], full=len(sys.argv) > 2 and sys.argv[2] == "full")
