"""Independent Gram/QR reconstruction of all fixed P5 endpoints; no LM forward."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh, solve_triangular
from intake import OUT, BASE, RELEASE, sha, write

RAW = OUT/"raw"
SPEC = json.loads((RELEASE/"spec.json").read_text())
INPUTS = json.loads((RELEASE/"inputs.json").read_text())["entries"]
errors = {}
scaled_errors = {}


def close(name, actual, expected, atol=1e-8, rtol=2e-7):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
    maximum = float(np.max(abs(a-b))) if a.size else 0.
    errors[name] = max(errors.get(name, 0.), maximum)
    scaled = float(np.max(abs(a-b)/(1+abs(b)))) if a.size else 0.
    scaled_errors[name] = max(scaled_errors.get(name, 0.), scaled)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def r(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spacing(a):
    a = np.sort(a)
    cuts = (a-a[0])/(a[-1]-a[0])
    index = np.minimum(19, np.floor(cuts*20).astype(int))
    return len(np.unique(index))/20, float(np.max(np.diff(cuts)))


def qr_fit(design, y):
    q, rr = np.linalg.qr(design, mode="reduced")
    return solve_triangular(rr, q.T@y)


def reconstruct(a, point, condition):
    original = a[point].astype(float)
    norms = np.linalg.norm(original, axis=1)
    mask = a["position"] > 0 if condition == "C1" else np.ones(len(original), bool)
    x = original[mask].copy()
    pos = a["position"][mask]
    if condition == "C4":
        x /= norms[mask, None]
    mu = x.mean(0)
    y = x-mu
    gram = y@y.T
    total = float(np.trace(gram))
    eigen, u = eigh(gram)
    eigen, u = eigen[::-1], u[:, ::-1]
    v = (y.T@u[:, :8]/np.sqrt(eigen[:8])).T
    sign = np.sign(v[np.arange(8), abs(v).argmax(1)])
    v *= sign[:, None]
    score = y@v.T
    ratio = eigen[:8]/total
    z = score[:, 0]/score[:, 0].std()
    folds = np.minimum(4, np.arange(len(x))*5//len(x))
    totals = np.array([total, np.sum(score[:, :2]**2), np.sum(score[:, :3]**2)])
    sse, insample, coefficients, full_predictions = [], [], [], []
    for degree in (1, 2):
        design = np.vander(z, degree+1, increasing=True)
        coef = qr_fit(design, y)
        coefficients.append(coef)
        e = y-design@coef
        ep = e@v[:3].T
        insample.append(1-np.array([np.sum(e**2), np.sum(ep[:, :2]**2), np.sum(ep**2)])/totals)
        pred = np.empty_like(y)
        by_fold = []
        for fold in range(5):
            te = folds == fold
            tr = ~te
            beta = qr_fit(design[tr], y[tr])
            pred[te] = design[te]@beta
            error = y[te]-pred[te]
            projected = error@v[:3].T
            by_fold.append([np.sum(error**2), np.sum(projected[:, :2]**2), np.sum(projected**2)])
        sse.append(by_fold)
        full_predictions.append(pred)
    sse = np.array(sse)
    q = 1-sse.sum(1)/totals
    thick = (y-np.vander(z, 3, increasing=True)@coefficients[1])@v[1:8].T
    occ, gap = spacing(score[:, 0])
    lo, hi = np.percentile(score[:, 0], [5, 95])
    occ90, gap90 = spacing(score[(score[:, 0] >= lo) & (score[:, 0] <= hi), 0])
    share = score[:, 0]**2/np.sum(score[:, 0]**2)
    m = dict(V1=float(ratio[0]), V2=float(ratio[:2].sum()), V3=float(ratio[:3].sum()),
        V8=float(ratio.sum()), pc2_variance=float(ratio[1]), participation=total**2/np.sum(gram**2),
        occupancy=occ, max_gap=gap, occupancy90=occ90, max_gap90=gap90,
        max_pc1_share=float(share.max()), max_pc1_position=int(pos[share.argmax()]),
        first_pc1_share=float(share[pos == 0][0]) if 0 in pos else None,
        pc1_norm_r=r(score[:, 0], norms[mask]))
    spread = occ >= .8 and gap <= .1 and share.max() <= .5
    for j, space in enumerate(["full", "2d", "3d"]):
        m["Q_line_"+space], m["Q_curve_"+space] = float(q[0, j]), float(q[1, j])
        m["curve_gain_"+space], m["Q_insample_"+space] = float(q[1, j]-q[0, j]), float(insample[1][j])
        m["graph_"+space] = bool(spread and q[1, j] >= .9)
        m["curved_graph_"+space] = bool(m["graph_"+space] and q[1, j]-q[0, j] >= .01)
    features = dict(position=pos, log_norm=np.log1p(norms[mask]),
                    log_chars=np.log1p(a["token_chars"][mask]), surprisal=a["surprisal"][mask])
    m["thickness_correlations"] = {k: [r(thick[:, col], value) for col in range(7)]
                                    for k, value in features.items()}
    xf = x.astype(np.float32)
    yf = xf-xf.mean(0)
    singular32 = np.linalg.svd(yf, full_matrices=False, compute_uv=False)
    m["float32_V1"] = float(singular32[0]**2/np.sum(yf*yf))
    diagnostic = {}
    if condition == "C0":
        squared = np.sum((y-full_predictions[1])**2, axis=1)
        tr = folds != 0
        diagnostic = dict(first_error_share=float(squared[0]/squared.sum()),
            largest_error_position=int(pos[squared.argmax()]), largest_error_share=float(squared.max()/squared.sum()),
            first_z=float(z[0]), fold0_train_z_min=float(z[tr].min()), fold0_train_z_max=float(z[tr].max()),
            first_norm=float(norms[0]), other_norm_min=float(norms[1:].min()), other_norm_max=float(norms[1:].max()))
    arrays = dict(mean=mu, components=v, scores=score, positions=pos, z=z, folds=folds,
        curve_fold_sse=sse, curve_q=q, line_coef=coefficients[0], curve_coef=coefficients[1], thickness=thick)
    return m, arrays, diagnostic


def main():
    started = time.monotonic()
    summary = json.loads((RAW/"summary.json").read_text())
    release = json.loads((RELEASE/"release_manifest.json").read_text())
    contract = json.loads((RAW/"environment_contract.json").read_text())
    assert contract["packages"] == SPEC["pins"]
    assert contract["dtype"] == "float32" and contract["attention"] == "eager" and contract["tf32"] is False
    for path in (RAW/"environments").glob("*.json"):
        env = json.loads(path.read_text())
        assert {k: env[k] for k in contract} == contract
    assert json.loads((RAW/"model_verified.json").read_text()) == json.loads((RELEASE/"model.json").read_text())
    observations = {}
    unit_checks = []
    for device in SPEC["devices"]:
        for entry in INPUTS:
            key = device+"_"+entry["id"]
            path = RAW/"units"/key
            meta = json.loads((path/"commit.json").read_text())
            assert meta["key"] == key and meta["device"] == device and meta["text_id"] == entry["id"]
            assert meta["text_sha256"] == entry["utf8_sha256"] and meta["capsule"] == release["capsule"]
            assert meta["sha256"] == sha(path/"raw.npz")
            assert meta["checks"]["hooks"] == {p: 1 for p in SPEC["points"]}
            assert all(meta["checks"][k] for k in ["hidden_state_alignment", "norm_reconstruction",
                                                 "logit_reconstruction", "cross_entropy_alignment"])
            a = load(path/"raw.npz")
            n = entry["tokens"]
            np.testing.assert_array_equal(a["token_id"], entry["token_ids"])
            np.testing.assert_array_equal(a["position"], np.arange(n))
            np.testing.assert_array_equal(a["logit_source_index"], np.arange(n)-1)
            np.testing.assert_array_equal(a["token_chars"], [len(t) for t in a["token_text"]])
            np.testing.assert_array_equal(a["token_bytes"], [len(t.encode()) for t in a["token_text"]])
            for p in SPEC["points"]:
                assert a[p].shape == (n, 768) and a[p].dtype == np.float32 and np.isfinite(a[p]).all()
            for name in ["surprisal", "logsumexp", "target_logit"]:
                assert np.isnan(a[name][0]) and np.isfinite(a[name][1:]).all()
            close("surprisal_alignment", a["surprisal"][1:], a["logsumexp"][1:]-a["target_logit"][1:], 1e-10, 1e-10)
            pre = a["block12_preln"].astype(float)
            centered = pre-pre.mean(1, keepdims=True)
            normed = centered/np.sqrt(np.mean(centered**2, 1, keepdims=True)+float(a["norm_eps"]))
            ref = normed*a["norm_weight"]+a["norm_bias"]
            close("raw_LayerNorm", ref, a["final_ln"], 3e-5, 3e-5)
            unit_checks.append(dict(key=key, n=n, norm_max_error=float(abs(ref-a["final_ln"]).max()),
                                    extraction_seconds=meta["seconds"]))
            observations[key] = a
    records, diagnostics = [], []
    seen = set()
    for row in summary["rows"]:
        key, point, condition = row["key"], row["point"], row["condition"]
        assert (key, point, condition) not in seen
        seen.add((key, point, condition))
        actual, arrays, diagnostic = reconstruct(observations[key], point, condition)
        expected = row["metrics"]
        if row["device"] == "cpu" and point == "block07" and condition == "C0":
            actual["historical_pc1_difference_pp"] = 100*(actual["V1"]-SPEC["reference_pc1"][row["text"]])
            actual["historical_pc2_difference_pp"] = 100*(actual["pc2_variance"]-SPEC["reference_pc2"][row["text"]])
            actual["historical_variance_agreement"] = abs(actual["historical_pc1_difference_pp"]) <= .5
        assert set(actual) == set(expected)
        for name, value in actual.items():
            if name == "thickness_correlations":
                for feature in value:
                    close("feature_correlations", value[feature], expected[name][feature], atol=1e-7)
            elif isinstance(value, bool) or value is None:
                assert value == expected[name], (key, point, condition, name)
            else:
                close("metric_"+name, value, expected[name], atol=2e-7 if name == "float32_V1" else 1e-8)
        saved = load(RAW/"analysis"/(key+"_"+point+"_"+condition+".npz"))
        for name, value in arrays.items():
            close("array_"+name, value, saved[name], atol=3e-6 if name in ("line_coef", "curve_coef") else 3e-7)
        close("singular_values_top8", np.sqrt(np.sum(arrays["scores"]**2, 0)), saved["singular_values"][:8], atol=1e-7)
        assert np.allclose(arrays["components"]@arrays["components"].T, np.eye(8), atol=1e-9)
        records.append(dict(key=key, device=row["device"], text=row["text"], point=point,
                            condition=condition, metrics=actual))
        if diagnostic:
            diagnostics.append(dict(key=key, point=point, **diagnostic))
    assert len(seen) == 72
    comparisons = []
    for stored in summary["cpu_gpu_comparisons"]:
        text, point = stored["text"], stored["point"]
        x, y = observations["cpu_"+text][point].astype(float), observations["cuda_"+text][point].astype(float)
        diff = abs(x-y)
        # np.allclose(x, y) scales relative tolerance against y, the GPU array.
        tolerance = 1e-4+1e-4*abs(y)
        exceeds = np.argwhere(diff > tolerance)
        close("device_max_abs", diff.max(), stored["max_abs"])
        close("device_relative_frobenius", np.linalg.norm(x-y)/np.linalg.norm(x), stored["relative_frobenius"])
        assert (len(exceeds) == 0) == stored["within_tolerance"]
        bad = [dict(position=int(i), coordinate=int(j), token=str(observations["cpu_"+text]["token_text"][i]),
                    cpu=float(x[i, j]), cuda=float(y[i, j]), abs_diff=float(diff[i, j]), tolerance=float(tolerance[i, j]))
               for i, j in exceeds]
        comparisons.append(dict(stored, exceeded_coordinates=bad, total_coordinates=int(x.size)))
    index = {(r["device"], r["text"], r["point"], r["condition"]): r["metrics"] for r in records}
    agreement = []
    for entry in INPUTS:
        for point in SPEC["points"]:
            for condition in SPEC["conditions"]:
                c = index["cpu", entry["id"], point, condition]
                g = index["cuda", entry["id"], point, condition]
                gates = [k for k in c if k.startswith(("graph_", "curved_graph_"))]
                assert all(c[k] == g[k] for k in gates)
                agreement.append(dict(text=entry["id"], point=point, condition=condition,
                    V1_abs_difference=abs(c["V1"]-g["V1"]), same_geometry_decisions=True))
    for entry in INPUTS:
        g = index["cuda", entry["id"], "block07", "C0"]
        assert abs(g["V1"]-SPEC["reference_pc1"][entry["id"]]) <= .005
    write(OUT/"recomputed_metrics.json", records)
    write(OUT/"numerical_diagnostics.json", dict(cpu_gpu=comparisons, fixed_curve_diagnostics=diagnostics,
                                                cpu_gpu_geometry=agreement))
    result = dict(status="PASS", at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        independent_methods="Gram eigenvectors and QR regressions; NumPy LayerNorm; no archived code execution",
        units=6, conditions=72, curve_fold_sse_values=72*2*5*3,
        thickness_correlations=72*7*4, cpu_gpu_comparisons=12,
        source_hashes_match_release=True, frozen_before_acquisition=True,
        extraction_checks=unit_checks, max_abs_differences=errors, max_scaled_differences=scaled_errors,
        cpu_gpu_allclose_pass=sum(x["within_tolerance"] for x in comparisons),
        cpu_gpu_allclose_fail=sum(not x["within_tolerance"] for x in comparisons),
        cpu_gpu_geometry_agreement=len(agreement),
        runtime_GPU=contract["gpu_name"], GPU_candidate_deviation=True,
        new_model_forward=0, seconds=time.monotonic()-started,
        reviewer="Same agent self-audit, distinct reconstruction path; not independent review")
    write(OUT/"verification.json", result)
    print(json.dumps({k: result[k] for k in ["status", "units", "conditions", "curve_fold_sse_values",
        "thickness_correlations", "cpu_gpu_allclose_fail", "cpu_gpu_geometry_agreement", "seconds"]}))


if __name__ == "__main__":
    main()
