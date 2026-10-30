"""One bounded saved-observation analysis. No torch, Transformers, network or GPU."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import numeric as n

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_io as disk
import p4_numeric as rawio
import p4_fit as original

RAW = BASE / "receipts/4b013b35152b_p4_v1_final/raw"
SPEC = disk.read(OUT / "spec.json")
FIXTURES = disk.fixtures()


def load_data(key, point, split, counts):
    arrays = {k: [] for k in ["x", "b", "ids", "y", "type", "frequency", "position"]}
    docs = []
    start = 0
    for f in FIXTURES:
        if f["split"] != split:
            continue
        folder = RAW / "docs" / disk.doc_key(key, f["id"])
        a = rawio.load_npz(folder / "raw.npz")
        x, b, ids, y = original.features(a, point, counts)
        meta = disk.read(folder / "doc.json")
        arrays["x"].append(x)
        arrays["b"].append(b)
        arrays["ids"].append(ids)
        arrays["y"].append(y)
        arrays["type"].append(np.array([n.token_type(t) for t in meta["decoded_tokens"][1:]]))
        arrays["frequency"].append(np.searchsorted([1, 10, 100], counts[ids], side="right"))
        arrays["position"].append(np.searchsorted([.25, .5, .75], b[:, 0], side="left"))
        docs.append(dict(id=f["id"], fold=f["fold"], start=start, end=start + len(y)))
        start += len(y)
    return {k: np.concatenate(v) for k, v in arrays.items()} | {"docs": docs}


def training_transforms(data, common):
    x = data["x"]
    radius = np.linalg.norm(x, axis=1)
    assert np.min(radius) > 1e-12
    unit = x / radius[:, None]
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, atol=1e-12)
    mean = unit.mean(0)
    centered = unit - mean
    cov = centered.T @ centered / (len(unit) - 1)
    eig, vec = np.linalg.eigh(cov)
    eig = eig[::-1].copy()
    pc = vec[:, ::-1][:, :8].T.copy()
    for v in pc:
        if v[np.argmax(abs(v))] < 0:
            v *= -1
    assert eig[7] > np.trace(cov) * 1e-12
    np.testing.assert_allclose(pc @ pc.T, np.eye(8), atol=1e-10)
    eigen_error = float(np.max(abs(cov @ pc.T - pc.T * eig[:8])) / np.trace(cov))
    assert eigen_error < 1e-10
    basis = np.concatenate([pc, common["random_components"].reshape(-1, x.shape[1])])
    us = n.scale(centered @ basis.T)
    knots = np.quantile(data["b"][:, 7], SPEC["norm_knot_quantiles"], method="linear")
    hinge = np.maximum(data["b"][:, 7, None] - knots, 0)
    hs = n.scale(hinge)
    return dict(unit_mean=mean, unit_components=pc, unit_eigenvalues=eig,
                unit_score_mean=us[0], unit_score_scale=us[1], unit_score_active=us[2],
                knots=knots, hinge_mean=hs[0], hinge_scale=hs[1], hinge_active=hs[2]), eigen_error


def prepare(data, common, t):
    x = data["x"]
    radii = np.linalg.norm(x, axis=1)
    assert radii.min() > 1e-12
    rawbasis = np.concatenate([common["components"], common["random_components"].reshape(-1, x.shape[1])])
    ubasis = np.concatenate([t["unit_components"], common["random_components"].reshape(-1, x.shape[1])])
    z = original.transform(data["b"], common, "feature")[:, [0, 1, 2, 3, 7]]
    r = original.transform((x - common["mean"]) @ rawbasis.T, common, "score")
    u = (x / radii[:, None] - t["unit_mean"]) @ ubasis.T
    u = (u - t["unit_score_mean"]) / t["unit_score_scale"] * t["unit_score_active"]
    hinge = np.maximum(data["b"][:, 7, None] - t["knots"], 0)
    hinge = (hinge - t["hinge_mean"]) / t["hinge_scale"] * t["hinge_active"]
    uncentered = np.column_stack([z, hinge, r, u])
    centered, targets, masks, weights, folds, article_index, docs = [], [], [], [], [], [], []
    offset = 0
    for j, d in enumerate(data["docs"]):
        sl = slice(d["start"], d["end"])
        c, mask, count = n.center(uncentered[sl], data["ids"][sl])
        cy, _, _ = n.center(data["y"][sl], data["ids"][sl])
        check = original.center(data["ids"][sl], uncentered[sl])[0]
        np.testing.assert_allclose(c, check, atol=1e-9, rtol=1e-7)
        groups = int(np.sum(1 / count[mask]) + .5)
        assert mask.sum() >= 32 and groups >= 8 and cy[mask] @ cy[mask] > 1e-12
        centered.append(c[mask])
        targets.append(cy[mask])
        masks.append(mask)
        weights.append(1 / count[mask])
        folds.append(np.repeat(d["fold"], mask.sum()))
        article_index.append(np.repeat(j, mask.sum()))
        docs.append(dict(id=d["id"], start=offset, end=offset + int(mask.sum()), groups=groups,
                         full_start=d["start"], full_end=d["end"]))
        offset += int(mask.sum())
    c = np.concatenate(centered)
    mask = np.concatenate(masks)
    return dict(z=c[:, :5], h=c[:, 5:10], r=c[:, 10:178].reshape(-1, 21, 8),
                u=c[:, 178:].reshape(-1, 21, 8), y=np.concatenate(targets),
                weights=np.concatenate(weights), folds=np.concatenate(folds),
                ids=data["ids"][mask], type=data["type"][mask],
                frequency=data["frequency"][mask], position=data["position"][mask],
                article_index=np.concatenate(article_index), docs=docs, full_mask=mask,
                uncentered=uncentered)


def design(a, family):
    if family in ["A1", "A7", "A8"]:
        ix = [0] if family == "A1" else list(range(1, 8)) if family == "A7" else list(range(8))
        return a["z"], a["r"][:, :, ix]
    z = a["z"][:, :4] if family == "PU" else np.column_stack([a["z"], a["h"]])
    return z, a["r"] if family == "NR" else a["u"]


def one(key, point):
    dest = OUT / "results" / key / point
    assert not dest.exists(), "Never overwrite an analysis result"
    dest.mkdir(parents=True)
    counts = rawio.load_npz(RAW / "fit" / key / "frequency.npz")["counts"]
    common = rawio.load_npz(RAW / "fit" / key / point / "common.npz")
    old = rawio.load_npz(RAW / "fit" / key / point / "W.npz")
    train = load_data(key, point, "train", counts)
    transforms, eigen_error = training_transforms(train, common)
    tr = prepare(train, common, transforms)
    del train
    alpha = float(old["alpha"])
    fit_arrays = dict(transforms)
    fit_arrays["alpha"] = np.array(alpha)
    cv = {}
    for name, z in [("P", tr["z"][:, :4]), ("B", tr["z"])]:
        beta = n.solve(z, tr["y"], np.repeat(alpha, z.shape[1]))
        n.audit_solution(z, tr["y"], np.repeat(alpha, z.shape[1]), beta)
        fit_arrays[name + "_beta"] = beta
    np.testing.assert_allclose(fit_arrays["B_beta"], old["beta"][0], atol=1e-8, rtol=1e-7)
    norm_model, cv["N"] = n.cv_block(tr["z"], tr["h"][:, None, :], tr["y"], tr["folds"], np.repeat(alpha, 5))
    fit_arrays["N_beta"] = norm_model["beta"][0]
    fit_arrays["N_gamma"] = norm_model["gamma"]
    models = {}
    for family in n.FAMILIES:
        z, w = design(tr, family)
        pen = (np.repeat(alpha, z.shape[1]) if family not in ["NR", "NU"]
               else np.r_[np.repeat(alpha, 5), np.repeat(norm_model["gamma"][0], 5)])
        models[family], cv[family] = n.cv_block(z, w, tr["y"], tr["folds"], pen)
        fit_arrays[family + "_beta"] = models[family]["beta"]
        fit_arrays[family + "_gamma"] = models[family]["gamma"]
        fit_arrays[family + "_base_penalty"] = pen
    np.testing.assert_array_equal(models["A8"]["gamma"], old["gammas"])
    np.testing.assert_allclose(models["A8"]["beta"], np.column_stack([old["beta"][1:], old["theta"]]),
                               atol=1e-8, rtol=1e-7)
    oldcv = disk.read(RAW / "fit" / key / point / "cv.json")["W"]
    np.testing.assert_allclose(cv["A8"]["mse"], oldcv["gamma_mse"], atol=1e-8, rtol=1e-7)
    del tr
    rawio.save_npz(dest / "fits.npz", fit_arrays)
    disk.write(dest / "cv.json", cv)
    # Read the just-saved fits before evaluation, preserving an explicit train-only seal.
    disk.write(dest / "FIT_SEAL.json", dict(sealed_at=disk.stamp(), source_fit_sha256=disk.digest(RAW / "fit/COMMIT.json"),
               files={p.name: disk.digest(p) for p in [dest / "fits.npz", dest / "cv.json"]}))
    loaded = rawio.load_npz(dest / "fits.npz")
    for f in n.FAMILIES:
        models[f] = {"beta": loaded[f + "_beta"], "gamma": loaded[f + "_gamma"]}
    data = load_data(key, point, "test", counts)
    te = prepare(data, common, loaded)
    pred = {
        "P": (te["z"][:, :4] @ loaded["P_beta"])[:, None],
        "B": (te["z"] @ loaded["B_beta"])[:, None],
        "N": (np.column_stack([te["z"], te["h"]]) @ loaded["N_beta"])[:, None],
    }
    for f in n.FAMILIES:
        pred[f] = n.predict(*design(te, f), models[f])
    max_original, max_reconstructed = 0., 0.
    articles = []
    for d in te["docs"]:
        sl, full = slice(d["start"], d["end"]), slice(d["full_start"], d["full_end"])
        repeat = te["full_mask"][full]
        y = te["y"][sl]
        local = {k: v[sl] for k, v in pred.items()}
        saved = rawio.load_npz(RAW / "evaluation" / key / point / d["id"] / "predictions.npz")["W"][repeat]
        got = np.column_stack([local["B"], local["A8"]])
        np.testing.assert_allclose(got, saved, atol=1e-8, rtol=1e-7)
        max_original = max(max_original, float(np.max(abs(got - saved))))
        # Alternative path: predict uncentered token rows, then center predictions.
        fmat = te["uncentered"][full]
        alt = dict(z=fmat[:, :5], h=fmat[:, 5:10], r=fmat[:, 10:178].reshape(-1, 21, 8),
                   u=fmat[:, 178:].reshape(-1, 21, 8))
        for family in n.FAMILIES:
            z, w = design(alt, family)
            q = np.column_stack([z, w[:, 0]]).shape[1]
            recon = np.column_stack([np.column_stack([z, w[:, j]]) @ loaded[family + "_beta"][j]
                                    for j in range(21)])
            recon = original.center(data["ids"][full], recon)[0][repeat]
            np.testing.assert_allclose(recon, local[family], atol=1e-8, rtol=1e-7)
            max_reconstructed = max(max_reconstructed, float(np.max(abs(recon - local[family]))))
        metrics, r2 = n.score(y, local)
        macro, macro_r2 = n.score(y, local, te["weights"][sl])
        token_delta = n.token_contrasts(n.errors(y, local))
        ss = float(y @ y)
        bias = {}
        for name, labels in n.STRATA.items():
            cats = te[name][sl]
            bias[name] = []
            for i, label in enumerate(labels):
                mask = cats == i
                bias[name].append(dict(label=label, tokens=int(mask.sum()),
                                       contributions={c: float(v[mask].sum() / ss) for c, v in token_delta.items()}))
            for c, v in token_delta.items():
                assert abs(sum(b["contributions"][c] for b in bias[name]) - v.sum() / ss) < 1e-10
        map_metrics = ["A1_gain", "A7_gain", "A8_gain", "N_minus_B", "NR_gain", "NU_gain", "A8_adv", "NU_adv"]
        for c, m in zip(n.CONTRASTS, map_metrics):
            assert abs(token_delta[c].sum() / ss - metrics[m]) < 1e-10
        articles.append(dict(document_id=d["id"], model=key, point=point, tokens=len(y), groups=d["groups"],
                             metrics=metrics, r2=r2, equal_ID_metrics=macro, equal_ID_r2=macro_r2, bias=bias))
    arrays = {k: np.asarray(v) for k, v in pred.items()}
    arrays.update({k: te[k] for k in ["y", "ids", "weights", "type", "frequency", "position", "article_index"]})
    np.savez_compressed(dest / "predictions.npz", **arrays)
    disk.write(dest / "articles.json", articles)
    audit = dict(status="PASS", training_articles=64, evaluation_articles=64,
                 unit_PCA_eigen_residual=eigen_error, P4_W_prediction_max_difference=max_original,
                 center_after_prediction_max_difference=max_reconstructed,
                 all_selected_fits_checked_by_augmented_lstsq=True,
                 all_selected_CV_checked_by_direct_predictions=True,
                 all_stratum_contributions_sum_to_article_gain=True,
                 fitted_models=3 + len(n.FAMILIES) * 21, saved_prediction_values=sum(v.size for v in pred.values()),
                 max_gradient=max(v["max_gradient_relative"] for v in cv.values()),
                 max_lstsq_difference=max(v["max_augmented_lstsq_difference"] for v in cv.values()),
                 max_cv_difference=max(v["max_cv_sse_difference"] for v in cv.values()))
    disk.write(dest / "audit.json", audit)
    disk.write(dest / "COMMIT.json", {p.name: disk.digest(p) for p in dest.iterdir() if p.is_file()})
    print(key, point, "completed; audit PASS; saved predictions", audit["saved_prediction_values"], flush=True)


def main():
    start = time.monotonic()
    freeze = disk.read(OUT / "plan_freeze.json")
    for path, h in freeze["plan_sha256"].items():
        assert disk.digest(PROJECT / path) == h
    disk.write(OUT / "execution_manifest.json", dict(
        started=disk.stamp(), plan_freeze_sha256=disk.digest(OUT / "plan_freeze.json"),
        code_sha256={p.name: disk.digest(p) for p in [OUT / "analyze.py", OUT / "numeric.py"]},
        numpy=np.__version__, python=sys.version, model_forwards=0, gpu_used=False,
        known_evaluation_data=True), immutable=True)
    try:
        for key in SPEC["models"]:
            for point in SPEC["points"]:
                assert time.monotonic() - start < 3600, "Numerical time budget"
                one(key, point)
        disk.write(OUT / "execution_result.json", dict(status="PASS", completed=disk.stamp(),
                   seconds=time.monotonic() - start, model_points=6, new_model_forwards=0))
    except BaseException:
        disk.write(OUT / "execution_failure.json", dict(at=disk.stamp(), seconds=time.monotonic() - start,
                   traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
