"""Separate reconstruction and stationarity audit of saved regressions."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.sparse import csr_matrix
import analyze as a


def feature_reconstruction(b, shape, meta):
    if meta["family"] == "ID":
        return b.copy()
    x = (b[:, [0, 4, 5, 6, 7]]-np.array(meta["spline_mean"]))/np.array(meta["spline_scale"])
    groups = []
    for j in range(5):
        groups.append(np.stack([x[:, j]]+[np.clip(x[:, j]-k, 0, None)
                            for k in meta["knots"][j]], axis=1))
    columns = [b, *groups]
    for j in range(5):
        for k in range(j+1, 5):
            columns.extend([(groups[j][:, u]*groups[k][:, v])[:, None]
                            for u in range(4) for v in range(4)])
    return np.concatenate([*columns, shape], axis=1)


def main():
    start = time.monotonic()
    commit = json.loads((a.HERE/"FIT_COMMIT.json").read_text())
    assert commit["source_sha256"] == a.digest(a.HERE/"analyze.py")
    assert commit["card_sha256"] == a.digest(a.CARD)
    for name, expected in commit["files"].items():
        assert a.digest(a.HERE/name) == expected
    train, test = a.read_data("train"), a.read_data("test")
    splits = json.loads((a.HERE/"splits.json").read_text())
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert splits["train"] == train["documents"] and splits["test"] == test["documents"]
    assert [list(splits["folds"].values()).count(i) for i in range(4)] == [16]*4
    counts = np.bincount(train["all_ids"].ravel(), minlength=a.VOCAB)
    with np.load(a.HERE/"evaluation_targets.npz") as t:
        np.testing.assert_array_equal(t["y"], test["y"].reshape(64, 511))
        np.testing.assert_array_equal(t["train_frequency"], counts[test["id"]].reshape(64, 511))
    cv = json.loads((a.HERE/"cv_results.json").read_text())
    for r in cv:
        assert r["selected_baseline"] == min(r["baseline_candidates"], key=lambda x: x["mean_mse"])
        for c in r["baseline_candidates"]:
            np.testing.assert_allclose(c["mean_mse"], np.mean(c["fold_mse"]), atol=1e-14)
        for c in r["conditions"].values():
            mean = np.array(c["fold_mse"]).mean(0)
            np.testing.assert_array_equal(np.array(a.GAMMAS)[mean.argmin(0)], c["selected_gamma"])
    results = json.loads((a.HERE/"results.json").read_text())
    y = test["y"].reshape(64, 511)
    sst = ((y-y.mean(1, keepdims=True))**2).sum(1)
    max_prediction_error = 0.
    max_normal_error = 0.
    max_original_error = 0.
    max_metric_error = 0.
    all_checks = []
    for point in a.POINTS:
        bt = a.controls(train, point, counts)
        bv = a.controls(test, point, counts)
        for condition in a.CONDITIONS:
            st, sv = a.scores(train, point, condition), a.scores(test, point, condition)
            original = a.HERE/"fits"/f"{point}__{condition}__original.npz"
            with np.load(original) as f:
                original_pred = f["prediction"]
            for j, name in enumerate(test["documents"]):
                with np.load(a.RAW/"evaluation"/name/f"{point}__{condition}"/"arrays.npz") as ref:
                    # C0/C4 retain position 0 in stored arrays; its target is NA.
                    keep = ref["kept_positions"] >= 1
                    np.testing.assert_array_equal(ref["kept_positions"][keep], np.arange(1, 512))
                    saved = ref["surprisal__predictions"].T[keep]
                    max_original_error = max(max_original_error, float(np.max(abs(original_pred[j]-saved))))
                    np.testing.assert_allclose(original_pred[j], saved, atol=1e-10, rtol=1e-10)
            orig_sse = ((y[:, :, None]-original_pred)**2).sum(1)
            orig_delta = (orig_sse[:, 0]-orig_sse[:, 1])/sst
            for family in a.FAMILIES:
                stem = f"{point}__{condition}__{family}"
                meta = json.loads((a.HERE/"fits"/(stem+".json")).read_text())
                with np.load(a.HERE/"fits"/(stem+".npz")) as f:
                    m = {k: f[k] for k in f.files}
                z = (feature_reconstruction(bt, train["shape"], meta["feature_transform"]) -
                     m["feature_mean"])/m["feature_scale"]*m["feature_active"]
                zv = (feature_reconstruction(bv, test["shape"], meta["feature_transform"]) -
                      m["feature_mean"])/m["feature_scale"]*m["feature_active"]
                z, zv = np.column_stack([np.ones(len(z)), z]), np.column_stack([np.ones(len(zv)), zv])
                w = (st-m["score_mean"])/m["score_scale"]*m["score_active"]
                wv = (sv-m["score_mean"])/m["score_scale"]*m["score_active"]
                ix = np.searchsorted(m["vocab"], train["id"])
                ixv = np.searchsorted(m["vocab"], test["id"])
                known = ixv < len(m["vocab"])
                ixv = np.minimum(ixv, len(m["vocab"])-1)
                known &= m["vocab"][ixv] == test["id"]
                u = csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), ix)),
                               shape=(len(ix), len(m["vocab"])))
                fitted_train = z @ m["beta"].T + m["token_effect"][:, ix].T
                fitted_test = zv @ m["beta"].T + m["token_effect"][:, ixv].T*known[:, None]
                for g in range(21):
                    sl = slice(g*8, g*8+8)
                    fitted_train[:, g+1] += w[:, sl] @ m["theta"][g]
                    fitted_test[:, g+1] += wv[:, sl] @ m["theta"][g]
                pred_error = float(np.max(abs(fitted_test.reshape(64, 511, 22)-m["prediction"])))
                np.testing.assert_allclose(fitted_test.reshape(64, 511, 22), m["prediction"],
                                           atol=1e-9, rtol=1e-10)
                residual = train["y"][:, None]-fitted_train
                gz = z.T @ residual
                gz[1:] -= len(z)*meta["alpha"]*m["beta"][:, 1:].T
                normal_error = np.max(abs(gz))/max(1., np.max(abs(z.T @ train["y"])))
                if family != "NL":
                    gu = np.asarray(u.T @ residual)-meta["tau"]*m["token_effect"].T
                    normal_error = max(normal_error, np.max(abs(gu))/max(1., np.max(abs(u.T @ train["y"]))))
                else:
                    assert np.all(m["token_effect"] == 0)
                for g in range(21):
                    sl = slice(g*8, g*8+8)
                    gs = w[:, sl].T @ residual[:, g+1]-len(z)*m["gammas"][g]*m["theta"][g]
                    normal_error = max(normal_error, np.max(abs(gs))/max(1., np.max(abs(w[:, sl].T @ train["y"]))))
                assert normal_error < 1e-8, (stem, normal_error)
                se = (y[:, :, None]-m["prediction"])**2
                sse = se.sum(1)
                pc_gain = (sse[:, 0]-sse[:, 1])/sst
                randadv = (sse[:, 2:].mean(1)-sse[:, 1])/sst
                direct = dict(baseline_r2=1-sse[:, 0]/sst, pc_r2=1-sse[:, 1]/sst,
                              random_r2=1-sse[:, 2:].mean(1)/sst, pc_gain=pc_gain,
                              random_advantage=randadv, gain_change_from_original=pc_gain-orig_delta)
                row = next(r for r in results["records"] if
                           (r["point"], r["condition"], r["family"]) == (point, condition, family))
                for key, value in direct.items():
                    old = np.array(row["article_values"][key])
                    err = float(np.max(abs(old-value)))
                    max_metric_error = max(max_metric_error, err)
                    np.testing.assert_allclose(old, value, atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(row["metrics"][key]["median"], np.median(value), atol=1e-12)
                max_prediction_error = max(max_prediction_error, pred_error)
                max_normal_error = max(max_normal_error, float(normal_error))
                all_checks.append(dict(case=stem, prediction_error=pred_error,
                                       normal_equation_relative_error=float(normal_error)))
                print("Verified", stem, flush=True)
    a.write_json(a.HERE/"numerical_verification.json", dict(status="PASS", checked_model_sets=18,
        reconstructed_predictions=18*64*511*22, checked_original_article_conditions=384,
        original_prediction_max_abs_error=max_original_error,
        new_prediction_max_abs_error=max_prediction_error,
        normal_equation_max_relative_error=max_normal_error,
        article_metric_max_abs_error=max_metric_error, folds_disjoint=True,
        hyperparameters_training_cv_argmin_verified=True, language_model_forwards=0,
        elapsed_seconds=time.monotonic()-start, cases=all_checks,
        reviewer="Same agent, separate saved-artifact numerical reconstruction; not independent review"))
    print("All numerical verification passed", flush=True)


if __name__ == "__main__":
    main()
