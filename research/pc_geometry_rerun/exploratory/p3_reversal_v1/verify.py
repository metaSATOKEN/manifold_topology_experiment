"""Re-read raw observations; audit saved fits/predictions using alternate algebra."""
import json
import time
import numpy as np
import analyze as a


def grouped(token, v):
    c = np.array(v, float, copy=True)
    mask = np.zeros(len(token), bool)
    for t in np.unique(token):
        ix = token == t
        c[ix] -= c[ix].mean(0)
        mask[ix] = ix.sum() >= 2
    return c, mask


def collapsed(raw, point, original, fitted, kind):
    """Expand standardized coefficients into weights on unstandardized controls/state."""
    x, ids, z, w, _ = a.fields(raw, point, original)
    # Reconstruct raw numeric variables; this is inverse of the recorded standardization.
    b = z*original["feature_scale"] + original["feature_mean"]
    if kind in ["O", "R"]:
        beta = fitted["beta"][:, 1:]
        intercept = fitted["beta"][:, 0].copy()
        lookup = {int(t): i for i, t in enumerate(fitted["vocab"])}
        effect = np.array([fitted["token_effect"][:, lookup[int(t)]] if int(t) in lookup
                           else np.zeros(22) for t in ids])
    else:
        beta = np.zeros((len(fitted["beta"]), 8))
        beta[:, a.CONTROLS] = fitted["beta"]
        intercept = np.zeros(len(beta))
        effect = np.zeros((len(x), len(beta)))
    slopes = beta*original["feature_active"]/original["feature_scale"]
    intercept -= slopes @ original["feature_mean"]
    pred = b @ slopes.T + intercept + effect
    if kind == "complement":
        bases = fitted["components"]
        scale, mean = fitted["score_scale"], fitted["score_mean"]
        active = np.ones_like(scale)
    else:
        bases = np.concatenate([original["components"][None], original["random_components"]])
        scale, mean, active = (original[k] for k in ["score_scale", "score_mean", "score_active"])
    weights = fitted["theta"]*(active/scale).reshape(-1, 8)
    state = np.einsum("gk,gkd->gd", weights, bases)
    offset = -state @ original["mean"] - (weights*mean.reshape(-1, 8)).sum(1)
    pred[:, 1:] += x @ state.T + offset
    return pred


start = time.monotonic()
commit = json.loads((a.HERE/"FIT_COMMIT.json").read_text())
evaluated = json.loads((a.HERE/"evaluation_start.json").read_text())
assert commit["sealed_at"] < evaluated["started_at"]
for path, sha in commit["files"].items():
    assert a.digest(a.HERE/path) == sha
assert commit["card_sha256"] == a.digest(a.CARD)
results = json.loads((a.HERE/"results.json").read_text())
rows = json.loads((a.HERE/"article_results.json").read_text())
errors = dict(predictions=0., metric=0., geometry_identity=0., fitted_qr=0.,
              ordinary_gradient=0., bootstrap=0., original_endpoints=0.)
counts = dict(predictions=0, article_points=0, direct_qr_fits=0, id_gradient_fits=0)
splits = json.loads((a.HERE/"splits.json").read_text())
assert set(splits["train"]).isdisjoint(splits["evaluation"])
assert len(splits["train"]) == len(splits["evaluation"]) == 64

for point in a.POINTS:
    original = a.load(a.SOURCE/f"frozen_{point}_C1.npz")
    fitted = {kind: a.load(a.HERE/"fits"/f"{point}__{kind}.npz")
              for kind in ["R", "W_fixed", "W_CV", "complement"]}
    d = {k: [] for k in ["z", "w", "y", "id", "zc", "wc", "yc", "cc"]}
    cm = fitted["complement"]
    for f in a.fixtures(True):
        raw = a.load(a.P2/"docs"/f["id"]/"raw.npz")
        x, ids, z, w, _ = a.fields(raw, point, original)
        y = raw["surprisal"][1:]
        combined, mask = grouped(ids, np.column_stack([y, z, w]))
        comp = ((x-original["mean"]) @ cm["components"].reshape(-1, 768).T-cm["score_mean"])/cm["score_scale"]
        cc, mask2 = grouped(ids, comp)
        assert np.array_equal(mask, mask2)
        for k, v in [("z", z[mask]), ("w", w[mask]), ("y", y[mask]), ("id", ids[mask]),
                     ("zc", combined[mask, 1:9][:, a.CONTROLS]),
                     ("wc", combined[mask, 9:]), ("yc", combined[mask, 0]), ("cc", cc[mask])]:
            d[k].append(v)
    d = {k: np.concatenate(v) for k, v in d.items()}
    n = len(d["y"])
    r = fitted["R"]
    pred = a.ordinary_predict(d["z"], d["w"], d["id"], r)
    ix = np.searchsorted(r["vocab"], d["id"])
    z1 = np.column_stack([np.ones(n), d["z"]])
    for j in range(22):
        residual = pred[:, j]-d["y"]
        gradient = z1.T @ residual + n*float(r["alpha"])*r["beta"][j]*np.r_[0., np.ones(8)]
        id_gradient = np.bincount(ix, weights=residual, minlength=len(r["vocab"])) + float(r["tau"])*r["token_effect"][j]
        maxima = [abs(gradient).max(), abs(id_gradient).max()]
        if j:
            extra = d["w"][:, (j-1)*8:j*8]
            maxima.append(abs(extra.T @ residual+n*r["gammas"][j-1]*r["theta"][j-1]).max())
        relative = max(maxima)/max(1., abs(z1.T @ d["y"]).max())
        errors["ordinary_gradient"] = max(errors["ordinary_gradient"], float(relative))
        assert relative < 1e-8
        counts["id_gradient_fits"] += 1
    for kind in ["W_fixed", "W_CV", "complement"]:
        m = fitted[kind]
        extra = d["cc"] if kind == "complement" else d["wc"]
        for j in range(len(m["beta"])):
            design = d["zc"] if j == 0 else np.column_stack([d["zc"], extra[:, (j-1)*8:j*8]])
            penalty = np.r_[np.repeat(n*float(m["alpha"]), 5),
                            np.repeat(n*m["gammas"][j-1], 8) if j else []]
            ref = np.linalg.lstsq(np.vstack([design, np.diag(np.sqrt(penalty))]),
                                 np.r_[d["yc"], np.zeros(len(penalty))], rcond=None)[0]
            coef = np.r_[m["beta"][j], m["theta"][j-1] if j else []]
            diff = float(abs(ref-coef).max())
            errors["fitted_qr"] = max(errors["fitted_qr"], diff)
            assert diff < 1e-8
            counts["direct_qr_fits"] += 1
    saved = a.load(a.HERE/"predictions"/f"{point}.npz")
    for i, f in enumerate(a.fixtures(False)):
        raw = a.load(a.P3/"docs"/f["id"]/"raw.npz")
        ids = raw["token_id"][1:]
        yc, mask = grouped(ids, raw["surprisal"][1:])
        row = next(r for r in rows if r["point"] == point and r["document_id"] == f["id"])
        for kind in ["O", "R", "W_fixed", "W_CV", "complement"]:
            m = original if kind == "O" else fitted[kind]
            pred = collapsed(raw, point, original, m, kind)
            diff = float(abs(pred-saved[kind][i]).max())
            errors["predictions"] = max(errors["predictions"], diff)
            assert diff < 1e-8
            counts["predictions"] += pred.size
            pc, _ = grouped(ids, pred)
            sse = np.einsum("ij,ij->j", pc[mask]-yc[mask, None], pc[mask]-yc[mask, None])
            sst = yc[mask] @ yc[mask]
            r2 = 1-sse/sst
            for key, v in [("r2", r2), ("sse", sse), ("sst", sst)]:
                old = row["models"][kind][key]
                diff = float(np.max(abs(np.array(old)-v)))
                errors["metric"] = max(errors["metric"], diff)
                np.testing.assert_allclose(old, v, atol=1e-8, rtol=1e-9)
            ratio = np.linalg.norm(pc[mask], axis=0)/np.linalg.norm(yc[mask])
            cov = yc[mask] @ pc[mask]/sst
            corr = np.array([np.corrcoef(yc[mask], pc[mask, j])[0, 1]
                             for j in range(pc.shape[1])])
            for key, value in [("sd_ratio", ratio), ("corr", corr),
                               ("covariance_term", 2*cov), ("variance_term", ratio**2)]:
                np.testing.assert_allclose(row["models"][kind][key], value, atol=1e-9, rtol=1e-9)
        x = raw["h_"+point][1:].astype(float)[mask]
        token = ids[mask]
        total = np.square(x-x.mean(0)).sum()
        within = between = 0.
        for t in np.unique(token):
            group = x[token == t]
            within += np.square(group-group.mean(0)).sum()
            between += len(group)*np.square(group.mean(0)-x.mean(0)).sum()
        relative = abs(total-within-between)/total
        errors["geometry_identity"] = max(errors["geometry_identity"], float(relative))
        for key, value in [("total_ss", total), ("within_ss", within), ("between_ss", between)]:
            np.testing.assert_allclose(value, row["geometry"][key], rtol=1e-10, atol=1e-8)
        counts["article_points"] += 1
    # Check the main endpoint bootstraps with separate sorted-draw median arithmetic.
    rs = [r for r in rows if r["point"] == point]
    for kind in ["O", "R", "W_fixed", "W_CV"]:
        for key in ["gain", "advantage"]:
            vals = np.array([r["models"][kind][key] for r in rs])
            rng = np.random.Generator(np.random.PCG64(20261027))
            draw = np.sort(vals[rng.integers(64, size=(10000, 64))], axis=1)
            limits = np.quantile((draw[:, 31]+draw[:, 32])/2, [.025, .975])
            target = results["points"][point]["models"][kind][key]
            diff = float(abs(limits-[target["lower"], target["upper"]]).max())
            errors["bootstrap"] = max(errors["bootstrap"], diff)
            assert diff < 1e-10
    print("Verified", point, flush=True)

reference = json.loads((a.ROOT/"evidence/p3_final_checks_2026-10-20.json").read_text())
for point in a.POINTS:
    old = reference["primary_endpoints"] if point == "block06" else reference["secondary_descriptive"]["final_ln"]
    for k, label in [("gain", "C"), ("advantage", "D")]:
        diff = abs(results["points"][point]["models"]["O"][k]["median"]-old[label]["median"])
        errors["original_endpoints"] = max(errors["original_endpoints"], diff)
        assert diff < 1e-10
a.write(a.HERE/"numerical_verification.json", dict(status="PASS", checked_at=a.stamp(),
        seconds=time.monotonic()-start, counts=counts, max_differences=errors,
        fit_precedes_new_evaluation=True, training_and_evaluation_disjoint=True,
        same_agent_self_audit=True, independent_reviewer=False, new_language_model_forwards=0))
print("PASS", counts, errors)
