"""Finite, CPU-only diagnostic of immutable P2/P3 observations (EDA-0002).

No language-model imports, downloads, or forward passes. Run `fit` before `evaluate`.
All new files use exclusive creation; historical artifacts are read-only.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import platform
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "p3_v1"
P2 = ROOT / "receipts/6966c146e980_p2_v1_final/raw"
P3 = ROOT / "receipts/0113330bdd64_p3_v1_final/raw"
CARD = ROOT / "prereg/EDA-0002_v1_p3_reversal.md"
POINTS = ["block06", "final_ln"]
GRID = np.array([.0001, .001, .01, .1, 1.])
CONTROLS = np.array([0, 1, 2, 3, 7])


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, obj):
    with Path(p).open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save(p, **a):
    with Path(p).open("xb") as f:
        np.savez_compressed(f, **a)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fixtures(train):
    p = ROOT / ("p2_v1/fixtures.json" if train else "p3_v1/fixtures.json")
    a = json.loads(p.read_text())
    return [f for f in a if f["split"] == "train"] if train else a


def center(token, values):
    """Center per ID, returning all rows and a repeated-token mask."""
    unique, inverse, counts = np.unique(token, return_inverse=True, return_counts=True)
    values = np.asarray(values, float)
    one = values.ndim == 1
    if one:
        values = values[:, None]
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    out = values - (sums/counts[:, None])[inverse]
    return (out[:, 0] if one else out), counts[inverse] >= 2, int((counts >= 2).sum())


def fields(raw, point, m):
    x = raw["h_"+point][1:].astype(float)
    t = raw["position"][1:].astype(float)
    p = t/511.
    ids = raw["token_id"][1:]
    counts = load(SOURCE/"frozen_frequency.npz")["counts"]
    b = np.column_stack([p, p*p, p**3, np.log1p(t)/np.log(512),
                         np.log1p(raw["token_chars"][1:]),
                         np.log1p(raw["token_utf8_bytes"][1:]),
                         -np.log((counts[ids]+1)/(counts.sum()+len(counts))),
                         np.log1p(np.linalg.norm(x, axis=1))])
    z = (b-m["feature_mean"])/m["feature_scale"]*m["feature_active"]
    bases = np.concatenate([m["components"], m["random_components"].reshape(-1, 768)])
    scores = (x-m["mean"]) @ bases.T
    w = (scores-m["score_mean"])/m["score_scale"]*m["score_active"]
    return x, ids, z, w, scores


def ordinary_predict(z, w, ids, m):
    z1 = np.column_stack([np.ones(len(z)), z])
    ix = np.searchsorted(m["vocab"], ids)
    known = ix < len(m["vocab"])
    ix = np.minimum(ix, len(m["vocab"])-1)
    known &= m["vocab"][ix] == ids
    pred = z1 @ m["beta"].T + m["token_effect"][:, ix].T*known[:, None]
    for j in range(21):
        pred[:, j+1] += w[:, j*8:j*8+8] @ m["theta"][j]
    return pred


def complement(m):
    q = m["components"]
    result = []
    singular = []
    for r in m["random_components"]:
        x = r - (r @ q.T) @ q
        singular.append(np.linalg.svd(x, compute_uv=False).min())
        result.append(np.linalg.qr(x.T, mode="reduced")[0].T)
    bases = np.stack(result)
    assert min(singular) > 1e-6
    assert np.max(abs(bases @ q.T)) < 1e-10
    assert np.max(abs(bases @ bases.transpose(0, 2, 1)-np.eye(8))) < 1e-10
    return bases, singular


def ordinary_fit(z, w, y, ids, alpha, tau, gammas):
    """Exact elimination of penalized token intercepts from joint ridge."""
    z = np.column_stack([np.ones(len(z)), z])
    x = np.column_stack([z, w])
    vocab, inv, count = np.unique(ids, return_inverse=True, return_counts=True)
    sums = np.zeros((len(vocab), x.shape[1]))
    sy = np.zeros(len(vocab))
    np.add.at(sums, inv, x)
    np.add.at(sy, inv, y)
    di = 1/(count+tau)
    gram = x.T @ x - sums.T @ (sums*di[:, None])
    rhs = x.T @ y - sums.T @ (sy*di)
    bs, es, ts, residuals = [], [], [], []
    for j in range(22):
        idx = np.arange(9) if j == 0 else np.r_[np.arange(9), np.arange(9+(j-1)*8, 9+j*8)]
        penalties = np.r_[0., np.repeat(alpha, 8)]
        if j:
            penalties = np.r_[penalties, np.repeat(gammas[j-1], 8)]
        mat = gram[np.ix_(idx, idx)] + len(y)*np.diag(penalties)
        coef = np.linalg.solve(mat, rhs[idx])
        residuals.append(float(abs(mat @ coef-rhs[idx]).max()/max(1., abs(rhs[idx]).max())))
        bs.append(coef[:9])
        es.append((sy-sums[:, idx] @ coef)*di)
        if j:
            ts.append(coef[9:])
    return dict(beta=np.array(bs), token_effect=np.array(es), theta=np.array(ts),
                vocab=vocab, normal_residual=np.array(residuals), alpha=np.array(alpha),
                tau=np.array(tau), gammas=np.array(gammas))


def stats(z, w, y):
    x = np.column_stack([z, w])
    return x.T @ x, x.T @ y, len(y)


def solve_within(stat, alpha, gammas):
    gram, rhs, n = stat
    q = 5
    bs, ts, residuals = [], [], []
    for j in range(len(gammas)+1):
        idx = np.arange(q) if j == 0 else np.r_[np.arange(q), np.arange(q+(j-1)*8, q+j*8)]
        penalty = np.repeat(alpha, q)
        if j:
            penalty = np.r_[penalty, np.repeat(gammas[j-1], 8)]
        mat = gram[np.ix_(idx, idx)] + n*np.diag(penalty)
        coef = np.linalg.solve(mat, rhs[idx])
        residuals.append(float(abs(mat @ coef-rhs[idx]).max()/max(1., abs(rhs[idx]).max())))
        bs.append(coef[:q])
        if j:
            ts.append(coef[q:])
    return dict(beta=np.array(bs), theta=np.array(ts), alpha=np.array(alpha),
                gammas=np.array(gammas), normal_residual=np.array(residuals))


def within_predict(z, w, model):
    pred = z @ model["beta"].T
    for j in range(len(model["theta"])):
        pred[:, j+1] += w[:, j*8:j*8+8] @ model["theta"][j]
    return pred


def cv_fit(z, w, y, folds):
    fold_data = []
    for fold in range(4):
        tr, va = folds != fold, folds == fold
        fold_data.append((stats(z[tr], w[tr], y[tr]), z[va], w[va], y[va]))
    aloss = []
    for alpha in GRID:
        sse = 0.
        for s, zv, wv, yv in fold_data:
            model = solve_within(s, alpha, [])
            sse += np.square(yv-zv @ model["beta"][0]).sum()
        aloss.append(float(sse/len(y)))
    alpha = GRID[np.argmin(aloss)]
    groups = w.shape[1]//8
    gloss = []
    for gamma in GRID:
        sse = np.zeros(groups)
        for s, zv, wv, yv in fold_data:
            model = solve_within(s, alpha, np.repeat(gamma, groups))
            pred = within_predict(zv, wv, model)
            sse += np.square(pred[:, 1:]-yv[:, None]).sum(0)
        gloss.append(sse/len(y))
    gloss = np.array(gloss)
    gammas = GRID[np.argmin(gloss, axis=0)]
    model = solve_within(stats(z, w, y), alpha, gammas)
    return model, dict(alpha=float(alpha), alpha_loss=aloss, gammas=gammas.tolist(),
                       gamma_loss=gloss.tolist(), grid=GRID.tolist(),
                       fold_tokens=[len(f[-1]) for f in fold_data])


def fit():
    started = time.monotonic()
    write(HERE/"execution_start.json", dict(started_at=stamp(), python=platform.python_version(),
        numpy=np.__version__, card_sha256=digest(CARD), code_sha256=digest(__file__),
        new_language_model_forwards=0, evaluation_is_previously_seen=True))
    (HERE/"fits").mkdir(exist_ok=False)
    fs = fixtures(True)
    assert len(fs) == 64
    ordered = sorted([f["id"] for f in fs],
                     key=lambda s: hashlib.sha256(("eda-p3-reversal-v1:"+s).encode()).hexdigest())
    fold_map = {name: i % 4 for i, name in enumerate(ordered)}
    write(HERE/"splits.json", dict(train=[f["id"] for f in fs], cv_fold=fold_map,
                                 evaluation=[f["id"] for f in fixtures(False)]))
    assert not (set(fold_map) & {f["id"] for f in fixtures(False)})
    hyper = json.loads((SOURCE/"frozen_manifest.json").read_text())["cases"]
    cvr, coverage = {}, []
    for point in POINTS:
        m = load(SOURCE/f"frozen_{point}_C1.npz")
        bases, singular = complement(m)
        data = {k: [] for k in ["z", "w", "y", "ids", "zc", "wc", "yc", "fold", "comp", "comp_all"]}
        for f in fs:
            raw = load(P2/"docs"/f["id"]/"raw.npz")
            x, ids, z, w, _ = fields(raw, point, m)
            y = raw["surprisal"][1:]
            combined, mask, groups = center(ids, np.column_stack([y, z, w]))
            assert groups >= 8 and mask.sum() >= 32
            assert abs(combined[:, [5, 6, 7]]).max() < 1e-11
            comp = (x-m["mean"]) @ bases.reshape(-1, 768).T
            cc, cmask, _ = center(ids, comp)
            assert np.array_equal(mask, cmask)
            for key, val in [
                ("z", z[mask]), ("w", w[mask]), ("y", y[mask]), ("ids", ids[mask]),
                ("zc", combined[mask, 1:9][:, CONTROLS]), ("wc", combined[mask, 9:]),
                ("yc", combined[mask, 0]), ("fold", np.repeat(fold_map[f["id"]], mask.sum())),
                ("comp", cc[mask]), ("comp_all", comp)]:
                data[key].append(val)
            if point == POINTS[0]:
                coverage.append(dict(document_id=f["id"], tokens=int(mask.sum()), groups=groups))
        d = {k: np.concatenate(v) for k, v in data.items()}
        hp = hyper[point]
        repeated = ordinary_fit(d["z"], d["w"], d["y"], d["ids"],
                                hp["alpha"], hp["tau"], hp["gamma"])
        save(HERE/"fits"/f"{point}__R.npz", **repeated)
        fixed = solve_within(stats(d["zc"], d["wc"], d["yc"]), hp["alpha"], hp["gamma"])
        save(HERE/"fits"/f"{point}__W_fixed.npz", **fixed)
        tuned, cv = cv_fit(d["zc"], d["wc"], d["yc"], d["fold"])
        save(HERE/"fits"/f"{point}__W_CV.npz", **tuned)
        cm = d["comp_all"].mean(0)
        cs = d["comp_all"].std(0)
        assert cs.min() > 1e-12
        comp_model, comp_cv = cv_fit(d["zc"], d["comp"]/cs, d["yc"], d["fold"])
        # The baseline search is identical; results must coincide.
        assert comp_cv["alpha"] == cv["alpha"]
        save(HERE/"fits"/f"{point}__complement.npz", **comp_model, components=bases,
             score_mean=cm, score_scale=cs, residual_min_singular=np.array(singular))
        cvr[point] = dict(within=cv, complement=comp_cv)
        print(f"Fitted {point}: {len(d['yc'])} within tokens, alpha={cv['alpha']}", flush=True)
    write(HERE/"cv_results.json", cvr)
    write(HERE/"train_coverage.json", coverage)
    files = [*sorted((HERE/"fits").glob("*.npz")), HERE/"cv_results.json", HERE/"splits.json",
             HERE/"train_coverage.json"]
    write(HERE/"FIT_COMMIT.json", dict(sealed_at=stamp(),
        files={str(p.relative_to(HERE)): digest(p) for p in files},
        card_sha256=digest(CARD), code_sha256=digest(__file__),
        new_language_model_forwards=0, seconds=time.monotonic()-started))


def describe(v):
    a = np.array(v, float)
    assert len(a) == 64 and np.isfinite(a).all()
    rng = np.random.Generator(np.random.PCG64(20261027))
    meds = np.median(a[rng.integers(64, size=(10000, 64))], axis=1)
    lo, hi = np.quantile(meds, [.025, .975])
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                q1=float(np.quantile(a, .25)), q3=float(np.quantile(a, .75)),
                minimum=float(a.min()), maximum=float(a.max()),
                positive=int((a > 0).sum()), negative=int((a < 0).sum()),
                lower=float(lo), upper=float(hi))


def metrics(y, pred):
    sst = y @ y
    sse = np.square(pred-y[:, None]).sum(0)
    cov = y @ pred/sst
    var = np.square(pred).sum(0)/sst
    r2 = 1-sse/sst
    assert np.max(abs(r2-(2*cov-var))) < 1e-10
    ratio = np.sqrt(var)
    corr = cov/ratio
    return dict(sst=float(sst), sse=sse.tolist(), r2=r2.tolist(),
                corr=corr.tolist(), sd_ratio=ratio.tolist(),
                covariance_term=(2*cov).tolist(), variance_term=var.tolist(),
                gain=float(r2[1]-r2[0]),
                advantage=float(r2[1]-r2[2:].mean()))


def variance(ids, x, scores, mask):
    xx = x[mask]
    ss = scores[mask]
    xc, _, _ = center(ids[mask], xx)
    sc, _, _ = center(ids[mask], ss)
    total = np.square(xx-xx.mean(0)).sum()
    within = np.square(xc).sum()
    between = np.square(xx-xc-xx.mean(0)).sum()
    st = np.square(ss-ss.mean(0)).sum(0).reshape(21, 8).sum(1)
    sw = np.square(sc).sum(0).reshape(21, 8).sum(1)
    sb = np.square(ss-sc-ss.mean(0)).sum(0).reshape(21, 8).sum(1)
    error = max(abs(total-within-between)/total, np.max(abs(st-sw-sb)/st))
    assert error < 1e-10 and total > within > 0 and between > 0
    return dict(total_ss=float(total), within_ss=float(within), between_ss=float(between),
        full_within_fraction=float(within/total), top8_within_fraction=float(sw[0]/st[0]),
        top8_total_capture=float(st[0]/total), top8_within_capture=float(sw[0]/within),
        top8_between_capture=float(sb[0]/between), random_total_capture=float((st/total)[1:].mean()),
        random_within_capture=float((sw/within)[1:].mean()), decomposition_error=float(error),
        all_projected_total_ss=st.tolist(), all_projected_within_ss=sw.tolist(),
        all_projected_between_ss=sb.tolist())


def evaluate():
    started = time.monotonic()
    commit = json.loads((HERE/"FIT_COMMIT.json").read_text())
    assert commit["card_sha256"] == digest(CARD) and commit["code_sha256"] == digest(__file__)
    for name, sha in commit["files"].items():
        assert digest(HERE/name) == sha
    write(HERE/"evaluation_start.json", dict(started_at=stamp(), fit_commit_sha256=digest(HERE/"FIT_COMMIT.json")))
    (HERE/"predictions").mkdir(exist_ok=False)
    fs = fixtures(False)
    rows, coverage = [], []
    max_old_error = 0.
    for point in POINTS:
        m = load(SOURCE/f"frozen_{point}_C1.npz")
        fits = {kind: load(HERE/"fits"/f"{point}__{kind}.npz")
                for kind in ["R", "W_fixed", "W_CV", "complement"]}
        predictions = {kind: [] for kind in ["O", *fits]}
        for f in fs:
            raw = load(P3/"docs"/f["id"]/"raw.npz")
            x, ids, z, w, scores = fields(raw, point, m)
            y = raw["surprisal"][1:]
            old = ordinary_predict(z, w, ids, m)
            stored = load(P3/"evaluation"/f["id"]/(point+"__C1")/"arrays.npz")
            err = float(abs(old.T-stored["predictions"]).max())
            max_old_error = max(max_old_error, err)
            assert err < 1e-9
            pred = dict(O=old, R=ordinary_predict(z, w, ids, fits["R"]))
            for kind in ["W_fixed", "W_CV"]:
                pred[kind] = within_predict(z[:, CONTROLS], w, fits[kind])
            cm = fits["complement"]
            wc = (((x-m["mean"]) @ cm["components"].reshape(-1, 768).T)-cm["score_mean"])/cm["score_scale"]
            pred["complement"] = within_predict(z[:, CONTROLS], wc, cm)
            yc, mask, groups = center(ids, y)
            assert mask.sum() >= 32 and groups >= 8 and yc[mask] @ yc[mask] > 0
            row = dict(document_id=f["id"], title=f["title"], point=point,
                       tokens=int(mask.sum()), groups=groups, models={},
                       geometry=variance(ids, x, scores, mask))
            for kind, pr in pred.items():
                pc, pmask, _ = center(ids, pr)
                assert np.array_equal(mask, pmask)
                met = metrics(yc[mask], pc[mask])
                if kind == "complement":
                    met["mean_gain"] = float(np.mean(met["r2"][1:])-met["r2"][0])
                    # These 20 models have no designated "PC"; omit misleading pair metrics.
                    del met["gain"], met["advantage"]
                row["models"][kind] = met
                predictions[kind].append(pr)
            rows.append(row)
            if point == POINTS[0]:
                coverage.append(dict(document_id=f["id"], tokens=int(mask.sum()), groups=groups))
        save(HERE/"predictions"/f"{point}.npz", **{k: np.stack(v) for k, v in predictions.items()})
        print(f"Evaluated {point}: all 64 articles", flush=True)
    write(HERE/"article_results.json", rows)
    write(HERE/"evaluation_coverage.json", coverage)
    result = {}
    for point in POINTS:
        rs = [r for r in rows if r["point"] == point]
        point_result = dict(models={}, geometry={}, paired={})
        for kind in ["O", "R", "W_fixed", "W_CV"]:
            mm = [r["models"][kind] for r in rs]
            d = {key: describe([m[key] for m in mm]) for key in ["gain", "advantage"]}
            for key in ["r2", "corr", "sd_ratio", "covariance_term", "variance_term"]:
                d[key] = dict(base=describe([m[key][0] for m in mm]),
                              pc=describe([m[key][1] for m in mm]),
                              random_mean=describe([np.mean(m[key][2:]) for m in mm]),
                              all_models=[describe([m[key][j] for m in mm]) for j in range(22)])
            point_result["models"][kind] = d
        mm = [r["models"]["complement"] for r in rs]
        point_result["models"]["complement"] = dict(
            mean_gain=describe([m["mean_gain"] for m in mm]),
            mean_r2=describe([np.mean(m["r2"][1:]) for m in mm]),
            all_gains=[describe([m["r2"][j]-m["r2"][0] for m in mm]) for j in range(1, 21)])
        for key in ["full_within_fraction", "top8_within_fraction", "top8_total_capture",
                    "top8_within_capture", "top8_between_capture", "random_total_capture",
                    "random_within_capture"]:
            point_result["geometry"][key] = describe([r["geometry"][key] for r in rs])
        for left, right in [("O", "R"), ("R", "W_fixed"), ("W_fixed", "W_CV"), ("O", "W_CV")]:
            point_result["paired"][f"{right}_minus_{left}"] = {
                key: describe([r["models"][right][key]-r["models"][left][key] for r in rs])
                for key in ["gain", "advantage"]}
        result[point] = point_result
    write(HERE/"results.json", dict(points=result, exploratory=True, bootstrap_seed=20261027,
          bootstrap_repetitions=10000, descriptive_interval=.95,
          original_prediction_max_difference=max_old_error, new_language_model_forwards=0,
          completed_at=stamp(), seconds=time.monotonic()-started))
    for p in POINTS:
        print(p, {k: (v.get("advantage") or v.get("mean_gain"))["median"]
                  for k, v in result[p]["models"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.mode == "fit" else evaluate()
