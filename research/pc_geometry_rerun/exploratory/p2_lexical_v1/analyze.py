"""Exploratory CPU-only regression on immutable P2 observations.

No model backend, tokenizer, network access, or mutation of previous results.
`fit` writes all fitted artifacts before `summarize` accesses evaluation errors.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import itertools
import json
import platform
import time
import unicodedata

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "receipts/6966c146e980_p2_v1_final/raw"
CARD = ROOT / "prereg/EDA-0001_v1_p2_lexical.md"
ALPHAS = [0.0001, 0.001, 0.01, 0.1]
TAUS = [1., 10., 100.]
GAMMAS = [0.0001, 0.001, 0.01, 0.1, 1.]
FAMILIES = ["ID", "NL", "NL_ID"]
CONDITIONS = ["C0", "C1", "C4"]
POINTS = ["block06", "final_ln"]
VOCAB = 50257


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(p, obj):
    with Path(p).open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_npz(p, **arrays):
    with Path(p).open("xb") as f:
        np.savez_compressed(f, **arrays)


def shapes(s):
    t = s.strip()
    return [
        bool(s) and s[0].isspace(), "\n" in s,
        any(c.isalpha() for c in s), any(c.isdigit() for c in s),
        any(unicodedata.category(c)[0] in "PS" for c in s),
        bool(s) and s.isspace(), bool(t) and t.isalpha(),
        bool(t) and t.isdigit(), bool(t) and t.isupper(),
        bool(t) and t[0].isupper(), "\ufffd" in s,
        any(ord(c) > 127 for c in s),
    ]


def read_data(split):
    fixtures = json.loads((ROOT / "p2_v1/fixtures.json").read_text())
    ids = [d["id"] for d in fixtures if d["split"] == split]
    result = {k: [] for k in ["id", "position", "chars", "bytes", "y", "shape"]}
    h = {p: [] for p in POINTS}
    all_ids = []
    for name in ids:
        with np.load(RAW / "docs" / name / "raw.npz", allow_pickle=False) as r:
            all_ids.append(r["token_id"].copy())
            for a, b in [("id", "token_id"), ("position", "position"),
                         ("chars", "token_chars"), ("bytes", "token_utf8_bytes"),
                         ("y", "surprisal")]:
                result[a].append(r[b][1:].copy())
            for p in POINTS:
                h[p].append(r["h_" + p].copy())
        meta = json.loads((RAW / "docs" / name / "doc.json").read_text())
        assert meta["split"] == split
        result["shape"].append(np.array([shapes(s) for s in meta["decoded_tokens"][1:]], float))
    result = {k: np.concatenate(v) for k, v in result.items()}
    result["h"] = {p: np.stack(v) for p, v in h.items()}
    result["all_ids"] = np.stack(all_ids)
    result["documents"] = ids
    result["article"] = np.repeat(np.arange(len(ids)), 511)
    assert len(ids) == 64 and np.isfinite(result["y"]).all()
    return result


def controls(data, point, counts):
    t = data["position"].astype(float)
    p = t / 511
    norm = np.linalg.norm(data["h"][point][:, 1:, :].astype(float), axis=2).ravel()
    rarity = -np.log((counts[data["id"]] + 1) / (counts.sum() + VOCAB))
    return np.column_stack([p, p*p, p**3, np.log1p(t)/np.log(512),
                            np.log1p(data["chars"]), np.log1p(data["bytes"]),
                            rarity, np.log1p(norm)])


class Features:
    def __init__(self, btrain, family):
        self.family = family
        x = btrain[:, [0, 4, 5, 6, 7]]
        self.mu = x.mean(0)
        self.sd = x.std(0)
        self.sd = np.where(self.sd > 1e-12, self.sd, 1.)
        self.knots = np.quantile((x-self.mu)/self.sd, [.25, .5, .75], axis=0).T

    def transform(self, b, shape):
        if self.family == "ID":
            return b.copy()
        x = (b[:, [0, 4, 5, 6, 7]] - self.mu) / self.sd
        basis = [np.column_stack([x[:, j], np.maximum(x[:, j, None]-self.knots[j], 0)])
                 for j in range(5)]
        pairs = [(basis[j][:, :, None]*basis[k][:, None, :]).reshape(len(x), 16)
                 for j, k in itertools.combinations(range(5), 2)]
        return np.column_stack([b, *basis, *pairs, shape])

    def metadata(self):
        return dict(family=self.family, spline_mean=self.mu.tolist(),
                    spline_scale=self.sd.tolist(), knots=self.knots.tolist())


class Context:
    """Joint ridge with an unpenalized intercept and optional token random intercepts.

The token block is eliminated exactly via its diagonal normal matrix.
All matrices include an intercept; no centering shortcut for token effects.
"""
    def __init__(self, x, y, ids, xv, idsv, use_id):
        self.mu = x.mean(0)
        sd = x.std(0)
        self.active = sd > 1e-12
        self.sd = np.where(self.active, sd, 1.)
        self.z = np.column_stack([np.ones(len(x)), (x-self.mu)/self.sd*self.active])
        self.zv = np.column_stack([np.ones(len(xv)), (xv-self.mu)/self.sd*self.active])
        self.y, self.ids, self.idsv = y, ids, idsv
        self.use_id = use_id
        self.n = len(y)
        self.vocab = np.unique(ids)
        self.ix = np.searchsorted(self.vocab, ids)
        self.ixv = np.searchsorted(self.vocab, idsv)
        self.knownv = self.ixv < len(self.vocab)
        self.ixv = np.minimum(self.ixv, len(self.vocab)-1)
        self.knownv &= self.vocab[self.ixv] == idsv
        self.u = csr_matrix((np.ones(self.n), (np.arange(self.n), self.ix)),
                            shape=(self.n, len(self.vocab)))
        self.counts = np.bincount(self.ix, minlength=len(self.vocab))
        self.t = np.asarray(self.u.T @ self.z)
        self.uy = np.asarray(self.u.T @ y)
        self.zz = self.z.T @ self.z
        self.zy = self.z.T @ y

    def fit_base(self, alpha, tau):
        self.alpha, self.tau = alpha, tau
        self.inv_d = 1/(self.counts+tau) if self.use_id else np.zeros(len(self.vocab))
        self.a = self.zz - self.t.T @ (self.t*self.inv_d[:, None])
        self.a.flat[::len(self.a)+1] += self.n*alpha
        self.a[0, 0] -= self.n*alpha
        b = self.zy - self.t.T @ (self.uy*self.inv_d)
        self.factor = cho_factor(self.a, lower=True)
        self.beta = cho_solve(self.factor, b)
        self.effects = (self.uy - self.t @ self.beta)*self.inv_d
        self.base_prediction = self.zv @ self.beta
        self.base_prediction += self.effects[self.ixv]*self.knownv
        self.normal_error = float(np.max(abs(self.a @ self.beta-b)) /
                                  max(1., np.max(abs(b))))
        return self.base_prediction

    def prepare_extra(self, s, sv):
        # s/ sv: rows x (PCA 8 + 20 random groups of 8)
        self.sm = s.mean(0)
        sd = s.std(0)
        self.sa = sd > 1e-12
        self.ss = np.where(self.sa, sd, 1.)
        w = (s-self.sm)/self.ss*self.sa
        wv = (sv-self.sm)/self.ss*self.sa
        self.us = np.asarray(self.u.T @ w)
        c = self.z.T @ w - self.t.T @ (self.us*self.inv_d[:, None])
        d = w.T @ self.y - self.us.T @ (self.uy*self.inv_d)
        self.k = cho_solve(self.factor, c)
        self.hg = (w.T @ w - self.us.T @ (self.us*self.inv_d[:, None]) -
                   c.T @ self.k)
        self.hg = (self.hg+self.hg.T)/2
        self.dg = d-c.T @ self.beta
        self.u_adjust = (self.us-self.t @ self.k)*self.inv_d[:, None]
        self.rv = (wv-self.zv @ self.k -
                   self.u_adjust[self.ixv]*self.knownv[:, None])

    def extra_prediction(self, gammas):
        pred = np.empty((len(self.zv), 22))
        pred[:, 0] = self.base_prediction
        betas, effects = [self.beta], [self.effects]
        theta = []
        residuals = [self.normal_error]
        for group, gamma in enumerate(gammas):
            sl = slice(group*8, (group+1)*8)
            mat = self.hg[sl, sl] + self.n*gamma*np.eye(8)
            coef = np.linalg.solve(mat, self.dg[sl])
            pred[:, group+1] = self.base_prediction + self.rv[:, sl] @ coef
            betas.append(self.beta-self.k[:, sl] @ coef)
            effects.append(self.effects-self.u_adjust[:, sl] @ coef)
            theta.append(coef)
            residuals.append(float(np.max(abs(mat @ coef-self.dg[sl])) /
                                   max(1., np.max(abs(self.dg[sl])))))
        return pred, dict(beta=np.stack(betas), token_effect=np.stack(effects),
                          theta=np.stack(theta), normal_residual=np.array(residuals),
                          feature_mean=self.mu, feature_scale=self.sd,
                          feature_active=self.active, vocab=self.vocab,
                          score_mean=self.sm, score_scale=self.ss, score_active=self.sa)


def scores(data, point, condition):
    with np.load(RAW/"fit"/f"{point}__{condition}"/"arrays.npz", allow_pickle=False) as f:
        bases = np.concatenate([f["components"], f["random_components"].reshape(-1, 768)])
        x = data["h"][point][:, 1:, :].reshape(-1, 768).astype(float)
        if condition == "C4":
            x /= np.linalg.norm(x, axis=1)[:, None]
        return (x-f["mean"]) @ bases.T


def original_predictions(b, s, point, condition):
    with np.load(RAW/"fit"/f"{point}__{condition}"/"arrays.npz", allow_pickle=False) as f:
        preds = []
        for m in range(22):
            score = np.zeros((len(b), 8)) if m == 0 else s[:, (m-1)*8:m*8]
            x = np.column_stack([b, score])
            preds.append(((x-f["surprisal__x_mean"][m])/f["surprisal__x_scale"][m] *
                          f["surprisal__active"][m]) @ f["surprisal__coef"][m] +
                         f["surprisal__y_mean"][m])
    return np.column_stack(preds)


def fit_all():
    start = time.monotonic()
    out = HERE/"fits"
    out.mkdir(exist_ok=False)
    freeze = dict(created_at=stamp(), card_sha256=digest(CARD),
                  code_sha256=digest(__file__), alphas=ALPHAS, taus=TAUS, gammas=GAMMAS,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  mode="Exploratory saved-data CPU regression; no language-model forwards",
                  test_used_for_tuning=False)
    write_json(HERE/"execution_start.json", freeze)
    train, test = read_data("train"), read_data("test")
    order = sorted(range(64), key=lambda j: hashlib.sha256(
        ("eda-p2-lexical-v1:"+train["documents"][j]).encode()).hexdigest())
    folds = np.empty(64, int)
    for rank, j in enumerate(order):
        folds[j] = rank % 4
    write_json(HERE/"splits.json", dict(train=train["documents"], test=test["documents"],
        folds={train["documents"][j]: int(folds[j]) for j in range(64)}))
    full_counts = np.bincount(train["all_ids"].ravel(), minlength=VOCAB)
    save_npz(HERE/"evaluation_targets.npz", y=test["y"].reshape(64, 511),
             token_id=test["id"].reshape(64, 511),
             train_frequency=full_counts[test["id"]].reshape(64, 511))
    cv_records = []
    for point in POINTS:
        print("Building controls", point, flush=True)
        full_bt, full_bv = controls(train, point, full_counts), controls(test, point, full_counts)
        fold_inputs = []
        for k in range(4):
            articles = folds != k
            mask = articles[train["article"]]
            counts = np.bincount(train["all_ids"][articles].ravel(), minlength=VOCAB)
            b = controls(train, point, counts)
            fold_inputs.append((mask, b[mask], b[~mask]))
        for family in FAMILIES:
            contexts = []
            for mask, bt, bv in fold_inputs:
                feat = Features(bt, family)
                contexts.append(Context(feat.transform(bt, train["shape"][mask]),
                    train["y"][mask], train["id"][mask],
                    feat.transform(bv, train["shape"][~mask]), train["id"][~mask],
                    use_id=family != "NL"))
            candidates = []
            for alpha in ALPHAS:
                for tau in (TAUS if family != "NL" else [0.]):
                    errors = []
                    for ctx, (mask, _, _) in zip(contexts, fold_inputs):
                        pred = ctx.fit_base(alpha, tau)
                        errors.append(float(np.mean((train["y"][~mask]-pred)**2)))
                    candidates.append(dict(alpha=alpha, tau=tau, fold_mse=errors,
                                           mean_mse=float(np.mean(errors))))
            chosen = min(candidates, key=lambda c: c["mean_mse"])
            for ctx in contexts:
                ctx.fit_base(chosen["alpha"], chosen["tau"])
            feat = Features(full_bt, family)
            full = Context(feat.transform(full_bt, train["shape"]), train["y"], train["id"],
                           feat.transform(full_bv, test["shape"]), test["id"],
                           use_id=family != "NL")
            full.fit_base(chosen["alpha"], chosen["tau"])
            record = dict(point=point, family=family, baseline_candidates=candidates,
                          selected_baseline=chosen, feature_count=len(full.mu), conditions={})
            print("Selected training-only baseline", point, family,
                  chosen["alpha"], chosen["tau"], "CV MSE", chosen["mean_mse"], flush=True)
            for condition in CONDITIONS:
                st, sv = scores(train, point, condition), scores(test, point, condition)
                errors = np.empty((4, len(GAMMAS), 21))
                for fold, (ctx, (mask, _, _)) in enumerate(zip(contexts, fold_inputs)):
                    ctx.prepare_extra(st[mask], st[~mask])
                    for j, gamma in enumerate(GAMMAS):
                        pred, _ = ctx.extra_prediction([gamma]*21)
                        errors[fold, j] = np.mean((train["y"][~mask, None]-pred[:, 1:])**2, axis=0)
                mean_errors = errors.mean(0)
                indices = np.argmin(mean_errors, axis=0)
                gammas = np.array(GAMMAS)[indices]
                full.prepare_extra(st, sv)
                pred, arrays = full.extra_prediction(gammas)
                assert np.isfinite(pred).all()
                assert arrays["normal_residual"].max() < 1e-8
                stem = f"{point}__{condition}__{family}"
                save_npz(out/(stem+".npz"), prediction=pred.reshape(64, 511, 22),
                         gammas=gammas, **arrays)
                write_json(out/(stem+".json"), dict(point=point, condition=condition,
                    family=family, alpha=chosen["alpha"], tau=chosen["tau"],
                    feature_transform=feat.metadata(), selected_gammas=gammas.tolist(),
                    created_at=stamp(), feature_count=len(full.mu),
                    source_fit_sha256=digest(RAW/"fit"/f"{point}__{condition}"/"arrays.npz")))
                record["conditions"][condition] = dict(fold_mse=errors.tolist(),
                    mean_mse=mean_errors.tolist(), selected_gamma=gammas.tolist())
                if family == FAMILIES[0]:
                    original = original_predictions(full_bv, sv, point, condition)
                    save_npz(out/f"{point}__{condition}__original.npz",
                             prediction=original.reshape(64, 511, 22))
                print("Saved", stem, flush=True)
            cv_records.append(record)
    write_json(HERE/"cv_results.json", cv_records)
    write_json(HERE/"FIT_COMMIT.json", dict(completed_at=stamp(),
        elapsed_seconds=time.monotonic()-start, output_models=18*22,
        reference_models=6*22, test_label_used_for_tuning=False,
        files={str(p.relative_to(HERE)): digest(p) for p in sorted(out.iterdir())},
        source_sha256=freeze["code_sha256"], card_sha256=freeze["card_sha256"]))
    print("All exploratory fits sealed before evaluation summary.", flush=True)


def describe(v, draws):
    v = np.asarray(v, float)
    assert v.shape == (64,) and np.isfinite(v).all()
    ci = np.quantile(np.median(v[draws], axis=1), [.025, .975])
    return dict(n=64, median=float(np.median(v)), q1=float(np.quantile(v, .25)),
                q3=float(np.quantile(v, .75)), minimum=float(v.min()), maximum=float(v.max()),
                positive=int((v > 0).sum()), negative=int((v < 0).sum()),
                zero=int((v == 0).sum()), descriptive_95_ci=ci.tolist())


def summarize():
    commit = json.loads((HERE/"FIT_COMMIT.json").read_text())
    for name, expected in commit["files"].items():
        assert digest(HERE/name) == expected
    with np.load(HERE/"evaluation_targets.npz") as z:
        y, ids, frequency = z["y"], z["token_id"], z["train_frequency"]
    sst = ((y-y.mean(1, keepdims=True))**2).sum(1)
    assert np.all(sst > 0)
    rng = np.random.Generator(np.random.PCG64(20261024))
    draws = rng.integers(0, 64, size=(10000, 64))
    records, strata = [], []
    original_deltas = {}
    rows = []
    for point in POINTS:
        for condition in CONDITIONS:
            for family in ["original", *FAMILIES]:
                with np.load(HERE/"fits"/f"{point}__{condition}__{family}.npz") as f:
                    pred = f["prediction"]
                se = (y[:, :, None]-pred)**2
                sse = se.sum(1)
                r2 = 1-sse/sst[:, None]
                delta = r2[:, 1]-r2[:, 0]
                adv = r2[:, 1]-r2[:, 2:].mean(1)
                if family == "original":
                    original_deltas[point, condition] = delta.copy()
                metrics = dict(baseline_r2=r2[:, 0], pc_r2=r2[:, 1],
                               random_r2=r2[:, 2:].mean(1), pc_gain=delta, random_advantage=adv,
                               gain_change_from_original=delta-original_deltas[point, condition])
                record = dict(point=point, condition=condition, family=family,
                              metrics={k: describe(v, draws) for k, v in metrics.items()},
                              retained_gain_ratio=float(np.median(delta)/
                                  np.median(original_deltas[point, condition])),
                              article_values={k: v.tolist() for k, v in metrics.items()})
                records.append(record)
                for j in range(64):
                    rows.append(dict(point=point, condition=condition, family=family, article_index=j,
                                     **{k: float(v[j]) for k, v in metrics.items()}))
                for label, mask in [
                    ("unseen_0", frequency == 0),
                    ("rare_1_4", (frequency >= 1) & (frequency <= 4)),
                    ("medium_5_19", (frequency >= 5) & (frequency <= 19)),
                    ("common_20plus", frequency >= 20)]:
                    target = y[mask]
                    ss = se[mask].sum(0)
                    denom = float(np.sum((target-target.mean())**2))
                    strata.append(dict(point=point, condition=condition, family=family, group=label,
                        tokens=int(mask.sum()), articles=int(mask.any(1).sum()),
                        distinct_test_ids=int(np.unique(ids[mask]).size), sst=denom,
                        baseline_r2=float(1-ss[0]/denom), pc_r2=float(1-ss[1]/denom),
                        pc_gain=float((ss[0]-ss[1])/denom),
                        random_advantage=float((ss[2:].mean()-ss[1])/denom),
                        mean_mse_reduction=float((ss[0]-ss[1])/mask.sum())))
    write_json(HERE/"results.json", dict(created_at=stamp(), scope="EXPLORATORY",
        fit_commit_sha256=digest(HERE/"FIT_COMMIT.json"), records=records, frequency_strata=strata,
        planned_article_condition_models=1536, analyzed_article_condition_models=len(rows),
        missing_article_condition_models=0, bootstrap=dict(repetitions=10000, seed=20261024,
        interval="95% descriptive percentile; fixed training fit; not multiplicity-adjusted")))
    write_json(HERE/"article_results.json", rows)
    print("Summarized", len(records), "condition/model sets;", len(rows), "article results", flush=True)
    for r in records:
        if r["condition"] == "C1":
            print(r["point"], r["family"],
                  {k: round(v["median"], 6) for k, v in r["metrics"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "summarize"])
    args = parser.parse_args()
    if args.action == "fit":
        fit_all()
    else:
        summarize()
