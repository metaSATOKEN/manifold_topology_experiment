"""Model-free finite EDA helpers; variable-size ridge blocks and article statistics."""
import numpy as np
import unicodedata

GRID = np.array([.0001, .001, .01, .1, 1.])
FAMILIES = ["A1", "A7", "A8", "NR", "PU", "NU"]
CONTRASTS = ["B_to_A1", "B_to_A7", "B_to_A8", "B_to_N",
             "N_to_NR", "N_to_NU", "A8_adv", "NU_adv"]
STRATA = {
    "type": ["whitespace", "numeric", "letters", "punct_symbol", "other"],
    "frequency": ["unseen", "1-9", "10-99", "100+"],
    "position": ["0-25%", "25-50%", "50-75%", "75-100%"],
}


def token_type(text):
    s = text.strip()
    if not s:
        return 0
    cats = [unicodedata.category(c)[0] for c in s if not c.isspace()]
    if "N" in cats and all(c in "NPS" for c in cats):
        return 1
    if "L" in cats:
        return 2
    if all(c in "PS" for c in cats):
        return 3
    return 4


def scale(x):
    mean, sd = x.mean(0), x.std(0)
    active = sd > 1e-12
    return mean, np.where(active, sd, 1.), active.astype(np.int64)


def standardize(x, params):
    mean, sd, active = params
    return (x - mean) / sd * active


def center(x, ids):
    """Group centering using sorted reduceat (different from frozen np.add.at)."""
    one = x.ndim == 1
    a = x[:, None] if one else x
    order = np.argsort(ids, kind="stable")
    uniq, start, count = np.unique(ids[order], return_index=True, return_counts=True)
    means = np.add.reduceat(a[order], start, axis=0) / count[:, None]
    ix = np.searchsorted(uniq, ids)
    y = a - means[ix]
    return (y[:, 0] if one else y), count[ix] >= 2, count[ix]


def solve(x, y, penalties):
    gram = x.T @ x
    return np.linalg.solve(gram + len(y) * np.diag(penalties), x.T @ y)


def audit_solution(x, y, penalties, beta):
    lhs = x.T @ (x @ beta - y) + len(y) * penalties * beta
    residual = float(np.max(abs(lhs)) / max(1., np.max(abs(x.T @ y))))
    assert residual < 1e-7, residual
    direct = np.linalg.lstsq(
        np.vstack([x, np.diag(np.sqrt(len(y) * penalties))]),
        np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]
    np.testing.assert_allclose(beta, direct, atol=1e-7, rtol=1e-6)
    return residual, float(np.max(abs(beta - direct)))


def cv_block(z, w, y, folds, base_penalty):
    """w: [tokens, groups, k]. Select gamma separately for each of the fixed groups."""
    n, groups, k = w.shape
    q = z.shape[1]
    x = np.column_stack([z, w.reshape(n, -1)])
    losses = np.zeros((len(GRID), groups))
    total_gram, total_rhs = x.T @ x, x.T @ y
    parts = []
    for fold in range(4):
        mask = folds == fold
        xv, yv = x[mask], y[mask]
        vg, vr, vy = xv.T @ xv, xv.T @ yv, float(yv @ yv)
        parts.append((mask, vg, vr, vy, total_gram - vg, total_rhs - vr))
    for j in range(groups):
        ix = np.r_[np.arange(q), np.arange(q + j * k, q + (j + 1) * k)]
        for gi, gamma in enumerate(GRID):
            pen = np.r_[base_penalty, np.repeat(gamma, k)]
            for mask, vg, vr, vy, tg, tr in parts:
                coef = np.linalg.solve(tg[np.ix_(ix, ix)] + (~mask).sum() * np.diag(pen), tr[ix])
                losses[gi, j] += vy - 2 * coef @ vr[ix] + coef @ vg[np.ix_(ix, ix)] @ coef
    losses /= n
    choices = np.argmin(losses, axis=0)
    gammas = GRID[choices]
    betas, max_grad, max_qr, max_cv = [], 0., 0., 0.
    for j, gamma in enumerate(gammas):
        xx = np.column_stack([z, w[:, j]])
        pen = np.r_[base_penalty, np.repeat(gamma, k)]
        beta = solve(xx, y, pen)
        grad, qr = audit_solution(xx, y, pen, beta)
        max_grad, max_qr = max(max_grad, grad), max(max_qr, qr)
        # Direct held-fold predictions cross-check sufficient-statistic CV.
        sse = 0.
        for fold in range(4):
            va = folds == fold
            b = solve(xx[~va], y[~va], pen)
            sse += np.square(xx[va] @ b - y[va]).sum()
        err = abs(sse / n - losses[choices[j], j])
        assert err < 1e-8, err
        max_cv = max(max_cv, float(err))
        betas.append(beta)
    return dict(beta=np.asarray(betas), gamma=gammas, base_penalty=np.asarray(base_penalty)), dict(
        grid=GRID.tolist(), mse=losses.tolist(), selected_gammas=gammas.tolist(),
        fold_tokens=[int((folds == f).sum()) for f in range(4)],
        max_gradient_relative=max_grad, max_augmented_lstsq_difference=max_qr,
        max_cv_sse_difference=max_cv)


def predict(z, w, m):
    q = z.shape[1]
    return z @ m["beta"][:, :q].T + np.einsum("ngk,gk->ng", w, m["beta"][:, q:])


def score(y, predictions, weight=None):
    weight = np.ones(len(y)) if weight is None else weight
    ss = float(np.sum(weight * y * y))
    assert ss > 1e-12
    r2 = {k: 1 - np.sum(weight[:, None] * (v - y[:, None]) ** 2, axis=0) / ss
          for k, v in predictions.items()}
    out = {}
    for family in FAMILIES:
        base = "N" if family in ["NR", "NU"] else "P" if family == "PU" else "B"
        gains = r2[family] - r2[base][0]
        out[family + "_gain"] = float(gains[0])
        out[family + "_adv"] = float(gains[0] - np.mean(gains[1:]))
    out.update(A8_minus_A1=float(r2["A8"][0] - r2["A1"][0]),
               A8_minus_A7=float(r2["A8"][0] - r2["A7"][0]),
               B_minus_P=float(r2["B"][0] - r2["P"][0]),
               N_minus_B=float(r2["N"][0] - r2["B"][0]),
               NR_gain_minus_A8_gain=out["NR_gain"] - out["A8_gain"],
               NR_minus_NU=float(r2["NR"][0] - r2["NU"][0]))
    return out, {k: v.tolist() for k, v in r2.items()}


def errors(y, predictions):
    return {k: np.square(v - y[:, None]) for k, v in predictions.items()}


def token_contrasts(e):
    return {
        "B_to_A1": e["B"][:, 0] - e["A1"][:, 0],
        "B_to_A7": e["B"][:, 0] - e["A7"][:, 0],
        "B_to_A8": e["B"][:, 0] - e["A8"][:, 0],
        "B_to_N": e["B"][:, 0] - e["N"][:, 0],
        "N_to_NR": e["N"][:, 0] - e["NR"][:, 0],
        "N_to_NU": e["N"][:, 0] - e["NU"][:, 0],
        "A8_adv": e["A8"][:, 1:].mean(1) - e["A8"][:, 0],
        "NU_adv": e["NU"][:, 1:].mean(1) - e["NU"][:, 0],
    }


def describe(values, mean=False):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(a):
        return dict(n=0, missing=len(values), estimate=None, lo=None, hi=None)
    rng = np.random.Generator(np.random.PCG64(20261029))
    stat = np.mean if mean else np.median
    draws = np.concatenate([stat(a[rng.integers(len(a), size=(200, len(a)))], axis=1)
                            for _ in range(100)])
    lo, hi = np.quantile(draws, [.025, .975])
    loo = [stat(np.delete(a, j)) for j in range(len(a))] if len(a) > 1 else [a[0]]
    positive = np.maximum(a, 0)
    return dict(n=len(a), missing=len(values) - len(a), estimate=float(stat(a)),
                statistic="mean" if mean else "median", lo=float(lo), hi=float(hi),
                q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                min=float(a.min()), max=float(a.max()),
                positive=int((a > 0).sum()), negative=int((a < 0).sum()), zero=int((a == 0).sum()),
                leave_one_out_min=float(min(loo)), leave_one_out_max=float(max(loo)),
                top5_positive_share=None if positive.sum() == 0 else float(np.sort(positive)[-5:].sum() / positive.sum()))
