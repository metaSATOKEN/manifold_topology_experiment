"""Fixed descriptive geometry, with no model/network imports."""
import numpy as np
from scipy.linalg import eigh


def pca(x):
    mean = x.mean(0)
    y = x - mean
    g = y @ y.T
    total = float(np.trace(g))
    ev, u = eigh(g, subset_by_index=(len(x)-8, len(x)-1), driver="evr")
    ev, u = ev[::-1].copy(), u[:, ::-1].copy()
    if not total > 1e-12 or not ev[-1] > total*1e-12:
        raise ValueError("rank<8 or zero variance")
    v = (y.T @ u / np.sqrt(ev)).T
    signs = np.sign(v[np.arange(8), np.argmax(abs(v), axis=1)])
    v *= signs[:, None]
    s = y @ v.T
    orth = float(np.max(abs(v @ v.T-np.eye(8))))
    eq = float(np.max(abs(g @ (u*signs)-(u*signs)*ev))/total)
    assert orth < 1e-7 and eq < 1e-9
    np.testing.assert_allclose(np.sum(s*s, 0), ev, atol=total*1e-8, rtol=1e-8)
    return dict(mean=mean, components=v, scores=s, eigenvalues=ev,
                total=total, participation=total**2/np.sum(g*g),
                orth_error=orth, eigen_error=eq), y


def spacing(s, bins=20):
    v = np.sort(s)
    span = np.ptp(v)
    if span <= 1e-12:
        raise ValueError("degenerate PC1 range")
    counts = np.histogram(v, bins=np.linspace(v[0], v[-1], bins+1))[0]
    return float(np.mean(counts > 0)), float(np.max(np.diff(v))/span)


def design(z, degree):
    return np.column_stack([z**i for i in range(degree+1)])


def curve_analysis(y, p, folds):
    s, v = p["scores"], p["components"]
    z = s[:, 0]/np.std(s[:, 0])
    coefs, predictions, sses, fullq = [], [], [], []
    total = np.array([p["total"], np.sum(s[:, :2]**2), np.sum(s[:, :3]**2)])
    for degree in [1, 2]:
        a = design(z, degree)
        coef = np.linalg.lstsq(a, y, rcond=1e-12)[0]
        coefs.append(coef)
        pred = np.empty_like(y)
        fss = []
        for fold in range(5):
            te = folds == fold
            tr = ~te
            assert tr.sum() >= 8 and te.sum() > 0
            c = np.linalg.lstsq(a[tr], y[tr], rcond=1e-12)[0]
            pred[te] = a[te] @ c
            e = y[te]-pred[te]
            ep = e @ v[:3].T
            fss.append([np.sum(e*e), np.sum(ep[:, :2]**2), np.sum(ep**2)])
        predictions.append(pred @ v[:3].T)
        sses.append(fss)
        e = y-a @ coef
        ep = e @ v[:3].T
        fullq.append(1-np.array([np.sum(e*e), np.sum(ep[:, :2]**2), np.sum(ep**2)])/total)
    sses = np.array(sses)
    q = 1-np.sum(sses, axis=1)/total
    thickness = s[:, 1:8]-design(z, 2) @ coefs[1] @ v[1:8].T
    return dict(z=z, line_coef=coefs[0], curve_coef=coefs[1],
                curve_cv_projection=np.array(predictions), curve_fold_sse=sses,
                curve_q=q, curve_full_q=np.array(fullq), thickness=thickness)


def ridge_predict(x, y, train, test, alpha=.01):
    mu, sd = x[train].mean(0), x[train].std(0)
    active = sd > 1e-12
    sd = np.where(active, sd, 1)
    a = np.column_stack([np.ones(train.sum()), ((x[train]-mu)/sd)*active])
    b = np.column_stack([np.ones(test.sum()), ((x[test]-mu)/sd)*active])
    penalty = np.r_[0, np.repeat(alpha*train.sum(), x.shape[1])]
    coef = np.linalg.solve(a.T @ a+np.diag(penalty), a.T @ y[train])
    grad = a.T @ (a @ coef-y[train])+penalty*coef
    assert np.max(abs(grad)) < 1e-5*max(1, np.max(abs(a.T @ y[train])))
    return b @ coef


def feature_analysis(a, positions, z, thickness, radius, frequency):
    keep = positions > 0
    p = positions[keep]
    rel = p/(len(a["position"])-1)
    norm = np.log1p(radius[keep])
    chars = np.log1p(a["token_chars"][p])
    surprisal = a["surprisal"][p]
    freq = np.log1p(frequency[a["token_id"][p]])
    targets = np.column_stack([rel, norm, chars, surprisal, surprisal])
    arc = np.column_stack([z[keep], z[keep]**2])
    controls = np.column_stack([rel, rel**2, norm, norm**2, chars, freq])
    folds = np.minimum(4, (p*5)//len(a["position"]))
    pred = np.empty((2, len(p), 5))
    for j in range(5):
        base = arc if j < 4 else np.column_stack([arc, controls])
        extra = np.column_stack([base, thickness[keep]])
        for mode, x in enumerate([base, extra]):
            for f in range(5):
                tr, te = folds != f, folds == f
                pred[mode, te, j] = ridge_predict(x, targets[:, j], tr, te)
    total = np.sum((targets-targets.mean(0))**2, axis=0)
    if not np.all(total > 1e-12):
        raise ValueError("degenerate feature target")
    ss = np.sum((pred-targets[None, :, :])**2, axis=1)
    r2 = 1-ss/total
    return dict(feature_targets=targets, feature_predictions=pred, feature_r2=r2,
                feature_gain=r2[1]-r2[0], feature_positions=p)


def metrics(p, curve, features, positions):
    s = p["scores"]
    # Centered squared norm from Gram is passed separately by caller.
    occ, gap = spacing(s[:, 0])
    lo, hi = np.quantile(s[:, 0], [.05, .95])
    occ90, gap90 = spacing(s[(s[:, 0] >= lo) & (s[:, 0] <= hi), 0])
    pcshare = s[:, 0]**2/np.sum(s[:, 0]**2)
    ev = p["eigenvalues"]/p["total"]
    m = {"V1": ev[0], "V2": ev[:2].sum(), "V3": ev[:3].sum(), "V8": ev.sum(),
         "participation": p["participation"], "occupancy": occ, "max_gap": gap,
         "occupancy90": occ90, "max_gap90": gap90,
         "max_pc1_share": pcshare.max(), "max_pc1_position": positions[pcshare.argmax()],
         "first_pc1_share": pcshare[positions == 0][0] if np.any(positions == 0) else None}
    for j, name in enumerate(["full", "2d", "3d"]):
        m["Q_line_"+name] = curve["curve_q"][0, j]
        m["Q_curve_"+name] = curve["curve_q"][1, j]
        m["curve_gain_"+name] = curve["curve_q"][1, j]-curve["curve_q"][0, j]
        m["Q_curve_insample_"+name] = curve["curve_full_q"][1, j]
    for j, name in enumerate(["position", "norm", "chars", "surprisal", "controlled_surprisal"]):
        m["thickness_gain_"+name] = features["feature_gain"][j]
        m["feature_base_r2_"+name] = features["feature_r2"][0, j]
        m["feature_full_r2_"+name] = features["feature_r2"][1, j]
    spread = occ >= .8 and gap <= .1 and pcshare.max() <= .5
    for space in ["2d", "full"]:
        m["graph_"+space] = bool(spread and m["Q_curve_"+space] >= .9)
        m["curved_graph_"+space] = bool(m["graph_"+space] and m["curve_gain_"+space] >= .01)
    return m
