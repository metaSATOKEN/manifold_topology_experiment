"""Fixed descriptive geometry. No network or language-model imports."""
import numpy as np


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return None
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den > 1e-12 else None


def spacing(v):
    v = np.sort(v)
    if len(v) < 2 or np.ptp(v) <= 1e-12:
        return None, None
    counts = np.histogram(v, np.linspace(v[0], v[-1], 21))[0]
    return float(np.mean(counts > 0)), float(np.diff(v).max()/np.ptp(v))


def analyze(raw, condition, spec):
    h = raw["h"].astype(np.float64)
    position = raw["position"]
    radius = np.linalg.norm(h, axis=1)
    keep = position > 0 if condition == "C1" else np.ones(len(h), dtype=bool)
    x = h[keep].copy()
    if condition == "C4":
        if np.any(radius[keep] <= 1e-12):
            raise ValueError("zero norm in C4")
        x /= radius[keep, None]
    positions = position[keep]
    mean = x.mean(0)
    y = x-mean
    u, singular, vt = np.linalg.svd(y, full_matrices=False)
    total = float(np.sum(y*y))
    if not total > 1e-12 or len(singular) < 8 or singular[7]**2 <= total*1e-12:
        raise ValueError("degenerate variance or rank < 8")
    components = vt[:8].copy()
    signs = np.sign(components[np.arange(8), np.argmax(abs(components), axis=1)])
    components *= signs[:, None]
    s = y @ components.T
    ev = singular**2/total
    z = s[:, 0]/s[:, 0].std()
    folds = np.minimum(4, np.arange(len(x))*5//len(x))
    denominator = np.array([total, np.sum(s[:, :2]**2), np.sum(s[:, :3]**2)])
    sses, in_sample, coefs = [], [], []
    for degree in [1, 2]:
        design = np.column_stack([z**j for j in range(degree+1)])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        coefs.append(coef)
        e = y-design@coef
        ep = e@components[:3].T
        in_sample.append(1-np.array([np.sum(e*e), np.sum(ep[:, :2]**2), np.sum(ep*ep)])/denominator)
        fs = []
        for fold in range(5):
            te = folds == fold
            tr = ~te
            beta = np.linalg.lstsq(design[tr], y[tr], rcond=1e-12)[0]
            error = y[te]-design[te]@beta
            projection = error@components[:3].T
            fs.append([np.sum(error*error), np.sum(projection[:, :2]**2), np.sum(projection*projection)])
        sses.append(fs)
    sses = np.array(sses)
    q = 1-sses.sum(1)/denominator
    thickness = s[:, 1:8]-np.column_stack([np.ones(len(z)), z, z*z])@coefs[1]@components[1:8].T
    occupancy, gap = spacing(s[:, 0])
    lo, hi = np.quantile(s[:, 0], [.05, .95])
    occ90, gap90 = spacing(s[(s[:, 0] >= lo) & (s[:, 0] <= hi), 0])
    share = s[:, 0]**2/np.sum(s[:, 0]**2)
    metrics = dict(V1=float(ev[0]), V2=float(ev[:2].sum()), V3=float(ev[:3].sum()),
        V8=float(ev[:8].sum()), pc2_variance=float(ev[1]), participation=float(1/np.sum(ev**2)),
        occupancy=occupancy, max_gap=gap, occupancy90=occ90, max_gap90=gap90,
        max_pc1_share=float(share.max()), max_pc1_position=int(positions[share.argmax()]),
        first_pc1_share=float(share[positions == 0][0]) if np.any(positions == 0) else None,
        pc1_norm_r=corr(s[:, 0], radius[keep]))
    spread = occupancy >= spec["occupancy_min"] and gap <= spec["max_gap_max"] and share.max() <= spec["max_pc1_share_max"]
    for j, space in enumerate(["full", "2d", "3d"]):
        metrics.update({f"Q_line_{space}": float(q[0, j]), f"Q_curve_{space}": float(q[1, j]),
            f"curve_gain_{space}": float(q[1, j]-q[0, j]), f"Q_insample_{space}": float(in_sample[1][j])})
        metrics[f"graph_{space}"] = bool(spread and q[1, j] >= spec["curve_q_min"])
        metrics[f"curved_graph_{space}"] = bool(metrics[f"graph_{space}"] and q[1, j]-q[0, j] >= spec["curve_gain_min"])
    targets = dict(position=positions, log_norm=np.log1p(radius[keep]),
                   log_chars=np.log1p(raw["token_chars"][keep]), surprisal=raw["surprisal"][keep])
    metrics["thickness_correlations"] = {name: [corr(thickness[:, k], values) for k in range(7)]
                                        for name, values in targets.items()}
    # Numerical sensitivity, not a second scientific condition or chosen solver.
    xf = x.astype(np.float32)
    yf = xf-xf.mean(0)
    sf = np.linalg.svd(yf, full_matrices=False, compute_uv=False)
    metrics["float32_V1"] = float(sf[0]**2/np.sum(yf*yf))
    arrays = dict(mean=mean, components=components, scores=s, singular_values=singular,
                  positions=positions, z=z, folds=folds, curve_fold_sse=sses, curve_q=q,
                  line_coef=coefs[0], curve_coef=coefs[1], thickness=thickness)
    return metrics, arrays
