"""Fixed out-of-sample evaluation and prespecified within-article/token-ID statistics."""
import numpy as np
import p3_io as disk


def within_center(token, y, predictions):
    """Demean y and each fixed prediction within article x token-ID groups."""
    unique, inverse, counts = np.unique(token, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 2
    yc = np.full(len(y), np.nan)
    pc = np.full_like(predictions, np.nan)
    for j in range(len(unique)):
        ix = inverse == j
        if counts[j] < 2:
            continue
        yc[ix] = y[ix]-y[ix].mean()
        pc[:, ix] = predictions[:, ix]-predictions[:, ix].mean(axis=1, keepdims=True)
    return yc, pc, keep, int((counts >= 2).sum())


def errors(y, prediction, *, centered=False):
    sst = float(np.sum(y*y)) if centered else float(np.sum((y-y.mean())**2))
    if len(y) < 2 or sst <= 0:
        return dict(valid=False, reason="zero_target_variance", rows=len(y),
                    gain=None, random_advantage=None, r2=None, sst=sst, sse=None)
    sse = np.sum((prediction-y[None])**2, axis=1)
    return dict(valid=True, reason=None, rows=len(y), sst=sst, sse=sse.tolist(),
                r2=(1-sse/sst).tolist(), gain=float((sse[0]-sse[1])/sst),
                random_advantage=float((sse[2:].mean()-sse[1])/sst))


def evaluate(raw, doc, point, condition, model, counts):
    from p3_frozen import predict
    disk.require(condition == "C1", "P3 v1 is limited to the fixed C1 condition")
    prediction, scores, known = predict(raw, point, model, counts)
    pos = np.arange(1, disk.spec()["tokens"], dtype=np.int64)
    y, token = raw["surprisal"][pos], raw["token_id"][pos]
    disk.require(np.isfinite(y).all(), "Nonfinite evaluation target")
    yc, pc, keep, groups = within_center(token, y, prediction)
    ordinary = errors(y, prediction)
    if int(keep.sum()) < disk.spec()["within_min_tokens"] or groups < disk.spec()["within_min_groups"]:
        within = dict(valid=False, reason="insufficient_repeated_tokens_or_groups",
                      rows=int(keep.sum()), gain=None, random_advantage=None, r2=None,
                      sst=None, sse=None)
    else:
        within = errors(yc[keep], pc[:, keep], centered=True)
    metrics = dict(document_id=doc, point=point, condition=condition, ordinary=ordinary,
                   within_id=within, repeated_groups=groups, repeated_tokens=int(keep.sum()),
                   singleton_tokens=int((~keep).sum()), unseen_id_tokens=int((~known).sum()))
    arrays = dict(kept_positions=pos, token_id=token, pc_scores=scores, target=y,
                  predictions=prediction, within_target=yc, within_predictions=pc,
                  within_mask=keep.astype(np.int64), known_id=known.astype(np.int64))
    return arrays, metrics


def interval(values):
    cfg = disk.spec()
    v = np.asarray([x for x in values if x is not None], np.float64)
    disk.require(np.isfinite(v).all(), "Nonfinite endpoint")
    if len(v) == 0:
        return dict(n=0, median=None, lower=None, upper=None, positive=0, negative=0)
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    draws = rng.integers(0, len(v), (cfg["bootstrap_repetitions"], len(v)))
    lo, hi = np.quantile(np.median(v[draws], axis=1), [.00625, .99375])
    return dict(n=len(v), median=float(np.median(v)), lower=float(lo), upper=float(hi),
                positive=int((v > 0).sum()), negative=int((v < 0).sum()))


def adjudicate(endpoints, keys, thresholds, complete):
    if not complete:
        return "PENDING"
    if any(endpoints[k]["n"] < disk.spec()["minimum_valid_documents"] for k in keys):
        return "INSUFFICIENT_DATA"
    if all(endpoints[k]["lower"] >= t for k, t in zip(keys, thresholds)):
        return "SUPPORTED"
    if any(endpoints[k]["upper"] < t for k, t in zip(keys, thresholds)):
        return "REFUTED"
    return "INCONCLUSIVE"


def summarize(rows, complete):
    cfg = disk.spec()
    primary = [r for r in rows if r["point"] == cfg["primary_point"] and r["condition"] == "C1"]
    mapping = {"A": ("ordinary", "gain"), "B": ("ordinary", "random_advantage"),
               "C": ("within_id", "gain"), "D": ("within_id", "random_advantage")}
    values = {key: [r[section][metric] for r in primary] for key, (section, metric) in mapping.items()}
    endpoints = {k: interval(v) for k, v in values.items()}
    missing = {k: [{"document_id": r["document_id"], "reason": r[section]["reason"]}
                    for r in primary if r[section][metric] is None]
               for k, (section, metric) in mapping.items()}
    outcomes = dict(
        H1=adjudicate(endpoints, ["A", "B"],
            [cfg["surprisal_min_gain"], cfg["surprisal_min_random_advantage"]], complete),
        H2=adjudicate(endpoints, ["C", "D"],
            [cfg["within_min_gain"], cfg["within_min_random_advantage"]], complete))
    secondary = {}
    for point in cfg["points"]:
        if point == cfg["primary_point"]:
            continue
        subset = [r for r in rows if r["point"] == point]
        # Same intervals are descriptive here, not additional adjudicated hypotheses.
        secondary[point] = {k: interval([r[s][m] for r in subset])
                            for k, (s, m) in mapping.items()}
    return dict(endpoints=endpoints, document_endpoints=[
        dict(document_id=r["document_id"], **{k:r[s][m] for k,(s,m) in mapping.items()})
        for r in primary], missing=missing, scientific_adjudication=outcomes,
        secondary_descriptive=secondary, complete=complete,
        interpretation="Fixed-probe replication and within-token-ID association; not causal identification")
