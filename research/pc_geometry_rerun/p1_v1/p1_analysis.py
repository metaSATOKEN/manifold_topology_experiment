"""Frozen P1 analysis. All scientific computations use saved observations only."""
import math
from pathlib import Path
import numpy as np
import p1_io as disk
import p1_numeric as num

CONDITIONS = ("C0", "C1", "C2", "C4")


def plans(x, doc, point):
    norms = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    n = len(x)
    count = math.ceil(n * disk.spec()["remove_fraction"])
    top = np.lexsort((np.arange(n), -norms))[:count]
    indices = dict(C0=np.arange(n), C1=np.arange(1, n),
                   C2=np.setdiff1d(np.arange(n), top), C4=np.arange(n))
    random = []
    for rep in range(disk.spec()["random_repetitions"]):
        seed = int(disk.sha(f"20261020:{doc}:{point}:{rep}".encode())[:16], 16)
        removed = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(n, count, replace=False))
        random.append(dict(repetition=rep, seed=seed, removed=removed.tolist()))
    return norms, top, indices, random


def gram_metrics(x):
    """Independent of SVD: leading Gram eigenvector yields PC1 scores up to sign."""
    x = np.asarray(x, np.float64)
    centered = x - x.mean(0)
    gram = centered @ centered.T / (len(x)-1)
    total = float(np.trace(gram))
    if not np.isfinite(x).all() or total <= 0:
        return dict(valid=False, reason="zero_or_nonfinite_total_variance")
    values, vectors = np.linalg.eigh(gram)
    r = num.correlation(vectors[:, -1], np.linalg.norm(x, axis=1))
    return dict(valid=True, R1=None if r is None else abs(r),
                V1=float(values[-1]/total), n_tokens=len(x), total_variance=total)


def fixed_basis(x, basis):
    centered = x - x.mean(0)
    total = float(np.square(centered).sum()/(len(x)-1))
    if total <= 0:
        return dict(valid=False, reason="zero_total_variance")
    scores = centered @ basis
    r = num.correlation(scores, np.linalg.norm(x, axis=1))
    return dict(valid=True, R1=None if r is None else abs(r),
                direction_variance_fraction=float(np.var(scores, ddof=1)/total))


def matrix_for(x, norms, index, condition):
    if condition == "C4":
        if np.any(norms == 0):
            return None, "zero_norm_unit_conversion"
        return x[index] / norms[index, None], None
    return x[index], None


def analyze_point(raw, doc, point):
    x = raw["h_"+point].astype(np.float64)
    features = {name:raw[name] for name in ("position", "token_chars", "surprisal")}
    norms, top, indices, random = plans(x, doc, point)
    arrays, conditions = {}, {}
    base_basis = None
    for c in CONDITIONS:
        index = indices[c]
        xx, reason = matrix_for(x, norms, index, c)
        if reason is None and np.square(xx-xx.mean(0)).sum() <= 0:
            reason = "zero_total_variance"
        if reason:
            conditions[c] = dict(valid=False, reason=reason, kept_positions=index.tolist())
            continue
        a, m = num.pca(xx, {k:v[index] for k,v in features.items()})
        if c == "C4":
            m["R1"] = m["signed_R1"] = None
        conditions[c] = dict(valid=True, metrics=m, kept_positions=index.tolist())
        for key, value in a.items():
            arrays[c+"__"+key] = value
        if c == "C0":
            base_basis = a["components"][0]
    for row in random:
        index = np.setdiff1d(np.arange(len(x)), row["removed"])
        row["metrics"] = gram_metrics(x[index])
    fixed = {} if base_basis is None else {
        c:fixed_basis(x[indices[c]], base_basis) for c in ("C1", "C2")}
    metadata = dict(document_id=doc, point=point, conditions=conditions,
        random_controls=random, fixed_C0_basis=fixed,
        diagnostics=dict(top_norm_positions=top.tolist(), position0_is_max=bool(np.argmax(norms)==0),
            position0_in_top=bool(0 in top), max_norm=float(norms.max()),
            median_norm=float(np.median(norms)),
            max_to_median=None if np.median(norms)==0 else float(norms.max()/np.median(norms))),
        feature_scope="Within-article exploratory correlations, not held-out prediction")
    return arrays, metadata


def verify_point(raw, doc, point, arrays, meta):
    """Audit saved SVD with Gram spectrum; rederive every exclusion and control."""
    x = raw["h_"+point].astype(np.float64)
    norms, top, indices, random = plans(x, doc, point)
    disk.require(meta["document_id"]==doc and meta["point"]==point, "Analysis identity")
    disk.require(set(meta["conditions"])==set(CONDITIONS), "Analysis conditions")
    expected_keys = set()
    for c, condition in meta["conditions"].items():
        index = indices[c]
        disk.require(condition["kept_positions"]==index.tolist(), "Exclusion indices changed")
        xx, reason = matrix_for(x, norms, index, c)
        if reason is None and np.square(xx-xx.mean(0)).sum() <= 0:
            reason = "zero_total_variance"
        if reason:
            disk.require(condition == dict(valid=False, reason=reason, kept_positions=index.tolist()),
                         "Invalid condition incorrectly represented")
            continue
        disk.require(condition["valid"], "Valid condition silently omitted")
        a = {key:arrays[c+"__"+key] for key in ("mean", "components", "scores", "eigenvalues")}
        expected_keys |= {c+"__"+key for key in a}
        num.verify_pca(xx, a, condition["metrics"],
                       {name:raw[name][index] for name in ("position","token_chars","surprisal")},
                       unit_norm=c=="C4")
    disk.require(set(arrays)==expected_keys, "Unexpected/omitted PCA arrays")
    disk.require(len(meta["random_controls"])==len(random), "Random control count")
    for actual, expected in zip(meta["random_controls"], random):
        index = np.setdiff1d(np.arange(len(x)), expected["removed"])
        expected["metrics"] = gram_metrics(x[index])
        num.compare_tree(actual, expected)
    expected_fixed = {} if not meta["conditions"]["C0"]["valid"] else {
        c:fixed_basis(x[indices[c]], arrays["C0__components"][0]) for c in ("C1","C2")}
    num.compare_tree(meta["fixed_C0_basis"], expected_fixed)
    num.compare_tree(meta["diagnostics"], dict(
        top_norm_positions=top.tolist(), position0_is_max=bool(np.argmax(norms)==0),
        position0_in_top=bool(0 in top), max_norm=float(norms.max()),
        median_norm=float(np.median(norms)),
        max_to_median=None if np.median(norms)==0 else float(norms.max()/np.median(norms))))


def interval(values, planned):
    values = np.array(values, np.float64)
    if not len(values):
        return dict(n=0, planned=planned, median=None, lower=None, upper=None)
    cfg = disk.spec()
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    sampled = values[rng.integers(0,len(values),(cfg["bootstrap_repetitions"],len(values)))]
    lo, hi = np.quantile(np.median(sampled, axis=1), [0.05/6,1-0.05/6])
    return dict(n=len(values), planned=planned, median=float(np.median(values)),
                lower=float(lo), upper=float(hi))


def adjudicate(a, b, d):
    cfg = disk.spec()
    if min(a["n"],b["n"]) < cfg["minimum_valid_documents"]:
        h1 = "INSUFFICIENT_DATA"
    elif a["lower"]>=cfg["high_R1"] and b["lower"]>=-cfg["allowed_R1_drop"]:
        h1 = "SUPPORTED"
    elif a["upper"]<cfg["high_R1"] or b["upper"]<-cfg["allowed_R1_drop"]:
        h1 = "REFUTED"
    else:
        h1 = "INCONCLUSIVE"
    if d["n"]<cfg["minimum_valid_documents"]:
        h2 = "INSUFFICIENT_DATA"
    elif d["upper"]<=-cfg["meaningful_V1_drop"]:
        h2 = "SUPPORTED"
    elif d["lower"]>-cfg["meaningful_V1_drop"]:
        h2 = "REFUTED"
    else:
        h2 = "INCONCLUSIVE"
    return dict(H1=h1,H2=h2)


def summarize(root, rows):
    cfg = disk.spec()
    measures, missing, document_endpoints = {k:[] for k in ("A","B","D")}, [], []
    for row in rows:
        if row["point"] != cfg["primary_point"]:
            continue
        c = row["conditions"]
        get = lambda key, metric: c[key]["metrics"][metric] if c[key]["valid"] else None
        r0,r1,v2 = get("C0","R1"),get("C1","R1"),get("C2","V1")
        v3 = [x["metrics"].get("V1") for x in row["random_controls"]]
        vals = dict(A=r1, B=None if r0 is None or r1 is None else r1-r0,
                    D=None if v2 is None or any(v is None for v in v3) else v2-float(np.mean(v3)))
        document_endpoints.append(dict(document_id=row["document_id"], **vals))
        for k,v in vals.items():
            if v is None:
                missing.append(dict(document=row["document_id"], endpoint=k,
                                    reason="undefined_required_condition_or_correlation"))
            else:
                measures[k].append(v)
    complete = len(rows)==cfg["documents"]*len(cfg["points"])
    estimated = {k:interval(v,cfg["documents"]) for k,v in measures.items()}
    return dict(planned_documents=cfg["documents"], analyzed_document_points=len(rows),
        planned_document_points=cfg["documents"]*len(cfg["points"]),
        primary_point=cfg["primary_point"], endpoints=estimated, missing=missing,
        scientific_adjudication=adjudicate(**{k.lower():v for k,v in estimated.items()})
            if complete else dict(H1="PENDING_COMPLETE_RUN",H2="PENDING_COMPLETE_RUN"),
        uncertainty="Article percentile bootstrap, 10000 replicates, 98.333333% per interval",
        document_endpoints=document_endpoints,
        observed_document_ids=sorted({r["document_id"] for r in rows}))
