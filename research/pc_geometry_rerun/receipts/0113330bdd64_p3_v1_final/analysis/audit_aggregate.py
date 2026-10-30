"""Saved-data audit and descriptive summaries; no LM imports, fitting or inference."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parent
RESEARCH = RECEIPT.parent.parent
SOURCE = RESEARCH/"p3_v1"
sys.path.insert(0, str(SOURCE))
import p3_io as disk
from p3_worker import compatibility

RAW = RECEIPT/"raw"
POINTS = ["block06", "final_ln"]


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def describe(values):
    a = np.array(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                q1=float(np.quantile(a, .25)), q3=float(np.quantile(a, .75)),
                minimum=float(a.min()), maximum=float(a.max()),
                positive=int((a > 0).sum()), negative=int((a < 0).sum()),
                zero=int((a == 0).sum()))


def direct_predictions(raw, point, m, counts):
    """Collapse the standardized PC readout into original hidden-state weights."""
    x = raw["h_"+point][1:].astype(float)
    pos = raw["position"][1:]
    p = pos/511.
    token = raw["token_id"][1:]
    b = np.stack([p, p**2, p**3, np.log1p(pos)/np.log(512.),
                  np.log1p(raw["token_chars"][1:]),
                  np.log1p(raw["token_utf8_bytes"][1:]),
                  -np.log((counts[token]+1)/(counts.sum()+len(counts))),
                  np.log1p(np.sqrt(np.einsum("ij,ij->i", x, x)))], axis=1)
    slopes = m["beta"][:, 1:] * (m["feature_active"]/m["feature_scale"])
    intercept = m["beta"][:, 0] - slopes @ m["feature_mean"]
    pred = b @ slopes.T + intercept
    lookup = {int(t): i for i, t in enumerate(m["vocab"])}
    effects = np.stack([m["token_effect"][:, lookup[int(t)]] if int(t) in lookup
                        else np.zeros(22) for t in token])
    known = np.array([int(t) in lookup for t in token])
    pred += effects
    bases = np.concatenate([m["components"][None], m["random_components"]], axis=0)
    w = m["theta"] * (m["score_active"]/m["score_scale"]).reshape(21, 8)
    state_weights = np.einsum("gk,gkd->gd", w, bases)
    offset = -state_weights @ m["mean"] - (w*m["score_mean"].reshape(21, 8)).sum(axis=1)
    pred[:, 1:] += x @ state_weights.T + offset
    return pred.T, known, effects.T, b


def group_center(token, values):
    """Different implementation: grouped sums via bincount, not per-ID slice means."""
    unique, inverse, counts = np.unique(token, return_inverse=True, return_counts=True)
    total = np.stack([np.bincount(inverse, weights=v, minlength=len(unique)) for v in values])
    centered = values - (total/counts)[:, inverse]
    mask = counts[inverse] >= 2
    centered[:, ~mask] = np.nan
    return centered, mask, int((counts >= 2).sum())


def metrics(y, prediction, centered=False):
    sst = float(y @ y) if centered else float(np.var(y)*len(y))
    assert sst > 0
    sse = np.einsum("ij,ij->i", prediction-y, prediction-y)
    r2 = 1-sse/sst
    return dict(sst=sst, sse=sse, r2=r2,
                gain=float((sse[0]-sse[1])/sst),
                random_advantage=float((np.mean(sse[2:])-sse[1])/sst))


def bootstrap(values):
    # Inverse empirical CDF bootstrap; shape differs from the released implementation.
    v = np.asarray(values, float)
    rng = np.random.Generator(np.random.PCG64(20261026))
    draws = rng.integers(len(v), size=(10000, len(v)))
    med = np.sort(v[draws], axis=1)
    mid = med.shape[1]//2
    medians = (med[:, mid-1]+med[:, mid])/2 if len(v)%2 == 0 else med[:, mid]
    return np.quantile(medians, [.00625, .99375])


def main():
    start = time.monotonic()
    cfg = disk.read(SOURCE/"spec.json")
    verified = disk.read(RECEIPT/"receipt_verification.json")
    assert verified["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    summary = disk.read(RAW/"summary.json")
    index = disk.read(RECEIPT/"archive_structure.json")
    fixtures = disk.read(SOURCE/"fixtures.json")
    assert len(fixtures) == 64
    old = disk.read(RESEARCH/"p1_v1/fixtures.json")+disk.read(RESEARCH/"p2_v1/fixtures.json")
    assert not ({f["id"] for f in fixtures} & {f["id"] for f in old})
    for name, meta in index["entries"].items():
        assert disk.digest(RAW/name) == meta["sha256"]
    release = disk.read(SOURCE/"release_manifest.json")
    run = disk.read(RAW/"run.json")
    fixed = disk.read(RAW/"fit/COMMIT.json")
    assert release["frozen_at"] < run["created"] <= fixed["sealed_at"]
    assert not fixed["new_training_performed"]
    assert run["sources"] == release["sources"]
    events = disk.events(RAW)
    starts = [e for e in events if e["kind"] == "FORWARD_STARTED"]
    commits = [e for e in events if e["kind"] == "DOCUMENT_COMMITTED"]
    expected_ids = {f["id"] for f in fixtures}
    assert Counter(e["document_id"] for e in starts) == Counter({d: 1 for d in expected_ids})
    assert Counter(e["document_id"] for e in commits) == Counter({d: 1 for d in expected_ids})
    assert all(e["at"] >= fixed["sealed_at"] and
               e["fit_commit_sha256"] == disk.digest(RAW/"fit/COMMIT.json") for e in starts)
    envs = [disk.read(p) for p in RAW.glob("attempts/*/environment.json")]
    contract = disk.read(RAW/"environment_contract.json")
    assert len(envs) == 4 and all(compatibility(e) == contract for e in envs)
    assert [e["forward_calls"] for e in events if e["kind"] == "NOOP"] == [0]
    assert not any(e["kind"] == "WORKER_FAILURE" for e in events)
    models = {p: read_npz(SOURCE/f"frozen_{p}_C1.npz") for p in POINTS}
    counts = read_npz(SOURCE/"frozen_frequency.npz")["counts"]
    rows, coverage = [], []
    errors = dict(predictions=0., within_predictions=0., within_target=0.,
                  metrics=0., bootstrap=0., centered_id_effect=0.)
    compared = 0
    for f in fixtures:
        doc = f["id"]
        raw = read_npz(RAW/"docs"/doc/"raw.npz")
        token, y = raw["token_id"][1:], raw["surprisal"][1:]
        assert np.array_equal(raw["token_id"], f["token_ids"])
        for point in POINTS:
            folder = RAW/"evaluation"/doc/(point+"__C1")
            stored = read_npz(folder/"arrays.npz")
            saved = disk.read(folder/"metrics.json")
            pred, known, effects, b = direct_predictions(raw, point, models[point], counts)
            errors["predictions"] = max(errors["predictions"], float(abs(pred-stored["predictions"]).max()))
            np.testing.assert_allclose(pred, stored["predictions"], atol=1e-9, rtol=1e-9)
            compared += pred.size
            centered, mask, groups = group_center(token, np.vstack([y, pred]))
            yc, pc = centered[0], centered[1:]
            assert groups >= cfg["within_min_groups"] and int(mask.sum()) >= cfg["within_min_tokens"]
            np.testing.assert_array_equal(mask, stored["within_mask"].astype(bool))
            np.testing.assert_array_equal(known, stored["known_id"].astype(bool))
            np.testing.assert_allclose(centered[0], stored["within_target"], atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(centered[1:], stored["within_predictions"],
                                       atol=1e-9, rtol=1e-9, equal_nan=True)
            for key, a, target in [("within_target", yc, stored["within_target"]),
                                   ("within_predictions", pc, stored["within_predictions"])]:
                errors[key] = max(errors[key], float(np.nanmax(abs(a-target))))
            ec, _, _ = group_center(token, effects)
            errors["centered_id_effect"] = max(errors["centered_id_effect"], float(np.nanmax(abs(ec))))
            bc, _, _ = group_center(token, b[:, 4:7].T)
            assert np.nanmax(abs(bc)) < 1e-12
            assert errors["centered_id_effect"] < 1e-12
            assert (groups, int(mask.sum()), int((~known).sum())) == (
                saved["repeated_groups"], saved["repeated_tokens"], saved["unseen_id_tokens"])
            row = dict(document_id=doc, title=f["title"], point=point, condition="C1")
            for kind, target, predictions in [
                ("ordinary", y, pred), ("within_id", yc[mask], pc[:, mask])]:
                direct = metrics(target, predictions, kind == "within_id")
                for key in direct:
                    difference = float(np.max(abs(np.asarray(direct[key])-np.asarray(saved[kind][key]))))
                    errors["metrics"] = max(errors["metrics"], difference)
                    np.testing.assert_allclose(direct[key], saved[kind][key], atol=1e-9, rtol=1e-9)
                row[kind] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in direct.items()}
            rows.append(row)
        coverage.append(dict(document_id=doc, title=f["title"], repeated_groups=groups,
                             repeated_tokens=int(mask.sum()), singleton_tokens=int((~mask).sum()),
                             unseen_id_tokens=int((~known).sum()),
                             unseen_id_within_tokens=int(((~known)&mask).sum())))
    mapping = dict(A=("ordinary", "gain"), B=("ordinary", "random_advantage"),
                   C=("within_id", "gain"), D=("within_id", "random_advantage"))
    endpoints, descriptions = {}, {}
    for point in POINTS:
        rs = [r for r in rows if r["point"] == point]
        endpoints[point] = {}
        declared = verified["endpoints"] if point == "block06" else verified["secondary_descriptive"][point]
        for key, (mode, metric) in mapping.items():
            vals = [r[mode][metric] for r in rs]
            lo, hi = bootstrap(vals)
            expected = [declared[key]["lower"], declared[key]["upper"]]
            errors["bootstrap"] = max(errors["bootstrap"], float(abs(np.array([lo, hi])-expected).max()))
            np.testing.assert_allclose([lo, hi], expected, atol=1e-10, rtol=1e-10)
            desc = describe(vals)
            assert desc["n"] == 64 and desc["positive"] == declared[key]["positive"]
            assert desc["negative"] == declared[key]["negative"]
            endpoints[point][key] = dict(**desc, lower=float(lo), upper=float(hi))
        descriptions[point] = {}
        for mode in ["ordinary", "within_id"]:
            descriptions[point][mode] = dict(
                id=describe([r[mode]["r2"][0] for r in rs]),
                pc=describe([r[mode]["r2"][1] for r in rs]),
                random_mean=describe([np.mean(r[mode]["r2"][2:]) for r in rs]),
                all_models=[describe([r[mode]["r2"][i] for r in rs]) for i in range(22)])
    for hypothesis, keys, thresholds in [("H1", ["A", "B"], [.02, .01]),
                                         ("H2", ["C", "D"], [.01, .005])]:
        es = [endpoints["block06"][k] for k in keys]
        outcome = ("SUPPORTED" if all(e["lower"] >= t for e, t in zip(es, thresholds)) else
                   "REFUTED" if any(e["upper"] < t for e, t in zip(es, thresholds)) else "INCONCLUSIVE")
        assert outcome == verified["scientific_adjudication"][hypothesis]
    previous = disk.read(RESEARCH/"exploratory/p2_lexical_v1/results.json")["records"]
    comparison = []
    for point in POINTS:
        for family in ["original", "ID"]:
            rec = next(r for r in previous if r["point"] == point and
                       r["condition"] == "C1" and r["family"] == family)
            comparison.append(dict(dataset="P2 known articles", point=point, family=family,
                **{k: rec["metrics"][k]["median"] for k in
                   ["baseline_r2", "pc_r2", "random_r2", "pc_gain", "random_advantage"]}))
        d = descriptions[point]["ordinary"]
        comparison.append(dict(dataset="P3 fresh articles", point=point, family="ID",
            baseline_r2=d["id"]["median"], pc_r2=d["pc"]["median"],
            random_r2=d["random_mean"]["median"], pc_gain=endpoints[point]["A"]["median"],
            random_advantage=endpoints[point]["B"]["median"]))
    totals = {k: sum(r[k] for r in coverage) for k in [
        "repeated_groups", "repeated_tokens", "singleton_tokens", "unseen_id_tokens", "unseen_id_within_tokens"]}
    assert totals["repeated_tokens"]+totals["singleton_tokens"] == 64*511
    aggregate = dict(run_id=run["run_id"], scope="P3 preregistered evaluation plus declared secondary summaries",
        primary_endpoints=endpoints["block06"], secondary_endpoints=endpoints["final_ln"],
        adjudication=verified["scientific_adjudication"], descriptions=descriptions,
        comparison=comparison, article_metrics=rows, coverage=coverage,
        coverage_totals=totals, coverage_descriptions={
            k: describe([r[k] for r in coverage]) for k in totals}, new_model_forwards=0)
    disk.write(HERE/"aggregates.json", aggregate)
    audit = dict(status="PASS", checked_at=disk.stamp(), runtime_python=sys.version,
        numpy=np.__version__, audit_seconds=time.monotonic()-start, maximum_absolute_differences=errors,
        fixed_predictions_checked=compared, article_points_checked=len(rows),
        archive_members_checked=len(index["entries"]), environments_matched=len(envs),
        forward_events=len(starts), commit_events=len(commits), no_op_added_forwards=0,
        release_precedes_run=True, fixed_model_reference_precedes_all_forwards=True,
        fixed_coefficients_unchanged=True, new_training_performed=False, old_article_id_overlap=0,
        ordinary_tokens=64*511, within_tokens=totals["repeated_tokens"],
        primary_endpoint_missing={k:64-v["n"] for k,v in endpoints["block06"].items()},
        gpu=dict(name=envs[0]["gpu_name"], vram_bytes=envs[0]["gpu_total_bytes"],
                 host_ram_bytes=envs[0]["host_ram"]["MemTotal"]),
        wall_seconds_through_summary=summary["wall_seconds_through_summary"],
        release_frozen_at=release["frozen_at"], run_created_at=run["created"],
        fit_reference_sealed_at=fixed["sealed_at"], first_forward_at=min(e["at"] for e in starts),
        last_commit_at=max(e["at"] for e in commits), archive_created_at=index["created"],
        archive_seconds_since_run_creation=(datetime.fromisoformat(index["created"])-
                                           datetime.fromisoformat(run["created"])).total_seconds(),
        new_language_model_forwards=0, independent_reviewer=False,
        note="Different arithmetic paths for fixed readouts and grouped sums; same agent self-audit")
    disk.write(RECEIPT/"extended_verification.json", audit)
    print(disk.canonical(audit))
    print(disk.canonical(dict(coverage_totals=totals, descriptions=descriptions,
                             comparison=comparison)))


if __name__ == "__main__":
    main()
