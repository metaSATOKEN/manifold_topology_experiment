"""Independent article-level aggregation and execution metadata audit.

Uses frozen per-article metrics, separately verified against saved observations.
No fitted coefficients, thresholds, features, or inclusion rules are changed.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys
import numpy as np

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parents[1]
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_io as disk

RAW = OUT / "raw"
spec = disk.spec()
fixtures = disk.fixtures()
test = [f for f in fixtures if f["split"] == "test"]
saved = disk.read(RAW / "summary.json")
checked = 0


def describe(values, confidence=.95):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if not len(a):
        return dict(n=0, missing=len(values), median=None, lo=None, hi=None, positive=0)
    rng = np.random.default_rng(20261028)
    medians = []
    for _ in range(100):  # Same fixed 20,000 draws, bounded independent batches.
        ix = rng.integers(0, len(a), size=(200, len(a)))
        ordered = np.sort(a[ix], axis=1)
        j = len(a) // 2
        medians.append((ordered[:, j] + ordered[:, j - 1]) / 2
                       if len(a) % 2 == 0 else ordered[:, j])
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(np.concatenate(medians), [tail, 1 - tail])
    return dict(n=len(a), missing=len(values) - len(a), median=float(np.median(a)),
                lo=float(lo), hi=float(hi), positive=int((a > 0).sum()), confidence=confidence)


def compare(a, b):
    global checked
    assert set(a) == set(b)
    for k in a:
        if a[k] is None:
            assert b[k] is None
        else:
            assert abs(a[k] - b[k]) < 1e-10, (k, a[k], b[k])
    checked += 1


articles = []
stats = {}
coverage = {}
cv = {}
for key in spec["models"]:
    stats[key] = {}
    cv[key] = {}
    for point in spec["points"]:
        rows = [disk.read(RAW / "evaluation" / key / point / f["id"] / "metrics.json")
                for f in test]
        articles.extend(rows)
        expected = saved["scientific_results"][key][point]
        table = {}
        r2 = {}
        for mode, names in [("ordinary", ["O", "R"]), ("within", ["O", "R", "W"])]:
            table[mode] = {}
            r2[mode] = {}
            for name in names:
                table[mode][name] = {}
                for metric in ["pc_gain", "advantage", "random_mean_gain"]:
                    confidence = 1 - .05 / 12 if (mode == "within" and name == "W"
                                                  and metric != "random_mean_gain") else .95
                    s = describe([r[mode][name].get(metric)
                                  if r[mode][name]["valid"] else None for r in rows], confidence)
                    compare(s, expected["metrics"][mode][name][metric])
                    table[mode][name][metric] = s
                r2[mode][name] = {
                    n: describe([float(np.mean(np.asarray(r[mode][name]["r2"])[ix]))
                                 if r[mode][name]["valid"] else None for r in rows])
                    for n, ix in [("baseline", [0]), ("PC", [1]), ("random_mean", list(range(2, 22)))]
                }
        geo = {}
        for c in ["C0", "C1"]:
            geo[c] = {}
            for q in ["R1", "V1", "V8", "relative_gap12"]:
                s = describe([r["geometry"][c].get(q) for r in rows])
                compare(s, expected["geometry"][c][q])
                geo[c][q] = s
        deltas = {q: describe([r["geometry"]["C1"][q] - r["geometry"]["C0"][q]
                              for r in rows]) for q in ["R1", "V1", "V8"]}
        for q, s in deltas.items():
            compare(s, expected["geometry_C1_minus_C0"][q])
        changes = {a + "_to_" + b:
                   describe([r["within"][b]["advantage"] - r["within"][a]["advantage"]
                             for r in rows])
                   for a, b in [("O", "R"), ("R", "W"), ("O", "W")]}
        for q, s in changes.items():
            compare(s, expected["within_advantage_changes"][q])
        capture = {q: describe([r["shared_top8_capture"][q] for r in rows])
                   for q in ["top8_total_fraction", "top8_within_fraction"]}
        for q, s in capture.items():
            compare(s, expected["shared_top8_capture"][q])
        g, a = table["within"]["W"]["pc_gain"], table["within"]["W"]["advantage"]
        decision = ("NOT_ADJUDICABLE" if not saved["complete"] or min(g["n"], a["n"]) < 58
                    else "SUPPORTS_MINIMUM_CRITERION" if g["lo"] > .01 and a["lo"] > .005
                    else "EVIDENCE_BELOW_MINIMUM_CRITERION" if g["hi"] <= .01 or a["hi"] <= .005
                    else "INCONCLUSIVE")
        assert decision == expected["decision"]
        stats[key][point] = dict(metrics=table, geometry=geo,
                                geometry_C1_minus_C0=deltas, within_advantage_changes=changes,
                                shared_top8_capture=capture, decision=decision)
        stats[key][point]["descriptive_r2"] = r2
        stats[key][point]["within_W_advantage_95_descriptive"] = describe(
            [r["within"]["W"]["advantage"] for r in rows])
        stats[key][point]["W_random_replicates"] = [
            dict(replicate=j + 1,
                 gain=describe([r["within"]["W"]["r2"][j + 2] - r["within"]["W"]["r2"][0]
                                for r in rows]),
                 pc_minus_random=describe([r["within"]["W"]["r2"][1] - r["within"]["W"]["r2"][j + 2]
                                          for r in rows]))
            for j in range(20)]
        c = disk.read(RAW / "fit" / key / point / "cv.json")
        cv[key][point] = {
            name: dict(base=c[name]["selected_base"], gammas=c[name]["selected_gammas"],
                       base_alpha_at_lower_boundary=c[name]["selected_base"]["alpha"] == .0001,
                       gamma_lower_count=c[name]["selected_gammas"].count(.0001),
                       gamma_upper_count=c[name]["selected_gammas"].count(1.))
            for name in ["O", "W"]
        }
        cv[key][point].update(train_tokens=c["train_tokens"],
                              repeated_train_tokens=c["repeated_train_tokens"])
        if point == "middle":
            coverage[key] = dict(
                test_articles=len(rows),
                tokens=sum(r["coverage"]["tokens"] for r in rows),
                repeated_tokens=sum(r["coverage"]["repeated_tokens"] for r in rows),
                repeated_groups=sum(r["coverage"]["repeated_groups"] for r in rows),
                minimum_repeated_tokens=min(r["coverage"]["repeated_tokens"] for r in rows),
                minimum_repeated_groups=min(r["coverage"]["repeated_groups"] for r in rows),
                training_ID_known_fraction=describe(
                    [r["coverage"]["training_ID_known_fraction"] for r in rows]),
            )
        print("Independent aggregation passed:", key, point, decision, flush=True)

ev = disk.events(RAW)
counter = Counter(e["kind"] for e in ev)
assert counter["FORWARD_STARTED"] == counter["DOCUMENT_COMMITTED"] == 384
assert counter["WORKER_FAILURE"] == 0 and counter["NOOP"] == 3
assert all(e["forward_calls"] == 0 for e in ev if e["kind"] == "NOOP")
env = disk.read(RAW / "environment_contract.json")
assert {p: dict(env["packages"])[p] for p in spec["pins"]} == spec["pins"]
environments = list((RAW / "attempts").glob("*/environment.json"))
assert len(environments) == 6
for path in environments:
    a = disk.read(path)
    assert all(a[k] == env[k] for k in env), path
    assert a["gpu_free_bytes"] >= spec["minimum_gpu_free_bytes"]
    assert a["host_ram"]["MemAvailable"] >= spec["minimum_host_available_bytes"]
assert env["dtype"] == "float32" and env["attention"] == "eager" and env["tf32"] is False
resources = {}
checks = []
for key in spec["models"]:
    m = disk.model(key)
    assert disk.read(RAW / "models" / key / "files.json") == m["files"]
    c = disk.read(RAW / "models" / key / "loaded_model_config.json")
    assert c.get("hidden_size", c.get("n_embd")) == m["hidden"]
    assert c.get("num_hidden_layers", c.get("n_layer")) == m["layers"]
    assert c["vocab_size"] == m["vocab"] and c["dtype"] == "float32"
    docs = [disk.read(RAW / "docs" / disk.doc_key(key, f["id"]) / "doc.json")
            for f in fixtures]
    checks.extend(d["extraction_checks"] for d in docs)
    resources[key] = dict(peak_allocated_bytes=max(d["gpu_peak_allocated_bytes"] for d in docs),
                          forward_seconds=sum(d["forward_seconds"] for d in docs))
seal = disk.read(RAW / "fit/COMMIT.json")["sealed_at"]
last_train = max(e["at"] for e in ev if e["kind"] == "DOCUMENT_COMMITTED" and e["split"] == "train")
first_test = min(e["at"] for e in ev if e["kind"] == "FORWARD_STARTED" and e["split"] == "test")
assert last_train < seal < first_test
release = disk.read(BASE / "p4_v1/release_manifest.json")
run = disk.read(RAW / "run.json")
freeze = release.get("frozen_at", release.get("created"))
assert freeze is not None and datetime.fromisoformat(freeze) < datetime.fromisoformat(run["created"])
time_record = disk.read(RAW / "active_time.json")
assert time_record["sessions"] == 1 and time_record["seconds"] < spec["total_seconds"]
assert not time_record["active"]
assert not list(RAW.rglob("error.json"))
assert len({f["id"] for f in fixtures}) == 128
old = disk.read(BASE / "p4_v1/input_manifest.json")["old_article_ids"]
assert not set(old).intersection(f["id"] for f in fixtures)

execution = dict(
    status="PASS", events=dict(counter), model_file_manifests_match=True,
    six_environment_contracts_match=True, session_count=1, technical_resumes=0,
    package_pins_match=True, frozen_at=freeze, run_created=run["created"],
    last_training_commit=last_train, fit_sealed_at=seal, first_test_forward=first_test,
    training_before_seal_before_test=True, resources=resources,
    gpu_name=env["gpu_name"], gpu_total_bytes=disk.read(environments[0])["gpu_total_bytes"],
    host_total_bytes=disk.read(environments[0])["host_ram"]["MemTotal"],
    seconds_through_summary=saved["cumulative_seconds_through_summary"],
    recorded_active_seconds=time_record["seconds"],
    excludes_final_zip_and_download_time=True,
    maximum_recorded_normalization_error=max(c["normalization_max_relative_error"] for c in checks),
    maximum_recorded_surprisal_ce_error=max(c["surprisal_ce_max_abs"] for c in checks),
    limitation="Weight files/pre-normalization tensors not in ZIP; validate saved hashes and extraction diagnostics, not a new model forward.",
)
disk.write(OUT / "analysis/article_metrics.json", articles)
disk.write(OUT / "analysis/aggregates.json", dict(
    status="PASS", fixed_summaries_compared=checked, independent_batched_bootstrap=True,
    scientific_results=stats, coverage=coverage, cv=cv, execution=execution,
    decision_counts=dict(Counter(v["decision"] for points in stats.values() for v in points.values())),
    commonality_across_all_three_models_and_both_points=False,
    extra_descriptives="R² levels, 95% W plotting intervals and per-random-basis summaries are descriptive only; primary decision unchanged.",
))
print(execution, flush=True)
