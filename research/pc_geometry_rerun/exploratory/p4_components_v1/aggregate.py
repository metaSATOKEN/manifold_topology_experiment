"""Aggregate every frozen contrast and independently audit saved prediction scores."""
from pathlib import Path
import json
import time
import numpy as np
import numeric as n
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_io as disk
import p4_numeric as rawio


def direct_metrics(y, predictions, w):
    den = np.dot(w, y * y)
    losses = {k: np.array([np.dot(w, (y - v[:, j]) ** 2) for j in range(v.shape[1])])
              for k, v in predictions.items()}
    metrics = {}
    for f in n.FAMILIES:
        base = "N" if f in ["NR", "NU"] else "P" if f == "PU" else "B"
        metrics[f + "_gain"] = float((losses[base][0] - losses[f][0]) / den)
        metrics[f + "_adv"] = float((losses[f][1:].mean() - losses[f][0]) / den)
    metrics.update(A8_minus_A1=float((losses["A1"][0] - losses["A8"][0]) / den),
                   A8_minus_A7=float((losses["A7"][0] - losses["A8"][0]) / den),
                   B_minus_P=float((losses["P"][0] - losses["B"][0]) / den),
                   N_minus_B=float((losses["B"][0] - losses["N"][0]) / den),
                   NR_minus_NU=float((losses["NU"][0] - losses["NR"][0]) / den))
    metrics["NR_gain_minus_A8_gain"] = metrics["NR_gain"] - metrics["A8_gain"]
    return metrics, losses


def independent_bootstrap(values, stat="median"):
    a = np.asarray(values)
    rng = np.random.default_rng(20261029)
    draws = a[rng.integers(0, len(a), size=(20000, len(a)))]
    if stat == "median":
        draws.sort(axis=1)
        m = (draws[:, 31] + draws[:, 32]) / 2  # all fixed endpoints have 64 articles
    else:
        m = draws.sum(1) / len(a)
    return np.quantile(m, [.025, .975])


start = time.monotonic()
assert disk.read(OUT / "execution_result.json")["status"] == "PASS"
summary, points_audit, allrows = {}, {}, []
counts = {"article_points": 0, "metric_checks": 0, "prediction_values": 0,
          "bootstrap_checks": 0, "stratum_checks": 0}
for key in ["gpt2", "opt", "qwen"]:
    summary[key] = {}
    for point in ["middle", "final_norm"]:
        folder = OUT / "results" / key / point
        for name, sha in disk.read(folder / "COMMIT.json").items():
            assert disk.digest(folder / name) == sha
        rows = disk.read(folder / "articles.json")
        assert len(rows) == 64
        saved = rawio.load_npz(folder / "predictions.npz")
        names = ["P", "B", "N", *n.FAMILIES]
        counts["prediction_values"] += sum(saved[k].size for k in names)
        for i, row in enumerate(rows):
            mask = saved["article_index"] == i
            y, ids = saved["y"][mask], saved["ids"][mask]
            preds = {k: saved[k][mask] for k in names}
            assert len(y) == row["tokens"]
            assert len(np.unique(ids)) == row["groups"]
            # Stored group weights and zero group means checked by direct loops.
            for token in np.unique(ids):
                group = ids == token
                np.testing.assert_allclose(saved["weights"][mask][group], 1 / group.sum(), atol=1e-15)
                assert abs(y[group].mean()) < 1e-10
            for weighted, expected_key in [(False, "metrics"), (True, "equal_ID_metrics")]:
                w = saved["weights"][mask] if weighted else np.ones(len(y))
                m, losses = direct_metrics(y, preds, w)
                assert set(m) == set(row[expected_key])
                for name, value in m.items():
                    assert abs(value - row[expected_key][name]) < 1e-10
                    counts["metric_checks"] += 1
            errs = {k: (y[:, None] - v) ** 2 for k, v in preds.items()}
            definitions = [
                ("B_to_A1", errs["B"][:, 0] - errs["A1"][:, 0]),
                ("B_to_A7", errs["B"][:, 0] - errs["A7"][:, 0]),
                ("B_to_A8", errs["B"][:, 0] - errs["A8"][:, 0]),
                ("B_to_N", errs["B"][:, 0] - errs["N"][:, 0]),
                ("N_to_NR", errs["N"][:, 0] - errs["NR"][:, 0]),
                ("N_to_NU", errs["N"][:, 0] - errs["NU"][:, 0]),
                ("A8_adv", errs["A8"][:, 1:].sum(1) / 20 - errs["A8"][:, 0]),
                ("NU_adv", errs["NU"][:, 1:].sum(1) / 20 - errs["NU"][:, 0]),
            ]
            for partition, strata in row["bias"].items():
                cats = saved[partition][mask]
                assert sum(v["tokens"] for v in strata) == len(y)
                for j, v in enumerate(strata):
                    selection = cats == j
                    assert selection.sum() == v["tokens"]
                    for name, token_delta in definitions:
                        expect = token_delta[selection].sum() / (y @ y)
                        assert abs(expect - v["contributions"][name]) < 1e-10
                        counts["stratum_checks"] += 1
            counts["article_points"] += 1
        stats = {}
        for column in ["metrics", "equal_ID_metrics"]:
            stats[column] = {}
            for name in rows[0][column]:
                values = [r[column][name] for r in rows]
                s = n.describe(values)
                lo, hi = independent_bootstrap(values)
                np.testing.assert_allclose([s["lo"], s["hi"]], [lo, hi], atol=1e-12)
                stats[column][name] = s
                counts["bootstrap_checks"] += 1
        stats["random_replicates"] = {}
        stats["r2_levels"] = {}
        for family in n.FAMILIES:
            base = "N" if family in ["NR", "NU"] else "P" if family == "PU" else "B"
            stats["random_replicates"][family] = [
                n.describe([r["r2"][family][j] - r["r2"][base][0] for r in rows])
                for j in range(1, 21)]
            stats["r2_levels"][family] = n.describe([r["r2"][family][0] for r in rows])
        for name in ["P", "B", "N"]:
            stats["r2_levels"][name] = n.describe([r["r2"][name][0] for r in rows])
        stats["bias"] = {}
        for partition, labels in n.STRATA.items():
            stats["bias"][partition] = []
            for j, label in enumerate(labels):
                nr = [r["bias"][partition][j]["tokens"] for r in rows]
                contributions = {}
                for contrast in n.CONTRASTS:
                    values = [r["bias"][partition][j]["contributions"][contrast] for r in rows]
                    s = n.describe(values, mean=True)
                    np.testing.assert_allclose([s["lo"], s["hi"]], independent_bootstrap(values, "mean"), atol=1e-12)
                    counts["bootstrap_checks"] += 1
                    contributions[contrast] = s
                stats["bias"][partition].append(dict(
                    label=label, total_tokens=sum(nr), articles_present=sum(v > 0 for v in nr),
                    token_fraction=sum(nr) / sum(r["tokens"] for r in rows),
                    contributions=contributions))
        stats["coverage"] = dict(articles=64, tokens=sum(r["tokens"] for r in rows),
                                 groups=sum(r["groups"] for r in rows), missing=0)
        stats["cv"] = disk.read(folder / "cv.json")
        summary[key][point] = stats
        points_audit[key + "/" + point] = disk.read(folder / "audit.json")
        allrows.extend(rows)
        print("Aggregate and independent audit PASS:", key, point, flush=True)
disk.write(OUT / "summary.json", dict(status="PASS", results=summary, article_points=384,
                                    confidence_interpretation="exploratory unadjusted descriptive 95%; P4 unchanged"))
disk.write(OUT / "article_metrics.json", allrows)
disk.write(OUT / "verification.json", dict(
    status="PASS", at=disk.stamp(), counts=counts, point_audits=points_audit,
    aggregation_seconds=time.monotonic() - start,
    analysis_seconds=disk.read(OUT / "execution_result.json")["seconds"],
    all_models_and_points_reported=True, source_P4_results_unchanged=True,
    new_language_model_forwards=0, independent_reviewer=False))
assert time.monotonic() - start + disk.read(OUT / "execution_result.json")["seconds"] < 3600
print(counts, flush=True)
