"""P2 receipt aggregation; read-only saved observations, no model execution.

Run after the frozen p2_verify.py receipt command passes, with NumPy 2.2.6.
The preregistered endpoints are copied from the verified result. Other summaries
describe the preregistered secondary outputs; no additional hypothesis is tested.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

RECEIPT = Path(__file__).resolve().parents[1]
BASE = RECEIPT.parents[1]
RAW = RECEIPT / "raw"
sys.path.insert(0, str(BASE / "p2_v1"))
import p2_io as disk
from p2_numeric import compare_tree
from p2_worker import require_compatible

POINTS = ["block06", "final_ln"]
CONDITIONS = ["C0", "C1", "C4"]
FEATURES = ["position", "token_chars", "surprisal", "norm"]


def read(path):
    return json.loads(path.read_text())


def stats(values):
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    assert np.isfinite(vals).all()
    if not len(vals):
        return dict(n=0, min=None, q1=None, median=None, q3=None, max=None,
                    positive=0, negative=0, zero=0)
    return dict(n=len(vals), **dict(zip(
        ["min", "q1", "median", "q3", "max"],
        map(float, np.quantile(vals, [0, .25, .5, .75, 1])))),
        positive=int((vals > 0).sum()), negative=int((vals < 0).sum()),
        zero=int((vals == 0).sum()))


def main():
    verified = read(RECEIPT / "receipt_verification.json")
    assert verified["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    declared = read(RAW / "summary.json")
    compare_tree(declared["endpoints"], verified["endpoints"])
    fixtures = disk.fixtures()
    train = [d["id"] for d in fixtures if d["split"] == "train"]
    test = [d["id"] for d in fixtures if d["split"] == "test"]
    assert len(train) == len(test) == 64 and not set(train) & set(test)
    rows, feature_rows, fits, summaries = [], [], {}, {}
    for point in POINTS:
        fits[point], summaries[point] = {}, {}
        for c in CONDITIONS:
            folder = RAW / "fit" / (point + "__" + c)
            meta = read(folder / "meta.json")
            assert meta["valid"]
            with np.load(folder / "arrays.npz", allow_pickle=False) as a:
                ratios = a["eigenvalues"] / a["eigenvalues"].sum()
                fits[point][c] = dict(
                    selected_pc=meta["position_selection"][0],
                    random_selections=meta["position_selection"][1:],
                    training_rows=meta["training_rows"], rank=meta["rank"],
                    training_pc_variance_fractions=ratios[:8].tolist(),
                    training_top8_variance_fraction=float(ratios[:8].sum()),
                    target_rows={h: m["rows"] for h, m in meta["heads"].items()})
            current = []
            for doc in test:
                m = read(RAW / "evaluation" / doc / (point + "__" + c) / "metrics.json")
                assert m["valid"] and all(h["valid"] for h in m["heads"].values())
                assert m["n_tokens"] == (511 if c == "C1" else 512)
                assert m["heads"]["surprisal"]["n"] == 511
                row = dict(document_id=doc, point=point, condition=c,
                    selected_pc=fits[point][c]["selected_pc"]["component"],
                    position_r=m["position_r"][0],
                    random_position_r=float(np.mean(m["position_r"][1:])),
                    position_advantage=m["position_random_advantage"],
                    pc1_variance_fraction=m["direction_variance_fractions"][0],
                    top8_variance_fraction=m["subspace_variance_fraction"])
                assert np.isclose(sum(m["direction_variance_fractions"]),
                                  row["top8_variance_fraction"])
                for head, values in m["heads"].items():
                    row.update({
                        head + "_baseline_r2": values["r2"][0],
                        head + "_pca_r2": values["r2"][1],
                        head + "_random_r2": float(np.mean(values["r2"][2:])),
                        head + "_delta_r2": values["delta_r2"],
                        head + "_random_advantage": values["random_advantage"]})
                current.append(row)
                rows.append(row)
                for feature in FEATURES:
                    for pc, v in enumerate(m["correlations"][feature], 1):
                        feature_rows.append(dict(document_id=doc, point=point,
                            condition=c, feature=feature, pc=pc, r=v,
                            abs_r=abs(v) if v is not None else None))
            numeric = [k for k in current[0] if k not in
                       ["document_id", "point", "condition", "selected_pc"]]
            summaries[point][c] = {k: stats(row[k] for row in current) for k in numeric}
    assert len(rows) == 384 and len(feature_rows) == 64 * 2 * 3 * 4 * 8
    feature_summary = {point: {c: {f: {
        str(pc): dict(
            signed=stats(row["r"] for row in feature_rows if
                         (row["point"], row["condition"], row["feature"], row["pc"]) ==
                         (point, c, f, pc)),
            absolute=stats(row["abs_r"] for row in feature_rows if
                           (row["point"], row["condition"], row["feature"], row["pc"]) ==
                           (point, c, f, pc)))
        for pc in range(1, 9)} for f in FEATURES} for c in CONDITIONS} for point in POINTS}

    events = disk.events(RAW)
    all_docs = train + test
    for kind in ["FORWARD_STARTED", "DOCUMENT_COMMITTED"]:
        counts = Counter(e["document_id"] for e in events if e["kind"] == kind)
        assert counts == {d: 1 for d in all_docs}
    seal = read(RAW / "fit" / "COMMIT.json")
    fit_hash = disk.digest(RAW / "fit" / "COMMIT.json")
    seal_events = [e for e in events if e["kind"] == "FIT_SEALED"]
    assert len(seal_events) == 1 and seal_events[0]["fit_commit_sha256"] == fit_hash
    assert seal_events[0]["test_documents_read"] == 0
    starts = {e["document_id"]: e for e in events if e["kind"] == "FORWARD_STARTED"}
    commits = {e["document_id"]: e for e in events if e["kind"] == "DOCUMENT_COMMITTED"}
    last_train = max(commits[d]["at"] for d in train)
    first_test = min(starts[d]["at"] for d in test)
    assert last_train < seal["sealed_at"] < first_test
    assert all(starts[d]["fit_commit_sha256"] == fit_hash for d in test)
    assert all(starts[d]["fit_commit_sha256"] is None for d in train)
    released = read(BASE / "p2_v1" / "release_manifest.json")["frozen_at"]
    created = read(RAW / "run.json")["created"]
    assert datetime.fromisoformat(released) < datetime.fromisoformat(created)
    envs = [read(f) for f in sorted((RAW / "attempts").glob("*/environment.json"))]
    assert len(envs) == 4
    for env in envs:
        require_compatible(read(RAW / "environment_contract.json"), env)
    config = read(RAW / "loaded_model_config.json")
    semantic = lambda d: {k: v for k, v in d.items() if k != "_name_or_path"}
    config_paths = list((RAW / "attempts").glob("*/loaded_model_config.json"))
    assert len(config_paths) == 4 and all(
        semantic(read(f)) == semantic(config) for f in config_paths)
    p1_model_files = BASE / "receipts/98a3faf68004_p1_v1_final/raw/model_files.json"
    assert read(RAW / "model_files.json") == read(p1_model_files)
    docmeta = [read(RAW / "docs" / d / "doc.json") for d in all_docs]
    operations = dict(
        event_counts=dict(Counter(e["kind"] for e in events)),
        every_document_forward_and_commit_once=True,
        environment_records_matched=len(envs), model_configurations_matched=4,
        model_file_hashes_match_p1=True, fit_commit_sha256=fit_hash,
        release_frozen_at=released, run_created_at=created,
        last_training_commit_at=last_train, fit_sealed_at=seal["sealed_at"],
        first_test_forward_at=first_test, fit_sealed_before_all_test_forwards=True,
        gpu_name=envs[0]["gpu_name"], gpu_total_bytes=envs[0]["gpu_total_bytes"],
        host_ram_total_bytes=envs[0]["host_ram"]["MemTotal"],
        peak_torch_allocated_bytes_max=max(d["gpu_peak_allocated_bytes"] for d in docmeta),
        forward_seconds=stats(d["forward_seconds"] for d in docmeta),
        wall_seconds_through_summary=declared["wall_seconds_through_summary"],
        no_op_added_forwards=declared["no_op_added_forwards"],
        failure_files=[str(f.relative_to(RAW)) for f in RAW.rglob("*failure*")],
        preln_to_final_max_abs=max(d["extraction_checks"]["preln_to_final_max_abs"] for d in docmeta),
        surprisal_ce_max_abs=max(d["extraction_checks"]["surprisal_ce_max_abs"] for d in docmeta),
        local_model_forwards=0)
    assert not operations["failure_files"]
    assert len(verified["missing"]) == 0
    result = dict(run_id=verified["run_id"], scope="Saved-observation descriptive aggregation",
        numpy=np.__version__, python=platform.python_version(),
        source_script_sha256=disk.digest(Path(__file__)),
        receipt_verification_sha256=disk.digest(RECEIPT / "receipt_verification.json"),
        primary_endpoints=verified["endpoints"],
        adjudication=verified["scientific_adjudication"],
        fits=fits, summaries=summaries, features=feature_summary,
        operations=operations, metrics=rows, feature_rows=feature_rows,
        undefined_feature_correlations=sum(row["r"] is None for row in feature_rows),
        secondary_notice="Secondary descriptive medians/IQRs, not additional confirmatory tests. "
                         "Random controls averaged within article. No selecting PCs on test.")
    target = RECEIPT / "analysis" / "aggregates.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(operations=operations, primary=summaries["block06"]["C1"],
        fits=fits, undefined_feature_correlations=result["undefined_feature_correlations"]),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
