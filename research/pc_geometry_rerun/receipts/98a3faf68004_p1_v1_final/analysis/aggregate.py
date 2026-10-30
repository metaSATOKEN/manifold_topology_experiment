"""Descriptive receipt analysis; no model loading or inference.

Run with the frozen NumPy 2.2.6 environment after p1_verify.py receipt passes.
The original ZIP, raw/ and preregistration are read-only inputs.
Post-hoc diagnostics are explicitly separated from preregistered endpoints.
"""
import collections
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

RECEIPT = Path(__file__).resolve().parents[1]
BASE = RECEIPT.parents[1]
RAW = RECEIPT / "raw"
sys.path.insert(0, str(BASE / "p1_v1"))
from p1_worker import require_compatible

POINTS = ["block01", "block06", "block12_preln", "final_ln"]
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
FEATURES = ["position", "token_chars", "surprisal"]


def read(path):
    return json.loads(path.read_text())


def stats(values):
    vals = np.asarray([x for x in values if x is not None], np.float64)
    assert np.isfinite(vals).all()
    if not len(vals):
        return dict(n=0, min=None, q1=None, median=None, q3=None, max=None)
    return dict(n=len(vals), **dict(zip(
        ["min", "q1", "median", "q3", "max"],
        map(float, np.quantile(vals, [0, .25, .5, .75, 1])))))


def main():
    verified = read(RECEIPT / "receipt_verification.json")
    assert verified["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS"
    summary = read(RAW / "summary.json")
    assert verified["endpoints"] == summary["endpoints"]
    docs = sorted(p.name for p in (RAW / "docs").iterdir() if p.is_dir())
    assert len(docs) == 64
    rows, diagnostics, feature_rows, condition_checks = [], [], [], []
    for doc in docs:
        with np.load(RAW / "docs" / doc / "raw.npz", allow_pickle=False) as raw:
            for point in POINTS:
                folder = RAW / "analysis" / doc / point
                m = read(folder / "metrics.json")
                for c in CONDITIONS:
                    if c == "C3":
                        metrics = {k: float(np.mean([r["metrics"][k] for r in m["random_controls"]]))
                                   for k in ["R1", "V1"]}
                    else:
                        assert m["conditions"][c]["valid"]
                        metrics = m["conditions"][c]["metrics"]
                        condition_checks.append(dict(document_id=doc, point=point, condition=c,
                            effective_rank=metrics["effective_rank"],
                            denominators=metrics["correlation_denominators"]))
                        for feature in FEATURES:
                            vals = metrics["correlations"][feature]
                            for pc, value in enumerate(vals, 1):
                                feature_rows.append(dict(document_id=doc, point=point,
                                    condition=c, feature=feature, pc=pc,
                                    signed_r=value, abs_r=None if value is None else abs(value)))
                    rows.append(dict(document_id=doc, point=point, condition=c,
                                     R1=metrics["R1"], V1=metrics["V1"]))
                # Additional diagnostics selected AFTER seeing the results.
                x = raw["h_" + point].astype(np.float64)
                centered = x - x.mean(0)
                norm = np.linalg.norm(x, axis=1)
                first_ss_fraction = float(np.square(centered[0]).sum() / np.square(centered).sum())
                with np.load(folder / "arrays.npz", allow_pickle=False) as arr:
                    diag = dict(
                        first_centered_ss_fraction=first_ss_fraction,
                        first_pc1_ss_fraction=float(arr["C0__scores"][0, 0] ** 2 /
                                                   np.square(arr["C0__scores"][:, 0]).sum()),
                        abs_cos_C0_PC1_C1_PC1=float(abs(arr["C0__components"][0] @ arr["C1__components"][0])),
                        abs_cos_C0_PC2_C1_PC1=float(abs(arr["C0__components"][1] @ arr["C1__components"][0])),
                        position0_norm_to_median=float(norm[0]/np.median(norm)))
                diagnostics.append(dict(document_id=doc, point=point,
                    preregistered=m["diagnostics"], fixed_basis=m["fixed_C0_basis"],
                    posthoc=diag,
                    random_controls=[dict(removed_first=0 in r["removed"],
                        repetition=r["repetition"], R1=r["metrics"]["R1"], V1=r["metrics"]["V1"])
                        for r in m["random_controls"]]))
    assert len(rows) == 64*4*5
    assert len(feature_rows) == 64*4*4*3*8
    aggregated, feature_summary, diagnostic_summary = {}, {}, {}
    for point in POINTS:
        pointrows = [r for r in rows if r["point"] == point]
        aggregated[point] = {c: {k: stats(r[k] for r in pointrows if r["condition"] == c)
                                for k in ["R1", "V1"]} for c in CONDITIONS}
        paired = {}
        for metric, left, right in [("R1","C1","C0"), ("V1","C1","C0"),
                                     ("V1","C2","C3"), ("V1","C2","C1")]:
            a = {r["document_id"]:r[metric] for r in pointrows if r["condition"]==left}
            b = {r["document_id"]:r[metric] for r in pointrows if r["condition"]==right}
            paired[f"{metric}_{left}_minus_{right}"] = stats(a[d]-b[d] for d in docs)
        aggregated[point]["paired"] = paired
        feature_summary[point] = {}
        for c in ["C0", "C1", "C2", "C4"]:
            feature_summary[point][c] = {}
            for f in FEATURES:
                selected = [r for r in feature_rows if r["point"]==point and
                            r["condition"]==c and r["feature"]==f]
                by_pc = [stats(r["abs_r"] for r in selected if r["pc"]==pc) for pc in range(1,9)]
                winners = collections.Counter()
                for doc in docs:
                    valid = [r for r in selected if r["document_id"]==doc and r["abs_r"] is not None]
                    if valid:
                        winners[str(max(valid, key=lambda r:r["abs_r"])["pc"])] += 1
                feature_summary[point][c][f] = dict(by_pc=by_pc,
                    best_pc_counts_within_article_exploratory=dict(sorted(winners.items())))
        ds = [d for d in diagnostics if d["point"]==point]
        controls = [r for d in ds for r in d["random_controls"]]
        diagnostic_summary[point] = dict(
            position0_is_max=sum(d["preregistered"]["position0_is_max"] for d in ds),
            position0_in_top6=sum(d["preregistered"]["position0_in_top"] for d in ds),
            max_to_median=stats(d["preregistered"]["max_to_median"] for d in ds),
            fixed_C0_basis_C1={k:stats(d["fixed_basis"]["C1"][k] for d in ds)
                              for k in ["R1", "direction_variance_fraction"]},
            posthoc={k:stats(d["posthoc"][k] for d in ds) for k in ds[0]["posthoc"]},
            posthoc_random_controls_by_first_removed={
                str(removed):dict(count=sum(r["removed_first"]==removed for r in controls),
                    V1=stats(r["V1"] for r in controls if r["removed_first"]==removed))
                for removed in [False, True]})
    events = [json.loads(line) for p in sorted((RAW / "attempts").glob("*/events.jsonl"))
              for line in p.read_text().splitlines()]
    event_counts = dict(collections.Counter(e["kind"] for e in events))
    for kind in ["FORWARD_STARTED", "DOCUMENT_COMMITTED"]:
        counts = collections.Counter(e["document_id"] for e in events if e["kind"]==kind)
        assert counts == {doc:1 for doc in docs}
    envs = [read(p) for p in sorted((RAW/"attempts").glob("*/environment.json"))]
    assert len(envs)==4
    for env in envs:
        require_compatible(read(RAW/"environment_contract.json"), env)
    config = read(RAW/"loaded_model_config.json")
    semantic = lambda x:{k:v for k,v in x.items() if k!="_name_or_path"}
    config_paths = sorted((RAW/"attempts").glob("*/loaded_model_config.json"))
    assert len(config_paths)==4
    assert all(semantic(read(p))==semantic(config) for p in config_paths)
    docmeta = [read(RAW/"docs"/doc/"doc.json") for doc in docs]
    assert all(m["extraction_checks"][key] for m in docmeta
               for key in ["block01_matches", "block06_matches", "final_ln_matches"])
    operations = dict(event_counts=event_counts, each_document_forward_and_commit_once=True,
        environment_contract_pass_attempts=len(envs), model_configuration_pass_attempts=4,
        gpu_name=envs[0]["gpu_name"], gpu_total_bytes=envs[0]["gpu_total_bytes"],
        host_ram_total_bytes=envs[0]["host_ram"]["MemTotal"],
        peak_torch_allocated_bytes_max=max(d["gpu_peak_allocated_bytes"] for d in docmeta),
        recorded_forward_seconds=stats(d["forward_seconds"] for d in docmeta),
        preln_to_final_max_abs=max(d["extraction_checks"]["preln_to_final_max_abs"] for d in docmeta),
        surprisal_ce_max_abs=max(d["extraction_checks"]["surprisal_ce_max_abs"] for d in docmeta),
        wall_seconds_through_summary=summary["wall_seconds_through_summary"],
        first_token_text_counts=dict(collections.Counter(d["decoded_tokens"][0] for d in docmeta)),
        failure_files=[str(p.relative_to(RAW)) for p in RAW.rglob("*failure*")],
        local_model_forwards=0)
    result = dict(run_id=verified["run_id"], scope="Saved-observation analysis only",
        python=platform.python_version(), numpy=np.__version__,
        receipt_verification_sha256=hashlib.sha256((RECEIPT/"receipt_verification.json").read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary_endpoints=verified["endpoints"], adjudication=verified["scientific_adjudication"],
        summaries=aggregated, features=feature_summary, diagnostics=diagnostic_summary,
        operations=operations, metrics=rows, feature_rows=feature_rows,
        document_diagnostics=diagnostics, condition_checks=condition_checks,
        undefined_feature_correlations=sum(r["abs_r"] is None for r in feature_rows),
        posthoc_notice="First-point sum-of-squares shares, direction cosines, random controls split by "
                      "first-point removal and first-token identity counts are post-hoc exploratory. "
                      "All other descriptive aggregation does not add confirmatory hypotheses.")
    dest = RECEIPT/"analysis"/"aggregates.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(path=str(dest),operations=operations,
        primary=diagnostic_summary["block06"],undefined_feature_correlations=result["undefined_feature_correlations"]),
        ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
