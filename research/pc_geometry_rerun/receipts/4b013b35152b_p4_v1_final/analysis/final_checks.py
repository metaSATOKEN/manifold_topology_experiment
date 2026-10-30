"""Final receipt integrity, report numbers/links, and historical preservation."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
PROJECT = ROOT.parents[3]
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_verify
read = lambda p: json.loads(p.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


before = read(ROOT / "preservation_before.json")
ops = read(ROOT / "operations_before.json")
allowed = set(ops)
preserved = {p: h for p, h in before["sha256"].items() if p not in allowed}
preserved[before["previous_anchor"]] = before["previous_anchor_sha256"]
for p, h in preserved.items():
    assert digest(PROJECT / p) == h, "Historical file changed: " + p
contract = lambda text: text[text.index("## 2."):text.index("## 4.")]
req = "research/pc_geometry_rerun/REQUIREMENTS.md"
assert contract((PROJECT / req).read_text()) == contract(ops[req])
ledger = "research/pc_geometry_rerun/audits/ledger_2026-10.md"
assert (PROJECT / ledger).read_text().startswith(ops[ledger])
assert "\n## 18." in (PROJECT / ledger).read_text()[len(ops[ledger]):]
receipt = read(ROOT / "archive_receipt.json")
for row in receipt["archives"]:
    for p in [PROJECT / row["source"], ROOT / "original_archives" / row["name"]]:
        assert p.stat().st_size == row["bytes"] and digest(p) == row["sha256"]
bundle = read(ROOT / "bundle_index.json")
raw_paths = {str(p.relative_to(ROOT / "raw")) for p in (ROOT / "raw").rglob("*") if p.is_file()}
assert raw_paths == set(bundle["entries"])
for n, entry in bundle["entries"].items():
    p = ROOT / "raw" / n
    assert p.stat().st_size == entry["bytes"] and digest(p) == entry["sha256"], n
recomputed = read(ROOT / "recomputed_summary.json")
execution = read(ROOT / "verification_execution.json")
ag = read(ROOT / "analysis/aggregates.json")
assert recomputed["audit"]["status"] == execution["status"] == ag["status"] == "PASS"
assert recomputed["complete"] and recomputed["committed_model_documents"] == 384
assert recomputed["analyzed_evaluation_article_points"] == 384
assert recomputed["audit"]["predictions_reconstructed"] == 12479148
assert ag["fixed_summaries_compared"] == 186
assert ag["decision_counts"] == {"SUPPORTS_MINIMUM_CRITERION": 4, "INCONCLUSIVE": 2}
assert all(v["metrics"]["within"]["W"][q]["n"] == 64
           for ps in recomputed["scientific_results"].values() for v in ps.values()
           for q in ["pc_gain", "advantage"])
originals = [ROOT / "raw/summary.json", *sorted((ROOT / "raw/sessions").glob("*/verified.json"))]
assert len(originals) == 2
for p in originals:
    p4_verify.near(read(p)["scientific_results"], recomputed["scientific_results"])

report_path = ROOT / "ANALYSIS_REPORT_JA.md"
supplement = BASE / "prereg/EXP-0004_v1_results_2026-10-20.md"
handoff = BASE / "HANDOFF/2026-10-20_zzzzzzzzzzzzz_p4_results.md"
anchor = BASE / "evidence/p4_final_checks_2026-10-20.json"
report, result = report_path.read_text(), supplement.read_text()
for ps in recomputed["scientific_results"].values():
    for v in ps.values():
        for q in ["pc_gain", "advantage"]:
            for n in ["median", "lo", "hi"]:
                token = f'{v["metrics"]["within"]["W"][q][n]:.6f}'
                assert token in report and token in result, token
detail = (ROOT / "analysis/DETAIL_TABLES.md").read_text()
assert detail.count("| wt2_train_") == 384
rows = read(ROOT / "analysis/article_metrics.json")
assert len(rows) == 384
for r in rows:
    prefix = "| " + " | ".join([r["document_id"],
                                *[f'{r["within"][m][q]:.6f}' for m in ["O", "R", "W"]
                                  for q in ["pc_gain", "advantage"]]]) + " |"
    assert prefix in detail, r["document_id"]
figures = sorted((ROOT / "figures").glob("*.png"))
assert len(figures) == 4
for p in figures:
    assert p.with_suffix(".svg").exists()
qa = dict(status="PASS", checked_by="same agent, direct image viewing",
          checks="All four PNGs viewed: labels, legends, intervals, axes and footnotes legible; no clipping or data-obscuring overlaps.",
          figures={p.name: dict(sha256=digest(p), svg_sha256=digest(p.with_suffix(".svg")))
                   for p in figures})
write(ROOT / "analysis/figure_review.json", qa)
for p in (ROOT / "analysis").glob("*.py"):
    ast.parse(p.read_text())
links = 0
for doc in [report_path, supplement, handoff]:
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        dest = (doc.parent / target.split("#")[0]).resolve()
        assert dest.exists() or dest in [anchor, ROOT / "receipt_verification.json"], (doc, target)
        links += 1
baseline = subprocess.run([sys.executable, str(BASE / "check_baseline.py")],
                          cwd=PROJECT, text=True, capture_output=True, check=True)
baseline_result = json.loads(baseline.stdout)
assert baseline_result["status"] == "PASS"
local = dict(
    status="LOCAL_RECEIPT_AND_RECOMPUTE_PASS", local_receipt_verified=True,
    checked_at=datetime.now(timezone.utc).isoformat(),
    run_id=receipt["run_id"], snapshot_id=receipt["snapshot_id"],
    capsule=receipt["capsule"], original_archive_files=11,
    archive_bytes=receipt["total_zip_bytes"], raw_evidence_files_checked=len(raw_paths),
    all_source_archives_and_copies_identical=True, all_raw_evidence_unchanged=True,
    both_archived_scientific_summaries_agree=True, numeric_audit=recomputed["audit"],
    operational_audit=ag["execution"], decision_counts=ag["decision_counts"],
    committed_model_documents=384, evaluation_article_points=384, valid_per_primary_endpoint=64,
    fixed_bootstrap_summaries_compared=186, independent_batched_bootstrap=True,
    original_status_fields_preserved=True, new_language_model_forwards=0,
    causal_mechanism="UNRESOLVED", final_comparison_closed=True,
    same_agent_self_audit=True, independent_reviewer=False)
write(ROOT / "receipt_verification.json", local)
assets = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
assets += [PROJECT / row["source"] for row in receipt["archives"]]
assets += [PROJECT / p for p in allowed] + [supplement, handoff]
hashes = {str(p.relative_to(PROJECT)): digest(p) for p in sorted(set(assets))}
final = dict(
    status="PASS", checked_at=datetime.now(timezone.utc).isoformat(), run_id=receipt["run_id"],
    snapshot_id=receipt["snapshot_id"], receipt_status=local["status"],
    previous_anchor_files_at_start=before["files_checked"],
    preserved_files_checked=len(preserved), preservation_mismatches=[],
    allowed_operational_updates=sorted(allowed),
    requirements_sections_2_3_unchanged=True, ledger_previous_prefix_unchanged=True,
    archive_count=11, archive_bytes=receipt["total_zip_bytes"],
    raw_evidence_files_checked=len(raw_paths), numeric_audit=recomputed["audit"],
    both_archived_scientific_summaries_agree=True, fixed_bootstrap_summaries_compared=186,
    report_primary_numeric_values_match=True, full_article_table_rows=384,
    local_markdown_links_checked=links, figures_visually_checked=4,
    decision_counts=ag["decision_counts"], new_language_model_forwards=0,
    final_comparison_closed=True, causal_mechanism="UNRESOLVED",
    original_papers_unchanged=True, frozen_p4_release_unchanged=True,
    fp_counts=dict(false_positive=1, false_negative=0, controls=1, unresolved=0),
    throughline_formal_trials=0, independent_reviewer=False, baseline=baseline_result,
    preserved_sha256=preserved, artifact_sha256=hashes)
write(anchor, final)
print(json.dumps({k: v for k, v in final.items()
                  if k not in ["artifact_sha256", "preserved_sha256", "allowed_operational_updates", "baseline"]},
                 ensure_ascii=False, indent=2))
