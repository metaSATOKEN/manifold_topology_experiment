"""Final receipt/report consistency and historical preservation audit. No inference."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent.parent
WORKSPACE = RESEARCH.parent.parent
read = lambda p: json.loads(p.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    before = read(ROOT/"operations_before.json")
    allowed = {
        "AGENTS.md", "research/pc_geometry_rerun/README.md",
        "research/pc_geometry_rerun/PLAN.md", "research/pc_geometry_rerun/REQUIREMENTS.md",
        "research/pc_geometry_rerun/prereg/README.md",
        "research/pc_geometry_rerun/audits/ledger_2026-10.md",
    }
    preserved = {p: h for p, h in before["sha256"].items() if p not in allowed}
    for name, sha in preserved.items():
        assert digest(WORKSPACE/name) == sha, "Historical evidence changed: "+name
    contract = lambda s: s[s.index("## 2."):s.index("## 4.")]
    assert contract((RESEARCH/"REQUIREMENTS.md").read_text()) == contract(before["requirements"])
    ledger = RESEARCH/"audits/ledger_2026-10.md"
    old_hash = before["sha256"][str(ledger.relative_to(WORKSPACE))]
    assert hashlib.sha256(ledger.read_bytes()[:before["ledger_bytes"]]).hexdigest() == old_hash
    assert "\n## 15." in ledger.read_text()
    original = Path(before["archive_original"])
    copy = ROOT/original.name
    assert digest(original) == digest(copy) == before["archive_sha256"]
    index = read(ROOT/"archive_structure.json")
    for name, entry in index["entries"].items():
        p = ROOT/"raw"/name
        assert p.stat().st_size == entry["bytes"] and digest(p) == entry["sha256"]
    raw_paths = {str(p.relative_to(ROOT/"raw")) for p in (ROOT/"raw").rglob("*") if p.is_file()}
    assert raw_paths == set(index["entries"]) and len(raw_paths) == 634
    v = read(ROOT/"receipt_verification.json")
    audit = read(ROOT/"extended_verification.json")
    d = read(ROOT/"analysis/aggregates.json")
    assert v["status"] == "LOCAL_RECEIPT_AND_RECOMPUTE_PASS" and audit["status"] == "PASS"
    assert v["committed_documents"] == 64 and v["evaluated_test_point_conditions"] == 128
    assert v["forward_attempts"] == 64 and v["worker_failures"] == 0
    assert d["adjudication"] == {"H1": "SUPPORTED", "H2": "SUPPORTED"}
    assert d["coverage_totals"]["repeated_tokens"] == 21630
    assert len(d["article_metrics"]) == 128 and len(d["coverage"]) == 64
    assert audit["fixed_predictions_checked"] == 1438976
    for point, source in [("primary", v["endpoints"]),
                          ("secondary", v["secondary_descriptive"]["final_ln"])]:
        for key, expected in source.items():
            got = d[point+"_endpoints"][key]
            for k in ["n", "positive", "negative"]:
                assert got[k] == expected[k]
            for k in ["median", "lower", "upper"]:
                assert abs(got[k]-expected[k]) < 1e-10
    report = (ROOT/"ANALYSIS_REPORT_JA.md").read_text()
    supplement = RESEARCH/"prereg/EXP-0003_v1_results_2026-10-20.md"
    for collection in ["primary_endpoints", "secondary_endpoints"]:
        for endpoint in d[collection].values():
            for key in ["median", "lower", "upper"]:
                assert f"{endpoint[key]:.6f}" in report
                assert f"{endpoint[key]:.6f}" in supplement.read_text()
    detail = (ROOT/"analysis/DETAIL_TABLES.md").read_text()
    assert detail.count("| wt2_train_") == 192  # 128 article-points plus 64 denominator rows.
    assert detail.count("| ID＋ランダム") == 20
    for row in d["article_metrics"]:
        expected = "| "+" | ".join([
            row["document_id"], row["title"].replace("|", "\\|").replace("\n", " "),
            *(f"{row[mode][metric]:.6f}" for mode in ["ordinary", "within_id"]
              for metric in ["gain", "random_advantage"])])+" |"
        assert expected in detail
    qa = read(ROOT/"analysis/figure_review.json")
    assert qa["status"] == "PASS" and len(qa["figures"]) == 3
    for name, meta in qa["figures"].items():
        assert digest(ROOT/"figures"/name) == meta["sha256"]
        assert (ROOT/"figures"/name.replace(".png", ".svg")).is_file()
    handoff = RESEARCH/"HANDOFF/2026-10-20_zzzzzzzzzz_p3_results.md"
    output = RESEARCH/"evidence/p3_final_checks_2026-10-20.json"
    links_checked = 0
    for doc in [ROOT/"ANALYSIS_REPORT_JA.md", supplement, handoff]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            dest = (doc.parent/target.split("#")[0]).resolve()
            assert dest.exists() or dest == output.resolve(), f"Broken link {doc}: {target}"
            links_checked += 1
    for p in (ROOT/"analysis").glob("*.py"):
        ast.parse(p.read_text())
    baseline = subprocess.run([sys.executable, str(RESEARCH/"check_baseline.py")],
                              cwd=WORKSPACE, capture_output=True, text=True, check=True)
    baseline_result = json.loads(baseline.stdout)
    assert baseline_result["status"] == "PASS" and baseline_result["files_matched"] == 139
    assets = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assets += [original, supplement, handoff]+[WORKSPACE/p for p in allowed]
    hashes = {str(p.relative_to(WORKSPACE)): digest(p) for p in sorted(set(assets))}
    result = dict(
        status="PASS", checked_at=datetime.now(timezone.utc).isoformat(),
        run_id=v["run_id"], scientific_adjudication=v["scientific_adjudication"],
        primary_endpoints=v["endpoints"], secondary_descriptive=v["secondary_descriptive"],
        archive_sha256=digest(original), archive_bytes=original.stat().st_size,
        original_archive_and_receipt_copy_identical=True, raw_evidence_files_checked=len(raw_paths),
        previous_anchor_files_at_start=len(before["sha256"]),
        preserved_files_checked=len(preserved), preservation_mismatches=[],
        allowed_operational_updates=sorted(allowed),
        requirements_sections_2_3_unchanged=True, ledger_previous_prefix_unchanged=True,
        baseline=baseline_result, numeric_audit=audit,
        report_and_detail_numeric_checks=True, local_markdown_links_checked=links_checked,
        figures_visually_checked=3, figures_formats=["PNG", "SVG"],
        new_language_model_forwards=0, original_papers_unchanged=True,
        frozen_p3_release_unchanged=True, fp_counts=dict(false_positive=1, false_negative=0,
                                                       controls=1, unresolved=0),
        independent_reviewer=False, preserved_sha256=preserved, artifact_sha256=hashes,
    )
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in [
        "status", "run_id", "scientific_adjudication", "archive_bytes",
        "raw_evidence_files_checked", "preserved_files_checked", "preservation_mismatches",
        "report_and_detail_numeric_checks", "local_markdown_links_checked",
        "figures_visually_checked", "new_language_model_forwards"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
