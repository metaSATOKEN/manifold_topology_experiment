"""Read-only evidence checks followed by writing the new EDA-0003 anchor."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
load = lambda p: json.loads(p.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(p):
    return str(p.relative_to(ROOT))


before = load(OUT / "preservation_before.json")
ops = before["operations"]
assert len(ops) == 6 and before["files_checked"] == 5110
preserved = {p: h for p, h in before["sha256"].items() if p not in ops}
preserved[before["previous_anchor"]] = before["previous_anchor_sha256"]
mismatches = [p for p, h in preserved.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
assert not mismatches, mismatches
print(f"Previous evidence preserved: {len(preserved)} files.", flush=True)

freeze = load(OUT / "plan_freeze.json")
for group in ["plan_sha256", "source_sha256"]:
    for p, h in freeze[group].items():
        assert sha(ROOT / p) == h, p
execution = load(OUT / "execution_manifest.json")
assert sha(OUT / "plan_freeze.json") == execution["plan_freeze_sha256"]
for p, h in execution["code_sha256"].items():
    assert sha(OUT / p) == h
assert datetime.fromisoformat(freeze["frozen_at"]) < datetime.fromisoformat(execution["started"])
assert not freeze["analysis_started"] and freeze["existing_P4_results_known"]
assert not freeze["confirmatory_preregistration"]
assert execution["model_forwards"] == 0 and not execution["gpu_used"]
points = [(m, p) for m in ["gpt2", "opt", "qwen"] for p in ["middle", "final_norm"]]
committed_files = 0
for m, p in points:
    folder = OUT / "results" / m / p
    for name, h in load(folder / "COMMIT.json").items():
        assert sha(folder / name) == h, (m, p, name)
        committed_files += 1
    assert load(folder / "audit.json")["status"] == "PASS"
assert load(OUT / "execution_result.json")["points"] if "points" in load(OUT / "execution_result.json") else True

# The governance contract stays byte-for-byte identical. Old audit history is append-only.
req_path = "research/pc_geometry_rerun/REQUIREMENTS.md"
contract = lambda s: s[s.index("## 2."):s.index("## 4.")]
assert contract(ops[req_path]) == contract((ROOT / req_path).read_text())
ledger_path = "research/pc_geometry_rerun/audits/ledger_2026-10.md"
assert (ROOT / ledger_path).read_text().startswith(ops[ledger_path])
assert "## 19. EDA-0003" in (ROOT / ledger_path).read_text()
assert "FINAL_SAVED_DATA_SET_CLOSED" in (ROOT / ledger_path).read_text()
for name in ["README.md", "PLAN.md", "prereg/README.md"]:
    path = BASE / name
    prior_tail = ops[relative(path)].split("\n", 1)[1]
    assert path.read_text().endswith(prior_tail), name
prior_agents = ops["AGENTS.md"]
section = "## 現在の承認境界\n\n"
prefix, suffix = prior_agents.split(section, 1)
assert (ROOT / "AGENTS.md").read_text().startswith(prefix + section)
assert (ROOT / "AGENTS.md").read_text().endswith(suffix)

# Verify all table renderings against the committed (unrounded) aggregates.
S = load(OUT / "summary.json")["results"]
rows = load(OUT / "article_metrics.json")
assert len(rows) == 384
report = (OUT / "ANALYSIS_REPORT_JA.md").read_text()
details = (OUT / "DETAIL_TABLES.md").read_text()
formatted = lambda d: f'{d["estimate"]:.5f} [{d["lo"]:.5f}, {d["hi"]:.5f}]'
metric_cells_checked = 0
for m, p in points:
    data = S[m][p]
    assert data["coverage"]["articles"] == 64 and data["coverage"]["missing"] == 0
    for key, stat in data["metrics"].items():
        assert formatted(stat) in report, (m, p, key)
    for category in ["metrics", "equal_ID_metrics", "r2_levels"]:
        for stat in data[category].values():
            assert formatted(stat) in details
            metric_cells_checked += 1
    for family in data["random_replicates"].values():
        assert len(family) == 20
        for stat in family:
            assert formatted(stat) in details
            metric_cells_checked += 1
    for strata in data["bias"].values():
        for stratum in strata:
            for stat in stratum["contributions"].values():
                assert formatted(stat) in details
                metric_cells_checked += 1
    selected_rows = [r for r in rows if r["model"] == m and r["point"] == p]
    assert len(selected_rows) == 64
    for r in selected_rows:
        assert r["document_id"] in details
        for key in ["A1_gain", "A7_gain", "A8_gain", "NR_gain", "NU_gain", "NU_adv"]:
            assert f'{r["metrics"][key]:.6f}' in details
            metric_cells_checked += 1
assert all(S[m][p]["metrics"]["NU_gain"]["lo"] > 0 for m, p in points)
assert S["opt"]["middle"]["metrics"]["NU_adv"]["lo"] < 0 < S["opt"]["middle"]["metrics"]["NU_adv"]["hi"]
assert all(S[m][p]["metrics"]["NU_adv"]["lo"] > 0 for m, p in points if (m, p) != ("opt", "middle"))
display = load(OUT / "report_data.json")
assert display["additional_mean_bootstrap_checks"] == 48
assert display["results"]["qwen/middle"]["B_to_A8"]["estimate"] < 0
assert S["qwen"]["middle"]["metrics"]["A8_gain"]["estimate"] > 0
assert display["results"]["qwen/middle"]["worst_A8_article"]["document_id"] == "wt2_train_27981"
fixture = next(f for f in load(BASE / "p4_v1/fixtures.json") if f["id"] == "wt2_train_27981")
assert fixture["title"] == "2nd Battalion 9th Marines"
assert fixture["title"] in report
assert "４条件支持・２条件不確定" in report and "この１セットで探索終了" in report
verification = load(OUT / "verification.json")
assert verification["status"] == "PASS"
assert verification["counts"] == dict(article_points=384, metric_checks=13824,
                                    prediction_values=16355394, bootstrap_checks=840, stratum_checks=39936)
assert verification["new_language_model_forwards"] == 0
seconds = verification["analysis_seconds"] + verification["aggregation_seconds"] + display["seconds"]
assert seconds < 3600
qa = load(OUT / "figure_qa.json")
assert qa["status"] == "PASS" and len(qa["sha256"]) == 8
for p, h in qa["sha256"].items():
    assert sha(OUT / p) == h

check = subprocess.run([sys.executable, str(BASE / "check_baseline.py")], cwd=ROOT,
                       capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
(OUT / "baseline_check.txt").write_text(check.stdout + check.stderr)

handoff = BASE / "HANDOFF" / ("2026-10-20_" + "z" * 14 + "_eda3_results.md")
new_extra = [
    BASE / "prereg/EDA-0003_v1_p4_components_radius_direction.md",
    BASE / "prereg/EDA-0003_v1_results_2026-10-20.md",
    handoff,
]
new_files = sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
target = BASE / "evidence/eda3_final_checks_2026-10-20.json"
link_checks = 0
for p in [OUT / "ANALYSIS_REPORT_JA.md", *new_extra]:
    for ref in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", p.read_text()):
        if ref.startswith(("http:", "https:", "#")):
            continue
        dest = (p.parent / ref.split("#")[0]).resolve()
        assert dest == target or dest.exists(), (p, ref)
        link_checks += 1
total_bytes = sum(p.stat().st_size for p in new_files + new_extra)
assert total_bytes < 512 * 1024 ** 2
artifact_files = new_files + new_extra + [ROOT / p for p in ops]
artifact = {relative(p): sha(p) for p in artifact_files}
result = dict(
    status="PASS", created_at=datetime.now(timezone.utc).isoformat(),
    state="COMPLETED / EXPLORATORY / FINAL_SAVED_DATA_SET_CLOSED",
    previous_anchor=before["previous_anchor"],
    preserved_file_count=len(preserved), preserved_sha256=preserved,
    artifact_file_count=len(artifact), artifact_sha256=artifact,
    plan_before_computation=True, frozen_plan_and_code_unchanged=True,
    frozen_source_files_checked=len(freeze["source_sha256"]),
    committed_result_files_checked=committed_files, table_cells_checked=metric_cells_checked,
    checked_markdown_links=link_checks, numerical_seconds=seconds,
    new_output_bytes_excluding_this_anchor=total_bytes,
    byte_budget=512 * 1024 ** 2,
    old_governance_contract_unchanged=True, old_audit_ledger_prefix_unchanged=True,
    existing_confirmatory_decisions_unchanged=True,
    new_model_downloads=0, new_model_forwards=0, gpu_used=False,
    independent_reviewer=False, figures_visually_checked=4,
    coverage=verification["counts"], additional_mean_bootstrap_checks=48,
    false_positive_corpus=dict(FP=1, FN=0, control=1, unresolved=0),
    TL_formal_evaluations=0,
    closure="No second exploration set, new data, or intervention; causal mechanism remains unresolved.",
)
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
assert total_bytes + target.stat().st_size < 512 * 1024 ** 2
print(json.dumps({k: v for k, v in result.items() if k not in ["preserved_sha256", "artifact_sha256"]},
                 ensure_ascii=False, indent=2))
