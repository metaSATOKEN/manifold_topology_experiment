"""Seal EDA-0002 artifacts and verify preservation of all previous evidence."""
import json
from pathlib import Path
import re
import subprocess
import sys
import analyze as a

repo = a.ROOT.parents[1]
before = json.loads((a.HERE/"preservation_before.json").read_text())
allowed = {"AGENTS.md", *[f"research/pc_geometry_rerun/{name}" for name in
    ["PLAN.md", "README.md", "REQUIREMENTS.md", "prereg/README.md", "audits/ledger_2026-10.md"]]}
preserved = {}
for name, sha in before["sha256"].items():
    if name not in allowed:
        assert a.digest(repo/name) == sha, name
        preserved[name] = sha
old_anchor = before["previous_anchor"]
assert a.digest(repo/old_anchor) == before["previous_anchor_sha256"]
preserved[old_anchor] = before["previous_anchor_sha256"]
ledger = a.ROOT/"audits/ledger_2026-10.md"
old_ledger_sha = before["sha256"][str(ledger.relative_to(repo))]
import hashlib
assert hashlib.sha256(ledger.read_bytes()[:before["ledger_bytes"]]).hexdigest() == old_ledger_sha
req = (a.ROOT/"REQUIREMENTS.md").read_text()
def contract(s):
    return s[s.index("## 2."):s.index("## 4.")]
assert contract(req) == contract(before["requirements"])
baseline = json.loads(subprocess.check_output([sys.executable, str(a.ROOT/"check_baseline.py")], cwd=repo))
assert baseline["status"] == "PASS"
verify = json.loads((a.HERE/"numerical_verification.json").read_text())
assert verify["status"] == "PASS"
assert json.loads((a.HERE/"test_results.json").read_text())["status"] == "PASS"
commit = json.loads((a.HERE/"FIT_COMMIT.json").read_text())
assert a.digest(a.CARD) == commit["card_sha256"]
assert a.digest(a.HERE/"analyze.py") == commit["code_sha256"]
for p, sha in commit["files"].items():
    assert a.digest(a.HERE/p) == sha
cv = json.loads((a.HERE/"cv_results.json").read_text())
for point in a.POINTS:
    for name in ["within", "complement"]:
        c = cv[point][name]
        assert c["alpha"] == c["grid"][min(range(5), key=lambda i: c["alpha_loss"][i])]
        for j, gamma in enumerate(c["gammas"]):
            assert gamma == c["grid"][min(range(5), key=lambda i: c["gamma_loss"][i][j])]
        assert sum(c["fold_tokens"]) == 21531 and min(c["fold_tokens"]) > 0
results = json.loads((a.HERE/"results.json").read_text())
report = (a.HERE/"ANALYSIS_REPORT_JA.md").read_text()
details = (a.HERE/"DETAIL_TABLES.md").read_text()
for point in a.POINTS:
    for kind in ["O", "R", "W_fixed", "W_CV"]:
        for key in ["gain", "advantage"]:
            v = results["points"][point]["models"][kind][key]
            assert v["n"] == 64 and v["positive"]+v["negative"] == 64
            assert f"{v['median']:.6f}" in report
            if key == "advantage":
                for bound in ["lower", "upper"]:
                    assert f"{v[bound]:.6f}" in report
        for v in results["points"][point]["models"][kind]["r2"]["all_models"]:
            assert f"{v['median']:.6f}" in details
fig = json.loads((a.HERE/"figure_review.json").read_text())
assert fig["status"] == "PASS" and len(fig["files"]) == 3
assert len(list((a.HERE/"figures").glob("*.svg"))) == 3
new_handoff = a.ROOT/"HANDOFF/2026-10-20_zzzzzzzzzzz_p3_reversal.md"
anchor = a.ROOT/"evidence/eda2_final_checks_2026-10-20.json"
mds = [*a.HERE.glob("*.md"), a.CARD, new_handoff, a.ROOT/"README.md",
       a.ROOT/"PLAN.md", a.ROOT/"prereg/README.md"]
links = []
pending = {(a.HERE/"final_checks.json").resolve(), anchor.resolve()}
for md in mds:
    for target in re.findall(r"\]\(([^)]+)\)", md.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        target = target.split("#")[0]
        path = (md.parent/target).resolve()
        assert path.exists() or path in pending, (md, target)
        links.append(path)
final = dict(status="PASS", checked_at=a.stamp(), baseline=baseline,
    previous_anchor_files_at_start=len(before["sha256"]),
    preserved_files_checked=len(preserved), preservation_mismatches=[],
    requirements_sections_2_3_unchanged=True, ledger_previous_prefix_unchanged=True,
    original_results_unchanged=True, fit_commit_unchanged=True, cv_selection_matches_losses=True,
    report_numbers_checked=True, local_markdown_links_checked=len(links),
    figures_visually_checked=3, numeric_audit=verify,
    finite_plan=dict(saved_diagnostic_completed=1, new_comparison_used=0, intervention_used=0,
                     new_comparison_maximum=1, conditional_intervention_maximum=1,
                     current_decision="Defer new acquisition; causal mechanism unresolved"),
    new_language_model_forwards=0, original_papers_unchanged=True, independent_reviewer=False,
    fp_counts=dict(false_positive=1, false_negative=0, controls=1, unresolved=0))
a.write(a.HERE/"final_checks.json", final)
artifacts = [repo/n for n in sorted(allowed)] + [a.CARD, new_handoff]
artifacts += [p for p in a.HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
artifact_hashes = {str(p.relative_to(repo)): a.digest(p) for p in sorted(set(artifacts))}
a.write(anchor, dict(**final, allowed_operational_updates=sorted(allowed),
    preserved_sha256=preserved, artifact_sha256=artifact_hashes))
assert all(p.exists() for p in links)
print(json.dumps({k: final[k] for k in ["status", "preserved_files_checked",
      "local_markdown_links_checked", "figures_visually_checked", "new_language_model_forwards"]}))
print("Current anchor:", anchor)
