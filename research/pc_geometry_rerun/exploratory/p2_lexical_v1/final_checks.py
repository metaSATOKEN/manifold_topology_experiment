"""Integrity, reporting and uncertainty checks; writes a new current anchor."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import numpy as np
import analyze as a

workspace = a.ROOT.parents[1]
before = json.loads((a.HERE/"preservation_before.json").read_text())
mutable = {
    "AGENTS.md", "research/pc_geometry_rerun/README.md",
    "research/pc_geometry_rerun/PLAN.md", "research/pc_geometry_rerun/REQUIREMENTS.md",
    "research/pc_geometry_rerun/prereg/README.md",
    "research/pc_geometry_rerun/audits/ledger_2026-10.md",
}
preserved = {}
for name, old in before["sha256"].items():
    if name not in mutable:
        assert a.digest(workspace/name) == old, name
        preserved[name] = old
ledger = a.ROOT/"audits/ledger_2026-10.md"
assert hashlib.sha256(ledger.read_bytes()[:before["ledger_bytes"]]).hexdigest() == before["sha256"][
    str(ledger.relative_to(workspace))]
req = (a.ROOT/"REQUIREMENTS.md").read_text()
for n in [2, 3]:
    old = before["requirements_text"].split(f"## {n}.", 1)[1].split(f"## {n+1}.", 1)[0]
    new = req.split(f"## {n}.", 1)[1].split(f"## {n+1}.", 1)[0]
    assert old == new
baseline = subprocess.run(["python3", str(a.ROOT/"check_baseline.py")],
                          cwd=workspace, capture_output=True, text=True, check=True)
baseline = json.loads(baseline.stdout)
assert baseline["status"] == "PASS"
numeric = json.loads((a.HERE/"numerical_verification.json").read_text())
assert numeric["status"] == "PASS"
data = json.loads((a.HERE/"results.json").read_text())
draws = np.random.Generator(np.random.PCG64(20261024)).integers(0, 64, (10000, 64))
checked = 0
for r in data["records"]:
    for key, v in r["article_values"].items():
        v = np.array(v)
        m = r["metrics"][key]
        np.testing.assert_allclose(np.percentile(np.median(v[draws], axis=1), [2.5, 97.5]),
                                   m["descriptive_95_ci"], atol=1e-12, rtol=0)
        assert m["n"] == 64 and m["positive"]+m["negative"]+m["zero"] == 64
        assert m["positive"] == int((v > 0).sum())
        checked += 1
assert len(json.loads((a.HERE/"article_results.json").read_text())) == 1536
detail = (a.HERE/"DETAIL_TABLES.md").read_text()
for r in data["records"]:
    m = r["metrics"]
    jp = {"original": "元P2", "ID": "ID", "NL": "NL", "NL_ID": "NL ID"}[r["family"]]
    row_start = f"| {r['point']} | {r['condition']} | {jp} | {m['baseline_r2']['median']:.5f} | {m['pc_r2']['median']:.5f} |"
    assert row_start in detail
report = (a.HERE/"ANALYSIS_REPORT_JA.md").read_text()
assert "探索的解析" in report and "言語モデルの再実行は0回" in report
assert "非線形モデルの失敗は隠さない" in report
paths = list(a.HERE.glob("*.md")) + [a.CARD, workspace/"AGENTS.md",
    a.ROOT/"README.md", a.ROOT/"PLAN.md", a.ROOT/"REQUIREMENTS.md",
    a.ROOT/"prereg/README.md", a.ROOT/"HANDOFF/2026-10-20_zzzzzzzz_p2_lexical.md"]
out = a.ROOT/"evidence/p2_lexical_checks_2026-10-20.json"
links, missing = [], []
for p in paths:
    for ref in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text()):
        if "://" in ref or ref.startswith("#"):
            continue
        target = (p.parent/ref.split("#")[0]).resolve()
        links.append(str(target))
        if target != out and not target.exists():
            missing.append(str(target))
assert not missing, missing
new = [p for p in a.HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
new += [workspace/p for p in mutable]
new += [a.CARD, a.ROOT/"HANDOFF/2026-10-20_zzzzzzzz_p2_lexical.md"]
assert len(new) == len(set(new))
evidence = dict(recorded_at=a.stamp(), status="EXPLORATORY_ANALYSIS_REPORTED_AND_VERIFIED",
    scope="User-authorized saved-P2-data lexical/nonlinear regression and report; old results preserved",
    primary=dict(point="block06", condition="C1", baseline="ID",
                 interpretation="PC gain remains smaller; no causal decomposition or confirmatory adjudication"),
    numerical_verification=numeric, planned_new_regressors=396, original_reference_regressors=132,
    analyzed_article_condition_models=1536, missing=0, new_language_model_forwards=0,
    verification_attempt_failures=1, failure="First audit retained C0/C4 NA row; verifier indexing corrected",
    source_artifacts_unchanged=len(preserved), preserved_sha256=preserved,
    ledger_prefix_preserved=True, adopted_contract_sections_2_3_unchanged=True,
    baseline_integrity=baseline, descriptive_bootstrap_summaries_verified=checked,
    full_table_rows_checked=24, local_links_checked=len(links), missing_links=missing,
    figures_visually_reviewed=["01_primary_gains.png", "02_prediction_levels.png", "03_frequency_strata.png"],
    tests=json.loads((a.HERE/"test_results.json").read_text()),
    fp_corpus_counts=dict(false_positive=1, false_negative=0, control=1, uncertain=0),
    tl0001_evaluations_added=0, reviewer="Same agent; separate numerical reconstruction, not independent review",
    artifact_sha256={str(p.relative_to(workspace)): a.digest(p) for p in sorted(new)})
# Numeric anchor values come from saved results, not hand-typed rounded examples.
primary = next(r for r in data["records"] if (r["point"], r["condition"], r["family"]) ==
               ("block06", "C1", "ID"))
evidence["primary"].update(metrics=primary["metrics"], retained_gain_ratio=primary["retained_gain_ratio"])
a.write_json(out, evidence)
print(json.dumps({k: evidence[k] for k in ["status", "source_artifacts_unchanged",
    "descriptive_bootstrap_summaries_verified", "full_table_rows_checked", "local_links_checked",
    "missing_links"]}, ensure_ascii=False, indent=2))
