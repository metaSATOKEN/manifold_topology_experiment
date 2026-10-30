"""Preservation and artifact closure, without changing scientific outputs."""
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
PROJECT=BASE.parents[1]
anchor=BASE/"evidence/eda4_final_checks_2026-10-20.json"
before=json.loads((OUT/"preservation_before.json").read_text())
mutable=before["mutable_before"]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def check(item):
    p,h=item
    return p if not (PROJECT/p).is_file() or digest(PROJECT/p)!=h else None


preserved={p:h for p,h in before["old_hashes"].items() if p not in mutable}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    bad=[p for p in pool.map(check,preserved.items()) if p]
assert not bad,bad
assert digest(PROJECT/before["previous_anchor"])==before["previous_anchor_sha256"]
preserved[before["previous_anchor"]]=before["previous_anchor_sha256"]
req=BASE/"REQUIREMENTS.md"
old=mutable[str(req.relative_to(PROJECT))]["text"]
new=req.read_text()
for a,b in [("## 2.","## 3."),("## 3.","## 4.")]:
    assert old.split(a,1)[1].split(b,1)[0]==new.split(a,1)[1].split(b,1)[0]
ledger=BASE/"audits/ledger_2026-10.md"
assert ledger.read_text().startswith(mutable[str(ledger.relative_to(PROJECT))]["text"])
for f in ["verification.json","output_checks.json","synthetic_checks.json"]:
    assert json.loads((OUT/f).read_text())["status"]=="PASS"
freeze=json.loads((OUT/"FREEZE.json").read_text())
start=json.loads((OUT/"execution_start.json").read_text())
assert freeze["at"]<start["at"]
for p,h in freeze["files"].items():
    assert digest(PROJECT/p)==h
for p,h in start["source_sha256"].items():
    assert digest(OUT/p)==h
bytes_total=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
assert bytes_total<512*1024**2
extra=[BASE/"prereg/EDA-0004_v1_arc_thickness.md",
       BASE/"prereg/EDA-0004_v1_results_2026-10-20.md",
       BASE/"HANDOFF/2026-10-20_zzzzzzzzzzzzzzzzzz_arc.md"]
files=[p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
files+=extra+[PROJECT/p for p in mutable]
links=0
for p in [q for q in files if q.suffix==".md"]:
    for target in re.findall(r"\]\(([^)]+)\)",p.read_text()):
        if re.match(r"^[a-z]+:|^#",target):continue
        linked=(p.parent/target.split("#")[0].strip("<>")).resolve()
        assert linked.exists() or linked==anchor, (p,target)
        links+=1
artifact={str(p.relative_to(PROJECT)):digest(p) for p in files}
record=dict(at=datetime.now(timezone.utc).isoformat(),status="PASS",operation="EDA-0004 saved-state arc and thickness; closed",
    previous_anchor=before["previous_anchor"],previous_anchor_sha256=before["previous_anchor_sha256"],
    preserved_sha256=preserved,artifact_sha256=artifact,
    checks=dict(old_nonmutable_files=len(preserved)-1,old_mismatches=0,
        original_manuscripts_unchanged=True,requirements_sections_2_3_unchanged=True,
        ledger_old_prefix_unchanged=True,freeze_before_analysis=True,
        analysis_sources_unchanged=True,conditions=1152,technical_failures=0,
        curve_sse_values=34560,feature_predictions=5672340,direct_svd=18,
        report_numeric_cells=270,figures=33,links_checked=links,artifact_bytes=bytes_total,
        limit_bytes=512*1024**2,new_lm_forward=0,gpu=0,
        visual_review="Representative GPT-2 middle/final, Qwen middle, overview, basis comparison, 64-article atlas, readout. Shared renderer and inventory for all 33.",
        plot_adjustments="Equal-unit axes, legible 3D limits and tick placement; no scientific changes",
        reviewer="Same agent, separate reconstruction and checks; not independent review"),
    conclusion="Conditional 2D band/curvature in GPT-2 final output; central outlier-cloud geometry; residual feature associations with major Qwen extrapolation failure. No universal one-dimensional arc claim.",
    stop="One authorized saved-data shape set completed. No new experiment or manuscript edit.")
anchor.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
assert all((p.parent/t.split("#")[0].strip("<>")).resolve().exists()
           for p in [OUT/"ANALYSIS_REPORT_JA.md",OUT/"GALLERY.md",*extra]
           for t in re.findall(r"\]\(([^)]+)\)",p.read_text())
           if not re.match(r"^[a-z]+:|^#",t))
print(json.dumps({"status":"PASS","preserved":len(preserved),"new_artifacts":len(artifact),
                  "bytes":bytes_total,"links":links},ensure_ascii=False))
