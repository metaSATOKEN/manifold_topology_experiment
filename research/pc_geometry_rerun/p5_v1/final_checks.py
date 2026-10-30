"""Release consistency and old-artifact preservation; no model execution."""
import ast
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib
import p5_io as disk

OUT = disk.HERE
BASE = OUT.parent
PROJECT = BASE.parents[1]


def main():
    before = disk.read(OUT/"preservation_before.json")
    mutable = before["mutable_before"]
    preserved = {p: h for p, h in before["old_hashes"].items() if p not in mutable}
    def check(item):
        p, h = item
        return p if not (PROJECT/p).is_file() or disk.digest(PROJECT/p) != h else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        bad = [p for p in pool.map(check, preserved.items()) if p]
    assert not bad, bad
    assert disk.digest(PROJECT/before["previous_anchor"]) == before["previous_anchor_sha256"]
    preserved[before["previous_anchor"]] = before["previous_anchor_sha256"]
    for p, value in mutable.items():
        assert value["sha256"] == disk.sha(value["text"].encode())
    req = "research/pc_geometry_rerun/REQUIREMENTS.md"
    old, new = mutable[req]["text"], (PROJECT/req).read_text()
    for a, b in [("## 2.", "## 3."), ("## 3.", "## 4.")]:
        assert old.split(a, 1)[1].split(b, 1)[0] == new.split(a, 1)[1].split(b, 1)[0]
    ledger = "research/pc_geometry_rerun/audits/ledger_2026-10.md"
    assert (PROJECT/ledger).read_text().startswith(mutable[ledger]["text"])

    release = disk.read(OUT/"release_manifest.json")
    tests = disk.read(OUT/"test_results.json")
    assert release["source_sha256"] == tests["source_sha256"] == disk.source_hashes()
    assert release["capsule"] == disk.capsule()
    assert release["tests_sha256"] == disk.digest(OUT/"test_results.json")
    assert tests["status"] == "PASS" and tests["pipeline"]["complete_synthetic_conditions"] == 72
    assert not tests["adapter"]["pretrained_weights_downloaded"] and not tests["pipeline"]["real_GPU_comparison"]
    onecell = OUT/"pcgeom_p5_v1_onecell.py"
    notebook = OUT/"pcgeom_p5_v1_onecell.ipynb"
    assert release["onecell_sha256"] == disk.digest(onecell)
    assert release["notebook_sha256"] == disk.digest(notebook)
    nb = disk.read(notebook)
    assert len(nb["cells"]) == 1 and "".join(nb["cells"][0]["source"]) == onecell.read_text()
    tree = ast.parse(onecell.read_text())
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in ("_PAYLOAD", "_PAYLOAD_SHA256")}
    raw = zlib.decompress(base64.b64decode(assignments["_PAYLOAD"], validate=True))
    assert disk.sha(raw) == assignments["_PAYLOAD_SHA256"] == release["payload_sha256"]
    packaged = json.loads(raw)
    assert set(packaged) == set(disk.SOURCE_NAMES)
    for name, encoded in packaged.items():
        assert base64.b64decode(encoded) == (OUT/name).read_bytes()
    for source in OUT.glob("*.py"):
        ast.parse(source.read_text())
    guarded = subprocess.run([sys.executable, str(onecell)], capture_output=True, text=True, timeout=10)
    assert guarded.returncode != 0 and "Run this cell in a Colab GPU runtime." in guarded.stderr
    card = BASE/"prereg/EXP-0005_v1_arc_reproduction.md"
    assert card.read_bytes() == (OUT/"experiment_card.md").read_bytes()
    oldsource = PROJECT/"experiments/utils/extraction.py"
    assert oldsource.read_bytes() == (OUT/"historical_extraction.py.txt").read_bytes()
    oldtree = ast.parse(oldsource.read_text())
    texts = next(ast.literal_eval(n.value) for n in oldtree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "SAMPLE_TEXTS" for t in n.targets))
    for e in disk.entries():
        assert e["text"] == texts[e["id"]] and disk.sha(e["text"].encode()) == e["utf8_sha256"]
        assert len(e["token_ids"]) == e["tokens"] <= 1024
    assert [e["tokens"] for e in disk.entries()] == [101, 85, 72]
    assert disk.read(OUT/"inputs.json")["raw_hidden_bytes"] == 2*4*258*768*4 == 6340608
    assert disk.spec()["primary"] == dict(device="cpu", point="block07", condition="C0")
    anchor = BASE/"evidence/p5_release_checks_2026-10-20.json"
    handoff = BASE/("HANDOFF/2026-10-20_"+"z"*19+"_p5_v1.md")
    targets = [card, handoff, OUT/"README.md"]+[PROJECT/p for p in mutable]
    links = 0
    for p in targets:
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if re.match(r"^[a-z]+:|^#", target):
                continue
            linked = (p.parent/target.split("#")[0].strip("<>")).resolve()
            assert linked.exists() or linked == anchor, (p, target)
            links += 1
    checks = dict(status="PASS", at=disk.stamp(), source_capsule=disk.capsule(),
        old_files_preserved=len(preserved), old_mismatches=0, old_manuscripts_unchanged=True,
        requirements_sections_2_3_unchanged=True, ledger_prefix_unchanged=True,
        payload_and_notebook_equal_sources=True, local_execution_guard=True,
        historical_texts_byte_exact=True, frozen_tokens=258, raw_hidden_bytes=6340608,
        synthetic_analysis_conditions=72, synthetic_zip_bytes=tests["pipeline"]["synthetic_zip_bytes"],
        source_checks=tests["numerical"], adapter_checks=tests["adapter"],
        transport_checks=tests["transport"], links_checked=links,
        research_model_forwards=0, new_pretrained_model_downloads=0, GPU_executions=0,
        reviewer="Same agent self-audit. Pinned Colab CUDA runtime has not been executed.")
    disk.write(OUT/"final_checks.json", checks)
    files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [card, handoff]+[PROJECT/p for p in mutable]
    artifact = {str(p.relative_to(PROJECT)): disk.digest(p) for p in files}
    disk.write(anchor, dict(at=disk.stamp(), status="PASS", operation="EXP-0005 frozen release; user execution pending",
        previous_anchor=before["previous_anchor"], previous_anchor_sha256=before["previous_anchor_sha256"],
        preserved_sha256=preserved, artifact_sha256=artifact, checks=checks,
        stop="Await user's Colab ZIP; then verify and report this single fixed reproduction set. No manuscript edit."))
    print(disk.canonical({k: checks[k] for k in ["status", "old_files_preserved", "old_mismatches",
        "payload_and_notebook_equal_sources", "synthetic_analysis_conditions", "links_checked"]}))


if __name__ == "__main__":
    main()
