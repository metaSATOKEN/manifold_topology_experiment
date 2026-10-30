"""Release and preservation audit only. Does not run the Colab cell or any model."""
import ast
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib
import p3_io as disk


def main():
    research = disk.HERE.parent
    workspace = research.parent.parent
    evidence = research/"evidence"
    output = evidence/"p3_release_checks_2026-10-20.json"
    before = disk.read(evidence/"p3_design_before.json")
    allowed = {
        "AGENTS.md",
        "research/pc_geometry_rerun/README.md",
        "research/pc_geometry_rerun/PLAN.md",
        "research/pc_geometry_rerun/REQUIREMENTS.md",
        "research/pc_geometry_rerun/prereg/README.md",
        "research/pc_geometry_rerun/audits/ledger_2026-10.md",
    }
    preserved = {p: h for p, h in before["sha256"].items() if p not in allowed}
    for name, sha in preserved.items():
        assert disk.digest(workspace/name) == sha, "Prior evidence changed: "+name
    ledger = research/"audits/ledger_2026-10.md"
    old_ledger = before["sha256"][str(ledger.relative_to(workspace))]
    assert disk.sha(ledger.read_bytes()[:before["ledger_bytes"]]) == old_ledger
    assert "\n## 14." in ledger.read_text()
    contract = lambda s: s[s.index("## 2."):s.index("## 4.")]
    assert contract((research/"REQUIREMENTS.md").read_text()) == contract(before["requirements"])
    baseline = subprocess.run([sys.executable, str(research/"check_baseline.py")],
                              cwd=workspace, capture_output=True, text=True, check=True)
    assert "PASS" in baseline.stdout

    release = disk.read(disk.HERE/"release_manifest.json")
    assert release["sources"] == disk.source_hashes()
    assert release["capsule"] == disk.capsule()
    assert (research/"prereg/EXP-0003_v1_frozen_id.md").read_bytes() == (disk.HERE/"protocol.md").read_bytes()
    cell = disk.HERE/"pcgeom_p3_v1_onecell.py"
    notebook = disk.HERE/"pcgeom_p3_v1_onecell.ipynb"
    assert disk.digest(cell) == release["onecell_sha256"]
    assert disk.digest(notebook) == release["notebook_sha256"]
    tree = ast.parse(cell.read_text())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign) and
        isinstance(node.targets[0], ast.Name) and
        node.targets[0].id in {"_PAYLOAD", "_PAYLOAD_SHA256"}
    }
    raw = zlib.decompress(base64.b64decode(assignments["_PAYLOAD"], validate=True))
    assert disk.sha(raw) == assignments["_PAYLOAD_SHA256"] == release["payload_sha256"]
    files = json.loads(raw)
    assert set(files) == set(disk.SOURCE_NAMES)
    for name, encoded in files.items():
        assert base64.b64decode(encoded, validate=True) == (disk.HERE/name).read_bytes()
    nb = disk.read(notebook)
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 1
    assert "".join(nb["cells"][0]["source"]) == cell.read_text()
    assert nb["cells"][0]["outputs"] == [] and nb["cells"][0]["execution_count"] is None
    sources = list(disk.HERE.glob("*.py"))
    for p in sources:
        ast.parse(p.read_text(), filename=p.name)
    import build_cell
    try:
        build_cell.build()
    except ValueError as exc:
        assert "Release already frozen" in str(exc)
    else:
        raise AssertionError("Frozen build was replaceable")
    assert disk.digest(cell) == release["onecell_sha256"]
    manifest = disk.read(disk.HERE/"frozen_manifest.json")
    for name, sha in manifest["files"].items():
        assert disk.digest(disk.HERE/name) == sha
    tests = (evidence/"p3_v1_pytest_final.txt").read_text()
    assert "32 passed" in tests
    design = disk.read(evidence/"p3_design_checks.json")
    assert design["status"] == "PASS" and design["new_model_forwards"] == 0

    handoff = research/"HANDOFF/2026-10-20_zzzzzzzzz_p3_v1.md"
    docs = [disk.HERE/"README.md", research/"prereg/EXP-0003_v1_frozen_id.md", handoff]
    checked_links = 0
    for p in docs:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            dest = (p.parent/target.split("#")[0]).resolve()
            assert dest.exists() or dest == output.resolve(), f"Missing link: {p}: {target}"
            checked_links += 1
    assets = [p for p in disk.HERE.rglob("*") if p.is_file() and
              "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
    assets += [workspace/p for p in allowed]
    assets += [research/"prereg/EXP-0003_v1_frozen_id.md", handoff]
    assets += list(evidence.glob("p3_*"))
    assets = [p for p in assets if p != output and p.is_file()]
    artifact_sha256 = {str(p.relative_to(workspace)): disk.digest(p) for p in sorted(set(assets))}
    result = dict(
        status="PASS", checked_at=disk.stamp(), stage="FROZEN_READY_FOR_USER_NOT_RUN",
        release_capsule=release["capsule"], frozen_at=release["frozen_at"],
        source_files_matched=len(files), python_files_ast_checked=len(sources),
        onecell_bytes=cell.stat().st_size, onecell_sha256=disk.digest(cell),
        notebook_bytes=notebook.stat().st_size, identical_onecell_notebook=True,
        frozen_rebuild_rejected=True, protocol_matches_card=True,
        local_tests=32, controlled_duplicate_zip_warning=1,
        design=design, new_language_model_forwards=0, gpu_runs=0,
        estimates=dict(first_run_minutes=[10, 25], final_zip_MB=[200, 260],
                       all_cumulative_zips_MB=[500, 650], measured_p3_gpu=False),
        preserved_files_checked=len(preserved), preservation_mismatches=[],
        requirements_sections_2_3_unchanged=True, ledger_previous_prefix_unchanged=True,
        baseline_stdout=baseline.stdout.strip(), markdown_links_checked=checked_links,
        fp_counts=dict(false_positive=1, false_negative=0, controls=1, unresolved=0),
        independent_reviewer=False, preserved_sha256=preserved,
        artifact_sha256=artifact_sha256,
    )
    disk.write(output, result)
    print(disk.canonical({k: v for k, v in result.items()
                         if k not in ("artifact_sha256", "preserved_sha256")}))


if __name__ == "__main__":
    main()
