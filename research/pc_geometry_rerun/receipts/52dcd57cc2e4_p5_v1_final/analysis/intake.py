"""Validate, preserve, and extract data only. Never execute archived Python."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parents[1]
PROJECT = BASE.parents[1]
RELEASE = BASE/"p5_v1"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    assert not (OUT/"archive_receipt.json").exists(), "Intake already recorded"
    original = PROJECT/"pcgeom_p5_v1_52dcd57cc2e4_FINAL_9c1e90.zip"
    anchor = BASE/"evidence/p5_release_checks_2026-10-20.json"
    previous = json.loads(anchor.read_text())
    mutable = ["AGENTS.md"]+[str((BASE/n).relative_to(PROJECT)) for n in [
        "README.md", "PLAN.md", "REQUIREMENTS.md", "prereg/README.md", "audits/ledger_2026-10.md"]]
    old = {**previous["preserved_sha256"], **previous["artifact_sha256"],
           str(original.relative_to(PROJECT)): sha(original)}
    write(OUT/"preservation_before.json", dict(previous_anchor=str(anchor.relative_to(PROJECT)),
        previous_anchor_sha256=sha(anchor), old_hashes=old,
        mutable_before={p: dict(sha256=sha(PROJECT/p), text=(PROJECT/p).read_text()) for p in mutable}))
    release = json.loads((RELEASE/"release_manifest.json").read_text())
    assert original.stat().st_size < 300000000
    with zipfile.ZipFile(original) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        assert len(names) == len(set(names)) <= 500
        assert sum(i.file_size for i in infos) < 128*1024**2
        for i in infos:
            p = PurePosixPath(i.filename)
            assert str(p) == i.filename and not p.is_absolute() and ".." not in p.parts
            assert "\\" not in i.filename and not i.is_dir() and not i.flag_bits & 1
            assert (i.external_attr >> 16) & 0o170000 != 0o120000
        manifest = json.loads(z.read("ARCHIVE.json"))
        assert manifest["label"] == "FINAL" and manifest["run_id"] == "52dcd57cc2e4"
        assert manifest["capsule"] == release["capsule"]
        assert set(names) == {"ARCHIVE.json"} | {"evidence/"+n for n in manifest["files"]}
        for name, expected in manifest["files"].items():
            b = z.read("evidence/"+name)
            assert len(b) == expected["bytes"] and hashlib.sha256(b).hexdigest() == expected["sha256"], name
        for name, expected in release["source_sha256"].items():
            assert hashlib.sha256(z.read("evidence/sources/"+name)).hexdigest() == expected
        run = json.loads(z.read("evidence/run.json"))
        final = json.loads(z.read("evidence/FINAL.json"))
        assert run["source_sha256"] == release["source_sha256"]
        assert run["capsule"] == final["capsule"] == release["capsule"]
        assert release["frozen_at"] < run["created"] < final["at"]
        assert hashlib.sha256(z.read("evidence/summary.json")).hexdigest() == final["summary_sha256"]
        for name, expected in final["analysis_sha256"].items():
            assert hashlib.sha256(z.read("evidence/"+name)).hexdigest() == expected
        assert z.testzip() is None
        raw = OUT/"raw"
        raw.mkdir(exist_ok=False)
        for name in manifest["files"]:
            target = raw/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read("evidence/"+name))
        write(OUT/"archive_manifest.json", manifest)
    destination = OUT/"original.zip"
    assert not destination.exists()
    shutil.copyfile(original, destination)
    assert sha(destination) == sha(original)
    summary = json.loads((raw/"summary.json").read_text())
    units = sorted(p.name for p in (raw/"units").iterdir())
    assert units == sorted(d+"_"+t for d in ["cpu", "cuda"] for t in ["narrative", "technical", "repetitive"])
    assert summary["analyzed_conditions"] == 72 and not summary["analysis_failures"]
    assert len(list((raw/"forward_attempts").glob("*.json"))) == 6
    assert not list((raw/"failures").glob("*.json"))
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), status="PASS",
        run_id=run["run_id"], capsule=release["capsule"], original=str(original),
        original_sha256=sha(original), original_bytes=original.stat().st_size,
        zip_entries=len(infos), evidence_files=len(manifest["files"]),
        uncompressed_bytes=sum(i.file_size for i in infos), committed_units=units,
        forward_attempts=6, analysis_conditions=72, failures=0,
        frozen_before_run=True, source_hashes_match_release=True,
        archived_code_executed=False, new_model_forward=0)
    write(OUT/"archive_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
