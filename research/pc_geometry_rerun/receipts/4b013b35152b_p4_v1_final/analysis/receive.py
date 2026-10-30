"""Receive the complete P4 snapshot using the local frozen transport code.

Never executes archived code or imports a language model. Refuses overwrites.
"""
from pathlib import Path
import json
import shutil
import sys

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[3]
BASE = PROJECT / "research/pc_geometry_rerun"
sys.path.insert(0, str(BASE / "p4_v1"))
import p4_io as disk

anchor = BASE / "evidence/p4_release_checks_2026-10-20.json"
prior = disk.read(anchor)
hashes = prior["preserved_sha256"] | prior["artifact_sha256"]
bad = [p for p, h in hashes.items() if disk.digest(PROJECT / p) != h]
disk.require(not bad, "Previous anchor mismatch: " + repr(bad))
disk.write(OUT / "preservation_before.json", {
    "checked_at": disk.stamp(), "status": "PASS",
    "previous_anchor": str(anchor.relative_to(PROJECT)),
    "previous_anchor_sha256": disk.digest(anchor),
    "files_checked": len(hashes), "sha256": hashes,
}, immutable=True)
operations = ["AGENTS.md"] + [
    "research/pc_geometry_rerun/" + n for n in [
        "README.md", "PLAN.md", "REQUIREMENTS.md",
        "prereg/README.md", "audits/ledger_2026-10.md",
    ]
]
disk.write(OUT / "operations_before.json",
           {p: (PROJECT / p).read_text() for p in operations}, immutable=True)
source = PROJECT / "p4_v1"
originals = sorted(source.glob("*.zip"))
disk.require(len(originals) == 11, "Expected one INDEX plus ten PART ZIPs")
rows = []
copies = []
for p in originals:
    dest = OUT / "original_archives" / p.name
    disk.require(not dest.exists(), "Never overwrite original archives")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(p, dest)
    h = disk.digest(p)
    disk.require(disk.digest(dest) == h, "Copy mismatch")
    rows.append(dict(source=str(p.relative_to(PROJECT)), name=p.name,
                     bytes=p.stat().st_size, sha256=h))
    copies.append(dest)
print("Copied and hashed all 11 original archives.", flush=True)
bundle = disk.restore(copies, OUT / "raw")
disk.require(bundle["status"] == "FINAL", "Not a FINAL snapshot")
disk.require(bundle["capsule"] == disk.capsule(), "Frozen capsule mismatch")
disk.write(OUT / "bundle_index.json", bundle, immutable=True)
disk.write(OUT / "archive_receipt.json", dict(
    checked_at=disk.stamp(), status="PASS", archives=rows,
    archive_count=len(rows), total_zip_bytes=sum(r["bytes"] for r in rows),
    evidence_files=len(bundle["entries"]),
    evidence_bytes=sum(r["bytes"] for r in bundle["entries"].values()),
    run_id=bundle["run_id"], snapshot_id=bundle["bundle_id"],
    capsule=bundle["capsule"], complete_independent_snapshot=True,
    all_members_sha256_and_crc_verified=True, unsafe_paths_rejected=True,
    original_files_preserved=True, restored_sources_executed=False,
    model_downloads=0, language_model_forward_calls=0,
), immutable=True)
print(json.dumps(disk.read(OUT / "archive_receipt.json") |
                 {"archives": "see archive_receipt.json"}, indent=2), flush=True)
