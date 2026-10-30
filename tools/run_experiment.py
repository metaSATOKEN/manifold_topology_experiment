"""Inspect a frozen source bundle, or explicitly launch it in Google Colab.

Default: hash source files only. --execute calls the unchanged stage launcher,
which installs its pinned environment, downloads weights, computes, and writes ZIPs.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"p0": "p0_v2", "p1": "p1_v1", "p2": "p2_v1",
          "p3": "p3_v1", "p4": "p4_v1", "p5": "p5_v1"}
SUCCESS = {"NO_NEW_COMPUTE", "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT",
           "COMPLETE", "COMPLETE_NOOP"}


def check_sources(folder):
    """Read bytes only; do not import the experiment or interpret its input data."""
    manifest = json.loads((folder / "release_manifest.json").read_text())
    hashes = manifest.get("sources", manifest.get("source_sha256"))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing source hashes")
    for name, expected in hashes.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Invalid source name: " + name)
        with (folder / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Source checksum mismatch: " + name)
    return len(hashes)


def execute(stage, folder, restore, auto_download):
    """Call visible Python modules directly; no encoded payload or unpacking."""
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("Execution requires a Google Colab runtime; use the default source check locally.")
    importlib.import_module("google.colab")
    sys.path.insert(0, str(folder))
    launcher = importlib.import_module(stage + "_launch")
    # P4 restores a complete PART + INDEX set. Other stages use one cumulative ZIP.
    restore_value = restore if stage == "p4" and restore else (restore[0] if restore else "")
    return launcher.run(restore_value, auto_download)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--execute", action="store_true", help="Start/resume computation in Colab after checking sources")
    parser.add_argument("--restore", nargs="+", default=[], metavar="ZIP", help="Uploaded ZIP path; P4 requires all PARTs and INDEX from one snapshot")
    parser.add_argument("--auto-download", action="store_true", help="Request browser downloads; default is manual download from Files")
    args = parser.parse_args(argv)
    if args.stage != "p4" and len(args.restore) > 1:
        parser.error("Only P4 accepts multiple restore ZIPs")
    if not args.execute and (args.restore or args.auto_download):
        parser.error("--restore and --auto-download require --execute")
    folder = ROOT / "research/pc_geometry_rerun" / STAGES[args.stage]
    count = check_sources(folder)
    print(json.dumps({"stage": args.stage, "source_files_verified": count,
                      "mode": "execute" if args.execute else "source_check_only"}), flush=True)
    if not args.execute:
        return 0
    result = execute(args.stage, folder, args.restore, args.auto_download)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("status") in SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
