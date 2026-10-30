"""Offline receipt verification. Only NumPy is needed; no model/network/GPU calls."""
import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile
import p0_io as disk
from p0_numeric import load_npz, save_npz, pca, validate_raw, verify_pca


def features(raw):
    return {n: raw[k] for n, k in (("position", "position"), ("token_chars", "token_chars"),
                                  ("surprisal", "surprisal"))}


def analyze(root):
    root = Path(root)
    done = disk.committed(root)
    disk.require(len(done) == 4, "Analysis requires all four committed documents")
    for doc_id in done:
        raw, _ = validate_raw(root, doc_id)
        for point in disk.spec()["points"]:
            arrays, metrics = pca(raw["h_"+point], features(raw), disk.spec()["components"])
            base = root / "analysis" / doc_id / point
            if base.with_suffix(".npz").exists() and base.with_suffix(".json").exists():
                verify_pca(raw["h_"+point], load_npz(base.with_suffix(".npz")),
                           disk.read(base.with_suffix(".json")), features(raw))
            else:
                save_npz(base.with_suffix(".npz"), arrays)
                disk.write(base.with_suffix(".json"), metrics, immutable=True)
            print("P0 analysis:", doc_id, point, flush=True)


def verify(root, complete=True):
    root = Path(root)
    run = disk.check_run(root)
    done = disk.committed(root)
    if "parent_capsule" in run:
        migration = disk.read(root / "migration_v2.json")
        disk.require(run["parent_capsule"] == disk.legacy()["capsule"] == migration["from_capsule"] and
                     migration["to_capsule"] == run["capsule"], "Migration code lineage mismatch")
        history = root / "code_history" / run["parent_capsule"]
        old = disk.read(history / "run.json")
        disk.require(old["run_id"] == run["run_id"] and old["capsule"] == run["parent_capsule"] and
                     old["sources"] == disk.legacy()["sources"] and
                     old["specification_sha256"] == run["specification_sha256"], "Legacy run lineage mismatch")
        disk.require(all(disk.digest(history/"sources"/n) == h for n, h in old["sources"].items()),
                     "Legacy source history changed")
        disk.require(all(done.get(d) == commit for d, commit in migration["retained_document_commits"].items()),
                     "Migrated observations changed")
    if complete:
        disk.require(len(done) == 4, "Missing committed documents")
    results = {}
    for doc_id in done:
        raw, _ = validate_raw(root, doc_id)
        results[doc_id] = {}
        for point in disk.spec()["points"]:
            base = root / "analysis" / doc_id / point
            if complete:
                arrays, metrics = load_npz(base.with_suffix(".npz")), disk.read(base.with_suffix(".json"))
                verify_pca(raw["h_"+point], arrays, metrics, features(raw))
                results[doc_id][point] = metrics
    expected = {str(Path(d) / (p + suffix)) for d in done for p in disk.spec()["points"]
                for suffix in (".npz", ".json")}
    if complete:
        actual = {str(p.relative_to(root / "analysis")) for p in disk.files(root / "analysis")}
        disk.require(actual == expected, "Unexpected/missing analysis files")
    rows = disk.events(root)
    report = dict(status="RAW_AND_ANALYSIS_VERIFIED" if complete else "PARTIAL_RAW_VERIFIED",
                  run_id=run["run_id"], capsule=run["capsule"], code_release=disk.RELEASE,
                  planned_documents=4, committed_documents=len(done),
                  planned_document_points=16, analyzed_document_points=4*len(done) if complete else 0,
                  forward_attempts=sum(r["kind"] == "FORWARD_STARTED" for r in rows),
                  commit_events=sum(r["kind"] == "DOCUMENT_COMMITTED" for r in rows),
                  worker_failures=sum(r["kind"] == "WORKER_FAILURE" for r in rows),
                  unique_committed_tokens=256*len(done), metrics=results)
    # Success of re-reading raw data is distinct from protocol/recovery completion.
    summary = root / "summary.json"
    if complete and summary.exists():
        s = disk.read(summary)
        for key in ("run_id", "capsule", "planned_documents", "committed_documents",
                    "planned_document_points", "analyzed_document_points",
                    "forward_attempts", "commit_events", "worker_failures", "unique_committed_tokens", "metrics"):
            disk.require(s[key] == report[key], "Summary differs from raw reaggregation: " + key)
    return report


def fault_checks(root, scratch):
    """Damage disposable copies only; test the detector, not the scientific hypothesis."""
    root, scratch = Path(root), Path(scratch)
    tests = {}
    first = next(iter(disk.committed(root)))
    for name in ("missing_raw", "corrupt_raw", "wrong_specification"):
        with tempfile.TemporaryDirectory(dir=scratch, prefix="p0-fault-") as td:
            copy = Path(td) / "copy"
            shutil.copytree(root, copy)
            raw = copy / "docs" / first / "raw.npz"
            if name == "missing_raw":
                raw.unlink()
            elif name == "corrupt_raw":
                data = bytearray(raw.read_bytes())
                data[-1] ^= 1
                raw.write_bytes(data)
            else:
                run = disk.read(copy / "run.json")
                run["specification_sha256"] = "0"*64
                disk.write(copy / "run.json", run)
            try:
                verify(copy, complete=False)
            except (ValueError, OSError, AssertionError, zipfile.BadZipFile) as exc:
                tests[name] = dict(rejected=True, reason=str(exc))
            else:
                raise ValueError("Fault escaped detection: " + name)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="p0-duplicate-") as td:
        path = Path(td) / "duplicate.zip"
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("run.json", "{}")
                z.writestr("run.json", "{}")
        try:
            disk.inspect_zip(path)
        except ValueError as exc:
            tests["duplicate_zip_members"] = dict(rejected=True, reason=str(exc))
        else:
            raise ValueError("Duplicate ZIP members accepted")
    return tests


def receipt(path, output):
    with tempfile.TemporaryDirectory(prefix="p0-receipt-") as td:
        root = Path(td) / "run"
        index = disk.restore(path, root)
        disk.require(index["status"] == "FINAL", "Not a final archive; inspect partial/failure data separately")
        report = verify(root)
        protocol = disk.read(root / "protocol_checks.json")
        roundtrip, replay, faults = (protocol[k] for k in
                                    ("archive_roundtrip", "no_op_replay", "fault_checks"))
        disk.require(roundtrip["passed"] and roundtrip["restored_documents"] == 2 and
                     replay["passed"] and replay["forward_calls_added"] == 0 and
                     replay["documents_unchanged"] == 4 and
                     set(faults) == {"missing_raw", "corrupt_raw", "wrong_specification", "duplicate_zip_members"} and
                     all(v["rejected"] for v in faults.values()), "P0 protocol checks incomplete")
        disk.require(disk.read(root/"status.json")["status"] == "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT" and
                     disk.read(root/"summary.json")["status"] == "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT",
                     "Final status missing")
        report.update(status="LOCAL_RECEIPT_AND_RECOMPUTE_PASS", archive_sha256=disk.digest(path),
                      archive_bytes=Path(path).stat().st_size, checked_at=disk.stamp(),
                      protocol_checks=protocol)
        disk.write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "verify", "faults", "receipt"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.mode == "analyze":
        analyze(args.path)
        result = verify(args.path)
    elif args.mode == "faults":
        result = fault_checks(args.path, args.path.parent)
    elif args.mode == "receipt":
        disk.require(args.output is not None, "Receipt needs --output")
        result = receipt(args.path, args.output)
    else:
        result = verify(args.path, not args.partial)
    if args.output:
        disk.write(args.output, result)
    print(disk.canonical({k: v for k, v in result.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
