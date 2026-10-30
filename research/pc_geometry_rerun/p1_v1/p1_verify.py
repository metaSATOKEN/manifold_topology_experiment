"""Model-free CLI. Receipt verification always uses locally trusted source."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
import p1_io as disk


def point_folder(root, doc, point):
    return Path(root)/"analysis"/doc/point


def point_commit(folder):
    names = ("arrays.npz","metrics.json")
    m = disk.read(folder/"COMMIT.json")
    disk.require(set(m)==set(names), "Analysis commit inventory")
    disk.require({p.name for p in folder.iterdir()}==set(names)|{"COMMIT.json"}, "Analysis files")
    for n,h in m.items():
        disk.require(disk.digest(folder/n)==h, "Analysis corrupted: "+str(folder/n))
    return m


def analyze(root):
    from p1_numeric import validate_raw, save_npz
    from p1_analysis import analyze_point
    done = disk.committed(root)
    for fixture in disk.fixtures():
        doc = fixture["id"]
        if doc not in done:
            continue
        raw,_ = validate_raw(root,doc)
        for point in disk.spec()["points"]:
            dest = point_folder(root,doc,point)
            if (dest/"COMMIT.json").exists():
                point_commit(dest)
                continue
            disk.require(sum(p.stat().st_size for p in disk.files(root)) < disk.RAW_LIMIT-16*1024**2,
                         "Evidence budget reached; no new analysis started")
            if dest.exists():
                orphan = Path(root)/"analysis_incomplete"/(uuid.uuid4().hex+"_"+doc+"_"+point)
                orphan.parent.mkdir(parents=True,exist_ok=True)
                os.replace(dest,orphan)
            started = time.monotonic()
            arrays, meta = analyze_point(raw,doc,point)
            save_npz(dest/"arrays.npz",arrays)
            disk.write(dest/"metrics.json",meta,immutable=True)
            disk.write(dest/"COMMIT.json",{n:disk.digest(dest/n) for n in ("arrays.npz","metrics.json")},
                       immutable=True)
            print(f"P1 analysis saved: {doc} / {point} ({time.monotonic()-started:.1f}s)",flush=True)
    return dict(status="ANALYSIS_SAVED_PENDING_RECOMPUTE",committed_documents=len(done))


def verify(root, partial=False):
    from p1_numeric import load_npz, validate_raw
    from p1_analysis import summarize, verify_point
    cfg = disk.spec()
    m = disk.check_run(root)
    done = disk.committed(root)
    if not partial:
        disk.require(len(done)==cfg["documents"],"Incomplete planned documents")
    rows = []
    for fixture in disk.fixtures():
        doc=fixture["id"]
        if doc not in done:
            continue
        raw,_=validate_raw(root,doc)
        for point in cfg["points"]:
            folder=point_folder(root,doc,point)
            if partial and not (folder/"COMMIT.json").exists():
                continue
            point_commit(folder)
            arrays=load_npz(folder/"arrays.npz")
            meta=disk.read(folder/"metrics.json")
            verify_point(raw,doc,point,arrays,meta)
            rows.append(meta)
        print(f"P1 verified: {doc}",flush=True)
    events=disk.events(root)
    commits=[e["document_id"] for e in events if e["kind"]=="DOCUMENT_COMMITTED"]
    # A crash between sealing and logging is visible, but sealed evidence is not discarded.
    disk.require(len(commits)==len(set(commits)), "Duplicate commit events")
    disk.require(set(commits)<=set(done), "Commit log refers to absent document")
    summary=summarize(root,rows)
    summary.update(status="RAW_AND_ANALYSIS_VERIFIED" if not partial else "PARTIAL_VERIFIED",
        run_id=m["run_id"],capsule=m["capsule"],committed_documents=len(done),
        forward_attempts=sum(e["kind"]=="FORWARD_STARTED" for e in events),
        worker_failures=sum(e["kind"]=="WORKER_FAILURE" for e in events),
        commit_events=len(commits),sealed_documents_without_commit_event=sorted(set(done)-set(commits)),
        unique_committed_tokens=len(done)*cfg["tokens"])
    if not partial:
        disk.require(len(rows)==cfg["documents"]*len(cfg["points"]),"Incomplete planned analyses")
    return summary


def receipt(path, partial=False):
    index=disk.inspect_zip(path)
    with tempfile.TemporaryDirectory(prefix="pcgeom-p1-receipt-") as td:
        root=Path(td)/"run"
        disk.restore(path,root)
        result=verify(root,partial=partial)
        if not partial:
            from p1_numeric import compare_tree
            declared=disk.read(root/"summary.json")
            for key in ("endpoints","scientific_adjudication","missing","observed_document_ids","document_endpoints"):
                compare_tree(declared[key],result[key])
            disk.require(declared["status"]=="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT",
                         "Not a completed Colab summary")
        result.update(status="LOCAL_RECEIPT_AND_RECOMPUTE_PASS" if not partial else "LOCAL_PARTIAL_VERIFIED",
            archive_sha256=disk.digest(path),archive_bytes=Path(path).stat().st_size,
            archive_label=index["status"],checked_at=disk.stamp())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("analyze","verify","receipt"))
    parser.add_argument("path")
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.action=="analyze":
        result=analyze(args.path)
    elif args.action=="verify":
        result=verify(args.path,args.partial)
    else:
        result=receipt(args.path,args.partial)
    if args.output:
        disk.write(args.output,result)
    print(disk.canonical(result),flush=True)


if __name__=="__main__":
    main()
