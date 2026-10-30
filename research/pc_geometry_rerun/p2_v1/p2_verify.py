"""Train fit, evaluate, audit saved observations, and verify transported P2 ZIPs."""
import argparse
import os
from pathlib import Path
import tempfile
import uuid
import numpy as np
import p2_io as disk
import p2_numeric as num
import p2_fit as fit
import p2_analysis as analysis


def folder_for(root,doc,point,condition):
    return Path(root)/"evaluation"/doc/(point+"__"+condition)


def point_commit(folder):
    folder=Path(folder)
    commit=disk.read(folder/"COMMIT.json")
    disk.require(set(commit)=={"arrays.npz","metrics.json"},"Evaluation commit inventory")
    disk.require({p.name for p in folder.iterdir()}==set(commit)|{"COMMIT.json"},"Unexpected evaluation member")
    for name,h in commit.items():
        disk.require(disk.digest(folder/name)==h,"Evaluation corruption")


def frozen_models(root):
    fit.check_frozen_fit(root)
    models={}
    for point in disk.spec()["points"]:
        for c in disk.spec()["conditions"]:
            folder=Path(root)/"fit"/(point+"__"+c)
            models[(point,c)]=(num.load_npz(folder/"arrays.npz"),disk.read(folder/"meta.json"))
    counts=num.load_npz(Path(root)/"fit"/"frequency.npz")["counts"]
    return models,counts


def evaluate_saved(root):
    root=Path(root)
    done=disk.committed(root)
    if not set(fit.split_ids("train"))<=set(done):
        disk.require(not (set(done)&set(fit.split_ids("test"))),"Test precedes training")
        return dict(status="TRAINING_INCOMPLETE",committed_documents=len(done))
    models,counts=frozen_models(root)
    for doc in fit.split_ids("test"):
        if doc not in done:
            continue
        raw,_=num.validate_raw(root,doc)
        for (point,c),(a,m) in models.items():
            folder=folder_for(root,doc,point,c)
            if (folder/"COMMIT.json").exists():
                point_commit(folder)
                continue
            if folder.exists():
                dest=root/"attempts"/("eval_"+uuid.uuid4().hex[:12])/"incomplete_before_retry"/doc/(point+"__"+c)
                dest.parent.mkdir(parents=True,exist_ok=True)
                os.replace(folder,dest)
            output,metrics=analysis.evaluate(raw,doc,point,c,a,m,counts)
            num.save_npz(folder/"arrays.npz",output)
            disk.write(folder/"metrics.json",metrics,immutable=True)
            disk.write(folder/"COMMIT.json",{n:disk.digest(folder/n) for n in ("arrays.npz","metrics.json")},immutable=True)
        print("P2 evaluation saved: "+doc,flush=True)
    return dict(status="EVALUATION_SAVED_PENDING_AUDIT",committed_documents=len(done))


def verify(root,partial=False):
    root=Path(root);cfg=disk.spec();run=disk.check_run(root);done=disk.committed(root)
    train=set(fit.split_ids("train"));test=set(fit.split_ids("test"))
    if not partial:
        disk.require(len(done)==cfg["documents"],"Incomplete planned documents")
    rows=[]
    for doc in done:
        num.validate_raw(root,doc)
    if (root/"fit"/"COMMIT.json").exists():
        fit.audit_fit(root)
        models,counts=frozen_models(root)
        for doc in fit.split_ids("test"):
            if doc not in done:
                continue
            raw,_=num.validate_raw(root,doc)
            for (point,c),(a,m) in models.items():
                folder=folder_for(root,doc,point,c)
                if partial and not (folder/"COMMIT.json").exists():
                    continue
                point_commit(folder)
                expected,metrics=analysis.evaluate(raw,doc,point,c,a,m,counts)
                stored=num.load_npz(folder/"arrays.npz")
                disk.require(set(stored)==set(expected),"Evaluation array fields")
                for key in expected:
                    disk.require(stored[key].dtype==expected[key].dtype,"Evaluation dtype")
                    np.testing.assert_allclose(stored[key],expected[key],atol=1e-8,rtol=1e-7,equal_nan=True)
                num.compare_tree(disk.read(folder/"metrics.json"),metrics)
                rows.append(metrics)
            print("P2 evaluation verified: "+doc,flush=True)
    else:
        disk.require(not (set(done)&test),"Test observations without frozen fit")
    complete=(len(done)==cfg["documents"] and
              len(rows)==cfg["test_documents"]*len(cfg["points"])*len(cfg["conditions"]))
    if not partial:
        disk.require(complete,"Incomplete evaluation")
    events=disk.events(root)
    commits=[e["document_id"] for e in events if e["kind"]=="DOCUMENT_COMMITTED"]
    disk.require(len(commits)==len(set(commits)) and set(commits)<=set(done),"Duplicate or invalid commits")
    if set(done)&test:
        sha=disk.digest(root/"fit"/"COMMIT.json")
        for e in events:
            if e["kind"]=="FORWARD_STARTED" and e["document_id"] in test:
                disk.require(e.get("fit_commit_sha256")==sha and e.get("split")=="test","Forward fit reference")
        sealed=disk.read(root/"fit"/"COMMIT.json")["sealed_at"]
        starts=[e["at"] for e in events if e["kind"]=="FORWARD_STARTED" and e["document_id"] in test]
        disk.require(all(t>=sealed for t in starts),"Test event precedes fit seal")
    result=analysis.summarize(rows,complete)
    result.update(status="RAW_FIT_AND_EVALUATION_VERIFIED" if complete else "PARTIAL_VERIFIED",
        run_id=run["run_id"],capsule=run["capsule"],planned_documents=cfg["documents"],
        committed_documents=len(done),train_documents=len(set(done)&train),test_documents=len(set(done)&test),
        evaluated_test_point_conditions=len(rows),
        planned_test_point_conditions=cfg["test_documents"]*len(cfg["points"])*len(cfg["conditions"]),
        unique_committed_tokens=len(done)*cfg["tokens"],
        forward_attempts=sum(e["kind"]=="FORWARD_STARTED" for e in events),
        worker_failures=sum(e["kind"]=="WORKER_FAILURE" for e in events),
        commit_events=len(commits),sealed_documents_without_commit_event=sorted(set(done)-set(commits)))
    return result


def receipt(path,partial=False):
    index=disk.inspect_zip(path)
    with tempfile.TemporaryDirectory(prefix="pcgeom-p2-receipt-") as td:
        root=Path(td)/"run"
        disk.restore(path,root)
        result=verify(root,partial)
        if not partial:
            declared=disk.read(root/"summary.json")
            disk.require(declared["status"]=="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT","Not a final summary")
            for key in ("endpoints","scientific_adjudication","document_endpoints","missing"):
                num.compare_tree(declared[key],result[key])
        result.update(status="LOCAL_RECEIPT_AND_RECOMPUTE_PASS" if not partial else "LOCAL_PARTIAL_VERIFIED",
                      archive_sha256=disk.digest(path),archive_bytes=Path(path).stat().st_size,
                      archive_label=index["status"],checked_at=disk.stamp())
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","evaluate","verify","receipt"))
    parser.add_argument("path")
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.action=="fit":
        result=fit.seal_fit(args.path)
    elif args.action=="evaluate":
        result=evaluate_saved(args.path)
    else:
        result=globals()[args.action](args.path,args.partial)
    if args.output:
        disk.write(args.output,result)
    print(disk.canonical(result),flush=True)


if __name__=="__main__":
    main()
