"""Colab orchestration, isolated install, controlled restart and bounded ZIP recovery."""
import importlib
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import p1_io as disk


def child_environment(base):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") and
           not k.startswith("PIP_")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE=os.devnull,
               HF_HOME=str(Path(base)/"hub"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_DOWNLOAD_TIMEOUT="120")
    return env


def process(command, deadline, log, env):
    disk.require(deadline > time.monotonic(), "Run time limit reached")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, start_new_session=True)
    messages = queue.Queue(maxsize=256)

    def reader():
        try:
            for line in proc.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    retained = ""
    try:
        eof = False
        while not eof or proc.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("P1 subprocess time limit reached")
            try:
                line = messages.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            print(line, end="", flush=True)
            retained = (retained+line)[-1024**2:]
        return proc.wait()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        disk.atomic(log, retained.encode())


def environment(base, logs, deadline):
    disk.require(sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and
                 platform.machine() == "x86_64", "P1 requires Colab CPython 3.11-3.13 / Linux x86_64")
    envdir = base / "venv"
    python = envdir / "bin" / "python"
    env = child_environment(base)
    until = min(deadline, time.monotonic()+disk.spec()["prepare_seconds"])
    if not python.exists():
        rc = process([sys.executable, "-I", "-m", "venv", "--without-pip", str(envdir)],
                     min(until, time.monotonic()+120), logs/"venv.log", env)
        disk.require(rc == 0, "Could not create isolated environment")
    pins = disk.spec()["pins"]
    code = ("import importlib.metadata as m,json; p="+repr(pins)+"; "
            "v={k:m.version(k) for k in p}; print(json.dumps(v)); assert v==p")
    check = subprocess.run([str(python), "-I", "-c", code], env=env,
                           capture_output=True, text=True, timeout=30)
    if check.returncode:
        prefix = [sys.executable, "-I", "-m", "pip", "--isolated", "--python", str(python),
                  "install", "--disable-pip-version-check", "--no-cache-dir", "--only-binary=:all:"]
        disk.require(process(prefix+["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
                     until, logs/"torch_install.log", env) == 0, "PyTorch installation failed")
        requirements = [f"{n}=={v}" for n, v in pins.items() if n != "torch"]
        disk.require(process(prefix+["--index-url", "https://pypi.org/simple"]+requirements,
                     until, logs/"python_install.log", env) == 0, "Analysis dependency installation failed")
    disk.require(process([str(python), "-I", "-c", code], until, logs/"versions.log", env) == 0,
                 "Installed versions differ from fixed contract")
    disk.require(process([str(python), "-m", "pip", "check"], until, logs/"pip_check.log", env) == 0,
                 "Dependency check failed")
    return python, env


def resolve(base, restore_zip):
    pointer=base/"current.json"
    if pointer.exists() and restore_zip != "NEW":
        disk.require(not restore_zip,"Existing run: leave RESTORE_ZIP blank, or NEW for a separate run")
        root=base/disk.safe_name(disk.read(pointer)["relative_root"])
        disk.check_run(root)
        disk.committed(root)
        return root
    if restore_zip == "UPLOAD":
        from google.colab import files
        uploaded=files.upload()
        disk.require(len(uploaded)==1,"Upload one cumulative P1 ZIP")
        restore_zip=str(Path(next(iter(uploaded))).resolve())
        del uploaded
    root=base/("run_"+uuid.uuid4().hex[:12])
    if restore_zip and restore_zip != "NEW":
        disk.restore(Path(restore_zip),root)
    else:
        disk.initialize(root)
    disk.write(pointer,dict(relative_root=root.name))
    return root


def run(restore_zip="",auto_download=True):
    disk.require(sys.platform=="linux" and Path("/content").is_dir(),"Paste into a Colab GPU cell")
    importlib.import_module("google.colab")
    import fcntl
    cfg=disk.spec()
    base=Path("/content")/(disk.RELEASE+"_"+disk.capsule()[:12])
    base.mkdir(exist_ok=True)
    lock=(base/"run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=None
    previous_handler=signal.getsignal(signal.SIGALRM)
    delivered=[]

    def download(path):
        print(f"\nZIP: {path} ({path.stat().st_size/1e6:.2f} MB)",flush=True)
        entry=dict(path=str(path),bytes=path.stat().st_size,sha256=disk.digest(path),
                   requested=False,local_save_confirmed=False)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(path))
                entry["requested"]=True
            except Exception as exc:
                entry["error"]=str(exc)
                print("Download request failed; download this ZIP from the Files panel.",flush=True)
        delivered.append(entry)
        disk.write(base/"delivery_latest.json",delivered)

    try:
        root=resolve(base,restore_zip)
        started=time.monotonic()
        deadline=started+cfg["total_seconds"]
        def timeout(*_):
            raise TimeoutError("P1 reached the 120-minute processing limit")
        signal.signal(signal.SIGALRM,timeout)
        signal.alarm(cfg["total_seconds"])
        m=disk.check_run(root)
        session=uuid.uuid4().hex[:12]
        logs=root/"sessions"/session
        logs.mkdir(parents=True)
        disk.write(logs/"start.json",dict(at=disk.stamp(),run_id=m["run_id"],
            total_seconds=cfg["total_seconds"],closure_seconds=cfg["closure_seconds"]))
        finals=sorted((base/"archives").glob(f"{disk.RELEASE}_{m['run_id']}_FINAL_*.zip"),
                      key=lambda p:p.stat().st_mtime)
        if finals and disk.read(root/"status.json")["status"]=="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT":
            disk.inspect_zip(finals[-1])
            download(finals[-1])
            return dict(status="NO_NEW_COMPUTE",zip=str(finals[-1]),root=str(root))
        disk.require(shutil.disk_usage(base).free>=cfg["initial_free_bytes"],
                     "P1 needs 20 GiB free disk for libraries, observations and cumulative ZIPs")
        print("P1: GPT-2; 64 articles x 512 tokens x 4 points. No training.",flush=True)
        print("First setup plus analysis estimate: 20-60 min; cap 120 min + 5 min recovery.",flush=True)
        python,env=environment(base,logs,deadline)
        def call(script,args,label):
            return process([str(python),"-u",str(disk.HERE/script),*map(str,args)],
                           deadline,logs/(label+".log"),env)
        for stop in range(cfg["checkpoint_documents"],cfg["documents"]+1,cfg["checkpoint_documents"]):
            if len(disk.committed(root)) > stop:
                continue
            disk.require(call("p1_worker.py",[root,stop],f"extract_{stop}")==0,"Extraction failed")
            disk.require(call("p1_verify.py",["analyze",root],f"analyze_{stop}")==0,"Analysis failed")
            label=f"CHECKPOINT_{stop:03d}"
            existing=list((base/"archives").glob(f"{disk.RELEASE}_{m['run_id']}_{label}_*.zip"))
            # Cumulative checkpoints need structural integrity; independent numerical audit is final.
            if stop<cfg["documents"] and not existing:
                disk.write(root/"status.json",dict(status="CHECKPOINT_PENDING_NUMERIC_AUDIT",
                    committed_documents=len(disk.committed(root))))
                download(disk.snapshot(root,base/"archives",label))
        starts=sum(e["kind"]=="FORWARD_STARTED" for e in disk.events(root))
        before=disk.committed(root)
        disk.require(call("p1_worker.py",[root,cfg["documents"]],"noop")==0,"No-op replay failed")
        disk.require(before==disk.committed(root) and starts==sum(
            e["kind"]=="FORWARD_STARTED" for e in disk.events(root)),"No-op changed evidence")
        disk.require(call("p1_verify.py",["verify",root,"--output",logs/"verified.json"],"verify")==0,
                     "Independent saved-array recomputation failed")
        summary=disk.read(logs/"verified.json")
        summary.update(status="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT",
            local_receipt_verified=False,wall_seconds_through_summary=time.monotonic()-started,
            no_op_added_forwards=0)
        disk.write(root/"summary.json",summary)
        disk.write(root/"status.json",dict(status=summary["status"]))
        final=disk.snapshot(root,base/"archives","FINAL")
        print("\nP1 computation and saved-array audit complete. Share the FINAL ZIP.",flush=True)
        print("Hypothesis outcomes:",summary["scientific_adjudication"],flush=True)
        print("Transfer receipt remains pending; a browser request is not a saved-file confirmation.",flush=True)
        download(final)
        return dict(status=summary["status"],zip=str(final),root=str(root))
    except BaseException as exc:
        signal.alarm(0)
        if root is None:
            root=base/("failure_"+uuid.uuid4().hex[:12])
            disk.initialize(root)
        disk.write(root/("failure_"+uuid.uuid4().hex[:8]+".json"),
            dict(at=disk.stamp(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        disk.write(root/"status.json",dict(status="FAILED_OR_INTERRUPTED"))
        traceback.print_exc()
        def recovery_timeout(*_):
            raise TimeoutError("P1 failure packaging exceeded five minutes")
        signal.signal(signal.SIGALRM,recovery_timeout)
        signal.alarm(cfg["closure_seconds"])
        try:
            path=disk.snapshot(root,base/"archives","PARTIAL_OR_FAILURE")
        except BaseException as recovery:
            print("Could not package ZIP:",recovery,"\nEvidence remains at",root,flush=True)
            return dict(status="FAILED_ARCHIVE_UNAVAILABLE",root=str(root))
        finally:
            signal.alarm(0)
        download(path)
        return dict(status="FAILED_OR_INTERRUPTED",zip=str(path),root=str(root))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous_handler)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


