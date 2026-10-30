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
import p5_io as disk


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
                raise TimeoutError("P5 subprocess time limit reached")
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
                 platform.machine() == "x86_64", "P5 requires Colab CPython 3.11-3.13 / Linux x86_64")
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
    pointer = base/"current.json"
    if pointer.exists():
        disk.require(not restore_zip, "Leave RESTORE_ZIP blank to resume the live run")
        name = disk.read(pointer)["relative_root"]
        disk.require(Path(name).name == name and name not in ("", ".", ".."), "Invalid run pointer")
        root = base/name
        disk.committed(root)
        return root
    if restore_zip == "UPLOAD":
        from google.colab import files
        uploaded = files.upload()
        disk.require(len(uploaded) == 1, "Select one latest cumulative P5 ZIP")
        restore_zip = str(Path(next(iter(uploaded))).resolve())
        del uploaded
    root = base/("run_"+uuid.uuid4().hex[:12])
    if restore_zip:
        disk.restore(Path(restore_zip), root)
    else:
        disk.initialize(root)
    disk.write(pointer, dict(relative_root=root.name))
    return root


def run(restore_zip="", auto_download=True):
    disk.require(sys.platform == "linux" and Path("/content").is_dir(), "Run this cell in Colab")
    importlib.import_module("google.colab")
    import fcntl
    cfg = disk.spec()
    base = Path("/content")/(cfg["release"]+"_"+disk.capsule()[:12])
    base.mkdir(exist_ok=True)
    lock = (base/"run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = None
    delivered = []
    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    budget, started, accounted = None, None, False

    def account():
        nonlocal accounted
        if budget is not None and started is not None and not accounted:
            budget["elapsed_seconds"] += time.monotonic()-started
            disk.write(root/"budget.json", budget)
            accounted = True

    def deliver(label):
        path = disk.archive(root, base/"archives", label)
        print("\nZIP:", path, f"({path.stat().st_size/1e6:.2f} MB)", flush=True)
        record = dict(path=str(path), sha256=disk.digest(path), bytes=path.stat().st_size,
            zip_verified=True, download_requested=False, user_disk_confirmed=False)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(path))
                record["download_requested"] = True
            except Exception as exc:
                record["error"] = str(exc)
                print("Use the Colab Files panel to download this ZIP.", flush=True)
        delivered.append(record)
        disk.write(base/"delivery_latest.json", delivered)
        return path

    try:
        root = resolve(base, restore_zip)
        if (root/"FINAL.json").exists():
            disk.committed(root)
            disk.require(set(disk.committed(root)) == set(disk.keys()), "Final unit inventory differs")
            summary = disk.read(root/"summary.json")
            final = disk.read(root/"FINAL.json")
            disk.require(disk.digest(root/"summary.json") == final["summary_sha256"], "Final summary changed")
            for name, expected in final["analysis_sha256"].items():
                disk.require(disk.digest(root/name) == expected, "Final analysis changed: "+name)
            disk.require(summary["status"] == "RAW_VERIFIED_COMPLETE", "Final status differs")
            print("NOOP: completed data reused; no model download or forward.", flush=True)
            path = deliver("FINAL")
            return dict(status="COMPLETE_NOOP", zip=str(path), root=str(root))
        budget = disk.read(root/"budget.json")
        disk.require(budget["attempts"] < 1+cfg["technical_resumes"], "Technical resume limit reached")
        disk.require(budget["elapsed_seconds"] < cfg["total_seconds"], "Cumulative processing budget exhausted")
        started = time.monotonic()
        deadline = started+cfg["total_seconds"]-budget["elapsed_seconds"]
        budget["attempts"] += 1
        disk.write(root/"budget.json", budget)
        session = uuid.uuid4().hex[:12]
        logs = root/"sessions"/session
        logs.mkdir(parents=True)
        disk.write(logs/"start.json", dict(at=disk.stamp(), budget=budget))

        def overall_timeout(*_):
            raise TimeoutError("P5 cumulative processing budget reached")

        signal.signal(signal.SIGALRM, overall_timeout)
        signal.alarm(max(1, int(deadline-time.monotonic())))
        disk.require(shutil.disk_usage(base).free >= cfg["initial_free_bytes"], "Need 15 GiB free disk space")
        python, env = environment(base, logs, deadline)

        def call(name, args, logname):
            return process([str(python), "-u", str(disk.HERE/name)]+[str(a) for a in args],
                           deadline, logs/(logname+".log"), env)

        disk.require(call("p5_worker.py", [root, "cpu"], "cpu") == 0, "CPU acquisition failed")
        disk.require(call("p5_analysis.py", [root], "cpu_verify") == 0, "CPU raw verification failed")
        # Keep an externally recoverable artifact before the GPU comparison.
        disk.write(root/"budget.json", dict(budget,
                   elapsed_seconds=budget["elapsed_seconds"]+time.monotonic()-started))
        deliver("CHECKPOINT")
        disk.require(call("p5_worker.py", [root, "cuda"], "cuda") == 0, "CUDA acquisition failed")
        disk.require(call("p5_analysis.py", [root, "full"], "full_verify") == 0, "Full raw verification failed")
        disk.require(call("p5_worker.py", [root, "cpu"], "cpu_noop") == 0, "CPU no-op check failed")
        disk.require(call("p5_worker.py", [root, "cuda"], "cuda_noop") == 0, "CUDA no-op check failed")
        account()
        disk.write(logs/"end.json", dict(at=disk.stamp(), status="RAW_VERIFIED_COMPLETE"))
        disk.write(root/"FINAL.json", dict(at=disk.stamp(), capsule=disk.capsule(),
                   summary_sha256=disk.digest(root/"summary.json"),
                   analysis_sha256={str(p.relative_to(root)): disk.digest(p)
                                    for p in sorted((root/"analysis").glob("*.npz"))}), immutable=True)
        path = deliver("FINAL")
        return dict(status="COMPLETE", zip=str(path), root=str(root),
                    next="Save FINAL locally and share it for receipt verification and reporting.")
    except BaseException as exc:
        signal.alarm(0)
        traceback.print_exc()
        path = None
        if root is not None:
            try:
                account()
                disk.write(root/"failures"/(uuid.uuid4().hex+".json"), dict(at=disk.stamp(),
                           type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
                def closure_timeout(*_):
                    raise TimeoutError("P5 recovery budget reached")
                signal.signal(signal.SIGALRM, closure_timeout)
                signal.alarm(cfg["closure_seconds"])
                path = deliver("PARTIAL_OR_FAILURE")
            except BaseException as recovery:
                print("Recovery error:", repr(recovery), "\nEvidence remains at:", root, flush=True)
        return dict(status="FAILED_OR_INTERRUPTED", zip=str(path) if path else None,
                    root=str(root) if root else None, error=type(exc).__name__+": "+str(exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
        account()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

