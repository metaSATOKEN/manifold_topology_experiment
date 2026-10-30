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
import p0_io as disk


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
                raise TimeoutError("P0 subprocess time limit reached")
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
                 platform.machine() == "x86_64", "P0 requires Colab CPython 3.11-3.13 / Linux x86_64")
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
    pointer = base / "current.json"
    if pointer.exists():
        disk.require(not restore_zip, "A live run exists: leave RESTORE_ZIP blank to resume it")
        root = base / disk.safe_name(disk.read(pointer)["relative_root"])
        disk.check_run(root)
        disk.committed(root)
        return root
    fresh = restore_zip == "NEW"
    if not restore_zip:
        # This exact archive was supplied and diagnosed; prefer it on the original VM.
        known = disk.legacy_base() / "archives" / (
            "pcgeom_p0_v1_7ec0f9a0ff0c_PARTIAL_OR_FAILURE_1a75e0.zip")
        if known.is_file():
            disk.require(disk.digest(known) == "7069b3138b9eca3c0d5ce78d46f2b3cc1d7c8b5a9711c8aa0c66ee99efec135c",
                         "Known failure archive changed")
            restore_zip = str(known)
        else:
            candidates = sorted(Path("/content").glob("pcgeom_p0_v*_*.zip"))
            disk.require(len(candidates) <= 1, "Multiple ZIPs found: set RESTORE_ZIP to the intended cumulative ZIP")
            restore_zip = str(candidates[0]) if candidates else "UPLOAD"
    if restore_zip == "UPLOAD":
        from google.colab import files
        uploaded = files.upload()
        disk.require(len(uploaded) == 1, "Upload one cumulative P0 ZIP")
        restore_zip = str(Path(next(iter(uploaded))).resolve())
        del uploaded
    root = base / ("run_"+uuid.uuid4().hex[:12])
    if fresh:
        disk.initialize(root)
    else:
        disk.restore(Path(restore_zip), root, allow_legacy=True)
    disk.write(pointer, dict(relative_root=root.name))
    return root


def run(restore_zip="", auto_download=True):
    disk.require(sys.platform == "linux" and Path("/content").is_dir(), "Paste into a Colab GPU cell")
    importlib.import_module("google.colab")
    import fcntl
    base = Path("/content") / (disk.RELEASE+"_"+disk.capsule()[:12])
    base.mkdir(exist_ok=True)
    lock = (base/"run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = None
    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    delivered = []

    def download(path):
        print("\nZIP:", path, f"({Path(path).stat().st_size/1e6:.2f} MB)", flush=True)
        entry = dict(path=str(path), sha256=disk.digest(path), bytes=Path(path).stat().st_size,
                     requested=False, local_save_confirmed=False)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(path))
                entry["requested"] = True
            except Exception as exc:
                entry["error"] = str(exc)
                print("Download could not be requested. Use the Colab Files panel.", flush=True)
        delivered.append(entry)
        disk.write(base/"delivery_latest.json", delivered)

    try:
        root = resolve(base, restore_zip)
        started = time.monotonic()
        deadline = started + disk.spec()["total_seconds"]
        def overall_timeout(*_):
            raise TimeoutError("P0 processing exceeded 40 minutes")
        signal.signal(signal.SIGALRM, overall_timeout)
        signal.alarm(disk.spec()["total_seconds"])
        m = disk.check_run(root)
        session = uuid.uuid4().hex[:12]
        logs = root/"sessions"/session
        logs.mkdir(parents=True)
        disk.write(logs/"start.json", dict(at=disk.stamp(), run_id=m["run_id"],
                                         total_seconds=disk.spec()["total_seconds"]))
        # A rerun after success offers the existing final ZIP again, even if a previous save was cancelled.
        finals = sorted((base/"archives").glob(f"{disk.RELEASE}_{m['run_id']}_FINAL_*.zip"),
                        key=lambda p: p.stat().st_mtime)
        if finals and disk.read(root/"status.json")["status"] == "COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT":
            disk.inspect_zip(finals[-1])
            download(finals[-1])
            return dict(status="NO_NEW_COMPUTE", zip=str(finals[-1]))
        disk.require(shutil.disk_usage(base).free >= disk.spec()["initial_free_bytes"],
                     "P0 requires 15 GiB free disk for isolated libraries, weights and checkpoints")
        print("P0: GPT-2, 4 texts x 256 tokens x 4 extraction points; no training.", flush=True)
        print("First setup estimate: 10-20 min; processing cap 40 min plus 2 min recovery.", flush=True)
        # Keep the original pinned venv/cache when resuming on the same VM.
        # environment() checks all pins; worker additionally checks the recorded environment.
        dependencies = disk.legacy_base() if (disk.legacy_base()/"venv/bin/python").is_file() else base
        python, env = environment(dependencies, logs, deadline)

        def call(script, args, name):
            return process([str(python), "-u", str(disk.HERE/script), *map(str, args)],
                           deadline, logs/(name+".log"), env)

        done = disk.committed(root)
        if len(done) < 2:
            disk.require(call("p0_worker.py", [root, 2], "extract_first") == 75, "First extraction did not reach controlled stop")
        check_path = root/"roundtrip.json"
        if not check_path.exists():
            disk.require(call("p0_verify.py", ["verify", root, "--partial"], "verify_checkpoint") == 0,
                         "Checkpoint raw verification failed")
            before = disk.committed(root)
            checkpoint = disk.snapshot(root, base/"archives", "CHECKPOINT")
            download(checkpoint)
            original_root = root.name
            restored = base/("run_restored_"+uuid.uuid4().hex[:12])
            disk.restore(checkpoint, restored)
            disk.require(disk.committed(restored) == before, "ZIP restore changed committed observations")
            # Keep the old directory as an immutable recovery source. Continue in the restored copy.
            disk.write(base/"current.json", dict(relative_root=restored.name))
            root = restored
            logs = root/"sessions"/session
            disk.write(root/"roundtrip.json", dict(passed=True, checkpoint_sha256=disk.digest(checkpoint),
                       restored_documents=len(before), original_relative_root=original_root,
                       checkpoint_filename=checkpoint.name,
                       scope="Separate worker and filesystem restore within the same VM; no browser receipt claim"))
        disk.require(call("p0_worker.py", [root, 4], "extract_rest") == 0, "Extraction failed")
        before = disk.committed(root)
        starts = sum(r["kind"] == "FORWARD_STARTED" for r in disk.events(root))
        disk.require(call("p0_worker.py", [root, 4], "noop_replay") == 0, "No-op replay failed")
        disk.require(disk.committed(root) == before and
                     sum(r["kind"] == "FORWARD_STARTED" for r in disk.events(root)) == starts,
                     "Replay changed evidence or repeated forward calls")
        disk.require(call("p0_verify.py", ["analyze", root, "--output", logs/"recomputed.json"], "analysis") == 0,
                     "Analysis/recomputation failed")
        disk.require(call("p0_verify.py", ["faults", root, "--output", logs/"faults.json"], "faults") == 0,
                     "Fault rejection checks failed")
        disk.write(root/"protocol_checks.json", dict(archive_roundtrip=disk.read(root/"roundtrip.json"),
                   no_op_replay=dict(passed=True, forward_calls_added=0, documents_unchanged=4),
                   fault_checks=disk.read(logs/"faults.json")))
        summary = disk.read(logs/"recomputed.json")
        summary.update(status="COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT",
                       wall_seconds_this_cell=time.monotonic()-started,
                       local_receipt_verified=False, note="P0 transport fixtures; no scientific generalization")
        disk.write(root/"summary.json", summary)
        disk.require(call("p0_verify.py", ["verify", root], "final_verify") == 0, "Final raw/summary audit failed")
        disk.write(root/"status.json", dict(status=summary["status"]))
        final = disk.snapshot(root, base/"archives", "FINAL")
        print("\nP0 Colab checks passed. Share the FINAL ZIP; it contains all four documents.", flush=True)
        print("Local receipt/recomputation remains pending. Browser request is not disk-save confirmation.", flush=True)
        download(final)
        return dict(status=summary["status"], zip=str(final), root=str(root))
    except BaseException as exc:
        if root is None:
            # Preserve diagnostics even if restoration/initialization failed.
            root = base/("failure_"+uuid.uuid4().hex[:12])
            disk.initialize(root)
        disk.write(root/("failure_"+uuid.uuid4().hex[:8]+".json"),
                   dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        disk.write(root/"status.json", dict(status="FAILED_OR_INTERRUPTED"))
        traceback.print_exc()

        def expired(*_):
            raise TimeoutError("Failure ZIP recovery exceeded two minutes")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(disk.spec()["closure_seconds"])
        try:
            path = disk.snapshot(root, base/"archives", "PARTIAL_OR_FAILURE")
        except BaseException as archive_exc:
            print("Could not package diagnostics:", archive_exc, "\nEvidence retained at", root, flush=True)
            return dict(status="FAILED_ARCHIVE_UNAVAILABLE", root=str(root))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        download(path)
        return dict(status="FAILED_OR_INTERRUPTED", zip=str(path), root=str(root))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
