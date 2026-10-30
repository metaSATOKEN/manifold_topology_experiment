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
import p4_io as disk


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


TICK = lambda: None


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
            TICK()
            if time.monotonic() >= deadline:
                raise TimeoutError("P4 subprocess time limit reached")
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
                 platform.machine() == "x86_64", "P4 requires Colab CPython 3.11-3.13 / Linux x86_64")
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


class Budget:
    def __init__(self,path):
        self.path=Path(path);previous=disk.read(path) if self.path.exists() else {}
        self.prior=previous.get('seconds',0.)+(60 if previous.get('active',False) else 0)
        self.sessions=previous.get('sessions',0)+1
        disk.require(self.sessions<=1+disk.spec()['technical_resumes'],'Technical resume limit reached; share existing evidence')
        self.start=time.monotonic();self.last=0.;self.closed=False
        self.tick(force=True)
    def seconds(self):return self.prior+time.monotonic()-self.start
    def tick(self,force=False):
        now=time.monotonic()
        if force or now-self.last>=5:
            disk.write(self.path,dict(seconds=self.seconds(),sessions=self.sessions,active=not self.closed,at=disk.stamp(),
                limitation='Journal every 5 seconds while awaiting subprocesses and at phase boundaries. Abrupt VM loss, especially during synchronous ZIP work, can lose unjournaled time; add 60 seconds on unclosed restore'))
            self.last=now
    def close(self):self.closed=True;self.tick(force=True)


def resolve(base,restore_zips):
    pointer=base/'current.json'
    if pointer.exists():
        disk.require(not restore_zips,'Existing VM: leave RESTORE_ZIPS blank; no overwrite')
        root=base/disk.safe_name(disk.read(pointer)['relative_root']);disk.check_run(root);disk.committed(root)
        return root
    if restore_zips=='UPLOAD':
        from p4_download import upload_streaming
        paths=upload_streaming(base/'uploads')
        disk.require(paths,'No restore files selected; refusing silent fresh run')
    elif restore_zips:
        paths=[Path(p).expanduser().resolve() for p in (restore_zips if isinstance(restore_zips,list) else [restore_zips])]
    else:paths=[]
    root=base/('run_'+uuid.uuid4().hex[:12])
    if paths:disk.restore(paths,root)
    else:disk.initialize(root)
    disk.write(pointer,dict(relative_root=root.name));return root


def existing_set(base,run_id,label):
    indexes=sorted((base/'archives').glob(f'{disk.RELEASE}_{run_id}_{label}_*_INDEX.zip'),key=lambda p:p.stat().st_mtime)
    if not indexes:return None
    index=indexes[-1];bundle=disk.inspect_zip(index)['bundle']
    paths=[index.parent/e['name'] for e in bundle['parts']]+[index]
    disk.inspect_bundle(paths);return paths


def run(restore_zips='',auto_download=True):
    global TICK
    disk.require(sys.platform=='linux' and Path('/content').is_dir(),'Paste into Colab GPU runtime')
    importlib.import_module('google.colab');import fcntl
    from p4_download import Delivery
    cfg=disk.spec();base=Path('/content')/(disk.RELEASE+'_'+disk.capsule()[:12]);base.mkdir(exist_ok=True)
    lock=(base/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_handler=signal.getsignal(signal.SIGALRM);root=None;budget=None;delivery=None
    def send(paths):
        print('\nZIP SET (keep ALL PARTs and INDEX of this set):',flush=True)
        for path in paths:
            print(f'{path} ({path.stat().st_size/1e6:.2f} MB)',flush=True)
            try:delivery.enqueue(path)
            except Exception as exc:print('Browser queue unavailable; download from Files panel:',exc,flush=True)
        print('The serial browser queue may continue after this cell ends. Save requests are not disk confirmation.',flush=True)
    try:
        root=resolve(base,restore_zips);runmeta=disk.check_run(root)
        try:delivery=Delivery(base/'delivery'/runmeta['run_id'],auto_download)
        except Exception as exc:
            print('Queue setup failed; files will remain available:',exc,flush=True)
            delivery=Delivery(base/'delivery'/runmeta['run_id'],False)
        if disk.read(root/'status.json')['status']=='COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT':
            paths=existing_set(base,runmeta['run_id'],'FINAL')
            if paths:
                send(paths);return dict(status='NO_NEW_COMPUTE',root=str(root),zips=[str(p) for p in paths])
        budget=Budget(root/'active_time.json');TICK=budget.tick
        remaining=cfg['total_seconds']-budget.seconds();disk.require(remaining>0,'Cumulative processing budget exhausted')
        deadline=time.monotonic()+remaining
        def timeout(*_):raise TimeoutError('P4 cumulative 180-minute limit reached')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
        session=uuid.uuid4().hex[:12];logs=root/'sessions'/session;logs.mkdir(parents=True)
        disk.write(logs/'start.json',dict(at=disk.stamp(),remaining_seconds=remaining,session_number=budget.sessions))
        disk.require(shutil.disk_usage(base).free>=cfg['initial_free_bytes'],'Need 40 GiB free disk for models, observations and bounded ZIP sets')
        print('P4: GPT-2 / OPT-1.3B / Qwen2.5-1.5B base; 64 train + 64 NEW test articles.',flush=True)
        print('Plan 25-60 min (slow network/CPU may need 60-120); cap 180 min + 10 recovery. ZIPs <300 MB each.',flush=True)
        python,env=environment(base,logs,deadline)
        def call(script,args,label):
            return process([str(python),'-u',str(disk.HERE/script),*map(str,args)],deadline,logs/(label+'.log'),env)
        if not (root/'fit/COMMIT.json').exists():
            for key in cfg['models']:
                disk.require(call('p4_worker.py',[root,key,'train'],'train_'+key)==0,'Training extraction failed: '+key)
            disk.require(call('p4_fit.py',[root],'fit_all')==0,'Training fit failed')
        disk.check_fit(root);budget.tick(force=True)
        trainpaths=existing_set(base,runmeta['run_id'],'TRAIN_SEALED')
        if trainpaths is None and not (set(disk.committed(root))&set(disk.split_keys('test'))):
            disk.write(root/'status.json',dict(status='TRAIN_SEALED'))
            trainpaths=disk.snapshot(root,base/'archives','TRAIN_SEALED')
        if trainpaths:send(trainpaths)
        for key in cfg['models']:
            disk.require(call('p4_worker.py',[root,key,'test'],'test_'+key)==0,'Evaluation extraction failed: '+key)
            disk.require(call('p4_analysis.py',[root,key],'evaluate_'+key)==0,'Saved evaluation failed: '+key)
            budget.tick(force=True)
            if key!=cfg['models'][-1] and existing_set(base,runmeta['run_id'],'TEST_'+key) is None:
                paths=disk.snapshot(root,base/'archives','TEST_'+key)
                print(f'Checkpoint set retained in {base / "archives"}: {len(paths)} ZIPs; manual download if needed.',flush=True)
        before=disk.committed(root);starts=sum(e['kind']=='FORWARD_STARTED' for e in disk.events(root))
        for key in cfg['models']:
            disk.require(call('p4_worker.py',[root,key,'test'],'noop_'+key)==0,'No-op failed')
        disk.require(before==disk.committed(root) and starts==sum(e['kind']=='FORWARD_STARTED' for e in disk.events(root)),'No-op changed raw evidence')
        disk.require(call('p4_verify.py',[root,logs/'verified.json'],'verify_all')==0,'Saved-observation audit failed')
        summary=disk.read(logs/'verified.json');summary.update(status='COLAB_VERIFIED_AWAITING_LOCAL_RECEIPT',
            no_op_added_forwards=0,cumulative_seconds_through_summary=budget.seconds())
        disk.write(root/'summary.json',summary);disk.write(root/'status.json',dict(status=summary['status']))
        budget.close();TICK=lambda:None
        paths=disk.snapshot(root,base/'archives','FINAL')
        send(paths)
        print('P4 science and saved-data audit complete. Share ALL FINAL PART ZIPs + the FINAL INDEX ZIP.',flush=True)
        print('No further model/layer/intervention experiment is scheduled. Causal mechanism remains unresolved.',flush=True)
        return dict(status=summary['status'],root=str(root),zips=[str(p) for p in paths],download_queue='May still be running; keep Colab tab open')
    except BaseException as exc:
        signal.alarm(0)
        if budget is not None:budget.close()
        TICK=lambda:None;traceback.print_exc()
        if root is None:
            disk.write(base/('restore_error_'+uuid.uuid4().hex[:8]+'.json'),dict(at=disk.stamp(),type=type(exc).__name__,message=str(exc)))
            return dict(status='RESTORE_OR_INITIALIZATION_FAILED',error=str(exc),no_silent_fresh_run=True)
        disk.write(root/('failure_'+uuid.uuid4().hex[:8]+'.json'),dict(at=disk.stamp(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        disk.write(root/'status.json',dict(status='FAILED_OR_INTERRUPTED'))
        def recovery_timeout(*_):raise TimeoutError('Failure packaging exceeded ten minutes; raw evidence retained')
        signal.signal(signal.SIGALRM,recovery_timeout);signal.alarm(cfg['closure_seconds'])
        try:
            paths=disk.snapshot(root,base/'archives','PARTIAL_OR_FAILURE');signal.alarm(0)
            if delivery is not None:send(paths)
            return dict(status='FAILED_OR_INTERRUPTED',root=str(root),zips=[str(p) for p in paths])
        except BaseException as recovery:
            print('Could not finish a ZIP set:',recovery,'; original evidence remains at',root,flush=True)
            return dict(status='FAILED_ARCHIVE_UNAVAILABLE',root=str(root))
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler);TICK=lambda:None
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
