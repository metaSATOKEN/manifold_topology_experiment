"""Immutable units, bounded cumulative ZIPs, and verified restore. Standard library only."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import uuid
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = [
    "p5_io.py", "p5_worker.py", "p5_numeric.py", "p5_analysis.py", "p5_launch.py",
    "spec.json", "inputs.json", "model.json", "experiment_card.md",
    "historical_extraction.py.txt", "historical_plot.py.txt", "historical_pointclouds.png",
    "LICENSE.txt", "THIRD_PARTY_NOTICE.txt",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == data, f"Immutable evidence changed: {path}")
        return
    tmp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, value, immutable=False):
    atomic(path, (canonical(value)+"\n").encode(), immutable)


def spec():
    return read(HERE/"spec.json")


def source_hashes():
    return {n: digest(HERE/n) for n in SOURCE_NAMES}


def capsule():
    return sha(canonical(source_hashes()).encode())


def entries():
    return read(HERE/"inputs.json")["entries"]


def keys():
    return [d+"_"+e["id"] for d in spec()["devices"] for e in entries()]


def initialize(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    write(root/"run.json", dict(run_id=uuid.uuid4().hex[:12], release=spec()["release"],
          capsule=capsule(), created=stamp(), source_sha256=source_hashes()), immutable=True)
    for n in SOURCE_NAMES:
        atomic(root/"sources"/n, (HERE/n).read_bytes(), immutable=True)
    write(root/"budget.json", dict(attempts=0, elapsed_seconds=0.0))


def check_run(root):
    root = Path(root)
    m = read(root/"run.json")
    require(m["capsule"] == capsule() and m["source_sha256"] == source_hashes(), "Run/code capsule differs")
    for n, h in m["source_sha256"].items():
        require(digest(root/"sources"/n) == h, "Archived source checksum differs: "+n)
    return m


def committed(root):
    root = Path(root)
    check_run(root)
    result = {}
    folder = root/"units"
    for unit in sorted(folder.iterdir()) if folder.exists() else []:
        require(unit.is_dir() and unit.name in keys(), "Unexpected committed unit")
        meta = read(unit/"commit.json")
        require(meta["key"] == unit.name and meta["capsule"] == capsule(), "Invalid unit identity")
        require(set(p.name for p in unit.iterdir()) == {"raw.npz", "commit.json"}, "Unexpected unit files")
        require(digest(unit/"raw.npz") == meta["sha256"], "Corrupt raw unit: "+unit.name)
        result[unit.name] = meta
    return result


def save_unit(root, key, arrays, metadata):
    import numpy as np
    root = Path(root)
    require(key in keys() and not (root/"units"/key).exists(), "Unit already committed or unknown")
    temp = root/"pending"/(key+"_"+uuid.uuid4().hex)
    temp.mkdir(parents=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic(temp/"raw.npz", stream.getvalue(), immutable=True)
    write(temp/"commit.json", dict(metadata, key=key, capsule=capsule(),
          sha256=digest(temp/"raw.npz"), committed_at=stamp()), immutable=True)
    (root/"units").mkdir(exist_ok=True)
    os.rename(temp, root/"units"/key)


def safe_member(name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and ".." not in p.parts and
            "\\" not in name and str(p) == name and not name.endswith("/"), "Unsafe ZIP member")
    return p


def inspect_zip(path):
    path = Path(path)
    require(path.stat().st_size < spec()["zip_limit"], "ZIP exceeds 300 MB")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [x.filename for x in infos]
        require(len(names) == len(set(names)) <= 500, "Duplicate/excess ZIP entries")
        require(sum(x.file_size for x in infos) < spec()["run_limit_bytes"], "Expanded ZIP exceeds limit")
        for n in names:
            safe_member(n)
        require("ARCHIVE.json" in names, "Archive manifest missing")
        manifest = json.loads(z.read("ARCHIVE.json"))
        require(manifest["capsule"] == capsule(), "ZIP belongs to another code/configuration")
        files = manifest["files"]
        require(set(names) == {"ARCHIVE.json"} | {"evidence/"+n for n in files}, "Archive inventory mismatch")
        for n, rec in files.items():
            safe_member(n)
            data = z.read("evidence/"+n)
            require(len(data) == rec["bytes"] and sha(data) == rec["sha256"], "Archive content mismatch: "+n)
        return manifest


def archive(root, folder, label):
    root, folder = Path(root), Path(folder)
    m = check_run(root)
    committed(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    require(not any(p.is_symlink() for p in files), "Symlink evidence forbidden")
    require(sum(p.stat().st_size for p in files) < spec()["run_limit_bytes"]-1000000, "Run evidence exceeds cap")
    inventory = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files}
    manifest = dict(capsule=capsule(), run_id=m["run_id"], label=label, created=stamp(), files=inventory)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/(spec()["release"]+"_"+m["run_id"]+"_"+label+"_"+uuid.uuid4().hex[:6]+".zip")
    temp = dest.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("ARCHIVE.json", canonical(manifest))
        for p in files:
            z.write(p, "evidence/"+p.relative_to(root).as_posix())
    require(temp.stat().st_size < spec()["zip_limit"], "ZIP exceeds 300 MB")
    inspect_zip(temp)
    os.rename(temp, dest)
    return dest


def restore(path, root):
    manifest = inspect_zip(path)
    root = Path(root)
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(path) as z:
        for n in manifest["files"]:
            atomic(root/n, z.read("evidence/"+n), immutable=True)
    check_run(root)
    committed(root)
    write(root/"restores"/(uuid.uuid4().hex+".json"), dict(at=stamp(),
          archive_sha256=digest(path), archive_name=Path(path).name), immutable=True)
    return root
