"""P0 evidence transport. Standard library only; restored sources are never run."""
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
import stat
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = (
    "spec.json", "fixtures.json", "protocol.md", "p0_io.py", "p0_numeric.py",
    "p0_worker.py", "p0_verify.py", "p0_launch.py",
    "legacy_release.json", "amendment.md",
)
EXP = "pcgeom_p0_v1"
RELEASE = "pcgeom_p0_v2"
ZIP_LIMIT = 64 * 1024**2
RAW_LIMIT = 128 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(path.read_bytes() == data, "Immutable evidence differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, obj, immutable=False):
    atomic(path, (canonical(obj) + "\n").encode(), immutable)


def spec():
    return read(HERE / "spec.json")


def fixtures():
    return read(HERE / "fixtures.json")


def source_hashes():
    return {n: digest(HERE / n) for n in SOURCE_NAMES}


def legacy():
    return read(HERE / "legacy_release.json")


def legacy_base():
    return Path("/content") / (EXP + "_" + legacy()["capsule"][:12])


def capsule():
    return sha(canonical(source_hashes()).encode())


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Unsafe archive path")
    p = PurePosixPath(name)
    require(bool(p.parts) and "\\" not in name and
            not p.is_absolute() and ".." not in p.parts and
            "." not in p.parts and str(p) == name, "Unsafe archive path")
    return Path(*p.parts)


def files(root):
    root = Path(root)
    result = []
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), "Symlink evidence rejected")
        if p.is_file() and not p.name.endswith(".tmp") and "__pycache__" not in p.parts:
            result.append(p)
    return result


def initialize(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "run.json", dict(experiment=EXP, run_id=uuid.uuid4().hex[:12],
          created=stamp(), capsule=capsule(), specification_sha256=digest(HERE / "spec.json"),
          sources=source_hashes(), code_release=RELEASE))
    for n in SOURCE_NAMES:
        atomic(root / "sources" / n, (HERE / n).read_bytes())
    write(root / "status.json", dict(status="NEW"))


def check_run(root, allow_legacy=False):
    root = Path(root)
    m = read(root / "run.json")
    old = allow_legacy and m["capsule"] == legacy()["capsule"]
    require(m["experiment"] == EXP and (m["capsule"] == capsule() or old), "Wrong experiment/code version")
    require(m["specification_sha256"] == digest(HERE / "spec.json"), "Wrong specification")
    require(m["sources"] == (legacy()["sources"] if old else source_hashes()), "Source manifest differs")
    for n, h in m["sources"].items():
        require(digest(root / "sources" / n) == h, "Saved source changed: " + n)
    return m


def seal_doc(folder):
    folder = Path(folder)
    names = ("raw.npz", "doc.json")
    require(all((folder / n).is_file() for n in names), "Incomplete document")
    write(folder / "COMMIT.json", {n: digest(folder / n) for n in names}, immutable=True)


def committed(root, allow_legacy=False):
    check_run(root, allow_legacy=allow_legacy)
    expected = {d["id"] for d in fixtures()}
    out = {}
    folder = Path(root) / "docs"
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        require(p.is_dir() and p.name in expected, "Unexpected document ID")
        if not (p / "COMMIT.json").exists():
            continue
        m = read(p / "COMMIT.json")
        require(set(m) == {"raw.npz", "doc.json"}, "Document commit inventory")
        require({f.name for f in p.iterdir() if f.is_file()} == set(m) | {"COMMIT.json"},
                "Unexpected committed document file")
        for n, h in m.items():
            require((p / n).is_file() and digest(p / n) == h, "Missing/corrupt document " + p.name)
        out[p.name] = m
    return out


def events(root):
    rows = []
    for p in sorted((Path(root) / "attempts").glob("*/events.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event(root, attempt, kind, **fields):
    p = Path(root) / "attempts" / attempt / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(canonical(dict(at=stamp(), attempt=attempt, kind=kind, **fields)) + "\n")
        f.flush()


def inspect_zip(path, allow_legacy=False):
    path = Path(path)
    require(path.stat().st_size < ZIP_LIMIT, "ZIP exceeds 64 MiB")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)) and len(names) <= 5000, "Duplicate/excess ZIP members")
        require(sum(i.file_size for i in infos) <= RAW_LIMIT, "Expanded ZIP exceeds 128 MiB")
        for i in infos:
            safe_name(i.filename)
            require(not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16), "Unsafe member type")
        require(z.getinfo("ARCHIVE_INDEX.json").file_size < 1024**2, "Index too large")
        index = json.loads(z.read("ARCHIVE_INDEX.json"))
        allowed = {capsule()} | ({legacy()["capsule"]} if allow_legacy else set())
        require(index["experiment"] == EXP and index["capsule"] in allowed, "Wrong ZIP version")
        require(set(names) == set(index["entries"]) | {"ARCHIVE_INDEX.json"}, "ZIP inventory mismatch")
        for n, entry in index["entries"].items():
            require(z.getinfo(n).file_size == entry["bytes"], "ZIP member size mismatch")
            with z.open(n) as f:
                require(hashlib.file_digest(f, "sha256").hexdigest() == entry["sha256"],
                        "ZIP member hash mismatch: " + n)
        m = json.loads(z.read("run.json"))
        require(m["run_id"] == index["run_id"] and m["capsule"] == index["capsule"], "ZIP run mismatch")
    return index


def snapshot(root, destination, label):
    root, destination = Path(root), Path(destination)
    m = check_run(root)
    paths = files(root)
    require(sum(p.stat().st_size for p in paths) < RAW_LIMIT - 1024**2, "Evidence exceeds archive budget")
    index = dict(experiment=EXP, capsule=m["capsule"], run_id=m["run_id"],
                 created=stamp(), status=label, entries={
                     str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in paths})
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{RELEASE}_{m['run_id']}_{label}_{uuid.uuid4().hex[:6]}.zip"
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in paths:
            z.write(p, str(p.relative_to(root)))
        z.writestr("ARCHIVE_INDEX.json", canonical(index))
    require(tmp.stat().st_size < ZIP_LIMIT, "ZIP exceeds 64 MiB; original files retained")
    inspect_zip(tmp)
    os.replace(tmp, target)
    return target


def migrate_v1(root, archive_sha256):
    """Only in a fresh staging copy, after validating the trusted v1 source hashes."""
    root = Path(root)
    old = check_run(root, allow_legacy=True)
    require(old["capsule"] == legacy()["capsule"], "Migration requires the frozen v1 release")
    before = committed(root, allow_legacy=True)
    require(len(before) <= 2 and not (root/"summary.json").exists() and not (root/"analysis").exists(),
            "v1 migration supports incomplete runs before analysis only")
    history = root / "code_history" / old["capsule"]
    history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "run.json", history / "run.json")
    shutil.copytree(root / "sources", history / "sources")
    for n in SOURCE_NAMES:
        atomic(root / "sources" / n, (HERE / n).read_bytes())
    updated = dict(old, capsule=capsule(), sources=source_hashes(), code_release=RELEASE,
                   parent_capsule=old["capsule"])
    write(root / "run.json", updated)
    write(root / "migration_v2.json", dict(
        at=stamp(), from_capsule=old["capsule"], to_capsule=capsule(),
        source_archive_sha256=archive_sha256, retained_document_commits=before,
        history=str(history.relative_to(root)),
        reason="Fix tuple/list environment false rejection; scientific specification unchanged"))
    require(committed(root) == before, "Migration changed committed evidence")


def restore(path, root, allow_legacy=False):
    root = Path(root)
    require(not root.exists(), "Restore destination already exists")
    index = inspect_zip(path, allow_legacy=allow_legacy)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix="restore-") as td:
        stage = Path(td) / "run"
        stage.mkdir()
        with zipfile.ZipFile(path) as z:
            for n in index["entries"]:
                target = stage / safe_name(n)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, target.open("wb") as dst:
                    while chunk := src.read(1024**2):
                        dst.write(chunk)
        check_run(stage, allow_legacy=allow_legacy)
        committed(stage, allow_legacy=allow_legacy)
        if index["capsule"] != capsule():
            migrate_v1(stage, digest(path))
        os.replace(stage, root)
    return index
