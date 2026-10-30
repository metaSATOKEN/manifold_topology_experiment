"""P4 evidence transport. Standard library only; restored sources are never run."""
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
SOURCE_NAMES = ("spec.json", "models.json", "fixtures.json", "input_manifest.json", "protocol.md",
    "DATA_LICENSE.md", "experiment_card.md", "p4_io.py", "p4_numeric.py", "p4_worker.py",
    "p4_fit.py", "p4_analysis.py", "p4_verify.py", "p4_launch.py", "p4_download.py",
    "LICENSE-transport.txt", "THIRD_PARTY_NOTICE.txt")
EXP = RELEASE = "pcgeom_p4_v1"
ZIP_LIMIT = 300_000_000
PART_TARGET = 250_000_000
RAW_LIMIT = 4 * 1024**3


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


def check_run(root):
    root = Path(root)
    m = read(root / "run.json")
    require(m["experiment"] == EXP and m["capsule"] == capsule(), "Wrong experiment/code version")
    require(m["specification_sha256"] == digest(HERE / "spec.json"), "Wrong specification")
    require(m["sources"] == source_hashes(), "Source manifest differs")
    for n, h in m["sources"].items():
        require(digest(root / "sources" / n) == h, "Saved source changed: " + n)
    return m


def seal_doc(folder):
    folder = Path(folder)
    names = ("raw.npz", "doc.json")
    require(all((folder / n).is_file() for n in names), "Incomplete document")
    write(folder / "COMMIT.json", {n: digest(folder / n) for n in names}, immutable=True)


def committed(root):
    check_run(root)
    expected = {k+"__"+d["id"] for k in spec()["models"] for d in fixtures()}
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



def model(key):
    return read(HERE / "models.json")[key]


def doc_key(key, doc_id):
    require(key in spec()["models"], "Unknown model")
    require(doc_id in {d["id"] for d in fixtures()}, "Unknown article")
    return key + "__" + doc_id


def split_keys(split):
    return [doc_key(k,d["id"]) for k in spec()["models"] for d in fixtures() if d["split"]==split]


def check_fit(root):
    root=Path(root); folder=root/"fit"; c=read(folder/"COMMIT.json")
    require(c["capsule"]==capsule(), "Fit capsule")
    require(set(c["files"]) == {str(p.relative_to(folder)) for p in files(folder)}-{"COMMIT.json"}, "Fit inventory")
    for n,h in c["files"].items():
        require(digest(folder/safe_name(n))==h, "Changed fit: "+n)
    done=committed(root)
    require(c["train_commits"]=={k:done.get(k) for k in split_keys("train")}, "Fit training commits changed")
    require(all(v is not None for v in c["train_commits"].values()), "Incomplete training")
    return c


def inspect_zip(path):
    """Verify one independent part/index; completeness requires inspect_bundle."""
    path=Path(path);require(path.stat().st_size<ZIP_LIMIT,'ZIP exceeds 300 MB')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        require(len(names)==len(set(names)) and len(names)<=10000,'Duplicate/excess ZIP members')
        require(sum(i.file_size for i in infos)<ZIP_LIMIT,'Expanded part exceeds 300 MB')
        for i in infos:
            safe_name(i.filename)
            require(not i.is_dir() and not stat.S_ISLNK(i.external_attr>>16),'Unsafe ZIP member type')
        require('ARCHIVE_INDEX.json' in names and z.getinfo('ARCHIVE_INDEX.json').file_size<8*1024**2,'Missing/large index')
        idx=json.loads(z.read('ARCHIVE_INDEX.json'))
        require(idx['experiment']==EXP and idx['capsule']==capsule(),'Wrong code/version')
        require(set(names)==set(idx['entries'])|{'ARCHIVE_INDEX.json'},'Part inventory')
        for n,e in idx['entries'].items():
            require(z.getinfo(n).file_size==e['bytes'],'Member size mismatch')
            with z.open(n) as f:require(hashlib.file_digest(f,'sha256').hexdigest()==e['sha256'],'Member hash mismatch: '+n)
        if idx['kind']=='index':
            require(set(idx['entries'])=={'BUNDLE_INDEX.json'},'Index ZIP inventory')
            bundle=json.loads(z.read('BUNDLE_INDEX.json'))
            for k in ['experiment','capsule','run_id','bundle_id','status']:require(bundle[k]==idx[k],'Bundle identity')
            idx['bundle']=bundle
        else:
            require(idx['kind']=='part','Unknown archive type')
            m=json.loads(z.read('run.json'))
            require(m['run_id']==idx['run_id'] and m['capsule']==idx['capsule'],'Part run mismatch')
    return idx


def partition(paths,root,target=None):
    target=PART_TARGET if target is None else target
    chunks=[];chunk=[];size=0
    for p in paths:
        n=p.stat().st_size
        require(n<target,'A single evidence file exceeds part target')
        if chunk and size+n>target:chunks.append(chunk);chunk=[];size=0
        chunk.append(p);size+=n
    if chunk:chunks.append(chunk)
    return chunks or [[]]


def snapshot(root,destination,label):
    """A complete snapshot set: bounded data PARTs plus one small INDEX ZIP.
    No dependency on earlier snapshots; all parts of this set are required.
    """
    root,destination=Path(root),Path(destination);m=check_run(root);paths=files(root)
    require(sum(p.stat().st_size for p in paths)<RAW_LIMIT-16*1024**2,'Total evidence exceeds 4 GiB')
    entries={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
    chunks=partition([p for p in paths if p!=root/'run.json'],root)
    bid=uuid.uuid4().hex[:12];destination.mkdir(parents=True,exist_ok=True)
    stem=f"{RELEASE}_{m['run_id']}_{label}_{bid}"
    common=dict(experiment=EXP,capsule=m['capsule'],run_id=m['run_id'],bundle_id=bid,status=label,created=stamp())
    output=[];parts=[]
    for j,chunk in enumerate(chunks,1):
        selected=[root/'run.json',*chunk]
        idx=dict(common,kind='part',part=j,parts=len(chunks),entries={str(p.relative_to(root)):entries[str(p.relative_to(root))] for p in selected})
        name=stem+f'_PART_{j:03d}_of_{len(chunks):03d}.zip';target=destination/name;tmp=target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for p in selected:z.write(p,str(p.relative_to(root)))
            z.writestr('ARCHIVE_INDEX.json',canonical(idx))
        require(tmp.stat().st_size<ZIP_LIMIT,'Part exceeds hard cap; raw evidence retained')
        inspect_zip(tmp);os.replace(tmp,target);output.append(target)
        parts.append(dict(name=name,bytes=target.stat().st_size,sha256=digest(target),part=j))
    bundle=dict(common,parts=parts,entries=entries,part_target_bytes=PART_TARGET,zip_limit_bytes=ZIP_LIMIT,independent_snapshot=True)
    content=(canonical(bundle)+'\n').encode()
    index=dict(common,kind='index',entries={'BUNDLE_INDEX.json':dict(bytes=len(content),sha256=sha(content))})
    target=destination/(stem+'_INDEX.zip');tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr('BUNDLE_INDEX.json',content);z.writestr('ARCHIVE_INDEX.json',canonical(index))
    inspect_zip(tmp);os.replace(tmp,target);output.append(target)
    # Every part already passed full member checks; verify exact global coverage
    # and the index's part checksums again before marking the set complete.
    inspect_bundle(output)
    return output


def inspect_bundle(paths):
    paths=list(dict.fromkeys(Path(p).resolve() for p in paths))
    rows=[(p,inspect_zip(p)) for p in paths]
    require(rows and len({r['run_id'] for _,r in rows})==1,'Mixed/empty runs')
    require(len({r['bundle_id'] for _,r in rows})==1,'Mixed snapshot sets: select one complete set')
    indexes=[(p,r) for p,r in rows if r['kind']=='index']
    require(len(indexes)==1,'Exactly one INDEX ZIP is required')
    bundle=indexes[0][1]['bundle'];parts={r['part']:(p,r) for p,r in rows if r['kind']=='part'}
    require(len(parts)==len(rows)-1,'Duplicate part numbers')
    required={e['part'] for e in bundle['parts']}
    missing=required-set(parts)
    require(not missing,'Missing PART numbers: '+','.join(map(str,sorted(missing))))
    require(set(parts)==required,'Unexpected PART numbers')
    seen={}
    for e in bundle['parts']:
        p,r=parts[e['part']]
        require(p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],'Wrong/corrupt part: '+e['name'])
        require(r['parts']==len(required),'Part count mismatch')
        for n,v in r['entries'].items():
            if n in seen:require(n=='run.json' and seen[n]==v,'Conflicting/duplicate evidence: '+n)
            seen[n]=v
    require(seen==bundle['entries'],'Global evidence coverage mismatch')
    require(sum(e['bytes'] for e in seen.values())<RAW_LIMIT,'Restored evidence exceeds 4 GiB')
    return bundle,[(parts[j][0],parts[j][1]) for j in sorted(parts)]


def restore(paths,root):
    root=Path(root);require(not root.exists(),'Never overwrite live run')
    bundle,rows=inspect_bundle(paths);root.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(root.parent).free>sum(e['bytes'] for e in bundle['entries'].values())+2*1024**3,'Insufficient restore space')
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='restore-') as td:
        stage=Path(td)/'run';stage.mkdir();seen=set()
        for p,r in rows:
            with zipfile.ZipFile(p) as z:
                for n in r['entries']:
                    if n in seen:continue
                    dest=stage/safe_name(n);dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(n) as src,dest.open('wb') as dst:
                        shutil.copyfileobj(src,dst,length=1024**2)
                    seen.add(n)
        check_run(stage);committed(stage)
        if (stage/'fit/COMMIT.json').exists():check_fit(stage)
        os.replace(stage,root)
    return bundle
