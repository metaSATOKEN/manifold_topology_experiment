"""Build distributable code without importing torch or executing the experiment."""
import ast
import base64
import json
from pathlib import Path
import zlib
import p2_io as disk


def build():
    disk.require(not (disk.HERE/"release_manifest.json").exists(),
                 "This release is already frozen; create a new version instead of overwriting it")
    sources={n:(disk.HERE/n).read_text() for n in disk.SOURCE_NAMES}
    raw=disk.canonical(sources).encode()
    payload=base64.b64encode(zlib.compress(raw,9)).decode()
    header='''# EXP-0002 / P2 v1: paste THIS ENTIRE FILE into ONE Colab GPU code cell.
# GPT-2; frozen WikiText-2 64 fit + 64 held-out articles x 512 tokens x 2 extraction points.
# C0 all / C1 drop first (primary) / C4 unit norm. 20 random 8D subspaces.
# PCA, feature selection and linear probes are sealed BEFORE held-out extraction.
# Estimated first run: 15-45 min. Cap: 120 min + 5 min failure packaging.
# FINAL ZIP estimate: 460-600 MB; checkpoints at 32/64/96 articles are cumulative.
# Download FINAL and share it. Retain checkpoints until FINAL is verified.
# Allow multiple downloads, or use the Files panel for the printed ZIP paths.
# First run: leave RESTORE_ZIP blank. Same VM rerun: leave blank to resume.
# New VM resume: set RESTORE_ZIP="UPLOAD" and upload the latest P2 ZIP.
# Only the same GPU/software environment can continue a run.
# To explicitly start another run, set RESTORE_ZIP="NEW"; previous evidence is retained.
# Requires >=20 GiB free disk. No Drive, API key, language-model training or notebook restart.
RESTORE_ZIP = ""
AUTO_DOWNLOAD = True
'''
    lines="\n".join("    "+repr(payload[i:i+100]) for i in range(0,len(payload),100))
    wrapper='''
import base64 as _b, hashlib as _h, importlib as _i, json as _j, sys as _s, zlib as _z
from pathlib import Path as _P
if _s.platform != "linux" or not _P("/content").is_dir():
    raise RuntimeError("Run this cell in a Colab GPU runtime.")
_i.import_module("google.colab")
_raw = _z.decompress(_b.b64decode(_PAYLOAD, validate=True))
if _h.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("P2 source payload checksum mismatch")
_sources = _j.loads(_raw)
_package = _P("/content") / ("pcgeom_p2_package_" + _PAYLOAD_SHA256[:12])
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _P(_name).name != _name or _name in ("", ".", ".."):
        raise RuntimeError("Invalid packaged source name")
    _dest = _package / _name
    if _dest.exists() and _dest.read_text(encoding="utf-8") != _body:
        raise RuntimeError("Packaged source has changed: " + _name)
    _dest.write_text(_body, encoding="utf-8")
_s.path.insert(0, str(_package))
for _name in ("p2_launch", "p2_io", "p2_worker", "p2_numeric", "p2_analysis", "p2_fit", "p2_verify"):
    _s.modules.pop(_name, None)
_result = _i.import_module("p2_launch").run(RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''
    text=header+"\n_PAYLOAD_SHA256 = "+repr(disk.sha(raw))+"\n_PAYLOAD = (\n"+lines+"\n)\n"+wrapper
    ast.parse(text)
    for name,source in sources.items():
        if name.endswith(".py"):
            ast.parse(source,filename=name)
    target=disk.HERE/"pcgeom_p2_v1_onecell.py"
    target.write_text(text)
    manifest=dict(experiment=disk.EXP,code_release=disk.RELEASE,capsule=disk.capsule(),frozen_at=disk.stamp(),
        payload_sha256=disk.sha(raw),sources=disk.source_hashes(),onecell_sha256=disk.digest(target),
        onecell_bytes=target.stat().st_size,raw_hidden_state_bytes=128*512*768*2*4)
    disk.write(disk.HERE/"release_manifest.json",manifest)
    # Alternative copy surface: one actual code cell, same exact source.
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        accelerator="GPU",kernelspec=dict(display_name="Python 3",language="python",name="python3")),
        cells=[dict(cell_type="code",id="pcgeom-p2-v1",execution_count=None,metadata={},
                    outputs=[],source=text.splitlines(keepends=True))])
    disk.write(disk.HERE/"pcgeom_p2_v1_onecell.ipynb",notebook)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    build()
