"""Build one self-contained Colab cell, including lossless frozen model files."""
import ast
import base64
import json
import zlib
import p3_io as disk


def build():
    disk.require(not (disk.HERE/"release_manifest.json").exists(),
                 "Release already frozen; create a new version")
    sources = {n: base64.b64encode((disk.HERE/n).read_bytes()).decode()
               for n in disk.SOURCE_NAMES}
    raw = disk.canonical(sources).encode()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    header = '''# EXP-0003 / P3 v1: paste THIS ENTIRE FILE into ONE Colab GPU code cell.
# Existing ID readouts, PCA/random bases and training frequencies are embedded.
# 64 NEW articles x 512 tokens; block06 (primary) and final_ln; C1 only.
# NO probe fitting/tuning. Ordinary and within-article/token-ID evaluation.
# First-run estimate 10-25 min; cap 90 min + 5 min recovery.
# FINAL ZIP estimate 200-260 MB; cumulative checkpoints at 16/32/48 articles.
# Download FINAL and share it; retain checkpoints until FINAL is verified.
# First run / same-VM resume: leave RESTORE_ZIP blank.
# New-VM resume: RESTORE_ZIP="UPLOAD", then upload the latest P3 v1 ZIP.
# Same GPU/software only. "NEW" starts a separate run, preserving old evidence.
# No prior P2 ZIP upload, API key, Drive mount or notebook restart needed.
# Frozen parameters make this cell several MB; the matching .ipynb has one identical cell.
RESTORE_ZIP = ""
AUTO_DOWNLOAD = True
'''
    lines = "\n".join("    "+repr(payload[i:i+120]) for i in range(0, len(payload), 120))
    wrapper = '''
import base64 as _b, hashlib as _h, importlib as _i, json as _j, sys as _s, zlib as _z
from pathlib import Path as _P
if _s.platform != "linux" or not _P("/content").is_dir():
    raise RuntimeError("Run this cell in a Colab GPU runtime.")
_i.import_module("google.colab")
_raw = _z.decompress(_b.b64decode(_PAYLOAD, validate=True))
if _h.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("P3 payload checksum mismatch")
_sources = _j.loads(_raw)
_package = _P("/content") / ("pcgeom_p3_package_" + _PAYLOAD_SHA256[:12])
_package.mkdir(parents=True, exist_ok=True)
for _name, _encoded in _sources.items():
    if _P(_name).name != _name or _name in ("", ".", ".."):
        raise RuntimeError("Invalid packaged source name")
    _body = _b.b64decode(_encoded, validate=True)
    _dest = _package / _name
    if _dest.exists():
        if _dest.read_bytes() != _body:
            raise RuntimeError("Packaged evidence has changed: " + _name)
    else:
        _dest.write_bytes(_body)
_s.path.insert(0, str(_package))
for _name in ("p3_launch", "p3_io", "p3_worker", "p3_numeric", "p3_analysis", "p3_frozen", "p3_verify"):
    _s.modules.pop(_name, None)
_result = _i.import_module("p3_launch").run(RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''
    text = (header+"\n_PAYLOAD_SHA256 = "+repr(disk.sha(raw))+
            "\n_PAYLOAD = (\n"+lines+"\n)\n"+wrapper)
    ast.parse(text)
    for name in disk.SOURCE_NAMES:
        if name.endswith(".py"):
            ast.parse((disk.HERE/name).read_text(), filename=name)
    target = disk.HERE/"pcgeom_p3_v1_onecell.py"
    target.write_text(text)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3",
                    language="python", name="python3")),
        cells=[dict(cell_type="code", id="pcgeom-p3-v1", execution_count=None,
                    metadata={}, outputs=[], source=text.splitlines(keepends=True))])
    disk.write(disk.HERE/"pcgeom_p3_v1_onecell.ipynb", notebook)
    manifest = dict(experiment=disk.EXP, code_release=disk.RELEASE,
        capsule=disk.capsule(), frozen_at=disk.stamp(), sources=disk.source_hashes(),
        payload_sha256=disk.sha(raw), onecell_sha256=disk.digest(target),
        onecell_bytes=target.stat().st_size,
        notebook_sha256=disk.digest(disk.HERE/"pcgeom_p3_v1_onecell.ipynb"),
        frozen_model_manifest_sha256=disk.digest(disk.HERE/"frozen_manifest.json"),
        raw_hidden_state_bytes=64*512*768*2*4, new_model_forwards_at_release=0)
    disk.write(disk.HERE/"release_manifest.json", manifest, immutable=True)
    print(disk.canonical(manifest))


if __name__ == "__main__":
    build()
