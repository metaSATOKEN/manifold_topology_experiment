"""Build one self-contained Colab cell, including immutable inputs and all runtime sources."""
import ast
import base64
import json
import zlib
import p4_io as disk


def build():
    disk.require(not (disk.HERE/"release_manifest.json").exists(),
                 "Release already frozen; create a new version")
    sources = {n: base64.b64encode((disk.HERE/n).read_bytes()).decode()
               for n in disk.SOURCE_NAMES}
    raw = disk.canonical(sources).encode()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    header = """# EXP-0004 / P4 v1: paste THIS ENTIRE FILE into ONE Colab GPU code cell.
# GPT-2 / OPT-1.3B / Qwen2.5-1.5B BASE; 64 train + 64 unseen test articles.
# All three-model fits are sealed before ANY evaluation forward.
# Planning estimate 25-60 min (slow CPU/network 60-120), cumulative cap 180 min.
# FINAL total ~2.0-2.5 GB, split into ~10 PART ZIPs + one small INDEX ZIP.
# EVERY ZIP < 300 MB. Serial browser queue; allow multiple downloads.
# Keep ALL FINAL PARTs + FINAL INDEX from the SAME snapshot set; share all.
# Cell completion does NOT mean browser downloads or local disk saves finished.
# First run / same-VM resume: leave RESTORE_ZIPS blank.
# New-VM resume: RESTORE_ZIPS="UPLOAD", select latest set's ALL PARTs + INDEX.
# Or set a list of exact /content/...zip paths already uploaded via Files panel.
# Same GPU/software contract only; no new data/condition on technical resume.
# No API key, Drive, earlier experiment ZIP, or notebook restart needed.
RESTORE_ZIPS = ""
AUTO_DOWNLOAD = True
"""
    lines = "\n".join("    "+repr(payload[i:i+120]) for i in range(0, len(payload), 120))
    wrapper = '''
import base64 as _b, hashlib as _h, importlib as _i, json as _j, sys as _s, zlib as _z
from pathlib import Path as _P
if _s.platform != "linux" or not _P("/content").is_dir():
    raise RuntimeError("Run this cell in a Colab GPU runtime.")
_i.import_module("google.colab")
_raw = _z.decompress(_b.b64decode(_PAYLOAD, validate=True))
if _h.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("P4 payload checksum mismatch")
_sources = _j.loads(_raw)
_package = _P("/content") / ("pcgeom_p4_package_" + _PAYLOAD_SHA256[:12])
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
for _name in ("p4_launch", "p4_io", "p4_worker", "p4_numeric", "p4_analysis", "p4_fit", "p4_verify", "p4_download"):
    _s.modules.pop(_name, None)
_result = _i.import_module("p4_launch").run(RESTORE_ZIPS, AUTO_DOWNLOAD)
print(_result)
'''
    text = (header+"\n_PAYLOAD_SHA256 = "+repr(disk.sha(raw))+
            "\n_PAYLOAD = (\n"+lines+"\n)\n"+wrapper)
    ast.parse(text)
    for name in disk.SOURCE_NAMES:
        if name.endswith(".py"):
            ast.parse((disk.HERE/name).read_text(), filename=name)
    target = disk.HERE/"pcgeom_p4_v1_onecell.py"
    target.write_text(text)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3",
                    language="python", name="python3")),
        cells=[dict(cell_type="code", id="pcgeom-p4-v1", execution_count=None,
                    metadata={}, outputs=[], source=text.splitlines(keepends=True))])
    disk.write(disk.HERE/"pcgeom_p4_v1_onecell.ipynb", notebook)
    manifest = dict(experiment=disk.EXP, code_release=disk.RELEASE,
        capsule=disk.capsule(), frozen_at=disk.stamp(), sources=disk.source_hashes(),
        payload_sha256=disk.sha(raw), onecell_sha256=disk.digest(target),
        onecell_bytes=target.stat().st_size,
        notebook_sha256=disk.digest(disk.HERE/"pcgeom_p4_v1_onecell.ipynb"),
        raw_hidden_state_bytes=disk.read(disk.HERE/"input_manifest.json")["raw_hidden_bytes"], new_model_forwards_at_release=0)
    disk.write(disk.HERE/"release_manifest.json", manifest, immutable=True)
    print(disk.canonical(manifest))


if __name__ == "__main__":
    build()
