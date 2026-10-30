"""Deterministic builder. Produces a copy/paste cell; never executes it."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import p0_io as disk


def build():
    sources = {n: (disk.HERE/n).read_text(encoding="utf-8") for n in disk.SOURCE_NAMES}
    raw = disk.canonical(sources).encode()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    header = '''# P0-0001 code v2: paste THIS WHOLE FILE into ONE Colab GPU code cell.
# Fixes the tuple/list false environment mismatch. Scientific settings unchanged.
# Same VM: automatically resumes the diagnosed v1 failure ZIP with two saved docs.
# If the VM was reset, a ZIP upload prompt appears. Choose your failure ZIP.
# To start a separate fresh run explicitly, set RESTORE_ZIP="NEW".
# GPT-2, four synthetic texts, 256 tokens, four extraction points. No training.
# Estimated first run 10-20 min. Cap: 40 min + up to 2 min failure recovery.
# Cumulative checkpoint ~6-8 MB; FINAL ~12-16 MB. Share the FINAL ZIP.
# Allow multiple downloads. A request does not confirm a save to your disk.
# Same runtime: rerun unchanged to resume or download FINAL again.
# New runtime: use the SAME GPU/environment; real changes are still rejected.
# A100 / RTX PRO 6000 Blackwell; actual device, VRAM and host RAM are recorded.
# No Drive, API key or runtime restart is required.
RESTORE_ZIP = ""
AUTO_DOWNLOAD = True
'''
    lines = "\n".join("    "+repr(payload[i:i+100]) for i in range(0, len(payload), 100))
    wrapper = '''
import base64 as _b, hashlib as _h, importlib as _i, json as _j
import os as _o, sys as _s, zlib as _z
from pathlib import Path as _P
if _s.platform != "linux" or not _P("/content").is_dir():
    raise RuntimeError("P0 extraction runs only in a Colab GPU runtime.")
_i.import_module("google.colab")
_raw = _z.decompress(_b.b64decode(_PAYLOAD, validate=True))
if _h.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("P0 source payload checksum mismatch")
_sources = _j.loads(_raw)
_package = _P("/content") / ("pcgeom_p0_package_" + _PAYLOAD_SHA256[:12])
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _P(_name).name != _name or _name in ("", ".", ".."):
        raise RuntimeError("Invalid source name")
    _dest = _package / _name
    if _dest.exists() and _dest.read_text(encoding="utf-8") != _body:
        raise RuntimeError("Existing packaged source changed: " + _name)
    _dest.write_text(_body, encoding="utf-8")
_s.path.insert(0, str(_package))
for _name in ("p0_launch", "p0_io", "p0_numeric", "p0_worker", "p0_verify"):
    _s.modules.pop(_name, None)
_result = _i.import_module("p0_launch").run(RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''
    text = header+"\n_PAYLOAD_SHA256 = "+repr(hashlib.sha256(raw).hexdigest())+"\n_PAYLOAD = (\n"+lines+"\n)\n"+wrapper
    ast.parse(text)
    for name, source in sources.items():
        if name.endswith(".py"):
            ast.parse(source, filename=name)
    target = disk.HERE/"pcgeom_p0_v2_onecell.py"
    target.write_text(text, encoding="utf-8")
    release = dict(experiment=disk.EXP, code_release=disk.RELEASE, version=2, capsule=disk.capsule(),
                   payload_sha256=hashlib.sha256(raw).hexdigest(), sources=disk.source_hashes(),
                   onecell_sha256=disk.digest(target), onecell_bytes=target.stat().st_size,
                   raw_hidden_state_bytes=4*256*768*4*4)
    disk.write(disk.HERE/"release_manifest.json", release)
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    build()
