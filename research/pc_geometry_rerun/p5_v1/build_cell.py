"""Build a standalone single cell; freeze only after local checks pass."""
import ast
import base64
import json
import zlib
import p5_io as disk


def build():
    disk.require(not (disk.HERE/"release_manifest.json").exists(), "Release already frozen; create a new version")
    tests = disk.read(disk.HERE/"test_results.json")
    disk.require(tests["status"] == "PASS" and tests["source_sha256"] == disk.source_hashes(),
                 "Tests absent or sources changed after tests")
    sources = {n: base64.b64encode((disk.HERE/n).read_bytes()).decode() for n in disk.SOURCE_NAMES}
    raw = disk.canonical(sources).encode()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    header = '''# EXP-0005 / P5 v1: paste THIS ENTIRE FILE into ONE Colab GPU code cell.
# Historical GPT-2 inputs: narrative (101 tokens), technical (85), repetitive (72).
# Primary: CPU float32, block07. Same weights/inputs on GPU for numerical comparison.
# Also save block06, before final LayerNorm, and after final LayerNorm.
# First run ~10-25 minutes; warm environment ~2-8 minutes. Processing cap 40 minutes.
# FINAL ZIP ~6-15 MB; weights are NOT included. Every ZIP < 300 MB.
# Save the CHECKPOINT as fallback; when complete, share the FINAL ZIP.
# First run / same-VM resume: leave RESTORE_ZIP blank.
# New-VM recovery: RESTORE_ZIP="UPLOAD", then choose latest cumulative P5 ZIP.
# Or set the exact path of the ZIP already uploaded through the Files panel.
# Same hardware/software contract required. Completed units are never recomputed.
# No API key, Drive, earlier experiment ZIP, or notebook restart required.
# Download requested != file saved on your computer; check the downloaded file.
RESTORE_ZIP = ""
AUTO_DOWNLOAD = True
'''
    wrapper = '''
import base64 as _b, hashlib as _h, importlib as _i, json as _j, sys as _s, zlib as _z
from pathlib import Path as _P
if _s.platform != "linux" or not _P("/content").is_dir():
    raise RuntimeError("Run this cell in a Colab GPU runtime.")
_i.import_module("google.colab")
_raw = _z.decompress(_b.b64decode(_PAYLOAD, validate=True))
if _h.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("P5 payload checksum mismatch")
_sources = _j.loads(_raw)
_package = _P("/content") / ("pcgeom_p5_package_" + _PAYLOAD_SHA256[:12])
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
for _name in ("p5_launch", "p5_io", "p5_worker", "p5_numeric", "p5_analysis"):
    _s.modules.pop(_name, None)
_result = _i.import_module("p5_launch").run(RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''
    lines = "\n".join("    "+repr(payload[i:i+120]) for i in range(0, len(payload), 120))
    code = header+"\n_PAYLOAD_SHA256 = "+repr(disk.sha(raw))+"\n_PAYLOAD = (\n"+lines+"\n)\n"+wrapper
    ast.parse(code)
    dest = disk.HERE/"pcgeom_p5_v1_onecell.py"
    dest.write_text(code)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3")),
        cells=[dict(cell_type="code", id="pcgeom-p5-v1", execution_count=None, metadata={}, outputs=[],
                    source=code.splitlines(keepends=True))])
    disk.write(disk.HERE/"pcgeom_p5_v1_onecell.ipynb", notebook)
    release = dict(experiment="EXP-0005", release=disk.spec()["release"], frozen_at=disk.stamp(),
        capsule=disk.capsule(), source_sha256=disk.source_hashes(), payload_sha256=disk.sha(raw),
        onecell_sha256=disk.digest(dest), notebook_sha256=disk.digest(disk.HERE/"pcgeom_p5_v1_onecell.ipynb"),
        tests_sha256=disk.digest(disk.HERE/"test_results.json"),
        raw_hidden_bytes=disk.read(disk.HERE/"inputs.json")["raw_hidden_bytes"],
        pretrained_model_downloads=0, research_model_forwards=0, GPU_executions=0)
    disk.write(disk.HERE/"release_manifest.json", release, immutable=True)
    print(disk.canonical(dict(capsule=release["capsule"], onecell_bytes=dest.stat().st_size,
                             frozen_at=release["frozen_at"])))


if __name__ == "__main__":
    build()
