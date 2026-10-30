"""Model-free NumPy analysis. No Transformers or torch imports."""
import io
import math
from pathlib import Path
import zipfile
import numpy as np
import p2_io as disk


def load_npz(path):
    """Bound NPY headers before allocation; never allow pickle."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        disk.require(len(names) == len(set(names)) and len(names) <= 32, "NPZ member inventory")
        disk.require(sum(i.file_size for i in z.infolist()) < 16 * 1024**2, "Oversized NPZ")
        for i in z.infolist():
            disk.require("/" not in i.filename and i.filename.endswith(".npy"), "Invalid NPZ member")
            with z.open(i) as f:
                version = np.lib.format.read_magic(f)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                disk.require(reader is not None, "Unsupported NPY version")
                shape, _, dtype = reader(f)
                disk.require(not dtype.hasobject and dtype.kind in "fiu" and
                             len(shape) <= 3 and all(0 <= n <= 100000 for n in shape),
                             "Invalid NPY type/shape")
                disk.require(math.prod(shape) * dtype.itemsize == i.file_size - f.tell(), "NPY size mismatch")
    with np.load(path, allow_pickle=False) as z:
        return {n: z[n].copy() for n in z.files}


def save_npz(path, arrays):
    b = io.BytesIO()
    np.savez(b, **arrays)
    if Path(path).exists():
        old = load_npz(path)
        disk.require(set(old) == set(arrays), "Existing NPZ fields differ")
        for n, a in arrays.items():
            disk.require(old[n].dtype == a.dtype and np.array_equal(old[n], a, equal_nan=True),
                         "Existing NPZ differs: " + n)
        return
    disk.atomic(path, b.getvalue(), immutable=True)


def correlation(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, np.float64)[ok], np.asarray(y, np.float64)[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def validate_raw(root, doc_id):
    cfg = disk.spec()
    folder = Path(root) / "docs" / doc_id
    meta = disk.read(folder / "doc.json")
    fixture = next(d for d in disk.fixtures() if d["id"] == doc_id)
    run = disk.read(Path(root) / "run.json")
    disk.require(meta["document_id"] == doc_id and meta["run_id"] == run["run_id"], "Document identity")
    disk.require(meta["text"] == fixture["text"] and meta["text_sha256"] == disk.sha(meta["text"].encode()),
                 "Input text mismatch")
    disk.require(meta["specification_sha256"] == run["specification_sha256"], "Document specification")
    disk.require(meta["model"] == cfg["model"] and meta["revision"] == cfg["revision"], "Model revision")
    disk.require(meta["split"] == fixture["split"], "Split mismatch")
    if fixture["split"] == "test":
        disk.require(meta["fit_commit_sha256"] == disk.digest(Path(root)/"fit"/"COMMIT.json"), "Test used another fitted artifact")
    else:
        disk.require(meta["fit_commit_sha256"] is None, "Train has test-fit reference")
    a = load_npz(folder / "raw.npz")
    n, d = cfg["tokens"], cfg["hidden"]
    fields = {"token_id", "position", "mask", "token_chars", "token_utf8_bytes", "logit_source_index",
              "surprisal", "target_logit", "logsumexp"} | {"h_" + p for p in cfg["points"]}
    disk.require(set(a) == fields, "Raw field inventory")
    for p in cfg["points"]:
        h = a["h_" + p]
        disk.require(h.shape == (n, d) and h.dtype == np.float32 and np.isfinite(h).all(), "Hidden shape/dtype")
    for key in fields - {"h_" + p for p in cfg["points"]}:
        disk.require(a[key].shape == (n,), "Token alignment/shape: " + key)
    for key in ("token_id", "position", "mask", "token_chars", "token_utf8_bytes", "logit_source_index"):
        disk.require(a[key].dtype == np.int64, "Integer feature dtype: " + key)
    disk.require(np.array_equal(a["position"], np.arange(n)) and np.all(a["mask"] == 1), "Position/mask")
    disk.require(np.array_equal(a["logit_source_index"], np.arange(-1, n-1)), "Surprisal off-by-one")
    disk.require(np.all((a["token_id"] >= 0) & (a["token_id"] < 50257)), "Token IDs")
    disk.require(len(meta["tokens"]) == len(meta["decoded_tokens"]) == n, "Token strings")
    disk.require(np.array_equal(a["token_chars"], [len(t) for t in meta["decoded_tokens"]]), "Token char feature")
    disk.require(np.array_equal(a["token_utf8_bytes"], [len(t.encode()) for t in meta["decoded_tokens"]]), "Token byte feature")
    disk.require(meta["input_ids_sha256"] == disk.sha(a["token_id"].tobytes()), "Token ID hash")
    disk.require(np.array_equal(a["token_id"], fixture["token_ids"]), "Frozen token IDs differ")
    for key in ("surprisal", "target_logit", "logsumexp"):
        disk.require(a[key].dtype == np.float64 and np.isnan(a[key][0]) and np.isfinite(a[key][1:]).all(),
                     "First-token NA/finite values: " + key)
    disk.require(np.all(a["surprisal"][1:] >= -cfg["surprisal_atol"]), "Negative surprisal")
    np.testing.assert_allclose(a["surprisal"][1:], a["logsumexp"][1:]-a["target_logit"][1:],
                               rtol=0, atol=cfg["surprisal_atol"])
    checks = meta["extraction_checks"]
    disk.require(checks["block01_matches"] and checks["block06_matches"] and checks["final_ln_matches"] and
                 checks["preln_to_final_max_abs"] <= 2e-5 and
                 checks["surprisal_ce_max_abs"] <= cfg["surprisal_atol"], "Extraction diagnostic failed")
    return a, meta


def compare_tree(actual, expected):
    if isinstance(expected, dict):
        disk.require(isinstance(actual, dict) and set(actual) == set(expected), "Metric keys")
        for k in expected:
            compare_tree(actual[k], expected[k])
    elif isinstance(expected, list):
        disk.require(isinstance(actual, list) and len(actual) == len(expected), "Metric length")
        for a, b in zip(actual, expected):
            compare_tree(a, b)
    elif expected is None:
        disk.require(actual is None, "Expected undefined metric")
    elif isinstance(expected, (str, bool)):
        disk.require(type(actual) is type(expected) and actual == expected, "Label mismatch")
    elif isinstance(expected, int):
        disk.require(type(actual) is int and actual == expected, "Count mismatch")
    else:
        disk.require(type(actual) in (int, float) and math.isclose(actual, expected,
                     rel_tol=disk.spec()["scalar_rtol"], abs_tol=disk.spec()["scalar_atol"]), "Metric mismatch")
