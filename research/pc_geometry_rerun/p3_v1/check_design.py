"""Validate input separation and frozen-model portability without any LM forward."""
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from transformers import AutoTokenizer
import p3_io as disk
import p3_frozen as fixed
import p3_numeric as num
import p3_analysis as analysis


def main():
    root = disk.HERE.parent
    fixtures = disk.fixtures()
    old = disk.read(root/"p1_v1/fixtures.json")+disk.read(root/"p2_v1/fixtures.json")
    assert len(fixtures) == 64 and all(f["split"] == "test" for f in fixtures)
    assert not ({f["id"] for f in fixtures} & {f["id"] for f in old})
    tokenizer = AutoTokenizer.from_pretrained(root/"design_inputs/p1_v1",
        local_files_only=True, use_fast=True, trust_remote_code=False)
    for f in fixtures:
        ids = tokenizer(f["text"], add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        assert ids == f["token_ids"]
    grams = lambda ids: {tuple(ids[i:i+5]) for i in range(len(ids)-4)}
    previous = [grams(f["token_ids"]) for f in old]
    max_jaccard = 0.
    for f in fixtures:
        g = grams(f["token_ids"])
        for h in previous:
            j = len(g & h)/len(g | h)
            max_jaccard = max(max_jaccard, j)
            assert j < .5
        previous.append(g)
    p0 = disk.read(root/"p0_v2/fixtures.json")
    p0ids = [tokenizer(d["text"], add_special_tokens=False)["input_ids"][:256] for d in p0]
    assert all(f["token_ids"][:256] not in p0ids for f in fixtures)
    eda = root/"exploratory/p2_lexical_v1"
    source = root/"receipts/6966c146e980_p2_v1_final/raw"
    for point in disk.spec()["points"]:
        new = num.load_npz(disk.HERE/f"frozen_{point}_C1.npz")
        with np.load(eda/"fits"/f"{point}__C1__ID.npz") as f:
            for k in f.files:
                if k not in ["prediction", "normal_residual"]:
                    np.testing.assert_array_equal(new[k], f[k])
        with np.load(source/"fit"/f"{point}__C1"/"arrays.npz") as f:
            for k in ["mean", "components", "random_components"]:
                np.testing.assert_array_equal(new[k], f[k])
    testids = disk.read(eda/"splits.json")["test"]
    refs = {}
    for point in disk.spec()["points"]:
        with np.load(eda/"fits"/f"{point}__C1__ID.npz") as f:
            refs[point] = f["prediction"].copy()
    started = time.monotonic()
    error = 0.
    with tempfile.TemporaryDirectory(prefix="pcgeom-p3-design-") as td:
        run = Path(td)/"run"
        disk.initialize(run)
        models, counts = fixed.models(run)
        measurement = Path(td)/"old-data-capacity.zip"
        with zipfile.ZipFile(measurement, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for j, doc in enumerate(testids):
                raw = num.load_npz(source/"docs"/doc/"raw.npz")
                for (point, condition), m in models.items():
                    output, _ = analysis.evaluate(raw, doc, point, condition, m, counts)
                    target = refs[point][j].T
                    error = max(error, float(np.max(abs(output["predictions"]-target))))
                    np.testing.assert_allclose(output["predictions"], target, atol=1e-10, rtol=1e-10)
                    if j < 16:
                        b = io.BytesIO();np.savez(b, **output)
                        z.writestr(f"evaluation/{doc}/{point}.npz", b.getvalue())
                if j < 16:
                    z.write(source/"docs"/doc/"raw.npz", f"docs/{doc}/raw.npz")
                    z.write(source/"docs"/doc/"doc.json", f"docs/{doc}/doc.json")
        measured = measurement.stat().st_size
    seconds = time.monotonic()-started
    fixed_bytes = sum((disk.HERE/n).stat().st_size for n in disk.SOURCE_NAMES)
    result = dict(status="PASS", new_model_forwards=0, new_articles=64,
        train_articles=0, excluded_old_ids=192, tokenizer_documents_matched=64,
        maximum_5gram_jaccard=max_jaccard, fixed_parameter_values_identical=True,
        original_test_article_points_checked=128, fixed_prediction_max_abs_error=error,
        numpy=np.__version__, reference_check_seconds_including_sample_zip=seconds,
        sample_zip_articles=16, sample_zip_bytes=measured,
        sample_zip_note="Existing P2 observations and new frozen readouts only; not a P3 GPU benchmark",
        extrapolated_final_bytes_before_small_metrics_logs=measured*4+fixed_bytes,
        raw_hidden_state_bytes=64*512*768*2*4,
        source_model_manifest_sha256=disk.digest(disk.HERE/"frozen_manifest.json"))
    disk.write(root/"evidence/p3_design_checks.json", result)
    print(disk.canonical(result))


if __name__ == "__main__":
    main()
