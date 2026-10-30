"""Package existing coefficients losslessly; do not fit or tune anything."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDA = ROOT/"exploratory/p2_lexical_v1"
OLD = ROOT/"receipts/6966c146e980_p2_v1_final/raw"


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build():
    assert not (HERE/"release_manifest.json").exists()
    assert not (HERE/"frozen_manifest.json").exists()
    assert json.loads((EDA/"numerical_verification.json").read_text())["status"] == "PASS"
    commit = json.loads((EDA/"FIT_COMMIT.json").read_text())
    for name, h in commit["files"].items():
        assert digest(EDA/name) == h
    cases = {}
    for point in ["block06", "final_ln"]:
        src = EDA/"fits"/f"{point}__C1__ID.npz"
        meta = json.loads(src.with_suffix(".json").read_text())
        with np.load(src, allow_pickle=False) as f:
            m = {k: f[k].copy() for k in f.files if k not in ["prediction", "normal_residual"]}
        # Integer masks have identical values and are accepted by the bounded NPZ reader.
        for k in ["feature_active", "score_active"]:
            m[k] = m[k].astype(np.int64)
        basis_path = OLD/"fit"/f"{point}__C1"/"arrays.npz"
        with np.load(basis_path, allow_pickle=False) as f:
            m.update({k: f[k].copy() for k in ["mean", "components", "random_components"]})
        dest = HERE/f"frozen_{point}_C1.npz"
        with dest.open("xb") as f:
            np.savez_compressed(f, **m)
        cases[point] = dict(file=dest.name, sha256=digest(dest), source_fit=str(src.relative_to(ROOT)),
            source_fit_sha256=digest(src), source_basis=str(basis_path.relative_to(ROOT)),
            source_basis_sha256=digest(basis_path), alpha=meta["alpha"], tau=meta["tau"],
            gamma=m["gammas"].tolist(), token_types=len(m["vocab"]), numeric_controls=8)
    for source, name in [(OLD/"fit/frequency.npz", "frozen_frequency.npz"),
                         (OLD/"model_files.json", "expected_model_files.json")]:
        with (HERE/name).open("xb") as f:
            f.write(source.read_bytes())
    manifest = dict(origin="P2 train-only ID fits selected in EDA-0001; frozen without refitting",
        original_p2_archive_sha256="730f2eb58af84fa6a6017eb2a07fe7324e4f6c4692dbec4ffb720fa2c4626d0a",
        original_fit_commit_sha256=digest(OLD/"fit/COMMIT.json"),
        exploratory_fit_commit_sha256=digest(EDA/"FIT_COMMIT.json"),
        train_ids=json.loads((EDA/"splits.json").read_text())["train"],
        selection_history="Family chosen after observing P2 results; no P3 observations acquired",
        cases=cases, files={n: digest(HERE/n) for n in
            ["frozen_block06_C1.npz", "frozen_final_ln_C1.npz",
             "frozen_frequency.npz", "expected_model_files.json"]},
        numeric_conversion="Boolean active masks converted losslessly to int64; float arrays copied exactly",
        new_training_or_tuning=False, new_language_model_forwards=0)
    (HERE/"frozen_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(cases=cases, total_bytes=sum((HERE/n).stat().st_size
                           for n in manifest["files"])), indent=2))


if __name__ == "__main__":
    build()
