"""Regression tests use the received evidence as data; no GPU/model code runs."""
import copy
import json
from pathlib import Path
import shutil
import zipfile
import pytest
import p0_io as disk
import p0_worker as worker
import p0_verify as verifier
import p0_launch as launcher
import p0_numeric as numeric

RECEIPT = disk.HERE.parent / "receipts" / "7ec0f9a0ff0c_v1_failure"
ARCHIVE = RECEIPT / "pcgeom_p0_v1_7ec0f9a0ff0c_PARTIAL_OR_FAILURE_1a75e0.zip"


def recorded_environment():
    # Recreate the tuple-valued in-memory inventory produced by hardware().
    env = disk.read(RECEIPT / "raw/attempts/f8022fecd0b4/environment.json")
    env["packages"] = [tuple(row) for row in env["packages"]]
    return env


def test_exact_incident_json_roundtrip():
    current = recorded_environment()
    saved = disk.read(RECEIPT / "raw/environment_contract.json")
    assert saved["packages"] != current["packages"]  # v1's false mismatch
    assert json.loads(disk.canonical(current))["packages"] == saved["packages"]
    worker.require_compatible(saved, current)
    assert worker.compatibility(current) == saved


@pytest.mark.parametrize("field", [
    "gpu_name", "compute_capability", "python", "torch", "cuda_build", "nvidia_smi",
    "driver_query_exit", "packages", "tf32", "dtype", "attention",
])
def test_real_environment_changes_still_rejected(field):
    env = recorded_environment()
    saved = json.loads(disk.canonical(worker.compatibility(env)))
    changed = copy.deepcopy(env)
    if field == "packages":
        changed[field][0] = (changed[field][0][0], "changed-version")
    elif field == "compute_capability":
        changed[field] = [0, 0]
    elif field == "tf32":
        changed[field] = True
    elif field == "driver_query_exit":
        changed[field] = 1
    else:
        changed[field] = "changed-value"
    with pytest.raises(ValueError, match=field):
        worker.require_compatible(saved, changed)


def test_actual_environment_records_agree_despite_free_disk_change():
    root = RECEIPT / "raw"
    first = disk.read(root / "attempts/0fc711ec27fd/environment.json")
    second = disk.read(root / "attempts/f8022fecd0b4/environment.json")
    assert first["disk_free_bytes"] != second["disk_free_bytes"]
    worker.require_compatible(first, second)


@pytest.mark.parametrize("document", ["p0_story", "p0_science"])
@pytest.mark.parametrize("point", disk.spec()["points"])
def test_existing_raw_arrays_pass_downstream_pca_check(document, point):
    # Only recompute from already received arrays; no forward or new observations.
    raw, _ = numeric.validate_raw(RECEIPT / "raw", document)
    features = verifier.features(raw)
    arrays, metrics = numeric.pca(raw["h_"+point], features)
    numeric.verify_pca(raw["h_"+point], arrays, metrics, features)


def test_received_archive_migration_preserves_all_observations(tmp_path):
    old_hash = disk.digest(ARCHIVE)
    root = tmp_path / "migrated"
    index = disk.restore(ARCHIVE, root, allow_legacy=True)
    assert index["capsule"] == disk.legacy()["capsule"]
    assert disk.digest(ARCHIVE) == old_hash
    for p in disk.files(RECEIPT / "raw/docs"):
        assert disk.digest(p) == disk.digest(root / "docs" / p.relative_to(RECEIPT / "raw/docs"))
    old = disk.read(root / "code_history" / disk.legacy()["capsule"] / "run.json")
    new = disk.read(root / "run.json")
    assert old["run_id"] == new["run_id"] == "7ec0f9a0ff0c"
    assert old["specification_sha256"] == new["specification_sha256"]
    assert new["capsule"] == disk.capsule()
    result = verifier.verify(root, complete=False)
    assert (result["committed_documents"], result["forward_attempts"], result["worker_failures"]) == (2, 2, 1)
    # The migrated result itself must remain restorable without a second migration.
    archive = disk.snapshot(root, tmp_path / "archives", "PARTIAL_OR_FAILURE")
    restored = tmp_path / "restored_v2"
    disk.restore(archive, restored)
    assert verifier.verify(restored, complete=False) == result


def test_legacy_migration_requires_explicit_permission(tmp_path):
    with pytest.raises(ValueError, match="Wrong ZIP version"):
        disk.restore(ARCHIVE, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_migrated_lineage_damage_is_rejected(tmp_path):
    root = tmp_path / "migrated"
    disk.restore(ARCHIVE, root, allow_legacy=True)
    p = root / "code_history" / disk.legacy()["capsule"] / "sources/p0_worker.py"
    p.write_text(p.read_text()+"\n# damaged\n")
    with pytest.raises(ValueError, match="source history"):
        verifier.verify(root, complete=False)


def test_model_cache_path_can_move_but_config_cannot_change(tmp_path):
    original = disk.read(RECEIPT / "raw/loaded_model_config.json")
    disk.write(tmp_path / "loaded_model_config.json", original)
    saved_hash = disk.digest(tmp_path / "loaded_model_config.json")
    current = copy.deepcopy(original)
    current["_name_or_path"] = "/content/another_cache/snapshot"
    # Transformers returns integer id2label keys; JSON persists string keys.
    current["id2label"] = {int(k): v for k, v in current["id2label"].items()}
    worker.record_model_config(tmp_path, "relocated", current)
    assert disk.digest(tmp_path / "loaded_model_config.json") == saved_hash
    current["layer_norm_epsilon"] = 0.01
    with pytest.raises(ValueError, match="configuration changed"):
        worker.record_model_config(tmp_path, "different", current)
    assert not (tmp_path / "attempts/different/loaded_model_config.json").exists()


def test_default_resume_uses_exact_diagnosed_archive(tmp_path, monkeypatch):
    legacy = tmp_path / "v1"
    (legacy / "archives").mkdir(parents=True)
    shutil.copy2(ARCHIVE, legacy / "archives" / ARCHIVE.name)
    monkeypatch.setattr(disk, "legacy_base", lambda: legacy)
    base = tmp_path / "v2"
    base.mkdir()
    root = launcher.resolve(base, "")
    before = disk.committed(root)
    assert len(before) == 2
    assert launcher.resolve(base, "") == root
    assert disk.committed(root) == before


def test_damaged_known_archive_never_starts_new_run(tmp_path, monkeypatch):
    legacy = tmp_path / "v1"
    (legacy / "archives").mkdir(parents=True)
    (legacy / "archives" / ARCHIVE.name).write_bytes(b"changed")
    monkeypatch.setattr(disk, "legacy_base", lambda: legacy)
    base = tmp_path / "v2"
    base.mkdir()
    with pytest.raises(ValueError, match="Known failure archive changed"):
        launcher.resolve(base, "")
    assert list(base.iterdir()) == []


def test_fresh_run_requires_explicit_new(tmp_path):
    base = tmp_path / "v2"
    base.mkdir()
    root = launcher.resolve(base, "NEW")
    assert disk.committed(root) == {}
