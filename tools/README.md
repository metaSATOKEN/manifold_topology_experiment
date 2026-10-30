# Verification, restoration, and experiment entry points

`verify_package.py` hashes repository files and, optionally, every release asset. It is read-only.

`prepare_workspace.py` first verifies the files, checks archive destinations, then copies the scientific tree and restores saved evidence into a new workspace. With `--dry-run` it does not extract files. It refuses an existing destination and conflicting duplicated members. It preserves each original archive metadata file under `_archive_metadata/`, including the distinct per-PART P4 indexes.

Both use Python 3.11+ and the standard library. Neither downloads data, runs experiment/analysis sources, invokes a model, nor uploads anything. See [reproducibility instructions](../docs/REPRODUCIBILITY.md).

`run_experiment.py` checks a selected stage's original source hashes by default. Only `--execute` invokes its unchanged Colab launcher, which installs dependencies, downloads weights, and performs computation. This command is separate from the two offline packaging tools above. Read the [source guide](../docs/CODE_GUIDE.md) before executing a stage.
