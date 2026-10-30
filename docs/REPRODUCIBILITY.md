# Reproducibility and saved evidence

## Three distinct tasks

**File integrity.** `tools/verify_package.py` checks bytes against the package manifest. With `--assets-dir`, it also hashes all 19 assets. No numerical estimate is recomputed.

**Saved evidence.** `tools/prepare_workspace.py` creates a new directory, copies the research tree, and restores raw states, fixed predictions, coefficients, and original environment records. It executes no model or analysis. P1–P4 raw contents go under the corresponding `receipts/<run>/raw/`; P5's `evidence/` prefix is removed to obtain the same structure. Transport manifests are retained under `_archive_metadata/`. The four exploratory bundles restore their original research-relative paths.

**New acquisition or numerical reproduction.** The original unpacked experiment modules and analysis sources are preserved. Running acquisition downloads model weights and invokes forward passes. This publication preparation does not rerun them or claim an end-to-end replication in the new layout.

## Python interpreter

Run local commands from the repository root using **`.venv/bin/python`**. This selects the project's interpreter directly and does not require activation or a change to your shell PATH. The integrity and restoration tools require Python 3.11 or later and only the standard library.

Use an existing suitable `.venv` at the repository root. If no environment exists there, create it with an explicitly selected Python executable. For example, replace `/absolute/path/to/python3.13` below with the actual installed executable; this is a one-time setup step for a new checkout:

```sh
/absolute/path/to/python3.13 -m venv .venv
```

Check the selected environment before using the tools:

```sh
.venv/bin/python --version
.venv/bin/python -c 'import sys; print(sys.executable)'
.venv/bin/python -m pip --version
```

Use `.venv/bin/python -m pip` for any later package operations. Scientific reproduction requires the experiment-specific dependency versions as well as a suitable Python version; the standard-library environment above is sufficient for file verification and restoration only.

## Offline preparation

From the repository root, with the 19 ZIP assets in the sibling `release-assets/` directory:

```sh
.venv/bin/python tools/verify_package.py --assets-dir ../release-assets
.venv/bin/python tools/prepare_workspace.py --assets-dir ../release-assets --destination /tmp/pcgeom-saved-work --dry-run
.venv/bin/python tools/prepare_workspace.py --assets-dir ../release-assets --destination /tmp/pcgeom-saved-work
```

Choose a nonexistent destination outside this repository. All assets are required by this restoration command. Existing directories are refused. Leave the published snapshot untouched; any scientific recomputation should operate on a working copy and write new outputs.

The archive inventory and expected hashes are in [ASSETS.json](../references/ASSETS.json). Original archives retain their exact bytes. The four additional ZIPs contain unchanged `.npz` members; no coefficient or prediction was refitted. Original model weights are not bundled.

## Acquisition entry points

Read the [source-code guide](CODE_GUIDE.md) for the module map and correspondence with the original Colab bundles. Use [run_experiment.py](../tools/run_experiment.py) to select P1–P5 and invoke the visible modules directly. No compressed cell is needed.

From the repository root, inspect a stage without running it:

```sh
.venv/bin/python tools/run_experiment.py --help
.venv/bin/python tools/run_experiment.py p1
```

The second command checks every file in P1's historical source manifest. It imports no experiment module and performs no installation, model download, or numerical calculation. Select `p2`, `p3`, `p4`, or `p5` to inspect the other bundles.

For a deliberate new acquisition, put a complete copy of the repository in a Colab runtime with a GPU. Keep the directory structure intact. Set `repository` below to the actual uploaded or checked-out folder. This is a short, readable notebook launcher; Colab's Python is selected explicitly with `sys.executable`:

```python
from pathlib import Path
import subprocess
import sys

repository = Path("/content/manifold_topology_experiment")
subprocess.run(
    [sys.executable, str(repository / "tools/run_experiment.py"), "p1", "--execute"],
    check=True,
)
```

Choose the stage after reading its `spec.json` and protocol. `--execute` creates the stage's pinned environment, downloads weights, and runs the experiment. The launcher writes outputs below `/content/` and prints the ZIP paths. Download them through Colab's Files panel. To resume, upload the saved ZIP and append `--restore` followed by its path to the argument list. For P4, supply **all PARTs and the INDEX from the same snapshot**. An empty restore argument list follows the original same-VM resume behavior. The optional `--auto-download` requests browser downloads; each file still needs to finish saving locally. P0 is a historical recovery utility; if intentionally using it for a fresh pilot, pass `--restore NEW` with `--execute`.

Source checks and command dispatch have been tested without executing the scientific launchers. A new end-to-end Colab run through this entry point has not been performed. The scientific sources retain their original environment and overwrite guards.

The archived runs record their actual Python, PyTorch/CUDA, hardware, and library environments. The manuscript summarizes PyTorch 2.8.0+cu128, Transformers 4.57.6, and NumPy 2.2.6 for the reference inference environment. The stage specifications and launchers contain the exact dependency pins. The saved CPU analyses use NumPy/SciPy; plotting uses Matplotlib. Use those recorded dependencies for numerical reproduction.

## Saved analyses and verification

The four exploration directories retain `analyze.py` and their numerical/plotting helpers, scientific specifications, results, and verification reports. The receipt directories retain the reports and checks produced when P1–P5 were received. Start from the [figure/table map](../references/FIGURE_TABLE_MAP.md).

Local invocations of these scripts also use `.venv/bin/python` from the working-copy root, after preparing the required fixed dependencies. Receipt verifiers and analysis programs can recompute scientific quantities and write outputs; run them in a separate working copy with new output paths when a numerical reproduction is intended.

Original scripts retain their source bytes and historical execution guards. Some write to their own directory or refuse to overwrite frozen results. Administrative helpers such as `final_checks.py`, `finalize.py`, and `closeout.py` refer to the original research workflow; they are not the package verification entry points. Historical document links to working-session notes may also refer to material excluded from the public candidate. The [historical reference inventory](HISTORICAL_REFERENCES.md) lists these targets. No such local operational records are required to browse the paper, full tables, or restored observations.

The scope of earlier numerical checking is in manuscript Appendix C.6. The present preparation checks file integrity, portability of the offline restoration path, reader-facing links, archive membership, and preservation of the PDF's displayed content. It does not add new confirmatory evidence or constitute independent replication.

## Rebuilding the paper

`paper/main.tex`, `paper/body.tex`, and `paper/figures/` form a LaTeX project using TeX Gyre and Latin Modern fonts. The checked PDF was produced from the latest author-contact edition; packaging changes only its relative link annotations. The equivalent link paths were updated in the TeX and Markdown copies. Use the repository layout when distributing the PDF with its materials. Once a public URL is chosen, permanent web links can be substituted in a later publication edit.
