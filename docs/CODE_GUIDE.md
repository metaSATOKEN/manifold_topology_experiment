# Reading the experiment code

The experiment is published as ordinary Python modules, JSON inputs, and binary numerical arrays. **Start with the stage's launcher, then follow the worker and analysis modules.** There is no compressed code payload to unpack or execute in the public entry point.

## A map of the sources

All stage directories below are under `research/pc_geometry_rerun/`.

| Stage | Question | Start reading | Main calculations |
| --- | --- | --- | --- |
| P1 | How sensitive are variance and norm associations to the first state? | [p1_launch.py](../research/pc_geometry_rerun/p1_v1/p1_launch.py) | [worker](../research/pc_geometry_rerun/p1_v1/p1_worker.py), [analysis](../research/pc_geometry_rerun/p1_v1/p1_analysis.py) |
| P2 | Do fixed PCs and readouts transfer to separate articles? | [p2_launch.py](../research/pc_geometry_rerun/p2_v1/p2_launch.py) | [fit](../research/pc_geometry_rerun/p2_v1/p2_fit.py), [worker](../research/pc_geometry_rerun/p2_v1/p2_worker.py), [analysis](../research/pc_geometry_rerun/p2_v1/p2_analysis.py) |
| P3 | Does the fixed token-ID comparison generalize to new articles? | [p3_launch.py](../research/pc_geometry_rerun/p3_v1/p3_launch.py) | [frozen readouts](../research/pc_geometry_rerun/p3_v1/p3_frozen.py), [analysis](../research/pc_geometry_rerun/p3_v1/p3_analysis.py) |
| P4 | How does within-ID readout vary across three models and two points? | [p4_launch.py](../research/pc_geometry_rerun/p4_v1/p4_launch.py) | [worker](../research/pc_geometry_rerun/p4_v1/p4_worker.py), [fit](../research/pc_geometry_rerun/p4_v1/p4_fit.py), [analysis](../research/pc_geometry_rerun/p4_v1/p4_analysis.py) |
| P5 | What shape is reproduced on the three known short texts? | [p5_launch.py](../research/pc_geometry_rerun/p5_v1/p5_launch.py) | [worker](../research/pc_geometry_rerun/p5_v1/p5_worker.py), [geometry](../research/pc_geometry_rerun/p5_v1/p5_numeric.py), [analysis](../research/pc_geometry_rerun/p5_v1/p5_analysis.py) |

Within each directory:

- `spec.json` records fixed settings, software pins, and resource limits. `protocol.md` or `experiment_card.md` explains the scientific comparison.
- `fixtures.json` and `input_manifest.json` hold fixed article inputs and their provenance for P1–P4. P5 uses `inputs.json`. P3's `frozen_*.npz` files contain numerical parameters, with an inventory in `frozen_manifest.json`.
- `p*_launch.py` coordinates environment setup, subprocesses, checkpoints, and final archives.
- `p*_worker.py` loads the model, extracts states, and records token features. `p*_fit.py`, where present, fits and seals training quantities.
- `p*_numeric.py` and `p*_analysis.py` implement measurements and aggregation. `p*_verify.py`, where present, checks saved evidence and recomputes quantities.
- `p*_io.py` implements hashing, immutable writes, archive checks, and restoration.

The exploratory analyses are separate: [lexical controls](../research/pc_geometry_rerun/exploratory/p2_lexical_v1/), [P3 reversal](../research/pc_geometry_rerun/exploratory/p3_reversal_v1/), [P4 components](../research/pc_geometry_rerun/exploratory/p4_components_v1/), and [local geometry](../research/pc_geometry_rerun/exploratory/arc_thickness_v1/). Each retains its analysis source and report. The [figure/table map](../references/FIGURE_TABLE_MAP.md) connects results to evidence.

`p0_v2/` contains pilot/recovery utilities. It is retained for provenance and dependencies; begin with P1–P5 for the paper's studies.

## What changed for publication

The scientific modules, fixed inputs, and parameters are **byte-identical to the files inside the original Colab cells**. The six source bundles were checked by decoding data literals without running those cells. [SOURCE_EQUIVALENCE.json](../references/SOURCE_EQUIVALENCE.json) lists each file and its SHA-256 hash.

The duplicated compressed `.py` cells and single-cell notebooks are omitted from this source tree. The author's local copies are preserved. Historical `release_manifest.json` files still contain the original cell/notebook hashes as provenance; those fields do not enumerate the current publication package. [MANIFEST.json](../references/MANIFEST.json) inventories the current package. The readable `build_cell.py` files describe the old packaging procedure and are retained as historical source, not required for reading or running the modules. Their frozen-release guards are unchanged.

The new [run_experiment.py](../tools/run_experiment.py) is a small command-line entry point. It checks the original source hashes and calls the existing launcher's `run` function. Its default action only checks files; `--execute` starts computation. Browser download requests are opt-in, while the original launchers retain their archival and recovery behavior.

## What execution does

The source check reads local files using the standard library. With `--execute`, the launcher creates its pinned Python environment, installs packages, downloads the selected model weights, runs model inference and analysis, and saves logs, checkpoints, and ZIPs under `/content/`. P4 emits a matched set of PART ZIPs and an INDEX. Restoring a previous run requires compatible recorded hardware and software.

The numerical launchers target Google Colab. They are not a general macOS or arbitrary-server runner. The new entry point has been checked for argument handling, source verification, and dispatch without executing the experiment; a new end-to-end GPU run has not been performed for this packaging change. See the [execution instructions](REPRODUCIBILITY.md) for inspecting sources locally and deliberately launching them in Colab.
