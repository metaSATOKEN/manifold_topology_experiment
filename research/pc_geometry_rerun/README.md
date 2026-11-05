# Scientific code and saved results

This tree retains the original directory relationships for the experiment and analysis sources. The top-level [paper](../../paper/transformer-pc-geometry-and-feature-readout.pdf), [figure/table map](../../references/FIGURE_TABLE_MAP.md), and [reproducibility guide](../../docs/REPRODUCIBILITY.md) are the current reader entry points.

- `p0_v2/`: pilot/recovery utilities retained as dependencies and provenance.
- `p1_v1/`–`p5_v1/`: unpacked acquisition programs, inputs, protocols, and support modules.
- `prereg/`: local scientific protocol cards and separate result addenda. These are not externally timestamp-certified registrations.
- `receipts/`: saved reports, numerical summaries, figures, and verification records for received P1–P5 outputs.
- `exploratory/`: lexical-control, readout-reversal, component/length/direction, and local-geometry analyses.

For a file-by-file reading route, start with the [source-code guide](../../docs/CODE_GUIDE.md). The [plain Python entry point](../../tools/run_experiment.py) checks sources by default and calls the existing launcher only with `--execute`.

Raw receipt contents and exploratory NPZ files are in the separate release assets. Use `.venv/bin/python tools/prepare_workspace.py` from the repository root, with the arguments in the reproducibility guide, to restore them into a new workspace. The current source snapshot is not a fresh execution environment; historical scripts may intentionally refuse to overwrite frozen results.

Scientific code, fixed scientific cards, settings, numerical summaries, and figures are copied unchanged. Compressed cell/notebook duplicates are omitted; their unpacked contents remain byte-identical. Superseded navigation READMEs, conversation excerpts, workstation-preservation manifests, and disposable logs were excluded. This curation does not discard negative or failed scientific outcomes.
