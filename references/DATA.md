# Data inventory

Original observations are retained in 15 unchanged ZIP files. Four additional ZIPs contain unchanged exploratory coefficient/prediction arrays. These ZIPs are stored separately by the author in `release-assets/`; they are not included in the Git repository. Asset upload and download URLs are pending.

See [ASSETS.json](ASSETS.json) for exact sizes, SHA-256, restoration paths, and the future URL fields. Use [offline verification and restoration](../docs/REPRODUCIBILITY.md).

| Asset | Role | Size (decimal MB) |
| --- | --- | --- |
| `pcgeom_p1_v1_98a3faf68004_FINAL_04cd3d.zip` | P1 / original observations | 469.78 |
| `pcgeom_p2_v1_6966c146e980_FINAL_f3efc8.zip` | P2 / original observations | 498.96 |
| `pcgeom_p3_v1_0113330bdd64_FINAL_4b0a6d.zip` | P3 / original observations | 217.83 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_INDEX.zip` | P4 / original observations | 0.12 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_001_of_010.zip` | P4 / original observations | 230.99 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_002_of_010.zip` | P4 / original observations | 229.14 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_003_of_010.zip` | P4 / original observations | 229.99 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_004_of_010.zip` | P4 / original observations | 229.26 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_005_of_010.zip` | P4 / original observations | 227.91 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_006_of_010.zip` | P4 / original observations | 227.19 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_007_of_010.zip` | P4 / original observations | 228.46 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_008_of_010.zip` | P4 / original observations | 228.97 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_009_of_010.zip` | P4 / original observations | 231.45 |
| `pcgeom_p4_v1_4b013b35152b_FINAL_76c5c40ed614_PART_010_of_010.zip` | P4 / original observations | 99.12 |
| `pcgeom_p5_v1_52dcd57cc2e4_FINAL_9c1e90.zip` | P5 / original observations | 12.46 |
| `pcgeom_arc_thickness_v1_saved_arrays.zip` | arc_thickness_v1 / saved exploratory arrays | 341.38 |
| `pcgeom_p2_lexical_v1_saved_arrays.zip` | p2_lexical_v1 / saved exploratory arrays | 147.16 |
| `pcgeom_p3_reversal_v1_saved_arrays.zip` | p3_reversal_v1 / saved exploratory arrays | 57.21 |
| `pcgeom_p4_components_v1_saved_arrays.zip` | p4_components_v1 / saved exploratory arrays | 122.78 |

P4 requires the INDEX plus all ten PART files from snapshot `76c5c40ed614`. Do not concatenate the ZIPs. The preparation tool verifies every asset before restoring them, retains the transport metadata separately, and checks repeated files for identity.

The P5 ZIP was called `original.zip` in the archived reference folder. Its descriptive asset filename here changes no bytes. Exploratory ZIPs use repository-relative member paths and contain arrays only; their summaries, source code, and scientific specifications remain in the repository.

The original archive hashes and internal commitments remain the scientific provenance records. The package manifest adds a transport check for this layout; it does not replace them or establish independent experimental replication.
