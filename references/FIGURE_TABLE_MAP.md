# From manuscript figures and tables to saved evidence

Numbers below refer to the [English manuscript](../paper/manuscript.md). Detailed sets D–M are archival labels, separate from manuscript Appendices A–C.

| Manuscript item | Result or method | Complete tables/specifications | Saved summaries and records |
| --- | --- | --- | --- |
| Tables 1–2; Appendix A | Stages, evidence levels, samples, methods, and decision rules | [Full specification](METHODS_AND_IDENTIFIERS.md); [identifiers M](tables/appendix_M.md) | [Fixed protocol cards](../research/pc_geometry_rerun/prereg/) |
| Tables 3–4, C1; Figures 1–4 | Short-text distributions, sensitivity, residual correlations, and curve fits | [Detailed L](tables/appendix_L.md); [extrapolation M](tables/appendix_M.md) | [Verified P5 quantities](../research/pc_geometry_rerun/receipts/52dcd57cc2e4_p5_v1_final/recomputed_metrics.json) |
| Tables 5, B1; Figure 5 | First-token/high-norm sensitivity and P1 decisions | [Detailed D](tables/appendix_D.md) | [P1 receipt analyses](../research/pc_geometry_rerun/receipts/98a3faf68004_p1_v1_final/analysis/) |
| Tables 6, B2 | Position and surprisal readout on separate articles | [Detailed E](tables/appendix_E.md) | [P2 receipt analyses](../research/pc_geometry_rerun/receipts/6966c146e980_p2_v1_final/analysis/) |
| Tables 7, B3 | Fixed ID control, within-ID evaluation, and final-layer reversal | [Detailed F](tables/appendix_F.md) | [P3 receipt analyses](../research/pc_geometry_rerun/receipts/0113330bdd64_p3_v1_final/analysis/) |
| Tables 8, 8b, B1; Figure 6 | All 6 confirmatory conditions, absolute R², and concentration | [Detailed G](tables/appendix_G.md) | [P4 aggregates](../research/pc_geometry_rerun/receipts/4b013b35152b_p4_v1_final/analysis/aggregates.json) |
| Tables 9, B3 | Exploratory lexical/nonlinear comparisons | [Detailed H](tables/appendix_H.md) | [Lexical exploration](../research/pc_geometry_rerun/exploratory/p2_lexical_v1/) |
| Tables 10, B3–B4 | Configuration-dependent reversal, correlations, and prediction amplitude | [Detailed I](tables/appendix_I.md); [complete prediction diagnostics](tables/prediction_diagnostics.md) | [Reversal diagnostic results](../research/pc_geometry_rerun/exploratory/p3_reversal_v1/results.json) |
| Tables 11–13, B3; Figures 7–9 | Component subsets, length/direction, and important failures | [Detailed J](tables/appendix_J.md) | [Component exploration summary](../research/pc_geometry_rerun/exploratory/p4_components_v1/summary.json) |
| Tables 14, C2–C3; Figure 10 | Article geometry, all 18 conditions, and local-residual failures | [Detailed K](tables/appendix_K.md) | [Article-geometry exploration](../research/pc_geometry_rerun/exploratory/arc_thickness_v1/) |
| Table C4; Section C.6 | CPU/GPU exception and acquisition/verification scope | [Detailed L](tables/appendix_L.md); [Detailed M](tables/appendix_M.md) | [P5 verification](../research/pc_geometry_rerun/receipts/52dcd57cc2e4_p5_v1_final/verification.json) |

Use the [archive inventory](DATA.md) to access original states, inputs, and predictions fixed at acquisition. Exploratory predictions and coefficients are also retained in the corresponding exploration folders.

The manuscript, detailed tables, and machine-readable summaries can display different precision. These are rounded representations of saved values, not separate estimates. Differences between article medians, medians of paired article-level increments, article means, and individual random sets versus their mean are distinct quantities.

Table 11 maps to all 6 `A7_adv` conditions in Detailed J. The count of 18 conditions and the smallest lower bound in Appendix C.4 map to `thickness_gain_controlled_surprisal` in Detailed K. All linked observations, numerical results, code, and archives remain unchanged.

The original-norm values, fold-training range, and position-1 error concentration added to Appendix C.4 come from the existing [local-residual failure diagnostics](../research/pc_geometry_rerun/exploratory/arc_thickness_v1/failure_diagnostics.json). They describe that local readout, separately from the shared-PC failure in Table 13. The revised figure labels are documented in the [figure consistency record](figure_checks.json); their numerical contents have not been re-estimated.
