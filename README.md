# Principal Component Geometry and Feature Readout in Transformer Representations

**Kentaro Sato · [info@metaclan.jp](mailto:info@metaclan.jp)**

**In GPT-2's middle layer, the first token drives much of the conspicuous variance concentration. After excluding its saved state from analysis, shared principal components still support surprisal readout on separate articles beyond lexical identity controls.**

The paper connects these two findings: which observations make representations look strongly concentrated, and what information remains readable when that concentration is reduced. It also shows why improvements on a typical article must be considered alongside failures on individual articles.

**[Read the paper (PDF)](paper/manuscript.pdf) · [Markdown version](paper/manuscript.md) · [Figures, tables, and supporting evidence](references/FIGURE_TABLE_MAP.md)**

## The question

PCA summarizes the directions in which a model's hidden states vary most. A striking long axis can suggest a simple organization of the representation space. To interpret it, we ask three questions: **What creates the shape? What features can be read from the remaining variation? Do those readouts transfer to other articles?**

We study GPT-2, OPT-1.3B, and Qwen2.5-1.5B base models using English WikiText articles, together with a geometric validation on three known short texts.

## Main findings

### 1. The first token strongly influences the apparent geometry

At GPT-2's middle layer, the first token has a strong influence on PC1's variance share and its association with hidden-state length. In the short texts, a distant first point and a dense cluster account for the prominent long axis. Excluding the first saved state sharply reduces the concentration; the input supplied to the model stays unchanged.

The article sensitivity tests are **confirmatory**. The short-text study is **geometric validation on known inputs**: strong concentration was reproduced, while the primary criterion for points distributed along a continuous arc was unmet in all three texts. Curve fits varied with extraction point and treatment.

### 2. Shared PCs retain information beyond token identity

With the first state excluded, a PCA basis fixed on training articles retains position correlations on separate articles. In a distinct test, shared PCs improve readout of **current-token surprisal**—how unexpected that token was given its preceding context—beyond a fixed token-ID baseline.

The prespecified GPT-2 middle-layer comparison also finds an advantage over random directions of the same dimensionality when comparing **repeated occurrences of the same token ID within the same article**. These **confirmatory** gains are measured as medians across evaluation articles. They show that the readout captures variation beyond differences between lexical identities.

### 3. The advantage depends on the evaluation conditions

A **confirmatory** comparison across three models and two extraction points used a training objective focused on variation within article × token-ID groups. **Four of six conditions met the prespecified minimum-effect criteria for surprisal gains over the baseline and over random directions; two were inconclusive.** The supported conditions were GPT-2 after final normalization, OPT at the middle layer, and Qwen at both points. GPT-2's middle layer and OPT after final normalization were inconclusive under this design.

In an earlier GPT-2 final-layer comparison within recurring token IDs, random directions outperformed PCs. Exploratory diagnostics showed that the ranking changed with the choice of training observations and objective.

## What the exploratory analyses add

Analyses of saved data help locate the additional readout and its weaknesses:

- **PC2–8 and vector direction matter.** Much of the additional surprisal association remains in PC2–8 after accounting for position and norm. Direction-based coordinates also add readout beyond flexible length controls, although their advantage over random directions is unclear at OPT's middle layer.
- **A positive median can conceal severe failures.** At Qwen's middle layer, a large loss on one article makes the mean increment negative despite a positive median. Reporting the article distribution changes how the aggregate gain should be interpreted.
- **Local residuals contain feature associations.** Coordinates fitted separately within each article reveal associations around its main axis. Their relationship to the fixed PCs shared across articles remains an open question.

These are **exploratory findings**, reported separately from the confirmatory tests.

## Contribution and scope

The contribution is an empirical connection between **first-token sensitivity of geometry, transferable readout beyond lexical identity, and variation in performance across articles**. The study combines point-distribution measurements with fixed PCA bases and predictors, token-ID controls, and matched random directions. This makes the link between a geometric visualization and a readout claim directly testable.

The results concern statistical readout in the selected models, extraction points, and English article passages. Causal mechanisms, dedicated semantic axes, generalization to other domains, and the correspondence between local residuals and shared PCs remain unresolved. The paper brings these limitations together in Section 8 and retains the unsupported criteria, inconclusive comparisons, and individual failures.

## Paper, data, and code

| To explore | Start here |
| --- | --- |
| Claims, methods, results, and compact appendices | [Paper PDF](paper/manuscript.pdf) or [Markdown](paper/manuscript.md) |
| Evidence behind a particular figure or table | [Figure and table map](references/FIGURE_TABLE_MAP.md) |
| Complete numerical tables and method details | [Reference materials](references/README.md) |
| Saved observations and exploratory arrays | [Data inventory](references/DATA.md) |
| Readable experiment code and its execution flow | [Source-code guide](docs/CODE_GUIDE.md) |
| Protocols, analysis code, and result summaries | [Research materials](research/pc_geometry_rerun/README.md) |
| File verification, data restoration, and execution instructions | [Reproducibility guide](docs/REPRODUCIBILITY.md) |

## Citation

Sato, Kentaro. 2026. *Principal Component Geometry and Feature Readout in Transformer Representations: Sensitivity to the First Token, Features in Local Residuals, and Readout from Fixed Shared Principal Components.*

[Citation metadata](CITATION.cff) · [Licensing and source attribution](licenses/README.md) · [Contact](mailto:info@metaclan.jp)
