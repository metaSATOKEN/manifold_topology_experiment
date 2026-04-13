# The Arc and Its Thickness: Geometric Decomposition of Pre-LayerNorm Transformer Hidden States

## What this is

Code, data, and paper for a study that asks: **what is the actual shape of the low-dimensional subspace that Pre-LN Transformer hidden states live on, and what does each direction in that subspace encode?**

Short answer: it's a 1D arc dominated by hidden-state norm, and its "thickness" encodes position, prediction difficulty, and part-of-speech — in that order of variance.

## The problem

It's well known that Pre-LayerNorm Transformer hidden states concentrate on low-dimensional subspaces (PCA top-2 explains >90% of variance). But "low-dimensional subspace" is vague. Is it a torus? A sphere? Something else? And what do the principal directions actually represent? Without answering these questions, we can't design principled interventions on hidden states or understand what PCA-based analyses are really measuring.

## What we found

1. **PC1 is norm.** The dominant PCA axis (80–99.7% of variance) is near-perfectly correlated with hidden-state norm (|r| > 0.96 across all 12 tested models, |r| = 1.000 at 40B scale). This means most PCA-based analyses of Pre-LN models are primarily measuring norm variation, not semantic structure.

2. **Remove norm, and the real structure appears.** Norm normalization collapses PC1 variance by 58–94 percentage points. What's left is a structured residual subspace where token position, prediction difficulty, and part-of-speech are encoded in approximately orthogonal directions.

3. **No interesting topology.** Persistent homology finds no evidence of torus, sphere, or other non-trivial structure. The manifold is a contractible arc. Linguistic information is encoded via linear geometry, not global topology — which explains why linear probing methods work so well.

4. **The arc has a lifecycle.** In GPT-2, it forms abruptly at Layer 1, stabilizes through middle layers, and partially dissolves in the final layer. At 7B scale, the final-layer dissolution becomes complete (PC1 drops to 8–10%). Three distinct dissolution patterns emerge across architectures: plateau (GPT-2), funnel (Pythia, Mistral), and cliff (Qwen).

5. **This scales to 40B.** We validated the core finding (PC1=norm) across 12 models, 5 architecture families, and 3 orders of magnitude in scale (124M to 40B). It holds everywhere except OPT, which is a consistent outlier.

6. **Direction doesn't help for intervention.** We tested whether steering hidden states along specific PCA axes (surprisal, position) outperforms random orthogonal perturbation. It doesn't (N=61 crisis events, all p > 0.15). The geometric decomposition is useful for understanding representations, but not directly exploitable for token-level intervention.

## Why it matters

- **For ML researchers:** PCA on Pre-LN hidden states is dominated by a norm artifact. If you're using PCA for interpretability or compression, normalize first — the real structure is in the residual.
- **For model builders:** The norm → position → difficulty hierarchy is an architectural invariant of Pre-LN. It holds across GPT-2, Pythia, Qwen, Mistral, and Falcon up to 40B. If you're designing new architectures or normalization schemes, this is the structure you're inheriting (or disrupting).
- **For intervention/safety researchers:** The negative steering result is important. Token-level hidden-state perturbations are direction-agnostic at the small-effect regime. Response-level strategies (e.g., checkpoint restart) may be fundamentally more effective than representation-level nudging.
- **For investors/industry:** This is basic science about what's inside the models. The practical implication is that representation geometry constrains what monitoring and intervention techniques can work at inference time. Teams building model observability tools should account for the norm-dominance artifact.

## What's in this repository

- **Paper:** LaTeX source and compiled PDF (23 pages, 12 figures)
- **16 experiments** — local (topology + probing) and Colab (7B–40B scaling + steering)
- **All generated figures** (PNG) and **raw data** (JSON) for full reproducibility
- Tested on Apple M1 (16GB) for local experiments, Google Colab (T4 / Blackwell 102GB) for large-scale

## Models tested

14 models across 5 architecture families, 110M–40B parameters:

| Scale | Models |
|-------|--------|
| 110M–500M | GPT-1 (Post-LN control), GPT-2, OPT-125m, Pythia-410M, Qwen2-0.5B |
| 1B–3B | OPT-1.3B, Pythia-2.8B |
| 7B | Pythia-6.9B, Mistral-7B, Qwen2-7B |
| 12B–40B | Pythia-12B, OPT-13B, Qwen2.5-14B, Falcon-40B |

## Reproducibility

```bash
# Local experiments (topology + GPT-2 probing)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/topology/exp2_persistent_homology.py
python experiments/probing/exp5_thickness_probing.py

# GPU experiments — copy-paste scripts from experiments/colab/ into Google Colab
```

Local experiments run on consumer hardware in ~10 minutes. GPU experiments require T4 (15GB) for 7B models and 40GB+ for 13B–40B.

## Code

- **This repository:** [github.com/metaSATOKEN/manifold_topology_experiment](https://github.com/metaSATOKEN/manifold_topology_experiment)

## License

- Paper content: CC BY 4.0
- Code: Apache License 2.0

Copyright 2026 Kentaro Sato.
