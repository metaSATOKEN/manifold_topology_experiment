# The Arc and Its Thickness

Geometric Decomposition of Pre-LayerNorm Transformer Hidden States

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19590036.svg)](https://doi.org/10.5281/zenodo.19590036)

**Kentaro Sato** — info@metaclan.jp

## What is this?

Pre-LayerNorm Transformers concentrate hidden states on a low-dimensional subspace (PCA top-2 explains >90% of variance). We investigate the **topology** and **geometry** of this subspace.

**Key findings (12 models, 124M–40B, 5 architecture families):**

1. **Topologically trivial.** Persistent homology finds no torus, sphere, or other non-trivial structure. The manifold is empirically indistinguishable from a contractible arc.
2. **The arc is norm.** PC1 (~90% variance) is near-perfectly correlated with hidden-state norm (|r| > 0.96 in all 12 tested models, |r| = 1.000 at 40B).
3. **Structured thickness.** The residual subspace (PC2+) encodes token position, prediction difficulty, and part-of-speech in orthogonal directions — consistently across text types and models, though PC indices are model-dependent.
4. **Layer-wise emergence.** The arc forms abruptly in early layers, stabilizes through middle layers, and dissolves in the final layer. Three distinct dissolution patterns: plateau (GPT-2), funnel (Pythia, Mistral), cliff (Qwen).
5. **Cross-model consistency up to 40B.** PC1=norm holds across GPT-2, Pythia (410M/2.8B/6.9B/12B), Qwen (0.5B/7B/14B), Mistral-7B, OPT (1.3B/13B), and Falcon-40B. OPT is a consistent outlier with weaker dominance.
6. **Direction doesn't help intervention.** PC-directed steering does not outperform random orthogonal perturbation (N=61 crises, all p > 0.15).

## Paper

`paper/the_arc_and_its_thickness.pdf` (22 pages, 11 figures)

```bash
# Recompile
cd paper
pdflatex the_arc_and_its_thickness.tex && pdflatex the_arc_and_its_thickness.tex
```

## Repository Structure

```
manifold_topology_experiment/
├── paper/
│   ├── the_arc_and_its_thickness.tex    # LaTeX source
│   ├── the_arc_and_its_thickness.pdf    # Compiled PDF (23pp)
│   ├── the_arc_and_its_thickness.md     # Markdown version
│   ├── figures/                         # fig1–fig12 (real files, not symlinks)
│   └── LICENSE                          # CC BY 4.0 (paper content)
│
├── experiments/
│   ├── topology/                        # Section 4.1 — local CPU
│   │   ├── exp1_winding_number.py
│   │   ├── exp2_persistent_homology.py
│   │   ├── exp2b_topology_deep_dive.py
│   │   ├── exp2c_highd_long_text.py
│   │   ├── exp3_gaussian_curvature.py
│   │   └── exp4_geodesic_ratio.py
│   │
│   ├── probing/                         # Sections 4.2–4.7 — local CPU
│   │   ├── exp5_thickness_probing.py    # GPT-2 PC-feature correlations
│   │   ├── exp5b_norm_normalized.py     # Norm normalization effect
│   │   ├── exp5c_layerwise.py           # GPT-2 layer-wise analysis
│   │   └── exp5d_cross_model.py         # Cross-model (<=2.8B)
│   │
│   ├── intervention/                    # Section 5.4 — local (M1 fallback)
│   │   └── exp6_pc_directed_steering.py
│   │
│   ├── colab/                           # GPU-required — copy-paste to Colab
│   │   ├── exp6_colab_n100.py           # Steering N=100
│   │   ├── exp7b_7b_probing.py          # 7B mid-layer probing
│   │   ├── exp7c_7b_layerwise.py        # 7B layer-wise (Pythia/Mistral)
│   │   ├── exp7d_qwen2_7b_layerwise.py  # Qwen2-7B RMSNorm test
│   │   └── exp7e_large_scale_probing.py # 13B–40B probing
│   │
│   ├── run_all.py                       # Batch runner (local experiments)
│   └── utils/
│       ├── __init__.py
│       └── extraction.py                # Shared hidden-state extraction
│
├── results/                             # Generated figures (PNG)
├── EXPERIMENT_REPORT.md                 # Detailed experiment log (18 sections)
├── README.md                            # This file
├── LICENSE                              # Apache 2.0 (code)
├── requirements.txt
└── .gitignore
```

## Quick Start

### Local experiments (M1 Mac, 16GB)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Topology (Section 4.1)
python experiments/topology/exp2_persistent_homology.py
python experiments/topology/exp2b_topology_deep_dive.py
python experiments/topology/exp2c_highd_long_text.py

# Probing (Sections 4.2–4.7)
python experiments/probing/exp5_thickness_probing.py
python experiments/probing/exp5b_norm_normalized.py
python experiments/probing/exp5c_layerwise.py
python experiments/probing/exp5d_cross_model.py
```

### Colab experiments (GPU required)

Copy-paste scripts from `experiments/colab/` into Google Colab cells:

```bash
# 7B probing (Section 5.1) — T4 GPU sufficient
experiments/colab/exp7b_7b_probing.py

# 7B layer-wise (Section 5.3) — T4 GPU sufficient
experiments/colab/exp7c_7b_layerwise.py

# Steering N=100 (Section 5.4) — any GPU
experiments/colab/exp6_colab_n100.py

# 13B–40B probing (Section 5.1) — 40GB+ VRAM recommended
experiments/colab/exp7e_large_scale_probing.py
```

## Models Tested

| Model | Params | d_model | LN Type | HuggingFace ID |
|-------|--------|---------|---------|----------------|
| GPT-1 | 110M | 768 | Post-LN | `openai-community/openai-gpt` |
| GPT-2 | 124M | 768 | Pre-LN | `gpt2` |
| OPT-125m | 125M | 768 | Pre-LN | `facebook/opt-125m` |
| Pythia-410M | 410M | 1024 | Pre-LN | `EleutherAI/pythia-410m` |
| Qwen2-0.5B | 0.5B | 896 | Pre-LN (RMSNorm) | `Qwen/Qwen2-0.5B` |
| OPT-1.3B | 1.3B | 2048 | Pre-LN | `facebook/opt-1.3b` |
| Pythia-2.8B | 2.8B | 2560 | Pre-LN | `EleutherAI/pythia-2.8b` |
| Pythia-6.9B | 6.9B | 4096 | Pre-LN | `EleutherAI/pythia-6.9b` |
| Mistral-7B | 7B | 4096 | Pre-LN (RMSNorm) | `mistralai/Mistral-7B-v0.1` |
| Qwen2-7B | 7B | 3584 | Pre-LN (RMSNorm) | `Qwen/Qwen2-7B` |
| Pythia-12B | 12B | 5120 | Pre-LN | `EleutherAI/pythia-12b` |
| OPT-13B | 13B | 5120 | Pre-LN | `facebook/opt-13b` |
| Qwen2.5-14B | 14B | 5120 | Pre-LN (RMSNorm) | `Qwen/Qwen2.5-14B` |
| Falcon-40B | 40B | 8192 | Pre-LN | `tiiuae/falcon-40b` |

## Hardware

- **Local:** Apple M1, 16 GB — topology + GPT-2 probing (~10 min)
- **Colab:** T4 GPU (15GB) for 7B; Blackwell (102GB) for 13B–40B

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19590036.svg)](https://doi.org/10.5281/zenodo.19590036)

```bibtex
@article{sato2026arc,
  title   = {The Arc and Its Thickness: Geometric Decomposition of
             Pre-LayerNorm Transformer Hidden States},
  author  = {Sato, Kentaro},
  year    = {2026},
  doi     = {10.5281/zenodo.19590036},
  url     = {https://doi.org/10.5281/zenodo.19590036}
}
```

## License

- **Paper content** (`.tex`, `.pdf`, `.md`, figures): CC BY 4.0
- **Experiment code** (`.py`): Apache License 2.0

Copyright 2026 Kentaro Sato.
