#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Experiment 2c: High-Dimensional Topology with Longer Text

Addresses two limitations of Exp 2/2b:
  1. Too few points (~100 tokens) for reliable topology estimation
  2. PCA-2D crushes the structure (PC1 eats 96%, everything else vanishes)

Approach:
  - Use much longer text (~500-1000 tokens)
  - Run persistent homology in PCA 2D, 5D, 10D, 20D
  - Compare: does topology emerge in the "remaining 4%" dimensions?
  - Also test: subtract PC1 (the dominant direction) and look at
    the residual manifold structure

Key question: Is there meaningful topology hidden in PC2-PC10
that gets crushed when we project to 2D?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)


# ── Long texts (~500-800 tokens) ───────────────────────────────────────────

LONG_TEXTS = {
    "narrative_long": (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm. The sea below crashed "
        "against the rocks with a rhythm that matched his breathing, and he "
        "found comfort in the predictability of waves and wind. Tonight, "
        "however, something was different. A strange light flickered on the "
        "horizon, pulsing with an irregular beat that made him uneasy. "
        "He set the lantern down on the iron railing and squinted into the "
        "darkness. The light grew brighter, then dimmed, then brightened again. "
        "It was not the steady beam of a ship's lantern, nor the rhythmic "
        "flash of another lighthouse. It moved in patterns he had never seen "
        "before, tracing arcs across the water like a living thing. "
        "The keeper reached for his telescope, hands trembling slightly. "
        "Through the lens, the light resolved into something he could not "
        "explain: a luminous sphere hovering just above the waves, its "
        "surface rippling with colors that shifted from deep violet to "
        "brilliant gold. Around it, the water was perfectly still, as if "
        "the ocean itself held its breath. He watched for what felt like "
        "hours, though the clock said only minutes had passed. The sphere "
        "drifted closer, and he could feel a warmth on his face, gentle "
        "as morning sunlight. Then, as suddenly as it had appeared, the "
        "light vanished. The waves resumed their ancient rhythm. The keeper "
        "stood alone in the darkness, wondering if he had imagined the whole "
        "thing. But the warmth lingered on his skin, and in the morning, "
        "he would find a single golden feather on the lighthouse steps. "
        "He picked up his lantern and continued his climb to the top. "
        "The great lamp needed tending, as it always did. Some things, "
        "at least, remained predictable. He wound the clockwork mechanism "
        "and watched the beam sweep across the black water, a faithful "
        "sentinel cutting through the night. Below, fishing boats rocked "
        "gently in the harbor, their crews sleeping soundly, trusting the "
        "light to guide them home when morning came. The keeper thought "
        "about the strange sphere and what it might mean. In forty years, "
        "he had seen storms that bent steel and waves that climbed the "
        "tower walls. He had seen ships appear from fog like ghosts and "
        "vanish again just as quickly. But nothing like that light. "
        "He made a note in his logbook, careful and precise, the way he "
        "recorded everything: wind speed, wave height, visibility, and now "
        "this. An unexplained luminous phenomenon, duration approximately "
        "seven minutes, bearing southwest, altitude approximately fifty "
        "meters above sea level. He closed the book and sat in his chair "
        "by the window, watching the empty horizon until dawn."
    ),
    "technical_long": (
        "Transformer architectures process sequential input through layers of "
        "multi-head self-attention and feed-forward networks. Each attention "
        "head computes query, key, and value projections, producing a weighted "
        "sum over value vectors. The weights are determined by the softmax of "
        "scaled dot-product similarities between queries and keys. Residual "
        "connections and layer normalization stabilize training across many "
        "layers, enabling models with billions of parameters to converge. "
        "The self-attention mechanism allows each token to attend to every "
        "other token in the sequence, creating a fully connected graph at "
        "each layer. This global receptive field distinguishes Transformers "
        "from convolutional and recurrent architectures, which build up "
        "context incrementally through local operations or sequential state "
        "updates. The computational cost of self-attention scales quadratically "
        "with sequence length, motivating research into efficient variants "
        "such as sparse attention, linear attention, and sliding window "
        "approaches. Feed-forward networks in each Transformer block "
        "typically expand the hidden dimension by a factor of four before "
        "projecting back down, creating a bottleneck structure that forces "
        "the model to form compressed representations. Recent work suggests "
        "these feed-forward layers act as key-value memories, storing "
        "factual associations learned during pretraining. The interplay "
        "between attention and feed-forward layers creates a complex "
        "information processing pipeline where attention routes information "
        "and feed-forward layers transform it. Layer normalization can be "
        "applied before each sublayer (Pre-LN) or after (Post-LN), with "
        "significant implications for training dynamics and hidden state "
        "geometry. Pre-LN architectures tend to produce hidden states that "
        "concentrate on low-dimensional manifolds, while Post-LN models "
        "distribute variance more broadly across dimensions. This geometric "
        "difference has consequences for phase extraction methods, as the "
        "concentration of variance determines whether angular coordinates "
        "in PCA space carry meaningful information. The training process "
        "itself shapes these geometric properties through gradient descent "
        "on the cross-entropy loss over large text corpora. Models learn "
        "to predict the next token given preceding context, developing "
        "internal representations that capture syntactic structure, semantic "
        "relationships, and world knowledge. The resulting hidden state "
        "trajectories reflect the sequential processing of language, with "
        "each token position contributing to an evolving representation "
        "that integrates information from the entire preceding context. "
        "Understanding the geometry of these representations is crucial "
        "for interpretability and for developing better architectures."
    ),
    "mixed_long": (
        "She opened the laptop and stared at the error message. The server "
        "had crashed again, the third time this week. Memory allocation "
        "failures in the gradient computation pipeline were corrupting the "
        "checkpoint files. She pulled up the monitoring dashboard and "
        "scrolled through the metrics. CPU utilization was spiking to "
        "ninety-eight percent every twelve minutes, exactly matching the "
        "batch processing interval. The pattern was too regular to be "
        "coincidental. She suspected a memory leak in the custom CUDA "
        "kernel they had written for the attention mechanism. The kernel "
        "allocated temporary buffers for the softmax computation but "
        "failed to release them when the backward pass encountered "
        "numerical overflow. Each iteration leaked a small amount of "
        "GPU memory until the system ran out and crashed. The fix was "
        "straightforward: add a try-finally block around the buffer "
        "allocation to ensure cleanup regardless of exceptions. But "
        "finding the bug had taken three days of careful analysis. "
        "She made the change, committed the code, and kicked off a new "
        "training run. While waiting, she reviewed the latest research "
        "papers on efficient attention mechanisms. One paper proposed "
        "replacing softmax attention with a linear kernel approximation "
        "that reduced the computational complexity from quadratic to "
        "linear in sequence length. Another suggested using sparse "
        "attention patterns learned through a reinforcement learning "
        "objective. A third paper took a completely different approach, "
        "arguing that attention itself was unnecessary and that simple "
        "feed-forward networks with appropriate positional encodings "
        "could match Transformer performance on many benchmarks. "
        "She found this last claim hard to believe but appreciated the "
        "boldness of the argument. Outside her window, the campus was "
        "quiet. It was past midnight, and the only other lights came "
        "from the physics building across the courtyard, where someone "
        "was probably running simulations of their own. She thought "
        "about how similar their work was, despite the different domains. "
        "Both were trying to understand complex systems by building "
        "simplified models and measuring how well those models captured "
        "the essential behavior. The physicist modeled particles and "
        "fields; she modeled language and meaning. Both wrestled with "
        "high-dimensional spaces and the challenge of extracting "
        "interpretable structure from overwhelming complexity. "
        "The training run finished without errors. She checked the "
        "loss curves and smiled. The fix had worked. Now she could "
        "focus on the actual research question: whether the geometric "
        "structure of the hidden states changed with training, or "
        "whether it was an architectural invariant present from "
        "initialization. She loaded checkpoints from different "
        "training stages and began her analysis."
    ),
}


# ── Analysis functions ─────────────────────────────────────────────────────

def run_multidim_topology(
    trajectory: np.ndarray,
    pca_dims: list[int] = [2, 5, 10, 20],
    max_homology_dim: int = 2,
    max_points: int = 300,
) -> dict:
    """
    Run persistent homology at multiple PCA dimensions.
    """
    T, d = trajectory.shape
    results = {}

    for k in pca_dims:
        k_actual = min(k, d, T - 1)
        pca = PCA(n_components=k_actual)
        projected = pca.fit_transform(trajectory)
        var_exp = pca.explained_variance_ratio_
        cumvar = np.cumsum(var_exp)

        # Subsample if needed
        if T > max_points:
            idx = np.random.choice(T, max_points, replace=False)
            idx.sort()
            points = projected[idx]
        else:
            points = projected

        # Normalize
        scaler = StandardScaler()
        points_norm = scaler.fit_transform(points)

        # Ripser
        max_dim = min(max_homology_dim, k_actual - 1)
        rips = ripser(points_norm, maxdim=max_dim)
        diagrams = rips["dgms"]

        # Analyse barcodes
        betti_at_thresholds = {}
        for thr in [0.05, 0.1, 0.15, 0.2, 0.3]:
            betti = {}
            for dim_h, dgm in enumerate(diagrams):
                if len(dgm) == 0:
                    betti[dim_h] = 0
                    continue
                finite = dgm[np.isfinite(dgm[:, 1])]
                n_inf = int(np.sum(~np.isfinite(dgm[:, 1])))
                if len(finite) == 0:
                    betti[dim_h] = n_inf
                    continue
                pers = finite[:, 1] - finite[:, 0]
                max_p = np.max(pers)
                sig = int(np.sum(pers > thr * max_p)) + n_inf if max_p > 0 else n_inf
                betti[dim_h] = sig
            betti_at_thresholds[thr] = betti

        # H1 barcode stats
        h1_stats = {}
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            finite = diagrams[1][np.isfinite(diagrams[1][:, 1])]
            if len(finite) > 0:
                pers = np.sort(finite[:, 1] - finite[:, 0])[::-1]
                gap = float(pers[0] / pers[1]) if len(pers) >= 2 and pers[1] > 1e-12 else float("inf")
                h1_stats = {
                    "n_features": len(pers),
                    "top5": pers[:5].tolist(),
                    "gap_ratio": gap,
                    "max_persistence": float(pers[0]),
                }

        results[k] = {
            "k_actual": k_actual,
            "var_explained": var_exp,
            "cumvar": cumvar,
            "betti_at_thresholds": betti_at_thresholds,
            "h1_stats": h1_stats,
            "diagrams": diagrams,
            "n_points": len(points),
        }

    return results


def run_residual_topology(
    trajectory: np.ndarray,
    max_points: int = 300,
) -> dict:
    """
    Remove PC1 (the dominant direction) and analyse topology
    of the residual manifold (PC2-PC10).

    This tests: is there structure in the "remaining 4%"?
    """
    T, d = trajectory.shape
    k = min(20, d, T - 1)
    pca = PCA(n_components=k)
    projected = pca.fit_transform(trajectory)
    var_exp = pca.explained_variance_ratio_

    # Residual: remove PC1
    residual = projected[:, 1:]  # PC2 onwards
    residual_var = var_exp[1:]

    # Subsample
    if T > max_points:
        idx = np.random.choice(T, max_points, replace=False)
        idx.sort()
        points = residual[idx]
    else:
        points = residual

    scaler = StandardScaler()
    points_norm = scaler.fit_transform(points)

    # Ripser on residual (up to H2)
    max_dim = min(2, points_norm.shape[1] - 1)
    rips = ripser(points_norm, maxdim=max_dim)
    diagrams = rips["dgms"]

    # Betti numbers
    betti_at_thresholds = {}
    for thr in [0.05, 0.1, 0.15, 0.2, 0.3]:
        betti = {}
        for dim_h, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                betti[dim_h] = 0
                continue
            finite = dgm[np.isfinite(dgm[:, 1])]
            n_inf = int(np.sum(~np.isfinite(dgm[:, 1])))
            if len(finite) == 0:
                betti[dim_h] = n_inf
                continue
            pers = finite[:, 1] - finite[:, 0]
            max_p = np.max(pers)
            sig = int(np.sum(pers > thr * max_p)) + n_inf if max_p > 0 else n_inf
            betti[dim_h] = sig
        betti_at_thresholds[thr] = betti

    # H1 stats
    h1_stats = {}
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        finite = diagrams[1][np.isfinite(diagrams[1][:, 1])]
        if len(finite) > 0:
            pers = np.sort(finite[:, 1] - finite[:, 0])[::-1]
            gap = float(pers[0] / pers[1]) if len(pers) >= 2 and pers[1] > 1e-12 else float("inf")
            h1_stats = {
                "n_features": len(pers),
                "top5": pers[:5].tolist(),
                "gap_ratio": gap,
                "max_persistence": float(pers[0]),
            }

    return {
        "residual_var": residual_var,
        "total_residual_var": float(np.sum(residual_var)),
        "betti_at_thresholds": betti_at_thresholds,
        "h1_stats": h1_stats,
        "diagrams": diagrams,
        "n_points": len(points),
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_multidim_results(all_data: dict, save_path: str):
    """
    Summary plot: Betti numbers across PCA dimensions for all models.
    """
    model_keys = list(all_data.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 4, figsize=(22, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        ln_type = MODELS[model_key]["ln_type"]
        # Use narrative_long
        text_key = "narrative_long"
        mdata = all_data[model_key][text_key]["multidim"]
        rdata = all_data[model_key][text_key]["residual"]

        pca_dims = sorted(mdata.keys())

        # ── (a) β₁ vs PCA dimension ──
        ax = axes[row, 0]
        for thr in [0.05, 0.1, 0.2]:
            b1_vals = []
            for k in pca_dims:
                b1 = mdata[k]["betti_at_thresholds"].get(thr, {}).get(1, 0)
                b1_vals.append(b1)
            ax.plot(pca_dims, b1_vals, "o-", label=f"thr={thr}", markersize=6)
        ax.set_xlabel("PCA dimensions")
        ax.set_ylabel("β₁ (number of loops)")
        ax.set_title(f"{model_key} ({ln_type}) — β₁ vs PCA dim")
        ax.legend(fontsize=8)
        ax.set_xticks(pca_dims)

        # ── (b) Cumulative variance explained ──
        ax = axes[row, 1]
        for k in pca_dims:
            cumvar = mdata[k]["cumvar"]
            ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "o-",
                    label=f"PCA-{k}", markersize=3)
        ax.axhline(96, color="red", ls="--", alpha=0.5, label="96%")
        ax.axhline(99, color="darkred", ls="--", alpha=0.5, label="99%")
        ax.set_xlabel("# components")
        ax.set_ylabel("Cumulative var explained (%)")
        ax.set_title(f"{model_key} — Variance by PCA dim")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 21)

        # ── (c) H1 gap ratio vs PCA dimension ──
        ax = axes[row, 2]
        gaps = []
        max_pers = []
        for k in pca_dims:
            h1 = mdata[k].get("h1_stats", {})
            gaps.append(h1.get("gap_ratio", 0))
            max_pers.append(h1.get("max_persistence", 0))
        ax.bar([f"PCA-{k}" for k in pca_dims], gaps,
               color="darkorange", edgecolor="black")
        ax.axhline(3, color="green", ls="--", alpha=0.5, label="Strong signal (3.0)")
        ax.set_ylabel("H1 Gap Ratio")
        ax.set_title(f"{model_key} — H1 Gap Ratio by PCA dim")
        ax.legend(fontsize=8)

        # Add max persistence as text
        for i, (g, mp) in enumerate(zip(gaps, max_pers)):
            ax.text(i, g + 0.1, f"max={mp:.3f}", ha="center", fontsize=7)

        # ── (d) Residual topology (PC2+ without PC1) ──
        ax = axes[row, 3]
        # Compare full PCA-10 vs residual (PC2-onward)
        full_b = mdata.get(10, mdata.get(5, {}))
        res_b = rdata

        labels = ["Full PCA-10", "Residual (PC2+)"]
        for thr in [0.1]:
            full_betti = full_b.get("betti_at_thresholds", {}).get(thr, {})
            res_betti = res_b.get("betti_at_thresholds", {}).get(thr, {})

            dims = [0, 1, 2]
            x = np.arange(len(dims))
            w = 0.35
            full_vals = [full_betti.get(d, 0) for d in dims]
            res_vals = [res_betti.get(d, 0) for d in dims]

            ax.bar(x - w / 2, full_vals, w, label="Full (with PC1)",
                   color="steelblue", edgecolor="black")
            ax.bar(x + w / 2, res_vals, w, label="Residual (PC2+)",
                   color="darkorange", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels([f"H{d}" for d in dims])
        ax.set_ylabel("Betti number (thr=0.1)")
        ax.set_title(f"{model_key} — Full vs Residual Topology")
        ax.legend(fontsize=8)

        # Annotate residual H1 stats
        rh1 = rdata.get("h1_stats", {})
        if rh1:
            txt = (f"Residual H1:\n"
                   f"  features={rh1.get('n_features', 0)}\n"
                   f"  gap={rh1.get('gap_ratio', 0):.2f}\n"
                   f"  max_pers={rh1.get('max_persistence', 0):.4f}\n"
                   f"  residual var={rdata['total_residual_var']:.1%}")
            ax.text(0.98, 0.95, txt, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.suptitle(
        "Experiment 2c: High-Dimensional Topology with Longer Text\n"
        "Does topology emerge in higher PCA dimensions? Is there structure in the residual (PC2+)?",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Figure saved to {save_path}")
    plt.close()


def plot_cross_text(all_data: dict, save_path: str):
    """
    Compare topology across text types at PCA-10.
    """
    model_keys = [m for m in all_data if MODELS[m]["ln_type"] == "Pre-LN"]
    text_keys = list(LONG_TEXTS.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        # (a) β₁ across texts at PCA-10
        ax = axes[row, 0]
        for thr in [0.05, 0.1, 0.2]:
            b1_vals = []
            for tk in text_keys:
                mdata = all_data[model_key][tk]["multidim"]
                k_use = 10 if 10 in mdata else max(mdata.keys())
                b1 = mdata[k_use]["betti_at_thresholds"].get(thr, {}).get(1, 0)
                b1_vals.append(b1)
            ax.bar(
                np.arange(len(text_keys)) + [0, 0.25, 0.5][
                    [0.05, 0.1, 0.2].index(thr)
                ] * 0.25,
                b1_vals, 0.25, label=f"thr={thr}",
            )
        ax.set_xticks(np.arange(len(text_keys)) + 0.25)
        ax.set_xticklabels([t[:12] for t in text_keys], fontsize=8)
        ax.set_ylabel("β₁")
        ax.set_title(f"{model_key} — β₁ by Text (PCA-10)")
        ax.legend(fontsize=7)

        # (b) H1 gap ratio across texts
        ax = axes[row, 1]
        gaps = []
        for tk in text_keys:
            mdata = all_data[model_key][tk]["multidim"]
            k_use = 10 if 10 in mdata else max(mdata.keys())
            h1 = mdata[k_use].get("h1_stats", {})
            gaps.append(h1.get("gap_ratio", 0))
        ax.bar(text_keys, gaps, color="darkorange", edgecolor="black")
        ax.axhline(3, color="green", ls="--", alpha=0.5)
        ax.set_ylabel("H1 Gap Ratio")
        ax.set_title(f"{model_key} — H1 Gap by Text (PCA-10)")
        ax.set_xticklabels([t[:12] for t in text_keys], fontsize=8)

        # (c) Residual topology across texts
        ax = axes[row, 2]
        for ti, tk in enumerate(text_keys):
            rdata = all_data[model_key][tk]["residual"]
            betti = rdata["betti_at_thresholds"].get(0.1, {})
            x = np.arange(3) + ti * 0.25
            vals = [betti.get(d, 0) for d in range(3)]
            ax.bar(x, vals, 0.22, label=tk[:12])
        ax.set_xticks(np.arange(3) + 0.25)
        ax.set_xticklabels(["H0", "H1", "H2"])
        ax.set_ylabel("Betti (thr=0.1)")
        ax.set_title(f"{model_key} — Residual Betti by Text")
        ax.legend(fontsize=7)

    plt.suptitle(
        "Experiment 2c: Cross-Text Topology Comparison (PCA-10 + Residual)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT 2c: High-Dimensional Topology with Longer Text")
    print("=" * 70)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for model_key in MODELS:
        print(f"\n{'━' * 60}")
        print(f"Model: {model_key} ({MODELS[model_key]['ln_type']})")
        print(f"{'━' * 60}")

        all_data[model_key] = {}

        for text_key, text in LONG_TEXTS.items():
            print(f"\n  Text: {text_key} ({len(text.split())} words approx)")
            trajectories = extract_hidden_states(model_key, text)
            n_layers = MODELS[model_key]["n_layers"]

            mid = n_layers // 2
            mid_idx = min(trajectories.keys(), key=lambda x: abs(x - mid))
            traj = trajectories[mid_idx]
            print(f"  Layer {mid_idx}, shape={traj.shape}")

            # Multi-dim topology
            print(f"  Running multi-dimensional topology...")
            multidim = run_multidim_topology(traj, pca_dims=[2, 5, 10, 20])

            for k in sorted(multidim.keys()):
                md = multidim[k]
                b_01 = md["betti_at_thresholds"].get(0.1, {})
                h1 = md.get("h1_stats", {})
                cumvar = md["cumvar"]
                k_act = md["k_actual"]
                print(
                    f"    PCA-{k} (actual={k_act}, {md['n_points']} pts, "
                    f"cumvar={cumvar[min(k_act-1, len(cumvar)-1)]:.1%}): "
                    f"β=({b_01.get(0,0)}, {b_01.get(1,0)}, {b_01.get(2,0)}) "
                    f"H1_gap={h1.get('gap_ratio', 0):.2f} "
                    f"H1_max={h1.get('max_persistence', 0):.4f}"
                )

            # Residual topology
            print(f"  Running residual topology (PC2+)...")
            residual = run_residual_topology(traj)
            rb = residual["betti_at_thresholds"].get(0.1, {})
            rh1 = residual.get("h1_stats", {})
            print(
                f"    Residual (var={residual['total_residual_var']:.1%}): "
                f"β=({rb.get(0,0)}, {rb.get(1,0)}, {rb.get(2,0)}) "
                f"H1_gap={rh1.get('gap_ratio', 0):.2f} "
                f"H1_max={rh1.get('max_persistence', 0):.4f}"
            )

            all_data[model_key][text_key] = {
                "multidim": multidim,
                "residual": residual,
            }

    # ── Summary ──
    print("\n" + "=" * 70)
    print("GRAND SUMMARY: β₁ by PCA dimension (threshold=0.1, narrative_long)")
    print("=" * 70)
    print(f"{'Model':>12} {'LN':>8} "
          f"{'PCA-2':>7} {'PCA-5':>7} {'PCA-10':>7} {'PCA-20':>7} "
          f"{'Resid':>7}")
    print("-" * 60)
    for model_key in all_data:
        mdata = all_data[model_key]["narrative_long"]["multidim"]
        rdata = all_data[model_key]["narrative_long"]["residual"]
        vals = []
        for k in [2, 5, 10, 20]:
            b1 = mdata.get(k, {}).get("betti_at_thresholds", {}).get(0.1, {}).get(1, 0)
            vals.append(f"{b1:>7}")
        rb1 = rdata.get("betti_at_thresholds", {}).get(0.1, {}).get(1, 0)
        vals.append(f"{rb1:>7}")
        print(f"{model_key:>12} {MODELS[model_key]['ln_type']:>8} " + " ".join(vals))

    print("\n" + "=" * 70)
    print("H1 GAP RATIO by PCA dimension (narrative_long)")
    print("=" * 70)
    print(f"{'Model':>12} {'LN':>8} "
          f"{'PCA-2':>7} {'PCA-5':>7} {'PCA-10':>7} {'PCA-20':>7} "
          f"{'Resid':>7}")
    print("-" * 60)
    for model_key in all_data:
        mdata = all_data[model_key]["narrative_long"]["multidim"]
        rdata = all_data[model_key]["narrative_long"]["residual"]
        vals = []
        for k in [2, 5, 10, 20]:
            gap = mdata.get(k, {}).get("h1_stats", {}).get("gap_ratio", 0)
            vals.append(f"{gap:>7.2f}")
        rgap = rdata.get("h1_stats", {}).get("gap_ratio", 0)
        vals.append(f"{rgap:>7.2f}")
        print(f"{model_key:>12} {MODELS[model_key]['ln_type']:>8} " + " ".join(vals))

    # ── Plots ──
    plot_multidim_results(all_data, str(out_dir / "exp2c_multidim.png"))
    plot_cross_text(all_data, str(out_dir / "exp2c_cross_text.png"))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "Key questions answered:\n\n"
        "1. Does β₁ INCREASE with PCA dimension?\n"
        "   Yes → real topological features exist in higher dims\n"
        "   No  → the 2D projection captured everything\n\n"
        "2. Does the RESIDUAL (PC2+) have topology?\n"
        "   β₁ > 0 with high gap ratio → the 'remaining 4%' has structure\n"
        "   β₁ = 0 → PC1 is all there is\n\n"
        "3. Is the H1 gap ratio LARGE in higher dims?\n"
        "   Gap > 3 → genuine topological feature\n"
        "   Gap < 1.5 → noise from high-dimensional sparsity\n\n"
        "4. Are results CONSISTENT across text types?\n"
        "   Same β₁ across texts → architectural property\n"
        "   Different β₁ → input-dependent topology"
    )


if __name__ == "__main__":
    main()
