#!/usr/bin/env python3
"""
Experiment 5: Probing the "Thickness" — What Do PC2-PC10 Encode?

The manifold topology experiments (Exp 2/2b/2c) established that
Pre-LN hidden states form a 1D arc (PC1 = 81-96% variance) with
"thickness" in PC2-PC10 (4-8% variance, no topological structure).

This experiment probes what linguistic features are encoded in
each thickness direction by correlating PC components with:

  1. Token position (sequential, relative)
  2. Surprisal (next-token prediction error)
  3. Part-of-speech categories (noun, verb, adj, function word, etc.)
  4. Token frequency (log unigram frequency proxy)
  5. Subword characteristics (is continuation token, token length)
  6. Local context change (cosim between consecutive hidden states)

Method:
  For each PC direction (PC1 through PC10), compute:
    - Pearson correlation with continuous features
    - ANOVA F-statistic for categorical features (POS)
    - Mutual information for nonlinear relationships

If specific PC directions carry specific linguistic features,
this informs Recync's orthogonal steering: push in the direction
that encodes the desired linguistic property.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, f_oneway
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import MODELS, SAMPLE_TEXTS


# ── Long texts for more tokens ─────────────────────────────────────────────

PROBE_TEXTS = {
    "narrative": (
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
        "explain. He watched for what felt like hours, though the clock "
        "said only minutes had passed. The sphere drifted closer, and he "
        "could feel a warmth on his face, gentle as morning sunlight. Then "
        "it vanished. The waves resumed their ancient rhythm. The keeper "
        "stood alone in the darkness, wondering if he had imagined it all."
    ),
    "technical": (
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
        "approaches. Recent work suggests these feed-forward layers act as "
        "key-value memories, storing factual associations learned during "
        "pretraining. Understanding the geometry of these representations "
        "is crucial for interpretability and better architectures."
    ),
}


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_all_features(
    text: str,
    model_name: str = "gpt2",
    target_layer: int | None = None,
    n_pca: int = 10,
) -> dict:
    """
    Extract hidden states + linguistic features for each token.

    Returns dict with:
      - projected: (T, n_pca) PCA-projected hidden states
      - var_explained: PCA variance ratios
      - features: dict of feature_name -> array of length T
      - tokens: list of token strings
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    input_ids = tokenizer.encode(text, return_tensors="pt")
    T = input_ids.shape[1]
    n_layers = model.config.n_layer

    if target_layer is None:
        target_layer = n_layers // 2

    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states
        logits = outputs.logits  # (1, T, vocab_size)

    # Hidden states at target layer
    h = hidden_states[target_layer + 1][0].cpu().numpy()  # (T, d)

    # PCA
    k = min(n_pca, h.shape[1], T - 1)
    pca = PCA(n_components=k)
    projected = pca.fit_transform(h)
    var_explained = pca.explained_variance_ratio_

    # ── Token strings ──
    tokens = [tokenizer.decode([input_ids[0, t].item()]) for t in range(T)]

    # ── Feature: position ──
    positions = np.arange(T, dtype=float)
    relative_positions = positions / T

    # ── Feature: surprisal (next-token prediction error) ──
    log_probs = torch.log_softmax(logits[0], dim=-1)  # (T, vocab)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        next_token = input_ids[0, t + 1].item()
        surprisals[t + 1] = -log_probs[t, next_token].item()
    surprisals[0] = surprisals[1]  # pad first position

    # ── Feature: token length (characters) ──
    token_lengths = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)

    # ── Feature: is continuation token (starts with non-space in GPT-2) ──
    is_continuation = np.array(
        [1.0 if not tokens[t].startswith(" ") and t > 0 else 0.0 for t in range(T)]
    )

    # ── Feature: cosim with previous hidden state ──
    cosims = np.zeros(T)
    for t in range(1, T):
        h_prev = h[t - 1]
        h_curr = h[t]
        cos = np.dot(h_prev, h_curr) / (np.linalg.norm(h_prev) * np.linalg.norm(h_curr) + 1e-12)
        cosims[t] = cos
    cosims[0] = cosims[1]

    # ── Feature: hidden state norm ──
    h_norms = np.linalg.norm(h, axis=1)

    # ── Feature: simple POS proxy via heuristics ──
    # (避けるためにspacy不要。GPT-2トークンから簡易分類)
    pos_categories = []
    function_words = {
        "the", "a", "an", "of", "in", "to", "and", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "for", "with", "on", "at", "from", "by", "as", "or", "but",
        "not", "no", "if", "that", "this", "it", "its", "he", "she", "they",
        "we", "you", "his", "her", "their", "our", "my", "your", "which",
        "what", "who", "how", "when", "where", "than", "then", "so",
    }
    punctuation = {".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"}

    for t in range(T):
        tok = tokens[t].strip().lower()
        if tok in punctuation or all(c in punctuation for c in tok):
            pos_categories.append("PUNCT")
        elif tok in function_words:
            pos_categories.append("FUNC")
        elif tok.endswith("ing"):
            pos_categories.append("VERB_ING")
        elif tok.endswith("ed"):
            pos_categories.append("VERB_ED")
        elif tok.endswith("ly"):
            pos_categories.append("ADV")
        elif tok.endswith("tion") or tok.endswith("ment") or tok.endswith("ness"):
            pos_categories.append("NOUN_SUFFIX")
        else:
            pos_categories.append("CONTENT")

    # ── Feature: windowed surprisal variance (Recync-style) ──
    window = 5
    surprisal_var = np.zeros(T)
    for t in range(window, T):
        surprisal_var[t] = np.var(surprisals[t - window : t])

    # ── Feature: phase velocity (atan2 angular change rate) ──
    phases = np.arctan2(projected[:, 1], projected[:, 0])
    phase_velocity = np.zeros(T)
    for t in range(1, T):
        dphase = phases[t] - phases[t - 1]
        # Wrap to [-pi, pi]
        dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
        phase_velocity[t] = abs(dphase)
    phase_velocity[0] = phase_velocity[1]

    features = {
        "position": positions,
        "relative_position": relative_positions,
        "surprisal": surprisals,
        "token_length": token_lengths,
        "is_continuation": is_continuation,
        "cosim_consecutive": cosims,
        "hidden_norm": h_norms,
        "surprisal_variance": surprisal_var,
        "phase_velocity": phase_velocity,
    }

    del outputs, hidden_states, logits, model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "projected": projected,
        "var_explained": var_explained,
        "features": features,
        "tokens": tokens,
        "pos_categories": pos_categories,
        "target_layer": target_layer,
        "T": T,
    }


# ── Correlation analysis ────────────────────────────────────────────────────

def compute_correlations(data: dict) -> dict:
    """
    Compute correlations between each PC direction and each feature.
    """
    projected = data["projected"]
    features = data["features"]
    pos = data["pos_categories"]
    T, K = projected.shape

    results = {}

    # ── Continuous features: Pearson & Spearman ──
    for feat_name, feat_vals in features.items():
        results[feat_name] = {}
        for pc in range(K):
            pc_vals = projected[:, pc]
            try:
                r_pearson, p_pearson = pearsonr(pc_vals, feat_vals)
                r_spearman, p_spearman = spearmanr(pc_vals, feat_vals)
            except Exception:
                r_pearson, p_pearson = 0.0, 1.0
                r_spearman, p_spearman = 0.0, 1.0

            results[feat_name][pc] = {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            }

    # ── Categorical feature (POS): ANOVA F-test ──
    pos_groups = defaultdict(list)
    for t, p in enumerate(pos):
        pos_groups[p].append(t)

    # Only use categories with >= 5 members
    valid_cats = {cat: idx for cat, idx in pos_groups.items() if len(idx) >= 5}

    results["POS_category"] = {}
    for pc in range(K):
        pc_vals = projected[:, pc]
        groups = [pc_vals[idx] for idx in valid_cats.values()]
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
            except Exception:
                f_stat, p_val = 0.0, 1.0
        else:
            f_stat, p_val = 0.0, 1.0

        # Per-category means
        cat_means = {cat: float(np.mean(pc_vals[idx])) for cat, idx in valid_cats.items()}

        results["POS_category"][pc] = {
            "f_stat": float(f_stat),
            "p_val": float(p_val),
            "cat_means": cat_means,
        }

    return results


def find_top_associations(correlations: dict, n_pca: int) -> list:
    """
    Find the strongest PC-feature associations.
    Returns sorted list of (|r|, pc, feature, r, p).
    """
    entries = []
    for feat_name, pc_data in correlations.items():
        if feat_name == "POS_category":
            continue
        for pc, stats in pc_data.items():
            r = stats["pearson_r"]
            p = stats["pearson_p"]
            entries.append((abs(r), pc, feat_name, r, p))

    entries.sort(reverse=True)
    return entries


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(correlations: dict, var_explained: np.ndarray,
                              text_key: str, save_path: str):
    """
    Heatmap: PC direction (x) vs feature (y), colored by Pearson r.
    """
    feat_names = [f for f in correlations if f != "POS_category"]
    n_pca = len(next(iter(correlations.values())).keys()) if correlations else 0

    # Build matrix
    r_matrix = np.zeros((len(feat_names), n_pca))
    p_matrix = np.zeros((len(feat_names), n_pca))
    for i, feat in enumerate(feat_names):
        for pc in range(n_pca):
            r_matrix[i, pc] = correlations[feat][pc]["pearson_r"]
            p_matrix[i, pc] = correlations[feat][pc]["pearson_p"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [3, 1]})

    # (a) Heatmap
    ax = axes[0]
    vmax = np.max(np.abs(r_matrix))
    im = ax.imshow(r_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    aspect="auto")
    ax.set_xticks(range(n_pca))
    ax.set_xticklabels([f"PC{i+1}\n({var_explained[i]:.1%})" for i in range(n_pca)],
                        fontsize=8)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.set_title(f"Pearson r: PC Direction vs Linguistic Feature — {text_key}")
    plt.colorbar(im, ax=ax, label="Pearson r")

    # Mark significant correlations (p < 0.01)
    for i in range(len(feat_names)):
        for j in range(n_pca):
            if p_matrix[i, j] < 0.01:
                ax.text(j, i, f"{r_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if abs(r_matrix[i, j]) > vmax * 0.6 else "black")
            elif p_matrix[i, j] < 0.05:
                ax.text(j, i, f"{r_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="gray")

    # (b) POS ANOVA F-statistics
    ax = axes[1]
    if "POS_category" in correlations:
        f_stats = [correlations["POS_category"][pc]["f_stat"] for pc in range(n_pca)]
        p_vals = [correlations["POS_category"][pc]["p_val"] for pc in range(n_pca)]
        colors = ["darkorange" if p < 0.01 else "steelblue" if p < 0.05 else "lightgray"
                  for p in p_vals]
        ax.barh(range(n_pca), f_stats, color=colors, edgecolor="black")
        ax.set_yticks(range(n_pca))
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_pca)], fontsize=9)
        ax.set_xlabel("ANOVA F-statistic")
        ax.set_title("POS Category\nDiscrimination")
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="darkorange", label="p < 0.01"),
            Patch(facecolor="steelblue", label="p < 0.05"),
            Patch(facecolor="lightgray", label="n.s."),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def plot_pc_vs_feature_scatter(data: dict, correlations: dict,
                                top_n: int = 6, save_path: str = ""):
    """
    Scatter plots for the top-N strongest PC-feature associations.
    """
    top = find_top_associations(correlations, data["projected"].shape[1])[:top_n]

    n_cols = 3
    n_rows = (len(top) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, (abs_r, pc, feat_name, r, p) in enumerate(top):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        pc_vals = data["projected"][:, pc]
        feat_vals = data["features"][feat_name]

        ax.scatter(feat_vals, pc_vals, s=15, alpha=0.6, c="steelblue", edgecolors="none")

        # Trend line
        if abs_r > 0.1:
            z = np.polyfit(feat_vals, pc_vals, 1)
            x_line = np.linspace(np.min(feat_vals), np.max(feat_vals), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r-", lw=2, alpha=0.7)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel(f"PC{pc+1}", fontsize=10)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"PC{pc+1} vs {feat_name}\nr={r:.3f}, p={p:.1e} {sig}", fontsize=10)

    # Hide unused
    for idx in range(len(top), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("Top PC-Feature Associations (GPT-2, mid-layer)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def plot_pos_breakdown(data: dict, correlations: dict, save_path: str):
    """
    Show how POS categories distribute across the top PCs.
    """
    pos = data["pos_categories"]
    projected = data["projected"]
    K = min(6, projected.shape[1])

    # Get categories with enough samples
    from collections import Counter
    counts = Counter(pos)
    valid_cats = [cat for cat, n in counts.items() if n >= 5]
    valid_cats.sort()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx in range(K):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        pc_vals = projected[:, idx]
        cat_data = defaultdict(list)
        for t, cat in enumerate(pos):
            if cat in valid_cats:
                cat_data[cat].append(pc_vals[t])

        labels = sorted(cat_data.keys())
        bp_data = [cat_data[c] for c in labels]

        bp = ax.boxplot(bp_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(f"PC{idx+1} value")
        f_stat = correlations["POS_category"][idx]["f_stat"]
        p_val = correlations["POS_category"][idx]["p_val"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        ax.set_title(
            f"PC{idx+1} ({data['var_explained'][idx]:.2%} var) — "
            f"F={f_stat:.1f}, p={p_val:.1e} {sig}"
        )
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("POS Category Distribution Across PC Directions (GPT-2)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT 5: Probing the Thickness — What Do PC2-PC10 Encode?")
    print("=" * 70)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_correlations = {}

    for text_key, text in PROBE_TEXTS.items():
        print(f"\n{'━' * 60}")
        print(f"Text: {text_key}")
        print(f"{'━' * 60}")

        data = extract_all_features(text, model_name="gpt2", n_pca=10)
        print(f"  Layer {data['target_layer']}, T={data['T']}")
        print(f"  PCA variance: {data['var_explained']}")

        correlations = compute_correlations(data)
        all_correlations[text_key] = correlations

        # ── Print top associations ──
        top = find_top_associations(correlations, 10)
        print(f"\n  TOP 15 PC-FEATURE ASSOCIATIONS:")
        print(f"  {'|r|':>6} {'PC':>4} {'Feature':>22} {'r':>8} {'p':>12}")
        print(f"  {'-'*56}")
        for abs_r, pc, feat, r, p in top[:15]:
            sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            print(f"  {abs_r:>6.3f} PC{pc+1:>2} {feat:>22} {r:>+8.3f} {p:>12.2e} {sig}")

        # ── POS ANOVA results ──
        print(f"\n  POS CATEGORY ANOVA:")
        print(f"  {'PC':>4} {'F-stat':>8} {'p':>12} {'Significant':>12}")
        print(f"  {'-'*40}")
        for pc in range(min(10, data["projected"].shape[1])):
            f_stat = correlations["POS_category"][pc]["f_stat"]
            p_val = correlations["POS_category"][pc]["p_val"]
            sig = "YES" if p_val < 0.05 else "no"
            print(f"  PC{pc+1:>2} {f_stat:>8.2f} {p_val:>12.2e} {sig:>12}")

        # ── Plots ──
        plot_correlation_heatmap(
            correlations, data["var_explained"],
            text_key,
            save_path=str(out_dir / f"exp5_heatmap_{text_key}.png"),
        )

        if text_key == "narrative":
            plot_pc_vs_feature_scatter(
                data, correlations,
                save_path=str(out_dir / "exp5_scatter_top.png"),
            )
            plot_pos_breakdown(
                data, correlations,
                save_path=str(out_dir / "exp5_pos_breakdown.png"),
            )

    # ── Cross-text consistency ──
    print("\n" + "=" * 70)
    print("CROSS-TEXT CONSISTENCY")
    print("=" * 70)
    print("Do the same PCs encode the same features across texts?\n")

    text_keys = list(all_correlations.keys())
    feat_names = [f for f in all_correlations[text_keys[0]] if f != "POS_category"]

    print(f"  {'Feature':>22} {'PC':>4} ", end="")
    for tk in text_keys:
        print(f"  {tk[:10]:>10}", end="")
    print("  Consistent?")
    print(f"  {'-'*70}")

    consistent_pairs = []
    for feat in feat_names:
        for pc in range(10):
            rs = []
            for tk in text_keys:
                if pc in all_correlations[tk].get(feat, {}):
                    rs.append(all_correlations[tk][feat][pc]["pearson_r"])
            if len(rs) >= 2 and all(abs(r) > 0.3 for r in rs):
                same_sign = all(r > 0 for r in rs) or all(r < 0 for r in rs)
                if same_sign:
                    print(f"  {feat:>22} PC{pc+1:>2} ", end="")
                    for r in rs:
                        print(f"  {r:>+10.3f}", end="")
                    print("  YES ✓")
                    consistent_pairs.append((feat, pc, rs))

    if not consistent_pairs:
        print("  No consistently strong associations found across texts.")

    # ── Summary for Recync ──
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR RECYNC STEERING")
    print("=" * 70)
    print(
        "If specific PC directions consistently encode specific features:\n"
        "  → Recync can steer orthogonally in MEANINGFUL directions\n"
        "  → Instead of random perpendicular push, push toward\n"
        "    the PC that encodes the desired property\n\n"
        "PC1 (the arc): position/sequential processing\n"
        "PC2+: look at the results above for feature-specific encodings\n\n"
        "Key question for Recync:\n"
        "  Does cosim crisis correspond to specific PC directions?\n"
        "  If so, Recync's detection can be decomposed into\n"
        "  PC-specific signals for targeted intervention."
    )


if __name__ == "__main__":
    main()
