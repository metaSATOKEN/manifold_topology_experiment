#!/usr/bin/env python3
"""
Experiment 5b: Norm-Normalized PCA

Verifies that PC1 dominance is driven by hidden-state norm variation.
Normalizes each hidden state to unit norm before PCA, then re-runs
the probing analysis from Exp 5.

Key prediction: norm normalization should collapse PC1 variance from
~95% to ~7%, causing position to become PC1 and POS/difficulty to
become PC2.

Paper reference: Section 4.6
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict
from pathlib import Path

TEXT = (
    "The old lighthouse keeper climbed the spiral staircase each evening, "
    "carrying a lantern that cast long shadows across the stone walls. "
    "He had performed this ritual for forty years, ever since the automated "
    "systems had failed during the great storm. The sea below crashed "
    "against the rocks with a rhythm that matched his breathing, and he "
    "found comfort in the predictability of waves and wind. Tonight, "
    "however, something was different. A strange light flickered on the "
    "horizon, pulsing with an irregular beat that made him uneasy. "
    "He set the lantern down on the iron railing and squinted into the "
    "darkness. The light grew brighter, then dimmed, then brightened again."
)

FUNC_WORDS = {
    "the","a","an","of","in","to","and","is","was","are","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","for","with","on","at","from","by",
    "as","or","but","not","no","if","that","this","it","its","he","she",
    "they","we","you","his","her","their","our","my","your","which","what",
    "who","how","when","where","than","then","so",
}
PUNCT = set(".,!?;:'\"--()")


def classify_pos(tok_str):
    t = tok_str.strip().lower()
    if t in PUNCT or all(c in PUNCT for c in t):
        return "PUNCT"
    if t in FUNC_WORDS:
        return "FUNC"
    return "CONTENT"


def analyze_pca(h, positions, surprisals, tok_lens, pos_cats, label, n_pcs=5):
    """Run PCA and compute correlations with linguistic features."""
    T, d = h.shape
    norms = np.linalg.norm(h, axis=1)
    K = min(10, d, T - 1)
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h)
    ve = pca.explained_variance_ratio_

    print(f"\n  --- {label} ---")
    print(f"  PCA variance: PC1={ve[0]:.1%}, PC2={ve[1]:.2%}, PC3={ve[2]:.2%}")
    print(f"  {'PC':>4} {'Var':>7} {'norm':>8} {'pos':>8} {'surp':>8} "
          f"{'tok_len':>8} {'POS F':>8}")
    print(f"  {'-'*55}")

    for pc in range(min(n_pcs, K)):
        pv = proj[:, pc]
        r_norm, _ = pearsonr(pv, norms)
        r_pos, _ = pearsonr(pv, positions)
        r_surp, _ = pearsonr(pv, surprisals)
        r_tl, _ = pearsonr(pv, tok_lens)

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0

        print(f"  PC{pc+1:>2} {ve[pc]:>6.2%} {r_norm:>+8.3f} {r_pos:>+8.3f} "
              f"{r_surp:>+8.3f} {r_tl:>+8.3f} {f_stat:>8.1f}")

    return ve


def main():
    print("=" * 60)
    print("EXPERIMENT 5b: Norm-Normalized PCA (GPT-2, Layer 6)")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    T = ids.shape[1]

    with torch.inference_mode():
        out = model(ids)

    h = out.hidden_states[7][0].float().cpu().numpy()  # Layer 6
    logits = out.logits[0].float().cpu()
    del out, model; gc.collect()

    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(T)]
    positions = np.arange(T, dtype=float)
    tok_lens = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)
    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, ids[0, t + 1].item()].item()
    surprisals[0] = surprisals[1]
    pos_cats = [classify_pos(tokens[t]) for t in range(T)]

    print(f"\n  Layer 6, T={T}, d={h.shape[1]}")

    # Raw PCA
    ve_raw = analyze_pca(h, positions, surprisals, tok_lens, pos_cats, "RAW")

    # Norm-normalized PCA
    h_normed = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    ve_norm = analyze_pca(h_normed, positions, surprisals, tok_lens, pos_cats,
                          "NORM-NORMALIZED")

    print(f"\n  COMPARISON:")
    print(f"  {'':>20} {'Raw':>10} {'Normalized':>12}")
    print(f"  {'-'*45}")
    print(f"  {'PC1 variance':>20} {ve_raw[0]:>9.1%} {ve_norm[0]:>11.1%}")
    print(f"  {'Δ PC1 variance':>20} {'':>10} {ve_norm[0]-ve_raw[0]:>+10.1%}")


if __name__ == "__main__":
    main()
