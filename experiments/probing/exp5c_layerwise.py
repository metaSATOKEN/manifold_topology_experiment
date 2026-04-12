#!/usr/bin/env python3
"""
Experiment 5c: Layer-Wise PCA Analysis

Extends the probing analysis from Exp 5 (mid-layer only) to all
transformer layers + embedding. Tracks emergence and dissolution
of the arc structure across depth.

Key findings:
  Phase 1 (embed/L0): Position dominates PC1, norm correlation weak
  Phase 2 (L1-L10): Norm locks onto PC1 at L1, position displaced to PC2
  Phase 3 (L11): Arc partially dissolves, position encoding collapses

Also runs norm-normalized PCA at each layer to verify that the
norm→position→POS hierarchy holds throughout.

Paper reference: Section 4.7
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def analyze_layer(h, positions, surprisals, tok_lens, pos_cats, normalize=False):
    """Analyze one layer's hidden states."""
    T, d = h.shape
    if normalize:
        h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)

    norms = np.linalg.norm(h, axis=1)
    K = min(10, d, T - 1)
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h)
    ve = pca.explained_variance_ratio_

    results = {"ve": ve}
    for pc in range(min(5, K)):
        pv = proj[:, pc]
        r_norm, _ = pearsonr(pv, norms)
        r_pos, _ = pearsonr(pv, positions)
        r_surp, _ = pearsonr(pv, surprisals)

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0

        results[pc] = {
            "norm": r_norm, "pos": r_pos, "surp": r_surp, "F": f_stat
        }

    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 5c: Layer-Wise PCA Analysis (GPT-2)")
    print("=" * 60)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    T = ids.shape[1]

    with torch.inference_mode():
        out = model(ids)

    all_hs = [hs[0].float().cpu().numpy() for hs in out.hidden_states]
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

    layer_names = ["embed"] + [f"L{i}" for i in range(12)]

    # ── Raw PCA ──
    print(f"\n  {'Layer':>6} {'PC1var':>7} {'PC1-n':>7} {'|pos|':>7} {'PC':>3} "
          f"{'|surp|':>7} {'PC':>3} {'F':>7} {'PC':>3}")
    print(f"  {'-'*58}")

    raw_results = []
    for li in range(13):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats)
        raw_results.append(r)
        best_pos = max(range(5), key=lambda pc: abs(r[pc]["pos"]))
        best_surp = max(range(5), key=lambda pc: abs(r[pc]["surp"]))
        best_f = max(range(5), key=lambda pc: r[pc]["F"])
        print(f"  {layer_names[li]:>6} {r['ve'][0]:>6.1%} {r[0]['norm']:>+7.3f} "
              f"{abs(r[best_pos]['pos']):>7.3f} PC{best_pos+1} "
              f"{abs(r[best_surp]['surp']):>7.3f} PC{best_surp+1} "
              f"{r[best_f]['F']:>7.1f} PC{best_f+1}")

    # ── Norm-normalized ──
    print(f"\n  --- NORM-NORMALIZED ---")
    print(f"  {'Layer':>6} {'PC1var':>7} {'|pos|':>7} {'PC':>3} {'F':>7} {'PC':>3}")
    print(f"  {'-'*38}")

    norm_results = []
    for li in range(13):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats,
                          normalize=True)
        norm_results.append(r)
        best_pos = max(range(5), key=lambda pc: abs(r[pc]["pos"]))
        best_f = max(range(5), key=lambda pc: r[pc]["F"])
        print(f"  {layer_names[li]:>6} {r['ve'][0]:>6.1%} "
              f"{abs(r[best_pos]['pos']):>7.3f} PC{best_pos+1} "
              f"{r[best_f]['F']:>7.1f} PC{best_f+1}")

    print("\n  Figure saved by exp5c or inline analysis script.")


if __name__ == "__main__":
    main()
