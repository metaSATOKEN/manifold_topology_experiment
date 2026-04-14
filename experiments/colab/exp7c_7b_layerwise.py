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
7B Layer-Wise Analysis — Does the 3-phase structure hold at 7B?
Google Colab version. Requires GPU with 16GB+ VRAM.

Setup cell (run first):
    !pip install torch transformers scikit-learn scipy accelerate matplotlib

Then paste this code into a cell and run.

Tests: Pythia-6.9B (ungated, no HF token needed)
"""

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

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
    "darkness. The light grew brighter, then dimmed, then brightened again. "
    "It was not the steady beam of a ship's lantern, nor the rhythmic "
    "flash of another lighthouse. It moved in patterns he had never seen "
    "before, tracing arcs across the water like a living thing. "
    "The keeper reached for his telescope, hands trembling slightly. "
    "Through the lens, the light resolved into something he could not "
    "explain. He watched for what felt like hours, though the clock "
    "said only minutes had passed."
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
    if t.endswith("ing"):
        return "VERB_ING"
    if t.endswith("ed"):
        return "VERB_ED"
    return "CONTENT"


def analyze_layer(h, positions, surprisals, tok_lens, pos_cats, normalize=False):
    """Analyze one layer."""
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
        try:
            r_norm, _ = pearsonr(pv, norms)
        except:
            r_norm = 0.0
        try:
            r_pos, _ = pearsonr(pv, positions[:T])
        except:
            r_pos = 0.0
        try:
            r_surp, _ = pearsonr(pv, surprisals[:T])
        except:
            r_surp = 0.0

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0

        results[pc] = {"norm": r_norm, "pos": r_pos, "surp": r_surp, "F": f_stat}

    return results


def run_layerwise(model_name, model_id, n_layers, ln_type):
    print(f"\n{'='*60}")
    print(f"LAYER-WISE ANALYSIS: {model_name}")
    print(f"  {model_id}, {n_layers} layers, {ln_type}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    device = next(model.parameters()).device
    ids = ids.to(device)
    max_len = getattr(model.config, "max_position_embeddings", 4096)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    T = ids.shape[1]

    print(f"  T={T}, running forward pass (all {n_layers + 1} hidden states)...")

    with torch.inference_mode():
        out = model(ids)

    # Extract ALL hidden states to CPU
    all_hs = []
    for li in range(len(out.hidden_states)):
        all_hs.append(out.hidden_states[li][0].float().cpu().numpy())

    logits = out.logits[0].float().cpu()
    del out, model
    gc.collect()
    torch.cuda.empty_cache()

    actual_layers = len(all_hs) - 1  # subtract embedding
    print(f"  Got {len(all_hs)} hidden states ({actual_layers} layers + embedding)")
    print(f"  d_model = {all_hs[0].shape[1]}")

    # Features
    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(T)]
    positions = np.arange(T, dtype=float)
    tok_lens = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)
    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, ids[0, t + 1].item()].item()
    surprisals[0] = surprisals[1]
    pos_cats = [classify_pos(tokens[t]) for t in range(T)]

    # ── Raw PCA ──
    print(f"\n  RAW PCA:")
    print(f"  {'Layer':>6} {'PC1var':>7} {'PC1-n':>7} {'|pos|':>7} {'PC':>3} "
          f"{'|surp|':>7} {'PC':>3} {'F':>7} {'PC':>3}")
    print(f"  {'-'*58}")

    raw_data = {"pc1_var": [], "pc1_norm": [], "best_pos": [], "best_f": []}

    for li in range(len(all_hs)):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats)

        best_pos = max(range(min(5, len(r) - 1)), key=lambda pc: abs(r[pc]["pos"]))
        best_surp = max(range(min(5, len(r) - 1)), key=lambda pc: abs(r[pc]["surp"]))
        best_f = max(range(min(5, len(r) - 1)), key=lambda pc: r[pc]["F"])

        layer_name = "embed" if li == 0 else f"L{li-1}"
        print(f"  {layer_name:>6} {r['ve'][0]:>6.1%} {r[0]['norm']:>+7.3f} "
              f"{abs(r[best_pos]['pos']):>7.3f} PC{best_pos+1} "
              f"{abs(r[best_surp]['surp']):>7.3f} PC{best_surp+1} "
              f"{r[best_f]['F']:>7.1f} PC{best_f+1}")

        raw_data["pc1_var"].append(r["ve"][0] * 100)
        raw_data["pc1_norm"].append(abs(r[0]["norm"]))
        raw_data["best_pos"].append(abs(r[best_pos]["pos"]))
        raw_data["best_f"].append(r[best_f]["F"])

    # ── Norm-normalized ──
    print(f"\n  NORM-NORMALIZED:")
    print(f"  {'Layer':>6} {'PC1var':>7} {'|pos|':>7} {'PC':>3} {'F':>7} {'PC':>3}")
    print(f"  {'-'*38}")

    nn_data = {"pc1_var": [], "best_pos": [], "best_f": []}

    for li in range(len(all_hs)):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats,
                          normalize=True)

        best_pos = max(range(min(5, len(r) - 1)), key=lambda pc: abs(r[pc]["pos"]))
        best_f = max(range(min(5, len(r) - 1)), key=lambda pc: r[pc]["F"])

        layer_name = "embed" if li == 0 else f"L{li-1}"
        print(f"  {layer_name:>6} {r['ve'][0]:>6.1%} "
              f"{abs(r[best_pos]['pos']):>7.3f} PC{best_pos+1} "
              f"{r[best_f]['F']:>7.1f} PC{best_f+1}")

        nn_data["pc1_var"].append(r["ve"][0] * 100)
        nn_data["best_pos"].append(abs(r[best_pos]["pos"]))
        nn_data["best_f"].append(r[best_f]["F"])

    # ── Phase detection ──
    print(f"\n  PHASE DETECTION:")
    # Find L1-equivalent: first layer where PC1-norm |r| > 0.95
    transition = None
    for li in range(len(all_hs)):
        if raw_data["pc1_norm"][li] > 0.95:
            transition = li
            break
    if transition is not None:
        layer_name = "embed" if transition == 0 else f"L{transition-1}"
        print(f"    Arc formation at: {layer_name} (PC1-norm |r| first exceeds 0.95)")
    else:
        print(f"    Arc formation: NOT DETECTED (PC1-norm never exceeds 0.95)")

    # Find dissolution: last layers where position encoding drops
    final_pos = raw_data["best_pos"][-1]
    peak_pos = max(raw_data["best_pos"])
    print(f"    Peak position |r|: {peak_pos:.3f}")
    print(f"    Final layer position |r|: {final_pos:.3f}")
    if final_pos < peak_pos * 0.5:
        print(f"    → Final layer dissolution DETECTED (>{50:.0f}% drop)")
    else:
        print(f"    → No clear final layer dissolution")

    # ── Plot ──
    n_total = len(all_hs)
    x = np.arange(n_total)
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_total - 1)]
    # Only show every Nth label to avoid crowding
    step = max(1, n_total // 16)
    tick_pos = list(range(0, n_total, step))
    tick_labels = [layer_labels[i] for i in tick_pos]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Phase coloring
    def add_phases(ax, trans, n):
        if trans is not None and trans > 0:
            ax.axvspan(-0.5, trans - 0.5, alpha=0.08, color="blue")
        if trans is not None:
            ax.axvspan(trans - 0.5, n - 1.5, alpha=0.08, color="green")
        ax.axvspan(n - 1.5, n - 0.5, alpha=0.08, color="red")

    # (a) PC1 variance + norm correlation
    ax = axes[0, 0]
    add_phases(ax, transition, n_total)
    ax2 = ax.twinx()
    ax.bar(x, raw_data["pc1_var"], color="steelblue", alpha=0.7, edgecolor="none")
    ax2.plot(x, raw_data["pc1_norm"], "o-", color="darkorange", lw=2, markersize=3)
    ax.set_ylabel("PC1 Variance (%)", color="steelblue")
    ax2.set_ylabel("|r(PC1, norm)|", color="darkorange")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(f"(a) PC1 Variance & Norm Correlation — {model_name}")
    ax.set_ylim(0, 105); ax2.set_ylim(0, 1.1)

    # (b) Position encoding
    ax = axes[0, 1]
    add_phases(ax, transition, n_total)
    ax.plot(x, raw_data["best_pos"], "s-", color="steelblue", lw=2, markersize=3, label="Raw")
    ax.plot(x, nn_data["best_pos"], "o--", color="darkorange", lw=2, markersize=3, label="Norm-normalized")
    ax.set_ylabel("Best |r| with position")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(f"(b) Position Encoding — {model_name}")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # (c) POS F-statistic
    ax = axes[1, 0]
    add_phases(ax, transition, n_total)
    ax.plot(x, raw_data["best_f"], "s-", color="steelblue", lw=2, markersize=3, label="Raw")
    ax.plot(x, nn_data["best_f"], "o--", color="darkorange", lw=2, markersize=3, label="Norm-normalized")
    ax.set_ylabel("Best ANOVA F (POS)")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(f"(c) POS Discrimination — {model_name}")
    ax.legend(fontsize=8)

    # (d) Comparison: GPT-2 vs 7B (raw PC1 var)
    ax = axes[1, 1]
    # GPT-2 data (hardcoded from our earlier experiment)
    gpt2_layers = 13
    gpt2_pc1 = [22.4, 14.8, 70.0, 96.9, 96.6, 96.3, 95.7, 94.8, 93.3, 90.9, 87.2, 81.4, 74.5]
    gpt2_x = np.linspace(0, 1, gpt2_layers)
    model_x = np.linspace(0, 1, n_total)
    ax.plot(gpt2_x, gpt2_pc1, "s-", color="steelblue", lw=2, markersize=4, label="GPT-2 (124M)")
    ax.plot(model_x, raw_data["pc1_var"], "o-", color="darkorange", lw=2, markersize=3, label=f"{model_name}")
    ax.set_xlabel("Relative depth (0=embed, 1=final)")
    ax.set_ylabel("PC1 Variance (%)")
    ax.set_title("(d) PC1 Variance: GPT-2 vs 7B")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    plt.suptitle(f"Layer-Wise Analysis: {model_name} ({n_layers} layers, d={all_hs[0].shape[1]})",
                 fontsize=14)
    plt.tight_layout()
    fname = f"exp7c_layerwise_{model_name.lower().replace('-','_').replace('.','')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"\n  Figure saved to {fname}")

    del all_hs, logits, log_probs
    gc.collect()

    return raw_data, nn_data


def main():
    print("=" * 60)
    print("7B LAYER-WISE ANALYSIS")
    print("  Does the 3-phase structure hold at 7B?")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    # Pythia-6.9B: ungated, no token needed
    raw, nn = run_layerwise(
        "Pythia-6.9B",
        "EleutherAI/pythia-6.9b",
        32,
        "Pre-LN",
    )

    gc.collect()
    torch.cuda.empty_cache()

    # If memory allows, also do Mistral-7B
    try:
        raw2, nn2 = run_layerwise(
            "Mistral-7B",
            "mistralai/Mistral-7B-v0.1",
            32,
            "Pre-LN (RMSNorm)",
        )
    except Exception as e:
        print(f"\n  Mistral-7B skipped: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        "Compare with GPT-2 3-phase structure:\n"
        "  Phase 1 (pre-arc): Position in PC1, weak norm correlation\n"
        "  Phase 2 (arc regime): Norm locks onto PC1, position → PC2\n"
        "  Phase 3 (final layer): Arc partially dissolves\n\n"
        "If the same pattern appears at 7B, the 3-phase structure\n"
        "is an architectural invariant of Pre-LN, not a small-model artifact."
    )


main()
