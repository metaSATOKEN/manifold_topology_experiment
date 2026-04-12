"""
Qwen2-7B Layer-Wise Analysis — RMSNorm hypothesis test.

If Mistral-7B (RMSNorm) hits PC1=100% at L1, does Qwen2-7B (also RMSNorm)?
If yes → RMSNorm drives extreme early-layer anisotropy.
If no  → Mistral-specific (architecture, training, etc.)

Also tests: Does Qwen2-7B show the same "funnel" (monotonic PC1 decay)?

Setup cell:
    !pip install torch transformers scikit-learn scipy accelerate matplotlib

Then paste and run.
"""

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict
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
    if t in PUNCT or all(c in PUNCT for c in t): return "PUNCT"
    if t in FUNC_WORDS: return "FUNC"
    if t.endswith("ing"): return "VERB_ING"
    if t.endswith("ed"): return "VERB_ED"
    return "CONTENT"

def analyze_layer(h, positions, surprisals, tok_lens, pos_cats, normalize=False):
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
        try: r_norm, _ = pearsonr(pv, norms)
        except: r_norm = 0.0
        try: r_pos, _ = pearsonr(pv, positions[:T])
        except: r_pos = 0.0
        try: r_surp, _ = pearsonr(pv, surprisals[:T])
        except: r_surp = 0.0
        groups = defaultdict(list)
        for t in range(T): groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0
        results[pc] = {"norm": r_norm, "pos": r_pos, "surp": r_surp, "F": f_stat}
    return results


def run_model(name, model_id, n_layers, ln_type):
    print(f"\n{'='*60}")
    print(f"LAYER-WISE: {name} ({ln_type})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    device = next(model.parameters()).device
    ids = ids.to(device)
    max_len = getattr(model.config, "max_position_embeddings", 4096)
    if ids.shape[1] > max_len: ids = ids[:, :max_len]
    T = ids.shape[1]

    print(f"  T={T}, forward pass...")
    with torch.inference_mode():
        out = model(ids)

    all_hs = [hs[0].float().cpu().numpy() for hs in out.hidden_states]
    logits = out.logits[0].float().cpu()
    del out, model; gc.collect(); torch.cuda.empty_cache()

    d = all_hs[0].shape[1]
    print(f"  {len(all_hs)} hidden states, d={d}")

    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(T)]
    positions = np.arange(T, dtype=float)
    tok_lens = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)
    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, ids[0, t + 1].item()].item()
    surprisals[0] = surprisals[1]
    pos_cats = [classify_pos(tokens[t]) for t in range(T)]

    # Raw
    raw = {"pc1_var": [], "pc1_norm": [], "best_pos": [], "best_f": []}
    print(f"\n  {'Layer':>6} {'PC1var':>7} {'PC1-n':>7} {'|pos|':>7} {'PC':>3} {'F':>7} {'PC':>3}")
    print(f"  {'-'*45}")
    for li in range(len(all_hs)):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats)
        bp = max(range(min(5,len(r)-1)), key=lambda pc: abs(r[pc]["pos"]))
        bf = max(range(min(5,len(r)-1)), key=lambda pc: r[pc]["F"])
        ln = "embed" if li == 0 else f"L{li-1}"
        print(f"  {ln:>6} {r['ve'][0]:>6.1%} {r[0]['norm']:>+7.3f} "
              f"{abs(r[bp]['pos']):>7.3f} PC{bp+1} {r[bf]['F']:>7.1f} PC{bf+1}")
        raw["pc1_var"].append(r["ve"][0] * 100)
        raw["pc1_norm"].append(abs(r[0]["norm"]))
        raw["best_pos"].append(abs(r[bp]["pos"]))
        raw["best_f"].append(r[bf]["F"])

    # Norm-normalized
    nn = {"pc1_var": [], "best_pos": [], "best_f": []}
    for li in range(len(all_hs)):
        r = analyze_layer(all_hs[li], positions, surprisals, tok_lens, pos_cats, normalize=True)
        bp = max(range(min(5,len(r)-1)), key=lambda pc: abs(r[pc]["pos"]))
        bf = max(range(min(5,len(r)-1)), key=lambda pc: r[pc]["F"])
        nn["pc1_var"].append(r["ve"][0] * 100)
        nn["best_pos"].append(abs(r[bp]["pos"]))
        nn["best_f"].append(r[bf]["F"])

    del all_hs, logits; gc.collect()
    return raw, nn, d


def main():
    print("=" * 60)
    print("RMSNorm HYPOTHESIS TEST: Qwen2-7B Layer-Wise")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    # Qwen2-7B
    qwen_raw, qwen_nn, qwen_d = run_model(
        "Qwen2-7B", "Qwen/Qwen2-7B", 28, "Pre-LN (RMSNorm)"
    )

    # Hardcoded comparison data from previous runs
    mistral_pc1 = [3.5, 73.2, 100.0, 99.9, 99.8, 99.6, 99.4, 99.1, 98.7, 98.4,
                   98.0, 97.5, 96.9, 96.3, 95.4, 94.2, 92.5, 90.0, 86.6, 82.4,
                   76.4, 71.3, 66.4, 62.2, 57.4, 53.6, 49.7, 45.7, 41.5, 37.1,
                   32.6, 24.4, 8.3]
    pythia_pc1 = [7.5, 19.6, 12.2, 8.6, 53.9, 86.8, 90.6, 91.5, 90.6, 89.8,
                  88.7, 88.0, 87.0, 85.4, 84.2, 82.9, 81.7, 80.3, 78.5, 76.3,
                  74.1, 71.0, 67.7, 64.2, 61.0, 58.0, 55.3, 52.5, 49.9, 47.6,
                  44.9, 39.8, 10.5]
    gpt2_pc1 = [22.4, 14.8, 70.0, 96.9, 96.6, 96.3, 95.7, 94.8, 93.3, 90.9,
                87.2, 81.4, 74.5]

    # ── Key comparison ──
    print("\n" + "=" * 60)
    print("RMSNorm HYPOTHESIS TEST")
    print("=" * 60)

    qwen_l1 = qwen_raw["pc1_var"][1] if len(qwen_raw["pc1_var"]) > 1 else 0
    print(f"\n  L1 PC1 variance:")
    print(f"    Mistral-7B (RMSNorm):  100.0%")
    print(f"    Qwen2-7B   (RMSNorm):  {qwen_l1:.1f}%")
    print(f"    Pythia-6.9B (LayerNorm): 19.6%")

    if qwen_l1 > 95:
        print(f"\n  → HYPOTHESIS SUPPORTED: Both RMSNorm models hit >95% at L1")
        print(f"    RMSNorm's lack of bias term likely drives extreme early anisotropy")
    elif qwen_l1 > 80:
        print(f"\n  → PARTIAL SUPPORT: Qwen2-7B high but not as extreme as Mistral")
    else:
        print(f"\n  → HYPOTHESIS REJECTED: Qwen2-7B does not show extreme L1 anisotropy")
        print(f"    The effect is Mistral-specific, not RMSNorm-general")

    # ── Funnel comparison ──
    print(f"\n  Peak PC1 variance:")
    print(f"    Mistral-7B:  {max(mistral_pc1):.1f}% at L{mistral_pc1.index(max(mistral_pc1))-1}")
    print(f"    Qwen2-7B:    {max(qwen_raw['pc1_var']):.1f}% at L{qwen_raw['pc1_var'].index(max(qwen_raw['pc1_var']))-1}")
    print(f"    Pythia-6.9B: {max(pythia_pc1):.1f}% at L{pythia_pc1.index(max(pythia_pc1))-1}")

    qwen_final = qwen_raw["pc1_var"][-1]
    print(f"\n  Final layer PC1 variance:")
    print(f"    Mistral-7B:   8.3%")
    print(f"    Qwen2-7B:    {qwen_final:.1f}%")
    print(f"    Pythia-6.9B: 10.5%")

    # ── Plot: 4-model comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) PC1 variance across depth
    ax = axes[0]
    for data, label, color in [
        (gpt2_pc1, "GPT-2 (124M, LN)", "steelblue"),
        (pythia_pc1, "Pythia-6.9B (LN)", "darkorange"),
        (mistral_pc1, "Mistral-7B (RMSNorm)", "forestgreen"),
        (qwen_raw["pc1_var"], "Qwen2-7B (RMSNorm)", "crimson"),
    ]:
        x = np.linspace(0, 1, len(data))
        ax.plot(x, data, "o-", color=color, markersize=3, lw=2, label=label)

    ax.set_xlabel("Relative depth (0=embed, 1=final)")
    ax.set_ylabel("PC1 Variance (%)")
    ax.set_title("(a) PC1 Variance Across Depth: 4 Models")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # (b) Norm correlation across depth
    ax = axes[1]
    # Qwen norm correlation
    qwen_x = np.linspace(0, 1, len(qwen_raw["pc1_norm"]))
    ax.plot(qwen_x, qwen_raw["pc1_norm"], "o-", color="crimson", markersize=3, lw=2,
            label="Qwen2-7B (RMSNorm)")

    # Add position encoding for Qwen
    ax2 = ax.twinx()
    ax2.plot(qwen_x, qwen_raw["best_pos"], "s--", color="gray", markersize=3, lw=1.5,
             alpha=0.7, label="Qwen2-7B position |r|")
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("|r(PC1, norm)|", color="crimson")
    ax2.set_ylabel("Best position |r|", color="gray")
    ax.set_title("(b) Qwen2-7B: Norm & Position Encoding by Layer")
    ax.legend(fontsize=8, loc="center left")
    ax2.legend(fontsize=8, loc="center right")

    plt.suptitle("RMSNorm Hypothesis Test: Does Qwen2-7B Match Mistral-7B's L1=100%?",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("exp7d_qwen2_7b_layerwise.png", dpi=150)
    plt.show()
    print(f"\n  Figure saved to exp7d_qwen2_7b_layerwise.png")


main()
