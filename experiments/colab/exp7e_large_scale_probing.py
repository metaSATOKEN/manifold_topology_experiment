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
Large-Scale Probing: 13B, 14B, 70B — Does PC1=norm hold?
Google Colab version. Requires GPU with 40GB+ VRAM.

Setup cell:
    !pip install torch transformers scikit-learn scipy accelerate bitsandbytes matplotlib

Then paste and run.
"""

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


MODELS = [
    {
        "name": "Llama-2-13B",
        "model_id": "meta-llama/Llama-2-13b-hf",
        "n_layers": 40,
        "ln_type": "Pre-LN (RMSNorm)",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"},
        "gated": True,
    },
    {
        "name": "Qwen2.5-14B",
        "model_id": "Qwen/Qwen2.5-14B",
        "n_layers": 48,
        "ln_type": "Pre-LN (RMSNorm)",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"},
        "gated": False,
    },
    {
        "name": "Llama-2-70B (INT4)",
        "model_id": "meta-llama/Llama-2-70b-hf",
        "n_layers": 80,
        "ln_type": "Pre-LN (RMSNorm)",
        "load_kwargs": {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": "auto",
        },
        "gated": True,
    },
]

# Ungated fallbacks
FALLBACKS = [
    {
        "name": "Pythia-12B",
        "model_id": "EleutherAI/pythia-12b",
        "n_layers": 36,
        "ln_type": "Pre-LN",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"},
        "gated": False,
    },
    {
        "name": "OPT-13B",
        "model_id": "facebook/opt-13b",
        "n_layers": 40,
        "ln_type": "Pre-LN",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"},
        "gated": False,
    },
    {
        "name": "Falcon-40B",
        "model_id": "tiiuae/falcon-40b",
        "n_layers": 60,
        "ln_type": "Pre-LN",
        "load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"},
        "gated": False,
    },
]


def probe_model(cfg, do_layerwise=False):
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"{name} ({cfg['model_id']}, {cfg['ln_type']})")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], output_hidden_states=True, **cfg["load_kwargs"]
        )
        model.eval()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    device = next(model.parameters()).device
    ids = ids.to(device)
    max_len = getattr(model.config, "max_position_embeddings", 4096)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    T = ids.shape[1]

    print(f"  T={T}, forward pass...")
    with torch.inference_mode():
        out = model(ids)

    n_hs = len(out.hidden_states)
    n_layers = n_hs - 1
    d = out.hidden_states[0][0].shape[1]
    print(f"  {n_hs} hidden states, {n_layers} layers, d={d}")

    # Mid-layer probing
    mid = n_layers // 2
    h_mid = out.hidden_states[mid + 1][0].float().cpu().numpy()
    logits = out.logits[0].float().cpu()

    # Prepare features
    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(T)]
    positions = np.arange(T, dtype=float)
    norms_mid = np.linalg.norm(h_mid, axis=1)
    tok_lens = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)
    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, ids[0, t + 1].item()].item()
    surprisals[0] = surprisals[1]
    pos_cats = [classify_pos(tokens[t]) for t in range(T)]

    # Mid-layer PCA
    K = min(10, d, T - 1)
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h_mid)
    ve = pca.explained_variance_ratio_

    print(f"\n  MID-LAYER (L{mid}): PC1={ve[0]:.1%}")
    print(f"  {'PC':>4} {'Var':>7} {'norm':>8} {'pos':>8} {'surp':>8} {'tok_len':>8} {'POS F':>8}")
    print(f"  {'-'*55}")

    mid_results = {}
    for pc in range(min(5, K)):
        pv = proj[:, pc]
        r_n, _ = pearsonr(pv, norms_mid)
        r_p, _ = pearsonr(pv, positions)
        r_s, _ = pearsonr(pv, surprisals)
        r_t, _ = pearsonr(pv, tok_lens)
        groups = defaultdict(list)
        for t in range(T): groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0
        print(f"  PC{pc+1:>2} {ve[pc]:>6.2%} {r_n:>+8.3f} {r_p:>+8.3f} "
              f"{r_s:>+8.3f} {r_t:>+8.3f} {f_stat:>8.1f}")
        mid_results[pc] = {"ve": ve[pc], "norm": r_n, "pos": r_p, "surp": r_s, "F": f_stat}

    # Norm-normalized
    h_nn = h_mid / (np.linalg.norm(h_mid, axis=1, keepdims=True) + 1e-12)
    pca_nn = PCA(n_components=K)
    proj_nn = pca_nn.fit_transform(h_nn)
    ve_nn = pca_nn.explained_variance_ratio_
    print(f"\n  NORM-NORMALIZED: PC1={ve_nn[0]:.1%} (Δ={ve_nn[0]-ve[0]:+.1%})")

    result = {
        "name": name, "d": d, "n_layers": n_layers, "ln_type": cfg["ln_type"],
        "mid_layer": mid,
        "pc1_var": ve[0] * 100, "pc1_norm_r": abs(mid_results[0]["norm"]),
        "nn_pc1_var": ve_nn[0] * 100,
        "best_pos": max(abs(mid_results[pc]["pos"]) for pc in mid_results),
        "best_f": max(mid_results[pc]["F"] for pc in mid_results),
    }

    # Layer-wise (optional, for very large models just do key layers)
    if do_layerwise:
        print(f"\n  LAYER-WISE (key layers):")
        print(f"  {'Layer':>6} {'PC1var':>7} {'PC1-n':>7} {'|pos|':>7} {'F':>7}")
        print(f"  {'-'*38}")

        key_layers = sorted(set([
            0, 1, 2,                    # early
            n_layers // 4,              # quarter
            n_layers // 2,              # mid
            3 * n_layers // 4,          # three-quarter
            n_layers - 2, n_layers - 1, n_layers  # final
        ]))
        key_layers = [li for li in key_layers if li < n_hs]

        lw_data = {"layers": [], "pc1_var": [], "pc1_norm": []}

        for li in key_layers:
            h_l = out.hidden_states[li][0].float().cpu().numpy()
            r = analyze_layer(h_l, positions, surprisals, tok_lens, pos_cats)
            bp = max(range(min(5, len(r)-1)), key=lambda pc: abs(r[pc]["pos"]))
            bf = max(range(min(5, len(r)-1)), key=lambda pc: r[pc]["F"])
            ln = "embed" if li == 0 else f"L{li-1}"
            print(f"  {ln:>6} {r['ve'][0]:>6.1%} {r[0]['norm']:>+7.3f} "
                  f"{abs(r[bp]['pos']):>7.3f} {r[bf]['F']:>7.1f}")
            lw_data["layers"].append(li)
            lw_data["pc1_var"].append(r["ve"][0] * 100)
            lw_data["pc1_norm"].append(abs(r[0]["norm"]))
            del h_l

        result["layerwise"] = lw_data

    del out, h_mid, h_nn, logits, log_probs, proj, proj_nn, model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 60)
    print("LARGE-SCALE PROBING: 13B / 14B / 70B")
    print("  Does PC1=norm hold at massive scale?")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU!")

    all_results = []

    # Try primary models, then fallbacks for any that fail
    for cfg in MODELS + FALLBACKS:
        result = probe_model(cfg, do_layerwise=True)
        if result is not None:
            all_results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("GRAND SUMMARY: PC1=NORM ACROSS ALL SCALES")
    print("=" * 60)

    # Include previous results for comparison
    prev = [
        {"name": "GPT-2", "d": 768, "pc1_var": 94.8, "pc1_norm_r": 0.999, "scale": "124M"},
        {"name": "Pythia-410M", "d": 1024, "pc1_var": 96.5, "pc1_norm_r": 1.000, "scale": "410M"},
        {"name": "Pythia-2.8B", "d": 2560, "pc1_var": 83.1, "pc1_norm_r": 0.999, "scale": "2.8B"},
        {"name": "Mistral-7B", "d": 4096, "pc1_var": 90.0, "pc1_norm_r": 0.999, "scale": "7B"},
        {"name": "Qwen2-7B", "d": 3584, "pc1_var": 99.5, "pc1_norm_r": 1.000, "scale": "7B"},
        {"name": "Pythia-6.9B", "d": 4096, "pc1_var": 80.3, "pc1_norm_r": 1.000, "scale": "6.9B"},
    ]

    print(f"\n  {'Model':>20} {'Params':>8} {'d':>6} {'PC1 var':>8} {'|r(PC1,norm)|':>14} {'Verdict':>10}")
    print(f"  {'-'*70}")

    for p in prev:
        v = "YES" if p["pc1_norm_r"] > 0.95 else "PARTIAL" if p["pc1_norm_r"] > 0.8 else "NO"
        print(f"  {p['name']:>20} {p['scale']:>8} {p['d']:>6} {p['pc1_var']:>7.1f}% "
              f"{p['pc1_norm_r']:>14.3f} {v:>10}")

    for r in all_results:
        v = "YES" if r["pc1_norm_r"] > 0.95 else "PARTIAL" if r["pc1_norm_r"] > 0.8 else "NO"
        scale = f"{r['n_layers']}L"
        print(f"  {r['name']:>20} {scale:>8} {r['d']:>6} {r['pc1_var']:>7.1f}% "
              f"{r['pc1_norm_r']:>14.3f} {v:>10}")

    print(f"\n  KEY QUESTION: Does PC1=norm scale to 70B?")
    for r in all_results:
        if r["pc1_norm_r"] > 0.95:
            print(f"    {r['name']:>20}: |r|={r['pc1_norm_r']:.3f}, var={r['pc1_var']:.1f}% → YES")
        elif r["pc1_norm_r"] > 0.8:
            print(f"    {r['name']:>20}: |r|={r['pc1_norm_r']:.3f}, var={r['pc1_var']:.1f}% → PARTIAL")
        else:
            print(f"    {r['name']:>20}: |r|={r['pc1_norm_r']:.3f}, var={r['pc1_var']:.1f}% → NO")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))

    # All models by parameter count (approximate)
    param_counts = {
        "GPT-2": 0.124, "Pythia-410M": 0.41, "Pythia-2.8B": 2.8,
        "Mistral-7B": 7, "Qwen2-7B": 7, "Pythia-6.9B": 6.9,
    }
    # Add new results
    model_params = {
        "Llama-2-13B": 13, "Qwen2.5-14B": 14,
        "Pythia-12B": 12, "OPT-13B": 13, "Falcon-40B": 40,
        "Llama-2-70B (INT4)": 70,
    }

    # Plot previous
    for p in prev:
        x = param_counts.get(p["name"], 1)
        ax.scatter(x, p["pc1_norm_r"], s=100, c="steelblue", zorder=5, edgecolors="black")
        ax.annotate(p["name"], (x, p["pc1_norm_r"]), fontsize=7, ha="right",
                    xytext=(-5, 5), textcoords="offset points")

    # Plot new
    for r in all_results:
        x = model_params.get(r["name"], r["n_layers"])
        ax.scatter(x, r["pc1_norm_r"], s=150, c="darkorange", zorder=6,
                   edgecolors="black", marker="D")
        ax.annotate(r["name"], (x, r["pc1_norm_r"]), fontsize=8, fontweight="bold",
                    ha="left", xytext=(5, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)", fontsize=12)
    ax.set_ylabel("|r(PC1, norm)|", fontsize=12)
    ax.set_title("PC1-Norm Alignment Across Model Scale", fontsize=14)
    ax.axhline(0.95, ls="--", color="green", alpha=0.5, label="|r|=0.95 threshold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.7, 1.02)

    plt.tight_layout()
    plt.savefig("exp7e_large_scale.png", dpi=150)
    plt.show()
    print(f"\n  Figure saved to exp7e_large_scale.png")


main()
