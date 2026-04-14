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
7B Model Probing — Does PC1=norm hold at 7B scale?
Google Colab version. Requires GPU with 16GB+ VRAM.

Setup cell (run first):
    !pip install torch transformers scikit-learn scipy bitsandbytes accelerate

Then paste this code and call main().
"""

# ============================================================
# CELL 1: Install
# ============================================================
# !pip install torch transformers scikit-learn scipy bitsandbytes accelerate

# ============================================================
# CELL 2: Full experiment
# ============================================================

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict

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


MODELS_7B = [
    # (display_name, hf_id, n_layers, ln_type, load_kwargs)
    (
        "Llama-2-7B",
        "meta-llama/Llama-2-7b-hf",
        32,
        "Pre-LN (RMSNorm)",
        {"torch_dtype": torch.float16, "device_map": "auto"},
    ),
    (
        "Mistral-7B-v0.1",
        "mistralai/Mistral-7B-v0.1",
        32,
        "Pre-LN (RMSNorm)",
        {"torch_dtype": torch.float16, "device_map": "auto"},
    ),
    (
        "Qwen2-7B",
        "Qwen/Qwen2-7B",
        28,
        "Pre-LN (RMSNorm)",
        {"torch_dtype": torch.float16, "device_map": "auto"},
    ),
]

# Fallbacks if gated models aren't accessible
MODELS_FALLBACK = [
    (
        "Pythia-6.9B",
        "EleutherAI/pythia-6.9b",
        32,
        "Pre-LN",
        {"torch_dtype": torch.float16, "device_map": "auto"},
    ),
    (
        "OPT-6.7B",
        "facebook/opt-6.7b",
        32,
        "Pre-LN",
        {"torch_dtype": torch.float16, "device_map": "auto"},
    ),
]


def probe_model(name, model_id, n_layers, ln_type, load_kwargs):
    print(f"\n{'='*60}")
    print(f"{name} ({model_id}, {ln_type})")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, output_hidden_states=True, **load_kwargs
        )
        model.eval()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    # Move to model's device
    device = next(model.parameters()).device
    ids = ids.to(device)

    max_len = getattr(model.config, "max_position_embeddings", 4096)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    T = ids.shape[1]

    print(f"  T={T}, running forward pass...")

    with torch.inference_mode():
        out = model(ids)

    mid = n_layers // 2
    h = out.hidden_states[mid + 1][0].float().cpu().numpy()
    logits = out.logits[0].float().cpu()

    del out, model
    gc.collect()
    torch.cuda.empty_cache()

    d = h.shape[1]
    print(f"  Layer {mid}, T={T}, d={d}")

    # Features
    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(T)]
    positions = np.arange(T, dtype=float)
    norms = np.linalg.norm(h, axis=1)
    tok_lens = np.array([len(tokens[t].strip()) for t in range(T)], dtype=float)

    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, ids[0, t + 1].item()].item()
    surprisals[0] = surprisals[1]

    pos_cats = [classify_pos(tokens[t]) for t in range(T)]

    # PCA
    K = min(10, d, T - 1)
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h)
    ve = pca.explained_variance_ratio_

    print(f"\n  PCA: PC1={ve[0]:.1%}, PC2={ve[1]:.2%}, PC3={ve[2]:.2%}")
    print(f"\n  {'PC':>4} {'Var':>7} {'norm':>8} {'pos':>8} {'surp':>8} "
          f"{'tok_len':>8} {'POS F':>8}")
    print(f"  {'-'*55}")

    results = {}
    for pc in range(min(5, K)):
        pv = proj[:, pc]
        r_n, _ = pearsonr(pv, norms)
        r_p, _ = pearsonr(pv, positions)
        r_s, _ = pearsonr(pv, surprisals)
        r_t, _ = pearsonr(pv, tok_lens)

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0

        print(f"  PC{pc+1:>2} {ve[pc]:>6.2%} {r_n:>+8.3f} {r_p:>+8.3f} "
              f"{r_s:>+8.3f} {r_t:>+8.3f} {f_stat:>8.1f}")
        results[pc] = {"ve": ve[pc], "norm": r_n, "pos": r_p,
                       "surp": r_s, "tl": r_t, "F": f_stat}

    # Norm-normalized
    h_nn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    pca_nn = PCA(n_components=K)
    proj_nn = pca_nn.fit_transform(h_nn)
    ve_nn = pca_nn.explained_variance_ratio_

    print(f"\n  NORM-NORMALIZED:")
    print(f"  PCA: PC1={ve_nn[0]:.1%}, PC2={ve_nn[1]:.2%}, PC3={ve_nn[2]:.2%}")
    print(f"\n  {'PC':>4} {'Var':>7} {'norm':>8} {'pos':>8} {'surp':>8} "
          f"{'tok_len':>8} {'POS F':>8}")
    print(f"  {'-'*55}")

    nn_results = {}
    norms_nn = np.linalg.norm(h_nn, axis=1)
    for pc in range(min(5, K)):
        pv = proj_nn[:, pc]
        r_n, _ = pearsonr(pv, norms_nn)
        r_p, _ = pearsonr(pv, positions)
        r_s, _ = pearsonr(pv, surprisals)
        r_t, _ = pearsonr(pv, tok_lens)

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = f_oneway(*valid)[0] if len(valid) >= 2 else 0.0

        print(f"  PC{pc+1:>2} {ve_nn[pc]:>6.2%} {r_n:>+8.3f} {r_p:>+8.3f} "
              f"{r_s:>+8.3f} {r_t:>+8.3f} {f_stat:>8.1f}")
        nn_results[pc] = {"ve": ve_nn[pc], "norm": r_n, "pos": r_p,
                          "surp": r_s, "tl": r_t, "F": f_stat}

    print(f"\n  RAW PC1 var: {ve[0]:.1%} → NORMALIZED PC1 var: {ve_nn[0]:.1%} "
          f"(Δ={ve_nn[0]-ve[0]:+.1%})")

    del h, h_nn, logits, log_probs, proj, proj_nn
    gc.collect()

    return {
        "name": name,
        "model_id": model_id,
        "ln_type": ln_type,
        "d_model": d,
        "n_layers": n_layers,
        "raw": {"ve": ve, "results": results},
        "normalized": {"ve": ve_nn, "results": nn_results},
    }


def main():
    print("=" * 60)
    print("7B MODEL PROBING: Does PC1=norm hold at 7B scale?")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU detected. 7B models will be very slow on CPU.")

    all_results = []

    # Try primary models first, fall back if needed
    for models_list, label in [(MODELS_7B, "primary"), (MODELS_FALLBACK, "fallback")]:
        for name, model_id, n_layers, ln_type, kwargs in models_list:
            result = probe_model(name, model_id, n_layers, ln_type, kwargs)
            if result is not None:
                all_results.append(result)
            gc.collect()
            torch.cuda.empty_cache()

            # Stop after we have at least 2 successful 7B models
            if len(all_results) >= 2 and label == "fallback":
                break

    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY: 7B PROBING RESULTS")
        print("=" * 60)
        print(f"\n  {'Model':>18} {'d':>6} {'PC1 var':>8} {'PC1-norm r':>11} "
              f"{'Best |pos|':>11} {'Best POS F':>11} {'Norm→PC1':>9}")
        print(f"  {'-'*75}")

        for r in all_results:
            raw = r["raw"]
            best_pos = max(range(5), key=lambda pc: abs(raw["results"][pc]["pos"]))
            best_f = max(range(5), key=lambda pc: raw["results"][pc]["F"])
            delta = r["normalized"]["ve"][0] - raw["ve"][0]
            print(f"  {r['name']:>18} {r['d_model']:>6} {raw['ve'][0]:>7.1%} "
                  f"{raw['results'][0]['norm']:>+11.3f} "
                  f"{abs(raw['results'][best_pos]['pos']):>11.3f} (PC{best_pos+1}) "
                  f"{raw['results'][best_f]['F']:>6.1f} (PC{best_f+1}) "
                  f"{delta:>+8.1%}")

        print("\n  KEY QUESTION: Does PC1=norm hold at 7B?")
        for r in all_results:
            rn = abs(r["raw"]["results"][0]["norm"])
            v = r["raw"]["ve"][0]
            if rn > 0.95:
                verdict = "YES — PC1=norm confirmed"
            elif rn > 0.8:
                verdict = "PARTIAL — strong but not dominant"
            else:
                verdict = "NO — PC1 ≠ norm at this scale"
            print(f"    {r['name']:>18}: |r|={rn:.3f}, var={v:.1%} → {verdict}")

    else:
        print("\n  No models could be loaded. Check HF_TOKEN for gated models.")


main()
