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
Experiment 5d: Cross-Model PCA Probing

Runs the thickness probing analysis (Exp 5) across multiple Pre-LN
models to verify that:
  1. PC1 = norm is consistent across architectures
  2. Position and POS/difficulty exist in the residual subspace
  3. The specific PC indices are model-dependent

Models: GPT-2, Pythia-410M, Qwen2-0.5B, OPT-1.3B

Paper reference: Section 5.1, Table 10
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    "darkness. The light grew brighter, then dimmed, then brightened again."
)

MODELS = [
    ("GPT-2",       "gpt2",                     12, "Pre-LN"),
    ("Pythia-410M", "EleutherAI/pythia-410m",    24, "Pre-LN"),
    ("Qwen2-0.5B",  "Qwen/Qwen2-0.5B",         24, "Pre-LN (RMS)"),
    ("OPT-1.3B",    "facebook/opt-1.3b",        24, "Pre-LN"),
]

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


def probe_model(name, model_id, n_layers, ln_type):
    print(f"\n{'='*50}")
    print(f"{name} ({model_id}, {ln_type})")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True, torch_dtype=torch.float32
    )
    model.eval()

    ids = tokenizer.encode(TEXT, return_tensors="pt")
    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    T = ids.shape[1]

    with torch.inference_mode():
        out = model(ids)

    mid = n_layers // 2
    h = out.hidden_states[mid + 1][0].float().cpu().numpy()
    logits = out.logits[0].float().cpu()
    del out, model; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"  Layer {mid}, T={T}, d={h.shape[1]}")

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

    K = min(10, h.shape[1], T - 1)
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h)
    ve = pca.explained_variance_ratio_

    print(f"  PCA: PC1={ve[0]:.1%}, PC2={ve[1]:.2%}, PC3={ve[2]:.2%}")

    print(f"\n  {'PC':>4} {'Var':>7} {'norm':>8} {'pos':>8} {'surp':>8} "
          f"{'tok_len':>8} {'POS F':>8}")
    print(f"  {'-'*55}")
    for pc in range(min(5, K)):
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

    del h, logits, log_probs; gc.collect()


def main():
    print("=" * 60)
    print("EXPERIMENT 5d: Cross-Model PCA Probing")
    print("=" * 60)

    for name, mid, nl, ln in MODELS:
        try:
            probe_model(name, mid, nl, ln)
        except Exception as e:
            print(f"\n  FAILED: {e}")
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
