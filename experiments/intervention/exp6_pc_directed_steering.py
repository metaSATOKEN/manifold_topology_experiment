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

"""Experiment 6: PC-Directed Steering vs Random Orthogonal
Runs each (prompt, strategy) pair in a subprocess to avoid memory leaks."""
import subprocess, sys, json, gc
import numpy as np
from pathlib import Path

N_RUNS = 8
PROMPTS = [
    "The scientist discovered that the ancient",
    "In a world where machines could dream,",
    "The last survivor of the expedition wrote",
    "Deep beneath the ocean surface, a strange",
    "The algorithm predicted something that no one",
    "When the lights went out across the city,",
    "The professor opened the dusty manuscript and",
    "At the edge of the known universe, a signal",
]
STRATEGIES = ["none", "orthogonal", "pc3", "pc2"]
LABELS = {"none":"Baseline","orthogonal":"Recync (ortho)","pc3":"PC3 (surprisal)","pc2":"PC2 (position)"}

WORKER = '''
import sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

args = json.loads(sys.argv[1])
prompt, seed, strat = args["prompt"], args["seed"], args["strategy"]
SIGMA, BL_WIN, K = 0.6, 15, 3
RATIO_B, RATIO_S = 0.08, 0.12

def is_crisis(sims):
    if len(sims) < max(K, BL_WIN): return False
    b = sims[-BL_WIN:]
    return all(v < np.mean(b)-SIGMA*np.std(b) for v in sims[-K:])

def get_ratio(sims, step):
    pre = np.mean(sims[max(0,step-K+1):step+1])
    bs = np.mean(sims[max(0,step-K+1-BL_WIN):max(0,step-K+1)]) if step>K else 0.9
    return RATIO_B + RATIO_S*max(0,min(1,(bs-pre)/(bs+1e-8)))

tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2"); model.eval()
hook_out = [None]
def hfn(m,i,o): hook_out[0] = (o[0] if isinstance(o,tuple) else o)[:,-1,:]
handle = model.transformer.h[-1].register_forward_hook(hfn)
pca_c = np.load("results/pca_components_distilgpt2.npy")

torch.manual_seed(seed); np.random.seed(seed)
ids = tok.encode(prompt, return_tensors="pt")
prev = None; sims = []; ni = 0; ctx = None

for step in range(50):
    with torch.inference_mode(): out = model(ids[:,-64:])
    h = hook_out[0].detach().clone(); hook_out[0] = None
    if prev is not None:
        sims.append(float(F.cosine_similarity(h.view(1,-1),prev.view(1,-1)).item()))
    if step==0 and strat=="orthogonal":
        ctx = (h/(torch.norm(h)+1e-8)).detach()
    do = len(sims)>0 and is_crisis(sims) and strat!="none"
    if do:
        ratio = get_ratio(sims, step)
        if strat=="orthogonal" and ctx is not None:
            d = ctx - (torch.sum(ctx*h)/(torch.sum(h*h)+1e-8))*h
        elif strat in ("pc3","pc2"):
            idx = 2 if strat=="pc3" else 1
            pv = torch.tensor(pca_c[idx],dtype=h.dtype).view_as(h)
            d = pv - (torch.sum(pv*h)/(torch.sum(h*h)+1e-8))*h
        else: d = torch.zeros_like(h)
        dn = d/(torch.norm(d)+1e-8)
        corrected = h + ratio*torch.norm(h)*dn
        with torch.inference_mode():
            logits = model.lm_head(model.transformer.ln_f(corrected.unsqueeze(0)))[0,0]
        ni += 1; prev = corrected.detach()
    else:
        logits = out.logits[0,-1]; prev = h.detach()
    t = torch.argmax(logits).view(1,1)
    ids = torch.cat([ids,t],dim=-1)
    del out, h, logits

handle.remove()
print(json.dumps({"sims": sims, "n_intv": ni}))
'''

def run_one(prompt, seed, strategy):
    """Run a single (prompt, strategy) in a subprocess."""
    args = json.dumps({"prompt": prompt, "seed": seed, "strategy": strategy})
    result = subprocess.run(
        [sys.executable, "-c", WORKER, args],
        capture_output=True, text=True, timeout=120,
    )
    for line in result.stdout.strip().split("\n"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    print(f"    WARN: no output for {strategy}, stderr: {result.stderr[-200:]}")
    return {"sims": [], "n_intv": 0}


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP 6: PC-Directed Steering vs Random Orthogonal")
    print(f"  {N_RUNS} prompts × {len(STRATEGIES)} strategies (subprocess isolation)")
    print("=" * 60)

    all_sims = {s: [] for s in STRATEGIES}

    for run in range(N_RUNS):
        prompt = PROMPTS[run % len(PROMPTS)]
        seed = 42 + run
        print(f"\n  Run {run+1}/{N_RUNS}: '{prompt[:40]}...'")

        for strat in STRATEGIES:
            r = run_one(prompt, seed, strat)
            all_sims[strat].append(r["sims"])
            if r["n_intv"]:
                print(f"    {strat:>12}: {r['n_intv']} interventions")

    # ── Analysis ──
    K, BL_WIN, SIGMA, REC_K = 3, 15, 0.6, 5

    def is_crisis(sims):
        if len(sims) < max(K, BL_WIN): return False
        b = sims[-BL_WIN:]
        return all(v < np.mean(b)-SIGMA*np.std(b) for v in sims[-K:])

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    pir = {s: [] for s in STRATEGIES}
    for run in range(N_RUNS):
        base = all_sims["none"][run]
        cs = None
        for t in range(K, len(base)):
            if is_crisis(base[:t+1]): cs = t; break
        if cs is None: continue
        for s in STRATEGIES:
            sm = all_sims[s][run]
            st = cs + 1; en = min(st + REC_K, len(sm))
            if st < len(sm):
                pir[s].append(float(np.mean(sm[st:en])))

    nc = len(pir["none"])
    print(f"\n  Crisis runs: {nc}/{N_RUNS}")

    print(f"\n  {'Strategy':>22} {'N':>3} {'Mean PIR':>10} {'Std':>8}")
    print(f"  {'-'*47}")
    for s in STRATEGIES:
        if pir[s]:
            print(f"  {LABELS[s]:>22} {len(pir[s]):>3} "
                  f"{np.mean(pir[s]):>10.4f} {np.std(pir[s]):>8.4f}")

    if nc >= 2:
        from scipy.stats import ttest_rel
        print(f"\n  PAIRWISE vs BASELINE:")
        for s in ["orthogonal", "pc3", "pc2"]:
            if pir["none"] and pir[s]:
                n = min(len(pir["none"]), len(pir[s]))
                ba, sa = np.array(pir["none"][:n]), np.array(pir[s][:n])
                d = sa - ba; md = np.mean(d)
                cd = md / np.std(d) if np.std(d) > 1e-12 else 0
                _, p = ttest_rel(ba, sa)
                sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
                print(f"    {LABELS[s]:>22}: Δ={md:+.4f} d={cd:+.3f} p={p:.3f} {sig}")

        if pir["orthogonal"] and pir["pc3"]:
            n = min(len(pir["orthogonal"]), len(pir["pc3"]))
            oa, pa = np.array(pir["orthogonal"][:n]), np.array(pir["pc3"][:n])
            d = pa - oa; md = np.mean(d)
            cd = md / np.std(d) if np.std(d) > 1e-12 else 0
            _, p = ttest_rel(oa, pa)
            sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
            print(f"\n  HEAD-TO-HEAD Ortho vs PC3: Δ={md:+.4f} d={cd:+.3f} p={p:.3f} {sig}")
    elif nc == 1:
        print("\n  Single crisis run — raw PIR:")
        for s in STRATEGIES:
            if pir[s]: print(f"    {LABELS[s]:>22}: {pir[s][0]:.4f}")
    else:
        print("\n  No crises detected in baseline runs.")
        print("  Mean cosim by strategy:")
        for s in STRATEGIES:
            means = [np.mean(sm) for sm in all_sims[s] if sm]
            if means: print(f"    {LABELS[s]:>22}: {np.mean(means):.4f}")

    # ── Plot ──
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["gray", "steelblue", "darkorange", "forestgreen"]
    if any(pir[s] for s in STRATEGIES):
        bp_d = [pir[s] for s in STRATEGIES if pir[s]]
        bp_l = [LABELS[s] for s in STRATEGIES if pir[s]]
        bp_c = [colors[i] for i,s in enumerate(STRATEGIES) if pir[s]]
        bp = ax.boxplot(bp_d, tick_labels=bp_l, patch_artist=True)
        for patch, c in zip(bp["boxes"], bp_c): patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel("PIR (mean cosim post-crisis)")
    else:
        means = {s: np.mean([np.mean(sm) for sm in all_sims[s] if sm]) for s in STRATEGIES}
        ax.bar(range(4), [means[s] for s in STRATEGIES], color=colors, edgecolor="black")
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=8)
        ax.set_ylabel("Mean cosim (all tokens)")
    ax.set_title("Exp 6: PC-Directed Steering vs Orthogonal (distilgpt2)")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    plt.tight_layout()
    plt.savefig(str(out_dir / "exp6_pc_steering.png"), dpi=150)
    print(f"\n  Figure: {out_dir}/exp6_pc_steering.png")


if __name__ == "__main__":
    main()
