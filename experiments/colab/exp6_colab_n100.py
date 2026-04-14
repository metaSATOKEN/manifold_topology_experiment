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
Experiment 6: PC-Directed Steering vs Random Orthogonal Steering
Google Colab version — N=100 runs for statistical power.

Setup cell (run first):
    !pip install torch transformers scikit-learn scipy numpy matplotlib

Then paste this code into a cell and run.
"""

import gc, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, ttest_rel
from collections import defaultdict
import matplotlib.pyplot as plt

# ── Config ──
MODEL_NAME = "gpt2"
MAX_TOKENS = 100
MAX_CTX = 256
N_RUNS = 100
SIGMA = 0.6
BL_WIN = 20
DEEP_K = 3
RATIO_BASE = 0.08
RATIO_SCALE = 0.12
RECOVERY_K = 5

PROMPTS = [
    "The scientist discovered that the ancient",
    "In a world where machines could dream,",
    "The last survivor of the expedition wrote",
    "Deep beneath the ocean surface, a strange",
    "The algorithm predicted something that no one",
    "When the lights went out across the city,",
    "The professor opened the dusty manuscript and",
    "At the edge of the known universe, a signal",
    "The detective noticed a pattern in the data",
    "In the abandoned laboratory, the screens still",
    "The philosopher argued that consciousness was",
    "A single note echoed through the empty hall",
    "The neural network began producing outputs that",
    "Under the microscope, the cells showed signs of",
    "The ancient library contained scrolls describing",
    "When the AI first spoke, it said something",
    "The gravitational wave detector picked up a",
    "In the depths of the archive, a forgotten",
    "The mathematician stared at the proof and",
    "As the spacecraft approached the anomaly,",
    "The historian found evidence that contradicted",
    "In the quantum laboratory, particles behaved",
    "The old bridge creaked under the weight of",
    "She realized the encryption key was hidden in",
    "The volcano had been dormant for centuries until",
    "At exactly midnight, every clock in the building",
    "The surgeon noticed something unusual during the",
    "In the desert, the nomads followed stars that",
    "The economist predicted a crash that would",
    "Below the ice shelf, sonar detected something",
    "The linguist decoded a message that changed",
    "When the dam finally broke, the water",
    "The architect designed a building that could",
    "In the courtroom, the evidence pointed to",
    "The astronomer captured an image that showed",
    "The chef combined ingredients that no one had",
    "Underground, the miners discovered a cavern",
    "The journalist uncovered a conspiracy involving",
    "At the bottom of the ocean, pressure crushed",
    "The painter mixed colors that seemed to glow",
    "In the server room, temperatures spiked as",
    "The biologist observed cells dividing in ways",
    "When the satellite transmitted its final signal",
    "The archaeologist brushed away sand to reveal",
    "In the operating room, the monitors suddenly",
    "The conductor raised the baton and the orchestra",
    "Deep in the forest, the expedition encountered",
    "The programmer found a bug that had been hiding",
    "When the earthquake struck, the city was",
    "The philosopher wrote that truth was merely",
    "The captain steered the ship toward the strange",
    "In the genetics lab, the sequence revealed",
    "The teacher explained that the equation meant",
    "Beneath the cathedral, workers found tunnels",
    "The robot began to exhibit behavior that",
    "In the refugee camp, a child drew pictures of",
    "The physicist calculated that the probability",
    "When the power grid failed, hospitals switched",
    "The novelist wrote a character who could",
    "At the summit, the climbers saw something",
    "The translator struggled with a phrase that",
    "In the bunker, the radio crackled with",
    "The dancer moved in ways that defied",
    "The security camera captured footage of",
    "In the greenhouse, plants grew at rates",
    "The diplomat received a message that would",
    "When the tide receded, it revealed",
    "The engineer realized the bridge design had",
    "In the museum, the artifact began to",
    "The pilot noticed the instruments showing",
    "The psychologist observed a pattern in",
    "When the fog lifted, the village appeared",
    "The chemist mixed the compounds and watched",
    "In the abandoned mine, echoes suggested",
    "The monk meditated until visions of",
    "The hacker bypassed the firewall and found",
    "When the glacier melted, it exposed",
    "The ornithologist tracked a species that",
    "In the courtyard, the statue seemed to",
    "The general received intelligence that changed",
    "The gardener noticed the flowers blooming in",
    "When the bell tolled twelve times, the",
    "The cartographer drew maps of territories",
    "In the wine cellar, behind the oldest bottles",
    "The veterinarian examined the animal and",
    "When the last star faded, the sky",
    "The mechanic found something wedged inside the",
    "In the clock tower, the gears suddenly",
    "The anthropologist lived among the tribe until",
    "The firefighter entered the building knowing",
    "When the letter arrived, its contents",
    "The glassblower created a piece that",
    "In the submarine, depth gauges indicated",
    "The librarian discovered a book that",
    "When the music stopped, everyone in the room",
    "The meteorologist warned that the approaching",
    "In the attic, behind stacks of newspapers",
    "The welder noticed cracks forming in the",
    "When the experiment concluded, the results",
]


# ── Helpers ──
def cosim(a, b):
    if a is None or b is None:
        return 0.0
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def is_crisis(sims):
    if len(sims) < max(DEEP_K, BL_WIN):
        return False
    base = sims[-BL_WIN:]
    thr = np.mean(base) - SIGMA * np.std(base)
    return all(v < thr for v in sims[-DEEP_K:])


def get_ratio(sims, step):
    pre = np.mean(sims[max(0, step - DEEP_K + 1):step + 1])
    be = max(0, step - DEEP_K + 1)
    bs = max(0, be - BL_WIN)
    bl = np.mean(sims[bs:be]) if be > bs else 0.9
    return RATIO_BASE + RATIO_SCALE * max(0, min(1, (bl - pre) / (bl + 1e-8)))


def recovery(sims, step):
    s = step + 1
    e = min(s + RECOVERY_K, len(sims))
    return float(np.mean(sims[s:e])) if s < len(sims) else float("nan")


def sample_token(logits, temperature=1.0, top_p=0.9):
    if temperature < 1e-8:
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        return idx, idx.item()
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[mask] = float("-inf")
    probs = F.softmax(sorted_logits, dim=-1)
    ix = torch.multinomial(probs, 1)
    token_id = sorted_indices.gather(-1, ix)
    return token_id.view(1, 1), token_id.item()


# ── PCA calibration ──
def calibrate_pca(model, tokenizer, n_pca=10):
    cal_text = (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm."
    )
    ids = tokenizer.encode(cal_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(ids, output_hidden_states=True)
    h = out.hidden_states[7][0].float().cpu().numpy()
    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pca = PCA(n_components=min(n_pca, h.shape[1], h.shape[0] - 1))
    pca.fit(h)
    print(f"  PCA calibration: var_explained={pca.explained_variance_ratio_[:5]}")
    return pca.components_


# ── Steering ──
def steer_ortho(h, ctx, ratio):
    proj = torch.sum(ctx * h) / (torch.sum(h * h) + 1e-8)
    d = ctx - proj * h
    dn = d / (torch.norm(d) + 1e-8)
    return h + ratio * torch.norm(h) * dn


def steer_pc(h, pc_vec_np, ratio, sign=+1):
    pc = torch.tensor(pc_vec_np, dtype=h.dtype, device=h.device).view_as(h)
    proj = torch.sum(pc * h) / (torch.sum(h * h) + 1e-8)
    d = pc - proj * h
    dn = d / (torch.norm(d) + 1e-8)
    return h + sign * ratio * torch.norm(h) * dn


# ── Generation ──
def generate(model, tokenizer, hook_out, prompt, seed, strategy, pca_c):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = model.device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prev = None
    ctx = None
    sims = []
    surprisals = []
    n_intv = 0

    for step in range(MAX_TOKENS):
        with torch.inference_mode():
            out = model(ids[:, -MAX_CTX:])
        h = hook_out[0].detach().clone()
        hook_out[0] = None
        sim = cosim(h, prev)
        sims.append(sim)

        if step > 0:
            log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)
            last_tok = ids[0, -1].item()
            surprisals.append(-log_probs[0, last_tok].item())
        else:
            surprisals.append(0.0)

        if step == 0 and strategy == "orthogonal":
            ctx = (h / (torch.norm(h) + 1e-8)).detach()

        do_int = len(sims) > 1 and is_crisis(sims) and strategy != "none"

        if do_int:
            ratio = get_ratio(sims, step)
            if strategy == "orthogonal" and ctx is not None:
                corrected = steer_ortho(h, ctx, ratio)
            elif strategy == "pc3":
                corrected = steer_pc(h, pca_c[2], ratio, sign=+1)
            elif strategy == "pc2":
                corrected = steer_pc(h, pca_c[1], ratio, sign=+1)
            else:
                corrected = h
            with torch.inference_mode():
                normed = model.transformer.ln_f(corrected.unsqueeze(0))
                c_logits = model.lm_head(normed)[0, 0]
            tok, _ = sample_token(c_logits, 1.0, 0.9)
            n_intv += 1
            prev = corrected.detach()
        else:
            logits = out.logits[0, -1]
            tok, _ = sample_token(logits, 1.0, 0.9)
            prev = h.detach()

        tok = tok.to(device)
        ids = torch.cat([ids, tok], dim=-1)
        del out, h
        if step % 30 == 0:
            gc.collect()

    del ids, prev
    gc.collect()
    return {"sims": sims, "surprisals": surprisals, "n_intv": n_intv}


# ── Main ──
def main():
    print("=" * 60)
    print("EXP 6: PC-Directed Steering vs Random Orthogonal")
    print(f"  Model: {MODEL_NAME}, {N_RUNS} runs, {MAX_TOKENS} tokens/run")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    hook_out = [None]
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            hook_out[0] = out[0][:, -1, :]
        else:
            hook_out[0] = out[:, -1, :]
    handle = model.transformer.h[-1].register_forward_hook(hook_fn)

    pca_c = calibrate_pca(model, tokenizer)

    strategies = ["none", "orthogonal", "pc3", "pc2"]
    labels = {
        "none": "Baseline",
        "orthogonal": "Recync (ortho)",
        "pc3": "PC3 (surprisal)",
        "pc2": "PC2 (position)",
    }

    all_results = {s: [] for s in strategies}

    for run in range(N_RUNS):
        prompt = PROMPTS[run % len(PROMPTS)]
        seed = 42 + run
        if run % 10 == 0:
            print(f"\n  Run {run+1}/{N_RUNS}: '{prompt[:40]}...'")

        for s in strategies:
            r = generate(model, tokenizer, hook_out, prompt, seed, s, pca_c)
            all_results[s].append(r)
            if r["n_intv"] and run % 10 == 0:
                print(f"    {s:>12}: {r['n_intv']} interventions")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    handle.remove()

    # ── Analysis ──
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    pir = {s: [] for s in strategies}
    surp_post = {s: [] for s in strategies}

    for run in range(N_RUNS):
        base_sims = all_results["none"][run]["sims"]
        crisis_step = None
        for t in range(DEEP_K, len(base_sims)):
            if is_crisis(base_sims[:t + 1]):
                crisis_step = t
                break
        if crisis_step is None:
            continue

        for s in strategies:
            sims = all_results[s][run]["sims"]
            if crisis_step < len(sims):
                r = recovery(sims, crisis_step)
                if not np.isnan(r):
                    pir[s].append(r)

            surps = all_results[s][run]["surprisals"]
            st = crisis_step + 1
            en = min(st + RECOVERY_K, len(surps))
            if st < len(surps):
                surp_post[s].append(float(np.mean(surps[st:en])))

    nc = len(pir["none"])
    print(f"\n  Crisis runs: {nc}/{N_RUNS}")

    print(f"\n  {'Strategy':>22} {'N':>4} {'Mean PIR':>10} {'Std':>8} {'Mean Surp':>10}")
    print(f"  {'-'*58}")
    for s in strategies:
        if pir[s]:
            ms = np.mean(surp_post[s]) if surp_post[s] else 0
            print(f"  {labels[s]:>22} {len(pir[s]):>4} "
                  f"{np.mean(pir[s]):>10.4f} {np.std(pir[s]):>8.4f} {ms:>10.3f}")

    if nc >= 3:
        print(f"\n  PAIRWISE vs BASELINE:")
        print(f"  {'Strategy':>22} {'Δ PIR':>8} {'Cohen d':>9} {'p':>10} {'Sig':>5}")
        print(f"  {'-'*58}")
        for s in ["orthogonal", "pc3", "pc2"]:
            if pir["none"] and pir[s]:
                n = min(len(pir["none"]), len(pir[s]))
                ba, sa = np.array(pir["none"][:n]), np.array(pir[s][:n])
                d = sa - ba
                md = np.mean(d)
                cd = md / np.std(d) if np.std(d) > 1e-12 else 0
                _, p = ttest_rel(ba, sa)
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
                print(f"  {labels[s]:>22} {md:>+8.4f} {cd:>+9.3f} {p:>10.4f} {sig:>5}")

        # Head-to-head
        if pir["orthogonal"] and pir["pc3"]:
            print(f"\n  HEAD-TO-HEAD: Orthogonal vs PC3:")
            n = min(len(pir["orthogonal"]), len(pir["pc3"]))
            oa, pa = np.array(pir["orthogonal"][:n]), np.array(pir["pc3"][:n])
            d = pa - oa
            md = np.mean(d)
            cd = md / np.std(d) if np.std(d) > 1e-12 else 0
            _, p = ttest_rel(oa, pa)
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            print(f"    Δ={md:+.4f}, d={cd:+.3f}, p={p:.4f} {sig}")
            if md > 0:
                print("    → PC3-directed OUTPERFORMS orthogonal")
            elif md < 0:
                print("    → Orthogonal OUTPERFORMS PC3-directed")
            else:
                print("    → No difference")

        # Head-to-head orthogonal vs pc2
        if pir["orthogonal"] and pir["pc2"]:
            print(f"\n  HEAD-TO-HEAD: Orthogonal vs PC2:")
            n = min(len(pir["orthogonal"]), len(pir["pc2"]))
            oa, pa = np.array(pir["orthogonal"][:n]), np.array(pir["pc2"][:n])
            d = pa - oa
            md = np.mean(d)
            cd = md / np.std(d) if np.std(d) > 1e-12 else 0
            _, p = ttest_rel(oa, pa)
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            print(f"    Δ={md:+.4f}, d={cd:+.3f}, p={p:.4f} {sig}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["gray", "steelblue", "darkorange", "forestgreen"]

    ax = axes[0]
    bp_d = [pir[s] for s in strategies if pir[s]]
    bp_l = [labels[s] for s in strategies if pir[s]]
    bp_c = [colors[i] for i, s in enumerate(strategies) if pir[s]]
    if bp_d:
        bp = ax.boxplot(bp_d, tick_labels=bp_l, patch_artist=True)
        for patch, c in zip(bp["boxes"], bp_c):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.set_ylabel("PIR (mean cosim post-crisis)")
    ax.set_title(f"(a) Post-Intervention Recovery (N={nc} crises)")
    ax.tick_params(axis="x", rotation=15, labelsize=9)

    ax = axes[1]
    means = [np.mean(pir[s]) if pir[s] else 0 for s in strategies]
    ses = [np.std(pir[s]) / np.sqrt(len(pir[s])) if pir[s] else 0 for s in strategies]
    ax.bar(range(4), means, yerr=ses, color=colors, edgecolor="black", capsize=5, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([labels[s] for s in strategies], fontsize=8, rotation=15)
    ax.set_ylabel("Mean PIR ± SE")
    ax.set_title("(b) Mean Recovery by Strategy")

    ax = axes[2]
    surp_means = [np.mean(surp_post[s]) if surp_post[s] else 0 for s in strategies]
    surp_ses = [np.std(surp_post[s]) / np.sqrt(len(surp_post[s])) if surp_post[s] else 0
                for s in strategies]
    ax.bar(range(4), surp_means, yerr=surp_ses, color=colors, edgecolor="black",
           capsize=5, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([labels[s] for s in strategies], fontsize=8, rotation=15)
    ax.set_ylabel("Mean Surprisal (post-crisis)")
    ax.set_title("(c) Post-Crisis Prediction Difficulty")

    plt.suptitle(f"Exp 6: PC-Directed Steering vs Orthogonal (GPT-2, N={N_RUNS} runs, {nc} crises)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("exp6_pc_steering_n100.png", dpi=150)
    plt.show()
    print(f"\n  Figure saved to exp6_pc_steering_n100.png")


main()
