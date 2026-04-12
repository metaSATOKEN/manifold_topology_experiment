#!/usr/bin/env python3
"""
Experiment 2b: Topology Deep Dive

Follow-up to Exp 2 — three key analyses:

1. Threshold sensitivity: How do Betti numbers change across
   significance thresholds (0.01 → 0.5)?
   If beta_1=2 is robust across thresholds, it's real.
   If it appears only at one threshold, it's noise.

2. Persistence barcode inspection: Plot the actual barcodes for
   GPT-2 PCA-2D. Long bars = real topological features.
   Short bars = noise. The gap between "long" and "short"
   bars is the key diagnostic.

3. Point cloud visualization: Direct 2D scatter of GPT-2 PCA
   projections, with convex hull and density, to visually
   check if the trajectory wraps around a hole (torus) or
   stays in a simply-connected region (disk/sphere).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)


# ── 1. Threshold sensitivity ───────────────────────────────────────────────

def threshold_sensitivity(
    diagrams: list[np.ndarray],
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Compute Betti numbers across a range of significance thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 50)

    results = {dim: [] for dim in range(len(diagrams))}

    for thr in thresholds:
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                results[dim].append(0)
                continue
            finite_mask = np.isfinite(dgm[:, 1])
            finite = dgm[finite_mask]
            n_inf = int(np.sum(~finite_mask))
            if len(finite) == 0:
                results[dim].append(n_inf)
                continue
            persistences = finite[:, 1] - finite[:, 0]
            max_pers = np.max(persistences)
            if max_pers > 0:
                significant = int(np.sum(persistences > thr * max_pers))
            else:
                significant = 0
            results[dim].append(significant + n_inf)

    return {
        "thresholds": thresholds,
        "betti_curves": results,
    }


# ── 2. Barcode analysis ───────────────────────────────────────────────────

def barcode_analysis(diagrams: list[np.ndarray]) -> dict:
    """
    Analyse persistence barcodes: identify long bars vs noise.
    Compute the persistence gap (ratio of longest to 2nd-longest bar).
    """
    results = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            results[dim] = {
                "persistences": np.array([]),
                "gap_ratio": 0.0,
                "n_features": 0,
            }
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite = dgm[finite_mask]

        if len(finite) == 0:
            results[dim] = {
                "persistences": np.array([]),
                "gap_ratio": 0.0,
                "n_features": int(np.sum(~finite_mask)),
            }
            continue

        persistences = np.sort(finite[:, 1] - finite[:, 0])[::-1]

        # Gap ratio: how much does the top feature stand out?
        if len(persistences) >= 2 and persistences[1] > 1e-12:
            gap_ratio = float(persistences[0] / persistences[1])
        else:
            gap_ratio = float("inf") if len(persistences) >= 1 else 0.0

        # Top-k persistences
        top_k = persistences[:10]

        results[dim] = {
            "persistences": persistences,
            "top_k": top_k,
            "gap_ratio": gap_ratio,
            "n_features": len(persistences),
            "births_deaths": finite,
        }

    return results


# ── Main analysis ──────────────────────────────────────────────────────────

def run_deep_dive(
    model_key: str,
    text_key: str,
    text: str,
) -> dict:
    """Run full deep dive on one model+text combo."""
    trajectories = extract_hidden_states(model_key, text)
    n_layers = MODELS[model_key]["n_layers"]
    mid = n_layers // 2
    mid_idx = min(trajectories.keys(), key=lambda x: abs(x - mid))
    traj = trajectories[mid_idx]

    # PCA-2D
    pca2 = PCA(n_components=2)
    proj2 = pca2.fit_transform(traj)
    var2 = pca2.explained_variance_ratio_

    # Normalize for ripser
    scaler = StandardScaler()
    proj2_norm = scaler.fit_transform(proj2)

    # Ripser
    rips = ripser(proj2_norm, maxdim=2)
    diagrams = rips["dgms"]

    # Analyses
    thr_sens = threshold_sensitivity(diagrams)
    barcodes = barcode_analysis(diagrams)

    return {
        "layer": mid_idx,
        "shape": traj.shape,
        "proj2": proj2,
        "proj2_norm": proj2_norm,
        "var_explained": var2,
        "diagrams": diagrams,
        "threshold_sensitivity": thr_sens,
        "barcodes": barcodes,
    }


def plot_deep_dive(all_results: dict, save_path: str):
    """
    Big figure: 4 models x 4 columns.
    Cols: (a) point cloud, (b) barcode H1, (c) threshold sensitivity, (d) persistence histogram H1
    """
    model_keys = list(all_results.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 4, figsize=(22, 5.5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    colors_dim = {0: "steelblue", 1: "darkorange", 2: "forestgreen"}

    for row, model_key in enumerate(model_keys):
        ln_type = MODELS[model_key]["ln_type"]
        # Use narrative
        data = all_results[model_key]["narrative"]
        proj = data["proj2"]
        diagrams = data["diagrams"]
        barcodes = data["barcodes"]
        thr_sens = data["threshold_sensitivity"]
        var_exp = data["var_explained"]

        # ── (a) Point cloud with trajectory ──
        ax = axes[row, 0]
        T = proj.shape[0]
        # Color by token position
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        # Draw trajectory as line segments
        segments = np.array([[proj[i], proj[i + 1]] for i in range(T - 1)])
        lc = LineCollection(segments, colors=colors[:-1], linewidths=0.8, alpha=0.6)
        ax.add_collection(lc)
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=15, zorder=5, edgecolors="none")
        # Mark start and end
        ax.scatter(*proj[0], s=100, c="red", marker="^", zorder=10, label="Start")
        ax.scatter(*proj[-1], s=100, c="blue", marker="v", zorder=10, label="End")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.2%})")
        ax.set_title(f"{model_key} ({ln_type}) — PCA-2D Trajectory")
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

        # ── (b) Persistence barcode (H0 + H1) ──
        ax = axes[row, 1]
        y_pos = 0
        for dim in [0, 1]:
            if dim not in barcodes or len(barcodes[dim].get("births_deaths", [])) == 0:
                continue
            bd = barcodes[dim]["births_deaths"]
            persistences = bd[:, 1] - bd[:, 0]
            order = np.argsort(-persistences)

            # Show top 20
            for idx in order[:20]:
                birth, death = bd[idx]
                if not np.isfinite(death):
                    continue
                pers = death - birth
                ax.barh(
                    y_pos, pers, left=birth,
                    height=0.8, color=colors_dim.get(dim, "gray"),
                    alpha=0.8, edgecolor="none",
                )
                y_pos += 1
            y_pos += 2

        ax.set_xlabel("Filtration value")
        ax.set_title(f"{model_key} — Barcode (H0=blue, H1=orange)")
        ax.invert_yaxis()

        # Annotate H1 gap ratio
        if 1 in barcodes and barcodes[1]["n_features"] > 0:
            gap = barcodes[1]["gap_ratio"]
            top = barcodes[1]["top_k"]
            txt = f"H1 gap ratio: {gap:.2f}\n"
            txt += f"Top H1 persistences:\n"
            for i, p in enumerate(top[:5]):
                txt += f"  #{i+1}: {p:.4f}\n"
            ax.text(
                0.98, 0.02, txt,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            )

        # ── (c) Threshold sensitivity ──
        ax = axes[row, 2]
        thrs = thr_sens["thresholds"]
        for dim in [0, 1, 2]:
            if dim in thr_sens["betti_curves"]:
                vals = thr_sens["betti_curves"][dim]
                ax.plot(thrs, vals, color=colors_dim.get(dim, "gray"),
                        lw=2, label=f"β{dim}")
        ax.set_xlabel("Significance threshold (fraction of max persistence)")
        ax.set_ylabel("Betti number")
        ax.set_title(f"{model_key} — β vs Threshold")
        ax.legend(fontsize=8)
        # Mark key thresholds
        for t_mark in [0.1, 0.2]:
            ax.axvline(t_mark, color="gray", ls=":", alpha=0.5)

        # ── (d) H1 persistence histogram ──
        ax = axes[row, 3]
        if 1 in barcodes and len(barcodes[1]["persistences"]) > 0:
            pers = barcodes[1]["persistences"]
            ax.hist(pers, bins=30, color="darkorange", edgecolor="black", alpha=0.7)
            ax.axvline(np.max(pers) * 0.1, color="red", ls="--", lw=1.5,
                       label=f"10% threshold ({np.max(pers)*0.1:.4f})")
            ax.axvline(np.max(pers) * 0.2, color="darkred", ls="--", lw=1.5,
                       label=f"20% threshold ({np.max(pers)*0.2:.4f})")
            ax.set_xlabel("Persistence (death - birth)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

            # How many survive each threshold?
            n_above_10 = int(np.sum(pers > np.max(pers) * 0.1))
            n_above_20 = int(np.sum(pers > np.max(pers) * 0.2))
            ax.set_title(
                f"{model_key} — H1 Persistence Distribution\n"
                f">{10}%: {n_above_10} features | >{20}%: {n_above_20} features"
            )
        else:
            ax.text(0.5, 0.5, "No H1 features", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{model_key} — H1 Persistence Distribution")

    plt.suptitle(
        "Experiment 2b: Topology Deep Dive\n"
        "Is GPT-2's β₁=2 real or noise? Check barcode gaps and threshold robustness.",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Figure saved to {save_path}")
    plt.close()


def plot_all_texts_pointcloud(all_results: dict, save_path: str):
    """
    Separate figure: GPT-2 point clouds for all 3 text types side by side.
    """
    if "GPT-2" not in all_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    text_keys = ["narrative", "technical", "repetitive"]

    for col, text_key in enumerate(text_keys):
        ax = axes[col]
        data = all_results["GPT-2"][text_key]
        proj = data["proj2"]
        var_exp = data["var_explained"]
        T = proj.shape[0]

        # Trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        segments = np.array([[proj[i], proj[i + 1]] for i in range(T - 1)])
        lc = LineCollection(segments, colors=colors[:-1], linewidths=1.0, alpha=0.7)
        ax.add_collection(lc)
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=20, zorder=5, edgecolors="none")
        ax.scatter(*proj[0], s=120, c="red", marker="^", zorder=10, label="Start")
        ax.scatter(*proj[-1], s=120, c="blue", marker="v", zorder=10, label="End")

        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.2%})")
        ax.set_title(f"GPT-2 — {text_key}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

        # Annotate Betti
        barcodes = data["barcodes"]
        b0 = len(data["diagrams"][0]) if len(data["diagrams"]) > 0 else 0
        b1_features = barcodes.get(1, {})
        b1_top = b1_features.get("top_k", [])
        gap = b1_features.get("gap_ratio", 0)
        txt = f"β₁ features (top): {b1_top[:3]}\nGap ratio: {gap:.2f}"
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    plt.suptitle(
        "GPT-2 PCA-2D Trajectories — Does the path wrap around a hole?",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 2b: Topology Deep Dive")
    print("  - Threshold sensitivity analysis")
    print("  - Persistence barcode inspection")
    print("  - Point cloud visualization")
    print("=" * 70)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key in MODELS:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_key} ({MODELS[model_key]['ln_type']})")
        print(f"{'─' * 60}")

        all_results[model_key] = {}

        for text_key, text in SAMPLE_TEXTS.items():
            print(f"\n  Text: {text_key}")
            data = run_deep_dive(model_key, text_key, text)
            all_results[model_key][text_key] = data

            print(f"  Layer {data['layer']}, shape={data['shape']}")
            print(f"  PCA var explained: {data['var_explained']}")

            # Print barcode summary
            for dim in [0, 1]:
                bc = data["barcodes"].get(dim, {})
                if bc.get("n_features", 0) > 0:
                    top = bc.get("top_k", [])[:5]
                    gap = bc.get("gap_ratio", 0)
                    print(f"    H{dim}: {bc['n_features']} features, "
                          f"gap_ratio={gap:.2f}, "
                          f"top={[f'{p:.4f}' for p in top]}")

            # Threshold sensitivity for H1
            thr_sens = data["threshold_sensitivity"]
            b1_curve = thr_sens["betti_curves"].get(1, [])
            thrs = thr_sens["thresholds"]
            if len(b1_curve) > 0:
                # Find threshold range where beta_1 = 2
                b1_arr = np.array(b1_curve)
                mask_2 = b1_arr == 2
                if np.any(mask_2):
                    t_range = thrs[mask_2]
                    print(f"    β₁=2 stable for threshold ∈ [{t_range[0]:.3f}, {t_range[-1]:.3f}]")
                else:
                    print(f"    β₁ never equals 2 at any threshold")

                mask_0 = b1_arr == 0
                if np.any(mask_0):
                    t_start = thrs[mask_0][0]
                    print(f"    β₁=0 (contractible) above threshold {t_start:.3f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTIC: H1 Gap Ratio")
    print("=" * 70)
    print("Gap ratio = (longest H1 bar) / (2nd longest H1 bar)")
    print("  > 3.0 : Strong signal (top feature clearly stands out)")
    print("  1.5-3 : Moderate (possible real feature)")
    print("  < 1.5 : Weak (probably noise)\n")

    print(f"{'Model':>12} {'Text':>12} {'LN':>8} "
          f"{'H1 gap':>8} {'H1 #feat':>9} {'β₁=2 range':>18}")
    print("-" * 75)
    for model_key in all_results:
        for text_key in all_results[model_key]:
            data = all_results[model_key][text_key]
            bc = data["barcodes"].get(1, {})
            gap = bc.get("gap_ratio", 0)
            n_feat = bc.get("n_features", 0)

            # beta_1=2 threshold range
            thr_sens = data["threshold_sensitivity"]
            b1_curve = np.array(thr_sens["betti_curves"].get(1, []))
            thrs = thr_sens["thresholds"]
            mask_2 = b1_curve == 2
            if np.any(mask_2):
                t_range = f"[{thrs[mask_2][0]:.3f}, {thrs[mask_2][-1]:.3f}]"
            else:
                t_range = "never"

            print(
                f"{model_key:>12} {text_key:>12} "
                f"{MODELS[model_key]['ln_type']:>8} "
                f"{gap:>8.2f} {n_feat:>9} {t_range:>18}"
            )

    # ── Plots ──
    plot_deep_dive(all_results, save_path=str(out_dir / "exp2b_deep_dive.png"))
    plot_all_texts_pointcloud(all_results, save_path=str(out_dir / "exp2b_gpt2_pointclouds.png"))

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(
        "1. BARCODE: Look at the H1 (orange) bars in the barcode plot.\n"
        "   If there are 2 bars MUCH longer than the rest → real torus.\n"
        "   If all bars are roughly the same length → noise.\n\n"
        "2. THRESHOLD SENSITIVITY: Check the β₁ vs threshold curve.\n"
        "   If β₁=2 persists over a wide threshold range → robust.\n"
        "   If β₁=2 only at one specific threshold → fragile/noise.\n\n"
        "3. POINT CLOUD: Look at the GPT-2 trajectory.\n"
        "   If it wraps around a visible hole → torus evidence.\n"
        "   If it's a dense cluster or arc → disk/sphere.\n\n"
        "4. GAP RATIO: The most compact diagnostic.\n"
        "   H1 gap > 3 + β₁=2 over wide range = real torus.\n"
        "   Anything else = probably not a clean torus."
    )


if __name__ == "__main__":
    main()
