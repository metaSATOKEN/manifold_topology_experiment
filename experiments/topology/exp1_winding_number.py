#!/usr/bin/env python3
"""
Experiment 1: Winding Number Analysis

Torus vs Sphere discrimination via phase winding.

Key idea:
  - On a torus (T^2), trajectories can wind around two independent cycles.
    The phase wraps around [-pi, pi] with net winding number != 0.
  - On a sphere (S^2), trajectories oscillate but cannot wind around
    a non-contractible loop. Net winding number ~ 0.

Method:
  1. Project hidden states to PCA 2D.
  2. Compute phase theta(t) = atan2(y, x) for each token.
  3. Unwrap the phase to get the continuous trajectory.
  4. Count the total winding number: W = (theta_unwrapped[-1] - theta_unwrapped[0]) / (2*pi).
  5. Also compute per-window winding to check local wrapping behavior.

If W >> 0 (or << 0) consistently across texts and layers, the manifold
has a non-contractible cycle -> torus-like.
If W ~ 0, the trajectory stays in a contractible region -> sphere-like.

We also decompose into PC1 and PC2 independently by projecting onto
higher PCA dimensions (3D -> two independent 2D planes) to check
for two independent winding directions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)


def compute_winding_number(phases: np.ndarray) -> float:
    """
    Total winding number from a sequence of phases.
    W = sum of angular increments / (2*pi).
    """
    unwrapped = np.unwrap(phases)
    return (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)


def compute_windowed_winding(
    phases: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """
    Winding number in sliding windows.
    Returns array of local winding numbers.
    """
    unwrapped = np.unwrap(phases)
    local_w = []
    for end in range(window_size, len(unwrapped) + 1):
        seg = unwrapped[end - window_size : end]
        w = (seg[-1] - seg[0]) / (2 * np.pi)
        local_w.append(w)
    return np.array(local_w)


def analyse_winding(trajectory: np.ndarray, n_pca: int = 3) -> dict:
    """
    Analyse winding behavior on PCA-projected hidden states.

    We project to 3D PCA, then examine winding in:
      - Plane (PC1, PC2): the primary 2D manifold
      - Plane (PC1, PC3): potential second independent cycle
      - Plane (PC2, PC3): potential second independent cycle

    For a torus, we expect non-zero winding in at least two
    independent planes.
    """
    T, d = trajectory.shape
    k = min(n_pca, d, T)
    pca = PCA(n_components=k)
    projected = pca.fit_transform(trajectory)  # (T, k)
    var_explained = pca.explained_variance_ratio_

    results = {
        "var_explained": var_explained,
        "T": T,
    }

    planes = {"PC1-PC2": (0, 1)}
    if k >= 3:
        planes["PC1-PC3"] = (0, 2)
        planes["PC2-PC3"] = (1, 2)

    for plane_name, (i, j) in planes.items():
        phases = np.arctan2(projected[:, j], projected[:, i])
        total_w = compute_winding_number(phases)
        local_w = compute_windowed_winding(phases)

        # Phase wrapping events: times the raw phase jumps by > pi
        dphase = np.diff(phases)
        wraps = np.sum(np.abs(dphase) > np.pi)

        results[plane_name] = {
            "phases": phases,
            "unwrapped": np.unwrap(phases),
            "total_winding": total_w,
            "local_winding": local_w,
            "abs_total_winding": abs(total_w),
            "n_wraps": int(wraps),
            "max_local_winding": float(np.max(np.abs(local_w))),
            "projected": projected[:, [i, j]],
        }

    return results


def plot_results(
    all_data: dict,
    save_path: str,
):
    """
    Plot winding analysis for all models and texts.

    Layout: rows = models, cols = [trajectory in PC1-PC2, unwrapped phase, winding summary]
    """
    model_keys = list(all_data.keys())
    text_keys = list(next(iter(all_data.values())).keys())

    fig, axes = plt.subplots(
        len(model_keys), 3, figsize=(18, 5 * len(model_keys))
    )
    if len(model_keys) == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        ln_type = MODELS[model_key]["ln_type"]

        # (a) 2D trajectories colored by token position
        ax = axes[row, 0]
        for text_key in text_keys:
            data = all_data[model_key][text_key]
            plane = data["PC1-PC2"]
            proj = plane["projected"]
            ax.plot(
                proj[:, 0], proj[:, 1],
                ".-", markersize=2, lw=0.5, alpha=0.7,
                label=f"{text_key} (W={plane['total_winding']:.2f})",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model_key} ({ln_type}) — Trajectory in PC1-PC2")
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

        # (b) Unwrapped phase
        ax = axes[row, 1]
        for text_key in text_keys:
            data = all_data[model_key][text_key]
            plane = data["PC1-PC2"]
            ax.plot(
                plane["unwrapped"],
                lw=1, alpha=0.8,
                label=f"{text_key} (W={plane['total_winding']:.2f})",
            )
        ax.axhline(0, color="black", lw=0.5)
        for k in range(-5, 6):
            ax.axhline(k * 2 * np.pi, color="gray", lw=0.3, ls=":")
        ax.set_xlabel("Token position")
        ax.set_ylabel("Unwrapped phase (rad)")
        ax.set_title(f"{model_key} — Unwrapped Phase (PC1-PC2)")
        ax.legend(fontsize=7)

        # (c) Winding summary across planes
        ax = axes[row, 2]
        plane_names = ["PC1-PC2", "PC1-PC3", "PC2-PC3"]
        x = np.arange(len(text_keys))
        width = 0.25
        for pi, pname in enumerate(plane_names):
            vals = []
            for text_key in text_keys:
                data = all_data[model_key][text_key]
                if pname in data:
                    vals.append(data[pname]["abs_total_winding"])
                else:
                    vals.append(0.0)
            ax.bar(
                x + pi * width, vals, width,
                label=pname, edgecolor="black",
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels(text_keys, fontsize=9)
        ax.set_ylabel("|Winding Number|")
        ax.set_title(f"{model_key} — |W| by Plane")
        ax.legend(fontsize=7)
        ax.axhline(0.5, ls="--", color="red", alpha=0.4, label="0.5 threshold")

    plt.suptitle(
        "Experiment 1: Winding Number Analysis\n"
        "Torus → |W| >> 0 in multiple planes; Sphere → |W| ≈ 0",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 1: Winding Number Analysis (Torus vs Sphere)")
    print("=" * 70)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for model_key in MODELS:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_key} ({MODELS[model_key]['ln_type']})")
        print(f"{'─' * 60}")

        all_data[model_key] = {}

        for text_key, text in SAMPLE_TEXTS.items():
            print(f"\n  Text: {text_key}")
            trajectories = extract_hidden_states(model_key, text)
            n_layers = MODELS[model_key]["n_layers"]

            # Use mid-layer
            mid = n_layers // 2
            mid_idx = min(trajectories.keys(), key=lambda x: abs(x - mid))
            traj = trajectories[mid_idx]
            print(f"  Using layer {mid_idx}, shape={traj.shape}")

            result = analyse_winding(traj)
            all_data[model_key][text_key] = result

            # Print summary
            for pname in ["PC1-PC2", "PC1-PC3", "PC2-PC3"]:
                if pname in result:
                    p = result[pname]
                    print(
                        f"    {pname}: W={p['total_winding']:+.3f}, "
                        f"|W|={p['abs_total_winding']:.3f}, "
                        f"wraps={p['n_wraps']}, "
                        f"max_local={p['max_local_winding']:.3f}"
                    )

            ve = result["var_explained"]
            print(f"    PCA var explained: {ve[0]:.1%}, {ve[1]:.1%}" +
                  (f", {ve[2]:.1%}" if len(ve) > 2 else ""))

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':>12} {'Text':>12} {'LN':>8} "
          f"{'W(1-2)':>8} {'W(1-3)':>8} {'W(2-3)':>8} "
          f"{'Wraps':>6}")
    print("-" * 72)
    for model_key in all_data:
        for text_key in all_data[model_key]:
            r = all_data[model_key][text_key]
            w12 = r["PC1-PC2"]["total_winding"]
            w13 = r.get("PC1-PC3", {}).get("total_winding", 0)
            w23 = r.get("PC2-PC3", {}).get("total_winding", 0)
            wraps = r["PC1-PC2"]["n_wraps"]
            print(
                f"{model_key:>12} {text_key:>12} "
                f"{MODELS[model_key]['ln_type']:>8} "
                f"{w12:>+8.3f} {w13:>+8.3f} {w23:>+8.3f} "
                f"{wraps:>6}"
            )

    # ── Interpretation ──
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "If |W| >> 0 consistently in PC1-PC2 plane:\n"
        "  -> Trajectories wind around a non-contractible cycle\n"
        "  -> Manifold has S^1 topology in that direction (torus-like)\n\n"
        "If |W| >> 0 in TWO independent planes:\n"
        "  -> Two independent non-contractible cycles exist\n"
        "  -> Strong evidence for torus (T^2) topology\n\n"
        "If |W| ~ 0 across all planes:\n"
        "  -> No winding; trajectories oscillate in a contractible region\n"
        "  -> Consistent with sphere (S^2) or disk topology\n\n"
        "Note: winding number is a necessary but not sufficient condition.\n"
        "Persistent homology (Exp 2) provides definitive Betti number evidence."
    )

    plot_results(all_data, save_path=str(out_dir / "exp1_winding_number.png"))


if __name__ == "__main__":
    main()
