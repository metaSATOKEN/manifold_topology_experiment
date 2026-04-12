#!/usr/bin/env python3
"""
Experiment 3: Local Gaussian Curvature Estimation

Differential-geometric approach to torus vs sphere discrimination.

Gauss-Bonnet theorem:
  ∫∫ K dA = 2π * χ(M)
where K = Gaussian curvature, χ = Euler characteristic.

  S^2: χ = 2, so ∫∫ K dA = 4π, K > 0 everywhere
  T^2: χ = 0, so ∫∫ K dA = 0, K has positive AND negative regions
  Flat torus: K = 0 everywhere

Method:
  1. Project hidden states to PCA 2D (or 3D for embedding curvature).
  2. For each point, estimate local Gaussian curvature using
     angular deficit method on k-nearest neighbor Delaunay triangulation.
  3. Analyse the distribution of curvature values:
     - All positive -> sphere-like
     - Mixed positive/negative summing to ~0 -> torus-like
     - All near zero -> flat torus
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)


def estimate_curvature_angular_deficit(
    points: np.ndarray,
    k_neighbors: int = 8,
) -> np.ndarray:
    """
    Estimate Gaussian curvature at each point using the angular deficit method.

    For each point p, find its k nearest neighbors, project them onto the
    local tangent plane, and compute the angular deficit:
        K(p) ≈ (2π - sum_of_angles) / A_local

    where sum_of_angles is the total angle at p in the Delaunay triangulation
    of the neighborhood, and A_local is the local area.

    This is a discrete approximation of Gaussian curvature.
    """
    N, d = points.shape
    curvatures = np.zeros(N)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N))
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    for i in range(N):
        neighbor_idx = indices[i, 1:]  # exclude self
        neighbors = points[neighbor_idx]
        center = points[i]

        # Vectors from center to neighbors
        vecs = neighbors - center
        dists = np.linalg.norm(vecs, axis=1)

        if np.min(dists) < 1e-12:
            curvatures[i] = 0.0
            continue

        # For 2D points, compute angular deficit directly
        if d == 2:
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            angles_sorted = np.sort(angles)
            # Consecutive angle differences
            diffs = np.diff(angles_sorted)
            diffs = np.append(diffs, 2 * np.pi + angles_sorted[0] - angles_sorted[-1])
            # Angular deficit = 2π - sum of sector angles = 0 for flat
            # We use a different approach: the angle sum in triangulation
            angle_sum = compute_angle_sum_2d(center, neighbors)
            deficit = 2 * np.pi - angle_sum
            # Local area (Voronoi area estimate)
            local_area = np.pi * np.mean(dists) ** 2 / len(neighbors)
            if local_area > 1e-12:
                curvatures[i] = deficit / local_area
            else:
                curvatures[i] = 0.0
        else:
            # For 3D+ points, project to local tangent plane
            angle_sum = compute_angle_sum_nd(center, neighbors)
            deficit = 2 * np.pi - angle_sum
            local_area = np.pi * np.mean(dists) ** 2 / len(neighbors)
            if local_area > 1e-12:
                curvatures[i] = deficit / local_area
            else:
                curvatures[i] = 0.0

    return curvatures


def compute_angle_sum_2d(center: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Compute the sum of angles around center formed by consecutive neighbors
    (sorted by angle).
    """
    vecs = neighbors - center
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    order = np.argsort(angles)
    sorted_vecs = vecs[order]

    total_angle = 0.0
    n = len(sorted_vecs)
    for j in range(n):
        v1 = sorted_vecs[j]
        v2 = sorted_vecs[(j + 1) % n]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_a = np.clip(cos_a, -1, 1)
        total_angle += np.arccos(cos_a)

    return total_angle


def compute_angle_sum_nd(center: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Compute the sum of angles at center in nD using local PCA projection to 2D.
    """
    vecs = neighbors - center
    # Project to 2D via local PCA
    if vecs.shape[0] < 2 or vecs.shape[1] < 2:
        return 2 * np.pi  # flat
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(vecs)
        # Now compute 2D angle sum
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        order = np.argsort(angles)
        sorted_proj = proj[order]

        total = 0.0
        n = len(sorted_proj)
        for j in range(n):
            v1 = sorted_proj[j]
            v2 = sorted_proj[(j + 1) % n]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            cos_a = np.clip(cos_a, -1, 1)
            total += np.arccos(cos_a)
        return total
    except Exception:
        return 2 * np.pi


def analyse_curvature(
    trajectory: np.ndarray,
    pca_dims: list[int] = [2, 3],
    k_neighbors: int = 8,
) -> dict:
    """Full curvature analysis."""
    results = {}

    for k in pca_dims:
        k_actual = min(k, trajectory.shape[1], trajectory.shape[0] - 1)
        pca = PCA(n_components=k_actual)
        projected = pca.fit_transform(trajectory)
        var_exp = pca.explained_variance_ratio_

        curvatures = estimate_curvature_angular_deficit(
            projected, k_neighbors=k_neighbors
        )

        # Curvature statistics
        K_pos = np.sum(curvatures > 0)
        K_neg = np.sum(curvatures < 0)
        K_zero = np.sum(np.abs(curvatures) < 1e-8)
        K_integral = float(np.sum(curvatures))  # ∝ 2πχ
        K_mean = float(np.mean(curvatures))
        K_std = float(np.std(curvatures))

        # Euler characteristic estimate
        # ∫K dA ≈ sum(K_i * A_i) ≈ 2πχ
        # We use a rough estimate
        chi_estimate = K_integral / (2 * np.pi) if abs(K_integral) > 1e-12 else 0.0

        # Classification
        if K_neg == 0 and K_pos > N_threshold(len(curvatures)):
            topology = "S^2-like (all positive curvature)"
        elif K_pos > 0 and K_neg > 0 and abs(K_mean) < K_std * 0.5:
            topology = "T^2-like (mixed curvature, mean ≈ 0)"
        elif abs(K_mean) < 1e-6 and K_std < 1e-6:
            topology = "Flat (K ≈ 0)"
        else:
            topology = f"Ambiguous (mean={K_mean:.4f}, std={K_std:.4f})"

        results[f"PCA-{k}"] = {
            "projected": projected,
            "var_explained": var_exp,
            "curvatures": curvatures,
            "K_pos": int(K_pos),
            "K_neg": int(K_neg),
            "K_zero": int(K_zero),
            "K_integral": K_integral,
            "K_mean": K_mean,
            "K_std": K_std,
            "chi_estimate": chi_estimate,
            "topology": topology,
        }

    return results


def N_threshold(n: int) -> int:
    """Minimum number of positive curvature points to classify as sphere."""
    return max(1, int(n * 0.7))


def plot_results(all_data: dict, save_path: str):
    """Plot curvature analysis."""
    model_keys = list(all_data.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        ln_type = MODELS[model_key]["ln_type"]
        data = all_data[model_key]["narrative"]
        pca_key = "PCA-2"
        pca_data = data[pca_key]

        projected = pca_data["projected"]
        curvatures = pca_data["curvatures"]

        # (a) 2D scatter colored by curvature
        ax = axes[row, 0]
        vmax = np.percentile(np.abs(curvatures), 95)
        sc = ax.scatter(
            projected[:, 0], projected[:, 1],
            c=curvatures, cmap="RdBu_r", s=30,
            vmin=-vmax, vmax=vmax, edgecolors="gray", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label="Gaussian curvature K")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model_key} ({ln_type}) — Curvature Map")
        ax.set_aspect("equal", adjustable="datalim")

        # (b) Curvature distribution
        ax = axes[row, 1]
        ax.hist(curvatures, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.axvline(np.mean(curvatures), color="darkorange", ls="-", lw=2,
                   label=f"mean={np.mean(curvatures):.4f}")
        ax.set_xlabel("Gaussian curvature K")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_key} — Curvature Distribution")
        ax.legend()

        # Annotate
        ax.text(
            0.98, 0.95,
            f"K>0: {pca_data['K_pos']}, K<0: {pca_data['K_neg']}\n"
            f"χ ≈ {pca_data['chi_estimate']:.2f}\n"
            f"{pca_data['topology']}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # (c) Cross-text comparison
        ax = axes[row, 2]
        text_keys = list(all_data[model_key].keys())
        x = np.arange(len(text_keys))
        means = []
        stds = []
        pos_fracs = []
        for tk in text_keys:
            d = all_data[model_key][tk][pca_key]
            means.append(d["K_mean"])
            stds.append(d["K_std"])
            n = len(d["curvatures"])
            pos_fracs.append(d["K_pos"] / n if n > 0 else 0)

        ax.bar(x - 0.2, means, 0.4, label="Mean K", color="steelblue", edgecolor="black")
        ax.bar(x + 0.2, pos_fracs, 0.4, label="Frac K>0", color="darkorange", edgecolor="black")
        ax.errorbar(x - 0.2, means, yerr=stds, fmt="none", color="black", capsize=3)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(text_keys, fontsize=9)
        ax.set_ylabel("Value")
        ax.set_title(f"{model_key} — Curvature by Text Type")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Experiment 3: Gaussian Curvature Estimation\n"
        "S^2: K > 0 everywhere (χ=2) | T^2: K mixed, ∫K=0 (χ=0) | Flat: K ≈ 0",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 3: Gaussian Curvature Estimation")
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

            mid = n_layers // 2
            mid_idx = min(trajectories.keys(), key=lambda x: abs(x - mid))
            traj = trajectories[mid_idx]
            print(f"  Using layer {mid_idx}, shape={traj.shape}")

            result = analyse_curvature(traj)
            all_data[model_key][text_key] = result

            for pca_key in sorted(result.keys()):
                d = result[pca_key]
                print(
                    f"    {pca_key}: K_mean={d['K_mean']:.4f}, "
                    f"K_std={d['K_std']:.4f}, "
                    f"K>0={d['K_pos']}, K<0={d['K_neg']}, "
                    f"χ≈{d['chi_estimate']:.2f}"
                )
                print(f"      → {d['topology']}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (PCA-2)")
    print("=" * 70)
    print(f"{'Model':>12} {'Text':>12} {'LN':>8} "
          f"{'K_mean':>8} {'K_std':>8} {'K>0':>5} {'K<0':>5} "
          f"{'χ':>6} {'Topology':>25}")
    print("-" * 100)
    for model_key in all_data:
        for text_key in all_data[model_key]:
            d = all_data[model_key][text_key]["PCA-2"]
            print(
                f"{model_key:>12} {text_key:>12} "
                f"{MODELS[model_key]['ln_type']:>8} "
                f"{d['K_mean']:>8.4f} {d['K_std']:>8.4f} "
                f"{d['K_pos']:>5} {d['K_neg']:>5} "
                f"{d['chi_estimate']:>6.2f} "
                f"{d['topology']:>25}"
            )

    plot_results(all_data, save_path=str(out_dir / "exp3_gaussian_curvature.png"))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "Gauss-Bonnet theorem constrains the integrated curvature:\n"
        "  ∫∫ K dA = 2πχ(M)\n\n"
        "All K > 0 with χ ≈ 2:\n"
        "  -> Sphere-like manifold\n\n"
        "Mixed K, mean ≈ 0, χ ≈ 0:\n"
        "  -> Torus-like manifold (or Klein bottle)\n\n"
        "K ≈ 0 everywhere:\n"
        "  -> Flat manifold (flat torus, plane, or cylinder)\n\n"
        "Caveat: curvature estimation from sparse point clouds is noisy.\n"
        "This experiment provides supporting evidence, not proof.\n"
        "Persistent homology (Exp 2) is the more reliable discriminator."
    )


if __name__ == "__main__":
    main()
