#!/usr/bin/env python3
"""
Experiment 4: Geodesic vs Euclidean Distance Ratio

Metric-geometric approach to torus vs sphere discrimination.

Key idea:
  On a sphere, geodesic distance grows with Euclidean distance but
  saturates at π*R (the great-circle half-circumference).
  The ratio d_geo / d_euc diverges for antipodal points.

  On a torus, there are multiple short geodesic paths between distant
  points (going "around the hole"). The ratio d_geo / d_euc behaves
  differently — it can decrease for points that are far in Euclidean
  space but close via a wrapping geodesic.

Method:
  1. Project hidden states to PCA 2D.
  2. Build a k-nearest-neighbor graph as a proxy for the manifold.
  3. Compute geodesic distances (shortest paths on the kNN graph).
  4. Compare geodesic vs Euclidean distances:
     - Plot d_geo vs d_euc relationship
     - Compute the ratio profile
     - Check for "shortcut" paths (d_geo << d_euc) that indicate wrapping

Additionally, we compute the "metric distortion profile":
  For each pair of points, compute d_geo / d_euc.
  On a sphere: this ratio increases monotonically with d_euc.
  On a torus: this ratio is non-monotonic (shortcuts exist).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)


def compute_geodesic_distances(
    points: np.ndarray,
    k_neighbors: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise geodesic distances via kNN graph.

    Returns:
        (geodesic_dists: (N,N), euclidean_dists: (N,N))
    """
    N = points.shape[0]

    # Build kNN graph
    graph = kneighbors_graph(
        points, n_neighbors=min(k_neighbors, N - 1),
        mode="distance", include_self=False,
    )
    # Make symmetric
    graph = graph.maximum(graph.T)

    # Shortest paths
    geo_dists = shortest_path(graph, directed=False)

    # Euclidean distances
    from scipy.spatial.distance import cdist
    euc_dists = cdist(points, points)

    return geo_dists, euc_dists


def analyse_distance_ratio(
    trajectory: np.ndarray,
    k_neighbors: int = 6,
    max_points: int = 150,
) -> dict:
    """
    Analyse the geodesic/Euclidean distance ratio profile.
    """
    T, d = trajectory.shape

    # PCA to 2D
    pca = PCA(n_components=min(2, d, T))
    projected = pca.fit_transform(trajectory)
    var_exp = pca.explained_variance_ratio_

    # Subsample for speed
    if T > max_points:
        idx = np.linspace(0, T - 1, max_points, dtype=int)
        points = projected[idx]
    else:
        points = projected

    N = points.shape[0]
    geo_dists, euc_dists = compute_geodesic_distances(points, k_neighbors)

    # Extract upper triangle (unique pairs)
    triu_idx = np.triu_indices(N, k=1)
    geo_flat = geo_dists[triu_idx]
    euc_flat = euc_dists[triu_idx]

    # Filter out infinite geodesic distances (disconnected components)
    finite_mask = np.isfinite(geo_flat) & (euc_flat > 1e-12)
    geo_flat = geo_flat[finite_mask]
    euc_flat = euc_flat[finite_mask]

    if len(geo_flat) == 0:
        return {"error": "No finite geodesic distances"}

    # Ratio
    ratios = geo_flat / euc_flat

    # Bin by Euclidean distance to see the profile
    n_bins = 20
    euc_bins = np.linspace(np.min(euc_flat), np.max(euc_flat), n_bins + 1)
    bin_means = []
    bin_stds = []
    bin_centers = []
    for i in range(n_bins):
        mask = (euc_flat >= euc_bins[i]) & (euc_flat < euc_bins[i + 1])
        if np.sum(mask) > 2:
            bin_means.append(np.mean(ratios[mask]))
            bin_stds.append(np.std(ratios[mask]))
            bin_centers.append((euc_bins[i] + euc_bins[i + 1]) / 2)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_centers = np.array(bin_centers)

    # Monotonicity test: is the ratio profile monotonically increasing?
    if len(bin_means) >= 3:
        diffs = np.diff(bin_means)
        n_increasing = np.sum(diffs > 0)
        n_decreasing = np.sum(diffs < 0)
        monotonicity = n_increasing / (n_increasing + n_decreasing + 1e-12)
    else:
        monotonicity = 0.5

    # Shortcut detection: pairs where geo << euc (ratio < 1 shouldn't happen
    # on a convex manifold, but on a torus you can go "around")
    # Actually ratio >= 1 always for geodesic, but on a torus the ratio
    # for distant points can be LOWER than for nearby points.
    # Better metric: check if ratio DECREASES at large Euclidean distance.
    ratio_decrease_at_large_d = False
    if len(bin_means) >= 5:
        last_third = bin_means[len(bin_means) * 2 // 3 :]
        first_third = bin_means[: len(bin_means) // 3]
        if np.mean(last_third) < np.mean(first_third) * 1.1:
            ratio_decrease_at_large_d = True

    # Classification
    if monotonicity > 0.7 and not ratio_decrease_at_large_d:
        topology = "Sphere-like (monotonically increasing ratio)"
    elif ratio_decrease_at_large_d:
        topology = "Torus-like (ratio plateaus/decreases at large d)"
    else:
        topology = f"Ambiguous (monotonicity={monotonicity:.2f})"

    return {
        "projected": projected,
        "var_explained": var_exp,
        "geo_flat": geo_flat,
        "euc_flat": euc_flat,
        "ratios": ratios,
        "bin_centers": bin_centers,
        "bin_means": bin_means,
        "bin_stds": bin_stds,
        "monotonicity": monotonicity,
        "ratio_decrease_at_large_d": ratio_decrease_at_large_d,
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "topology": topology,
    }


def plot_results(all_data: dict, save_path: str):
    """Plot geodesic ratio analysis."""
    model_keys = list(all_data.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        ln_type = MODELS[model_key]["ln_type"]

        # Use narrative text
        data = all_data[model_key]["narrative"]
        if "error" in data:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, data["error"],
                                     ha="center", va="center")
            continue

        # (a) Geodesic vs Euclidean scatter
        ax = axes[row, 0]
        ax.scatter(
            data["euc_flat"], data["geo_flat"],
            s=1, alpha=0.3, color="steelblue",
        )
        # Identity line
        max_val = max(np.max(data["euc_flat"]), np.max(data["geo_flat"]))
        ax.plot([0, max_val], [0, max_val], "r--", lw=1, label="d_geo = d_euc")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Geodesic distance")
        ax.set_title(f"{model_key} ({ln_type}) — Geodesic vs Euclidean")
        ax.legend(fontsize=8)

        # (b) Ratio profile (binned)
        ax = axes[row, 1]
        if len(data["bin_centers"]) > 0:
            ax.errorbar(
                data["bin_centers"], data["bin_means"],
                yerr=data["bin_stds"], fmt="o-",
                color="darkorange", markersize=4, capsize=3,
            )
        ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5, label="ratio=1")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("d_geo / d_euc (mean)")
        ax.set_title(f"{model_key} — Ratio Profile")
        ax.legend(fontsize=8)

        # Annotate
        ax.text(
            0.98, 0.95,
            f"Monotonicity: {data['monotonicity']:.2f}\n"
            f"Decrease at large d: {data['ratio_decrease_at_large_d']}\n"
            f"{data['topology']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # (c) Cross-text ratio comparison
        ax = axes[row, 2]
        text_keys = list(all_data[model_key].keys())
        for text_key in text_keys:
            td = all_data[model_key][text_key]
            if "error" in td:
                continue
            if len(td["bin_centers"]) > 0:
                ax.plot(
                    td["bin_centers"], td["bin_means"],
                    "o-", markersize=3, label=text_key,
                )
        ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("d_geo / d_euc")
        ax.set_title(f"{model_key} — Ratio by Text Type")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Experiment 4: Geodesic vs Euclidean Distance Ratio\n"
        "Sphere: ratio monotonically increases | Torus: ratio non-monotonic (shortcuts)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 4: Geodesic vs Euclidean Distance Ratio")
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

            result = analyse_distance_ratio(traj)
            all_data[model_key][text_key] = result

            if "error" in result:
                print(f"    ERROR: {result['error']}")
                continue

            print(
                f"    Mean ratio={result['mean_ratio']:.3f} ± {result['std_ratio']:.3f}"
            )
            print(f"    Monotonicity={result['monotonicity']:.2f}")
            print(f"    Decrease at large d: {result['ratio_decrease_at_large_d']}")
            print(f"    → {result['topology']}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':>12} {'Text':>12} {'LN':>8} "
          f"{'Ratio':>8} {'Mono':>6} {'Decrease':>9} {'Topology':>30}")
    print("-" * 95)
    for model_key in all_data:
        for text_key in all_data[model_key]:
            d = all_data[model_key][text_key]
            if "error" in d:
                print(f"{model_key:>12} {text_key:>12} "
                      f"{MODELS[model_key]['ln_type']:>8} {'ERROR':>8}")
                continue
            print(
                f"{model_key:>12} {text_key:>12} "
                f"{MODELS[model_key]['ln_type']:>8} "
                f"{d['mean_ratio']:>8.3f} "
                f"{d['monotonicity']:>6.2f} "
                f"{str(d['ratio_decrease_at_large_d']):>9} "
                f"{d['topology']:>30}"
            )

    plot_results(all_data, save_path=str(out_dir / "exp4_geodesic_ratio.png"))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "The geodesic/Euclidean ratio profile reveals metric geometry:\n\n"
        "Monotonically increasing ratio (monotonicity > 0.7):\n"
        "  -> Points that are far in Euclidean space are even farther\n"
        "     along the manifold. No shortcuts exist.\n"
        "  -> Consistent with a convex, sphere-like manifold.\n\n"
        "Non-monotonic ratio (ratio decreases at large d):\n"
        "  -> Distant Euclidean points can be reached via short\n"
        "     geodesic paths that 'wrap around' the manifold.\n"
        "  -> Indicates non-trivial topology: torus or other.\n\n"
        "This is complementary to persistent homology (Exp 2).\n"
        "Together, they provide both topological (Betti) and\n"
        "metric (curvature/geodesic) characterization."
    )


if __name__ == "__main__":
    main()
