#!/usr/bin/env python3
"""
Experiment 2: Persistent Homology — Betti Number Estimation

Definitive topological discrimination between S^2 and T^2.

Betti numbers (homology ranks):
  S^2 (sphere):   beta_0=1, beta_1=0, beta_2=1
  T^2 (torus):    beta_0=1, beta_1=2, beta_2=1
  S^1 (circle):   beta_0=1, beta_1=1
  Disk:           beta_0=1, beta_1=0

The critical discriminator is beta_1:
  beta_1 = 0 -> sphere-like (no holes)
  beta_1 = 1 -> circle-like (one hole)
  beta_1 = 2 -> torus-like (two independent holes)

Method:
  1. Project hidden states to PCA 2D (and 3D for comparison).
  2. Compute Vietoris-Rips persistent homology via ripser.
  3. Count significant persistent features (long bars in barcode)
     for H0 (components), H1 (loops), H2 (voids).
  4. Compare PreLN vs PostLN models.

We use a persistence threshold to distinguish signal from noise:
  features with persistence > alpha * max_persistence are counted as
  significant.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.extraction import (
    extract_hidden_states,
    MODELS,
    SAMPLE_TEXTS,
)

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not installed. Install with: pip install ripser")
    print("Falling back to gudhi if available...")
    try:
        import gudhi
        HAS_GUDHI = True
    except ImportError:
        HAS_GUDHI = False
        print("WARNING: Neither ripser nor gudhi available.")


def compute_persistence(
    points: np.ndarray,
    max_dim: int = 2,
    max_points: int = 200,
) -> dict:
    """
    Compute persistent homology of a point cloud.

    Args:
        points: (N, d) point cloud
        max_dim: maximum homology dimension
        max_points: subsample if N > this (for speed)

    Returns:
        dict with 'diagrams' (list of (birth, death) arrays per dimension)
        and 'betti' (estimated Betti numbers)
    """
    N = points.shape[0]

    # Subsample if needed
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        idx.sort()
        points = points[idx]

    # Normalize
    scaler = StandardScaler()
    points = scaler.fit_transform(points)

    if HAS_RIPSER:
        result = ripser(points, maxdim=max_dim)
        diagrams = result["dgms"]
    elif HAS_GUDHI:
        rips = gudhi.RipsComplex(points=points, max_edge_length=np.inf)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()
        diagrams = []
        for dim in range(max_dim + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                diagrams.append(np.array(pairs))
            else:
                diagrams.append(np.empty((0, 2)))
    else:
        raise RuntimeError("No persistent homology library available")

    return {"diagrams": diagrams}


def count_significant_features(
    diagrams: list[np.ndarray],
    significance_threshold: float = 0.1,
) -> dict:
    """
    Count significant persistent features.

    A feature is "significant" if its persistence (death - birth)
    exceeds significance_threshold * max_persistence.
    """
    betti = {}
    persistence_stats = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            betti[dim] = 0
            persistence_stats[dim] = {
                "n_total": 0,
                "n_significant": 0,
                "max_persistence": 0.0,
                "persistences": np.array([]),
            }
            continue

        # Filter out infinite death values for analysis
        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]

        if len(finite_dgm) == 0:
            # Only infinite features (H0 connected component)
            betti[dim] = np.sum(~finite_mask)
            persistence_stats[dim] = {
                "n_total": len(dgm),
                "n_significant": int(np.sum(~finite_mask)),
                "max_persistence": np.inf,
                "persistences": np.array([]),
            }
            continue

        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
        max_pers = float(np.max(persistences))

        if max_pers > 0:
            threshold = significance_threshold * max_pers
            significant = np.sum(persistences > threshold)
        else:
            significant = 0

        # Add infinite features
        n_infinite = int(np.sum(~finite_mask))
        significant += n_infinite

        betti[dim] = int(significant)
        persistence_stats[dim] = {
            "n_total": len(dgm),
            "n_significant": int(significant),
            "max_persistence": max_pers,
            "persistences": persistences,
        }

    return {"betti": betti, "stats": persistence_stats}


def analyse_topology(
    trajectory: np.ndarray,
    pca_dims: list[int] = [2, 3],
    significance_thresholds: list[float] = [0.05, 0.1, 0.2],
) -> dict:
    """
    Full topological analysis of a hidden-state trajectory.
    """
    results = {}

    for k in pca_dims:
        k_actual = min(k, trajectory.shape[1], trajectory.shape[0])
        pca = PCA(n_components=k_actual)
        projected = pca.fit_transform(trajectory)
        var_exp = pca.explained_variance_ratio_

        max_dim = min(k_actual - 1, 2)
        ph = compute_persistence(projected, max_dim=max_dim)

        threshold_results = {}
        for thr in significance_thresholds:
            features = count_significant_features(ph["diagrams"], thr)
            threshold_results[thr] = features

        results[f"PCA-{k}"] = {
            "projected": projected,
            "var_explained": var_exp,
            "diagrams": ph["diagrams"],
            "threshold_results": threshold_results,
        }

    return results


def classify_topology(betti: dict) -> str:
    """
    Classify the topology based on Betti numbers.
    """
    b0 = betti.get(0, 0)
    b1 = betti.get(1, 0)
    b2 = betti.get(2, 0)

    if b1 == 0 and b2 >= 1:
        return "S^2 (sphere)"
    elif b1 == 2 and b2 >= 1:
        return "T^2 (torus)"
    elif b1 == 1 and b2 == 0:
        return "S^1 (circle)"
    elif b1 == 0 and b2 == 0:
        return "Contractible (disk)"
    elif b1 >= 2:
        return f"Higher genus (beta_1={b1})"
    else:
        return f"Unknown (beta=({b0},{b1},{b2}))"


def plot_persistence_diagrams(
    all_data: dict,
    save_path: str,
):
    """
    Plot persistence diagrams and barcodes for all models.
    """
    model_keys = list(all_data.keys())
    n_models = len(model_keys)

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    colors_dim = {0: "steelblue", 1: "darkorange", 2: "forestgreen"}

    for row, model_key in enumerate(model_keys):
        # Use narrative text, PCA-3 (or PCA-2 as fallback)
        text_key = "narrative"
        data = all_data[model_key][text_key]
        pca_key = "PCA-3" if "PCA-3" in data else "PCA-2"
        pca_data = data[pca_key]
        diagrams = pca_data["diagrams"]
        ln_type = MODELS[model_key]["ln_type"]

        # (a) Persistence diagram
        ax = axes[row, 0]
        max_val = 0
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                continue
            finite_mask = np.isfinite(dgm[:, 1])
            finite = dgm[finite_mask]
            if len(finite) > 0:
                ax.scatter(
                    finite[:, 0], finite[:, 1],
                    s=20, alpha=0.6, color=colors_dim.get(dim, "gray"),
                    label=f"H{dim} ({len(finite)} features)",
                )
                max_val = max(max_val, np.max(finite))
        if max_val > 0:
            ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", lw=0.5)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"{model_key} ({ln_type}) — Persistence Diagram\n{pca_key}, {text_key}")
        ax.legend(fontsize=7)

        # (b) Barcode
        ax = axes[row, 1]
        y_pos = 0
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                continue
            finite_mask = np.isfinite(dgm[:, 1])
            finite = dgm[finite_mask]
            if len(finite) == 0:
                continue
            persistences = finite[:, 1] - finite[:, 0]
            order = np.argsort(-persistences)
            for idx in order[:30]:  # top 30
                ax.barh(
                    y_pos, persistences[idx],
                    left=finite[idx, 0],
                    height=0.8,
                    color=colors_dim.get(dim, "gray"),
                    alpha=0.7,
                )
                y_pos += 1
            y_pos += 2  # gap between dimensions
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Feature index")
        ax.set_title(f"{model_key} — Barcode (top features)")

        # (c) Betti number summary
        ax = axes[row, 2]
        thresholds = [0.05, 0.1, 0.2]
        x = np.arange(3)  # H0, H1, H2
        width = 0.25
        for ti, thr in enumerate(thresholds):
            thr_data = pca_data["threshold_results"][thr]
            betti = thr_data["betti"]
            vals = [betti.get(d, 0) for d in range(3)]
            ax.bar(
                x + ti * width, vals, width,
                label=f"threshold={thr}",
                edgecolor="black",
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels(["H0 (components)", "H1 (loops)", "H2 (voids)"])
        ax.set_ylabel("Significant features")
        ax.set_title(f"{model_key} — Betti Numbers")
        ax.legend(fontsize=7)

        # Annotate classification
        mid_thr = pca_data["threshold_results"][0.1]
        classification = classify_topology(mid_thr["betti"])
        ax.text(
            0.98, 0.95, f"Classification: {classification}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )

    plt.suptitle(
        "Experiment 2: Persistent Homology\n"
        "S^2: β=(1,0,1) | T^2: β=(1,2,1) | S^1: β=(1,1,0)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Persistent Homology (Betti Number Estimation)")
    print("=" * 70)

    if not HAS_RIPSER and not HAS_GUDHI:
        print("\nERROR: No persistent homology library available.")
        print("Install with: pip install ripser")
        print("Or: pip install gudhi")
        return

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

            result = analyse_topology(traj)
            all_data[model_key][text_key] = result

            # Print summary
            for pca_key in sorted(result.keys()):
                pca_data = result[pca_key]
                ve = pca_data["var_explained"]
                print(f"    {pca_key}: var_explained={ve[:3]}")

                for thr in [0.1]:
                    thr_data = pca_data["threshold_results"][thr]
                    betti = thr_data["betti"]
                    classification = classify_topology(betti)
                    print(
                        f"      threshold={thr}: "
                        f"β=({betti.get(0,0)}, {betti.get(1,0)}, {betti.get(2,0)}) "
                        f"→ {classification}"
                    )

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (threshold=0.1, PCA-3)")
    print("=" * 70)
    print(f"{'Model':>12} {'Text':>12} {'LN':>8} "
          f"{'β0':>4} {'β1':>4} {'β2':>4} {'Classification':>20}")
    print("-" * 72)
    for model_key in all_data:
        for text_key in all_data[model_key]:
            r = all_data[model_key][text_key]
            pca_key = "PCA-3" if "PCA-3" in r else "PCA-2"
            thr_data = r[pca_key]["threshold_results"][0.1]
            betti = thr_data["betti"]
            classification = classify_topology(betti)
            print(
                f"{model_key:>12} {text_key:>12} "
                f"{MODELS[model_key]['ln_type']:>8} "
                f"{betti.get(0,0):>4} {betti.get(1,0):>4} {betti.get(2,0):>4} "
                f"{classification:>20}"
            )

    plot_persistence_diagrams(
        all_data, save_path=str(out_dir / "exp2_persistent_homology.png")
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "Persistent homology provides the definitive topological fingerprint.\n\n"
        "β1 = 0 across all conditions:\n"
        "  -> The manifold is simply connected (no holes)\n"
        "  -> Consistent with S^2 (sphere) or contractible (disk)\n\n"
        "β1 = 1:\n"
        "  -> One non-contractible loop exists\n"
        "  -> The manifold has S^1 (circle) structure in one direction\n\n"
        "β1 = 2:\n"
        "  -> Two independent loops exist\n"
        "  -> STRONG evidence for T^2 (torus) topology\n\n"
        "β2 > 0:\n"
        "  -> An enclosed void exists\n"
        "  -> Combined with β1, distinguishes sphere from torus\n\n"
        "Compare PreLN vs PostLN: do they live on topologically\n"
        "different manifolds, or the same manifold with different\n"
        "PCA concentration?"
    )


if __name__ == "__main__":
    main()
