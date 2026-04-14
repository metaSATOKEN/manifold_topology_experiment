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
Runner for all manifold topology experiments.

Usage:
    python run_all.py               # Run all experiments
    python run_all.py --exp 1       # Run only experiment 1
    python run_all.py --exp 1 2     # Run experiments 1 and 2
    python run_all.py --exp 2 --skip-if-no-lib  # Skip exp2 if ripser missing
"""

import argparse
import importlib
import time
import sys
from pathlib import Path

EXPERIMENTS = {
    1: ("exp1_winding_number", "Winding Number Analysis"),
    2: ("exp2_persistent_homology", "Persistent Homology (Betti Numbers)"),
    3: ("exp3_gaussian_curvature", "Gaussian Curvature Estimation"),
    4: ("exp4_geodesic_ratio", "Geodesic vs Euclidean Distance Ratio"),
    5: ("exp5_thickness_probing", "PCA Probing (GPT-2, Layer 6)"),
    "5b": ("exp5b_norm_normalized", "Norm-Normalized PCA"),
    "5c": ("exp5c_layerwise", "Layer-Wise Analysis"),
    "5d": ("exp5d_cross_model", "Cross-Model Comparison"),
}


def main():
    parser = argparse.ArgumentParser(description="Run manifold topology experiments")
    parser.add_argument(
        "--exp", nargs="*", default=None,
        help="Experiment IDs to run, e.g. 2 5 5b 5c (default: all)",
    )
    parser.add_argument(
        "--skip-if-no-lib", action="store_true",
        help="Skip experiments if required libraries are missing",
    )
    args = parser.parse_args()

    if args.exp:
        exp_nums = []
        for e in args.exp:
            try:
                exp_nums.append(int(e))
            except ValueError:
                exp_nums.append(e)
    else:
        exp_nums = sorted(EXPERIMENTS.keys(), key=str)

    print("=" * 70)
    print("MANIFOLD TOPOLOGY EXPERIMENT SUITE")
    print("Investigating Torus vs Sphere Structure in Transformer Hidden States")
    print("=" * 70)
    print(f"\nExperiments to run: {exp_nums}")
    print()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timings = {}

    for num in exp_nums:
        if num not in EXPERIMENTS:
            print(f"WARNING: Experiment {num} not found. Skipping.")
            continue

        module_name, title = EXPERIMENTS[num]
        print(f"\n{'#' * 70}")
        print(f"# Experiment {num}: {title}")
        print(f"{'#' * 70}\n")

        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
            elapsed = time.time() - t0
            timings[num] = elapsed
            print(f"\n  [Experiment {num} completed in {elapsed:.1f}s]")
        except ImportError as e:
            if args.skip_if_no_lib:
                print(f"\n  [Experiment {num} SKIPPED — missing library: {e}]")
                timings[num] = -1
            else:
                raise
        except Exception as e:
            elapsed = time.time() - t0
            timings[num] = -1
            print(f"\n  [Experiment {num} FAILED after {elapsed:.1f}s: {e}]")
            import traceback
            traceback.print_exc()

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for num in exp_nums:
        if num not in EXPERIMENTS:
            continue
        _, title = EXPERIMENTS[num]
        t = timings.get(num, -1)
        status = f"{t:.1f}s" if t >= 0 else "SKIPPED/FAILED"
        print(f"  Exp {num}: {title:50s} [{status}]")

    print(f"\nTotal time: {sum(t for t in timings.values() if t > 0):.1f}s")
    print(f"Results saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
