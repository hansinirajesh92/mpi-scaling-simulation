"""
Profiling and Benchmarking Module
=================================
Runs the simulation at varying processor counts and particle counts
to measure parallel speedup and efficiency.

Key metrics (interview-ready):

    Speedup S(p) = T(1) / T(p)
        where T(1) is serial wall time and T(p) is wall time on p processors.
        Ideal speedup = p (linear scaling).

    Efficiency E(p) = S(p) / p
        How well you are utilizing each processor.
        E = 1.0 means perfect scaling. E < 1.0 means overhead.

    Amdahl's Law: S(p) = 1 / (f + (1-f)/p)
        where f is the serial fraction of the code.
        Even with infinite processors, speedup is bounded by 1/f.
        If 5% of your code is serial, max speedup = 20x regardless of p.

    Gustafson's Law: S(p) = p - f*(p - 1)
        Assumes you scale the problem size with processor count.
        More realistic for scientific computing where bigger clusters
        let you run bigger simulations (not just faster ones).

    Strong scaling: fixed problem size, increase processor count
        -> measures how much faster you can solve the SAME problem
    Weak scaling: increase problem size proportionally with processors
        -> measures if you can solve BIGGER problems in the same time
"""

import numpy as np
import json
import os
from typing import List, Dict


def compute_speedup(serial_time: float, parallel_times: Dict[int, float]) -> Dict[int, float]:
    """
    Compute speedup for each processor count.

    Args:
        serial_time: wall time with 1 processor
        parallel_times: {n_procs: wall_time} for each run

    Returns:
        {n_procs: speedup} dictionary
    """
    return {p: serial_time / t for p, t in parallel_times.items()}


def compute_efficiency(speedups: Dict[int, float]) -> Dict[int, float]:
    """
    Compute parallel efficiency for each processor count.

    Returns:
        {n_procs: efficiency} dictionary
    """
    return {p: s / p for p, s in speedups.items()}


def estimate_serial_fraction(speedups: Dict[int, float]) -> float:
    """
    Estimate the serial fraction f using Amdahl's Law.

    From Amdahl's Law: S(p) = 1 / (f + (1-f)/p)
    Rearranging: f = (1/S - 1/p) / (1 - 1/p)

    We average f across all processor counts for a robust estimate.

    Interview note: this tells you the theoretical ceiling for your code.
    If f = 0.10 (10% serial), max possible speedup is 10x no matter
    how many processors you throw at it. The serial fraction includes
    initialization, I/O, and any inherently sequential computation.
    """
    f_estimates = []
    for p, s in speedups.items():
        if p > 1:
            f = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
            f_estimates.append(f)

    return np.mean(f_estimates) if f_estimates else 0.0


def save_benchmark_results(
    results: List[Dict],
    output_path: str = "benchmark_results.json",
) -> None:
    """Save benchmark results to JSON for later analysis."""
    # Convert numpy types to Python natives for JSON serialization
    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                continue  # skip large arrays
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            else:
                clean[k] = v
        clean_results.append(clean)

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)


def print_benchmark_summary(
    results: List[Dict],
    serial_time: float,
) -> None:
    """Print a formatted summary table of benchmark results."""
    parallel_times = {r["n_ranks"]: r["wall_time_max"] for r in results}
    speedups = compute_speedup(serial_time, parallel_times)
    efficiencies = compute_efficiency(speedups)
    f_serial = estimate_serial_fraction(speedups)

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Ranks':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: x["n_ranks"]):
        p = r["n_ranks"]
        t = r["wall_time_max"]
        s = speedups[p]
        e = efficiencies[p]
        print(f"{p:<8} {t:<12.4f} {s:<10.2f} {e:<12.2%}")

    print("-" * 65)
    print(f"Estimated serial fraction (Amdahl's Law): {f_serial:.2%}")
    print(f"Theoretical max speedup: {1.0 / f_serial:.1f}x" if f_serial > 0 else "")
    print("=" * 65)
