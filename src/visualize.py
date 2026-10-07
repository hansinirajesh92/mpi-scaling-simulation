"""
Visualization Module
====================
Generates publication-quality plots for the scaling analysis.
These are the plots that go in the case study and portfolio.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC environments


def plot_speedup(
    proc_counts: list,
    speedups: list,
    output_path: str = "speedup.png",
) -> None:
    """
    Plot parallel speedup vs processor count.
    Includes ideal (linear) scaling line for comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(proc_counts, speedups, "o-", color="#2563eb", linewidth=2,
            markersize=8, label="Measured speedup", zorder=3)
    ax.plot(proc_counts, proc_counts, "--", color="#94a3b8", linewidth=1.5,
            label="Ideal (linear) scaling", zorder=2)

    ax.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax.set_ylabel("Speedup (T1 / Tp)", fontsize=12)
    ax.set_title("Strong Scaling: Speedup vs Processor Count", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(proc_counts)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency(
    proc_counts: list,
    efficiencies: list,
    output_path: str = "efficiency.png",
) -> None:
    """
    Plot parallel efficiency vs processor count.
    Efficiency = Speedup / p. Perfect efficiency = 1.0.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(proc_counts, efficiencies, color="#2563eb", alpha=0.8, width=0.6)
    ax.axhline(y=1.0, color="#94a3b8", linestyle="--", linewidth=1.5,
               label="Ideal efficiency")

    ax.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax.set_ylabel("Efficiency (Speedup / p)", fontsize=12)
    ax.set_title("Parallel Efficiency vs Processor Count", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(proc_counts)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_runtime_vs_particles(
    particle_counts: list,
    runtimes_serial: list,
    runtimes_parallel: list,
    n_ranks: int,
    output_path: str = "runtime_scaling.png",
) -> None:
    """
    Plot wall time vs problem size for serial and parallel runs.
    Shows the O(N^2) scaling of the brute-force approach.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(particle_counts, runtimes_serial, "s-", color="#dc2626",
            linewidth=2, markersize=8, label="Serial (1 rank)")
    ax.plot(particle_counts, runtimes_parallel, "o-", color="#2563eb",
            linewidth=2, markersize=8, label=f"Parallel ({n_ranks} ranks)")

    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Wall Time (seconds)", fontsize=12)
    ax.set_title("Runtime vs Problem Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_communication_overhead(
    proc_counts: list,
    compute_times: list,
    comm_times: list,
    output_path: str = "comm_overhead.png",
) -> None:
    """
    Stacked bar chart showing compute vs communication time.
    This is the KEY diagnostic plot for parallel performance:
    as you add more ranks, communication overhead grows.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(proc_counts))
    width = 0.5

    ax.bar(x, compute_times, width, label="Computation", color="#2563eb", alpha=0.85)
    ax.bar(x, comm_times, width, bottom=compute_times,
           label="Communication (MPI)", color="#f59e0b", alpha=0.85)

    ax.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Compute vs Communication Overhead", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(proc_counts)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
