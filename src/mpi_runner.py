"""
MPI-Parallelized N-Body Simulation Runner
==========================================
This is the heart of the project: it distributes the N-body force
computation across multiple MPI ranks (processes) using domain
decomposition and collective communication.

Key MPI concepts used here (interview-ready):
    - MPI.COMM_WORLD: the default communicator containing all processes
    - rank: each process's unique ID (0 to size-1)
    - size: total number of processes
    - Scatter: rank 0 splits data and sends chunks to each rank
    - Allgather: every rank shares its chunk with every other rank
    - Bcast: rank 0 sends the same data to all ranks
    - Reduce: all ranks contribute partial results, combined at rank 0

Why domain decomposition for N-body?
    Each particle needs to know the positions of ALL other particles
    to compute forces (it is a global interaction problem). So we:
    1. Split PARTICLES (not space) across ranks
    2. Each rank computes forces on its own subset
    3. But each rank needs ALL positions -> Allgather after each step
    4. Each rank only updates velocities/positions for its own particles

This is sometimes called "particle decomposition" as opposed to
"spatial decomposition" (which partitions the physical volume).
"""

import numpy as np
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

# mpi4py import: this must happen before any MPI calls.
# When you run: mpiexec -n 4 python mpi_runner.py - over here size = 4
# MPI spawns 4 copies of this script, each with a different rank.
from mpi4py import MPI

from src.simulation import (
    initialize_particles,
    compute_forces_vectorized,
    leapfrog_step,
    G,
    EPSILON,
)


def compute_local_forces_threaded(
    local_positions: np.ndarray,
    all_positions: np.ndarray,
    all_masses: np.ndarray,
    n_threads: int = 2,
) -> np.ndarray:
    """
    Compute forces on LOCAL particles from ALL particles using threads.
v
    This is the HYBRID parallelism part: MPI across nodes + threads
    within each node. This matters because modern HPC nodes have
    many cores (32-128) but MPI communication overhead grows with
    process count. So you use fewer MPI ranks with more threads each.

    Interview question: "Why not just use more MPI ranks instead?"
    Answer: Each MPI rank has its own memory space. With N ranks you
    replicate ALL positions N times (memory overhead). Threads share
    memory within a rank so they do not duplicate data. The sweet spot
    is usually 1-4 MPI ranks per node with threads filling the cores.

    Args:
        local_positions: (n_local, 3) positions of this rank's particles
        all_positions:   (N, 3) positions of ALL particles (from Allgather)
        all_masses:      (N,) masses of ALL particles
        n_threads:       number of threads per rank

    Returns:
        forces: (n_local, 3) forces on this rank's particles
    """
    n_local = local_positions.shape[0]
    forces = np.zeros_like(local_positions)

    def compute_chunk(start: int, end: int) -> np.ndarray:
        """Compute forces for particles[start:end] against all particles."""
        chunk_forces = np.zeros((end - start, 3))

        for i_local, i_global in enumerate(range(start, end)):
            pos_i = local_positions[i_local + start if start == 0 else i_local]
            # Recompute: use actual local index
            pos_i = local_positions[start + i_local - start]

            diff = all_positions - pos_i  # (N, 3)
            dist_sq = np.sum(diff ** 2, axis=1) + EPSILON ** 2  # (N,)
            inv_dist_cubed = dist_sq ** (-1.5)  # (N,)

            # Force on particle i from all others
            force_contributions = (
                G * all_masses * inv_dist_cubed
            )[:, np.newaxis] * diff  # (N, 3)

            chunk_forces[i_local] = np.sum(force_contributions, axis=0)

        return chunk_forces

    # Split local particles among threads
    chunk_size = max(1, n_local // n_threads)
    futures = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for t in range(n_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n_local)
            if start >= n_local:
                break
            futures.append(executor.submit(compute_chunk, start, end))

    # Reassemble results
    offset = 0
    for future in futures:
        chunk_result = future.result()
        forces[offset:offset + chunk_result.shape[0]] = chunk_result
        offset += chunk_result.shape[0]

    return forces


def run_parallel_simulation(
    n_particles: int = 1000,
    n_steps: int = 50,
    dt: float = 0.01,
    seed: int = 42,
    n_threads: int = 2,
    use_hybrid: bool = False,
) -> Dict[str, Any]:
    """
    Run the N-body simulation distributed across MPI ranks.

    Workflow per timestep:
        1. Allgather positions so every rank has the full position array
        2. Each rank computes forces on its LOCAL subset of particles
        3. Each rank updates its local velocities and positions (leapfrog)
        4. Repeat

    Args:
        n_particles: total number of particles (split across ranks)
        n_steps:     number of simulation timesteps
        dt:          timestep size
        seed:        random seed for reproducibility
        n_threads:   threads per MPI rank (for hybrid mode)
        use_hybrid:  if True use threaded force computation

    Returns:
        Dictionary with timing and metadata for benchmarking
    """
    comm = MPI.COMM_WORLD 
    # the group of all processes - every MPI program starts
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- INITIALIZATION (rank 0 only) ---
    # Only rank 0 generates particles. This avoids issues where different
    # ranks might generate slightly different floating-point values despite
    # using the same seed (can happen with different NumPy/OS versions).
    if rank == 0:
        positions, velocities, masses = initialize_particles(n_particles, seed)
    else:
        positions = None
        velocities = None
        masses = None

    # one rank sends the same data to all the ranks
    # and we use this to broadcast the particle masses at the start
    # coz masses don't change during the simulation
    masses = comm.bcast(masses, root=0)

    # DISTRIBUTING THE PARTICLES
    # we split the particles as evenly as possible across all the ranks
    # if n_particles = 10 and size = 3, ranks get [4, 3, 3] particles.
    counts = np.array([n_particles // size] * size)
    counts[:n_particles % size] += 1  # distribute remainder

    # Scatterv - where v stands for variable-length
    # coz the data does not spread evenly across 3 chunks
    # displacements for Scatterv: cumulative sum of counts
    # if suppose, counts = [4, 3, 3] -> displs = [0, 4, 7]
    displs = np.zeros(size, dtype=int)
    displs[1:] = np.cumsum(counts[:-1])

    n_local = counts[rank]

    # allocate local arrays
    local_positions = np.zeros((n_local, 3))
    local_velocities = np.zeros((n_local, 3))

    # rank 0 splits the data into chunks
    # sends different chunks into each rank
    if rank == 0:
        # flatten for MPI - each particle is 3 doubles
        send_counts_pos = counts * 3
        send_displs_pos = displs * 3
        comm.Scatterv(
            [positions, send_counts_pos, send_displs_pos, MPI.DOUBLE],
            local_positions,
            root=0,
        )
        comm.Scatterv(
            [velocities, send_counts_pos, send_displs_pos, MPI.DOUBLE],
            local_velocities,
            root=0,
        )
    else:
        comm.Scatterv(None, local_positions, root=0)
        comm.Scatterv(None, local_velocities, root=0)

    # SIMULATION LOOP
    comm.Barrier()  # synchronize before timing
    t_start = MPI.Wtime()  # wall-clock timer (more precise than time.time)

    all_positions = np.zeros((n_particles, 3))

    for step in range(n_steps):
        # Step 1: Allgather positions
        # After this call, EVERY rank has the full position array.
        # This is the main communication cost: O(N) data per rank per step.
        recv_counts = counts * 3
        recv_displs = displs * 3
        comm.Allgatherv(
            local_positions,
            [all_positions, recv_counts, recv_displs, MPI.DOUBLE],
        )

        # Step 2: Compute forces on LOCAL particles
        if use_hybrid:
            local_forces = compute_local_forces_threaded(
                local_positions, all_positions, masses, n_threads
            )
            local_forces *= masses[displs[rank]:displs[rank] + n_local, np.newaxis]
        else:
            # Use the vectorized computation on the local subset
            # Each rank computes forces for its own particles against ALL
            local_forces = np.zeros_like(local_positions)
            for i in range(n_local):
                global_i = displs[rank] + i
                diff = all_positions - local_positions[i]
                dist_sq = np.sum(diff ** 2, axis=1) + EPSILON ** 2
                inv_dist_cubed = dist_sq ** (-1.5)
                inv_dist_cubed[global_i] = 0.0  # no self-interaction

                force_i = G * masses * inv_dist_cubed
                local_forces[i] = np.sum(
                    force_i[:, np.newaxis] * diff, axis=0
                ) * masses[global_i]

        # Step 3: Leapfrog integration (local only)
        local_acc = local_forces / masses[
            displs[rank]:displs[rank] + n_local, np.newaxis
        ]
        local_velocities += local_acc * dt
        local_positions += local_velocities * dt

    comm.Barrier()  # synchronize before stopping timer
    t_end = MPI.Wtime()
    wall_time = t_end - t_start

    # --- GATHER RESULTS ---
    # Collect final positions back to rank 0 for output/validation
    if rank == 0:
        final_positions = np.zeros((n_particles, 3))
    else:
        final_positions = None

    send_counts_pos = counts * 3
    send_displs_pos = displs * 3

    comm.Gatherv(
        local_positions,
        [final_positions, send_counts_pos, send_displs_pos, MPI.DOUBLE]
        if rank == 0
        else None,
        root=0,
    )

    # Collect timing from all ranks
    all_times = comm.gather(wall_time, root=0)

    results = {}
    if rank == 0:
        results = {
            "n_particles": n_particles,
            "n_steps": n_steps,
            "n_ranks": size,
            "n_threads": n_threads if use_hybrid else 1,
            "wall_time_max": max(all_times),
            "wall_time_min": min(all_times),
            "wall_time_mean": np.mean(all_times),
            "final_positions": final_positions,
        }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPI N-Body Simulation")
    parser.add_argument("-n", "--particles", type=int, default=500)
    parser.add_argument("-s", "--steps", type=int, default=50)
    parser.add_argument("-t", "--threads", type=int, default=2)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()

    results = run_parallel_simulation(
        n_particles=args.particles,
        n_steps=args.steps,
        n_threads=args.threads,
        use_hybrid=args.hybrid,
    )

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print(f"Particles: {results['n_particles']}")
        print(f"Steps:     {results['n_steps']}")
        print(f"Ranks:     {results['n_ranks']}")
        print(f"Wall time: {results['wall_time_max']:.4f}s (slowest rank)")
        print(f"           {results['wall_time_mean']:.4f}s (mean)")
