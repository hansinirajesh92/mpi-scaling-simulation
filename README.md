# Scaling Simulation Pipelines with MPI (Astrophysics)

**Parallelized N-body gravitational simulation using MPI for high-performance computing.**

This project implements a brute-force N-body gravitational simulation and distributes the computation across multiple processors using MPI (Message Passing Interface). The goal is to measure parallel speedup and efficiency as processor count increases and to identify the bottlenecks that limit scaling.

## The Problem

Simulating gravitational dynamics for N particles requires computing pairwise forces between every pair of particles at each timestep. This is an O(N^2) computation: for 10,000 particles that means 100 million force calculations per step. On a single core this becomes impractical for large N. MPI lets us split the work across multiple processors to bring the runtime down.

## Architecture

```
src/
    simulation.py       # Core physics: force computation, leapfrog integrator
    mpi_runner.py       # MPI parallelization: domain decomposition, Allgather
    profiler.py         # Benchmarking: speedup, efficiency, Amdahl's Law
    visualize.py        # Matplotlib plots for scaling analysis
tests/
    test_simulation.py  # Unit tests for numerical accuracy
scripts/
    run_benchmark.sh    # Automated benchmark across 1, 2, 4, 8 ranks
```

## Key Concepts

**Particle Decomposition**: particles are split evenly across MPI ranks. Each rank computes forces on its own subset but needs ALL particle positions (via `Allgather`) since gravity is a long-range force.

**Hybrid Parallelism**: optional multithreading within each MPI rank using `concurrent.futures`. This reduces memory duplication compared to using more MPI ranks and better utilizes multi-core nodes.

**Leapfrog Integration**: a symplectic integrator that conserves energy over long timescales (unlike Euler which drifts). This is the standard choice in astrophysical N-body codes.

**Softening Parameter**: prevents numerical blow-up when particles get close. Without this, 1/r^2 forces diverge as r approaches 0.

## Scaling Results

| Ranks | Wall Time (s) | Speedup | Efficiency |
|-------|--------------|---------|------------|
| 1     | 12.34        | 1.00x   | 100%       |
| 2     | 6.52         | 1.89x   | 94.5%      |
| 4     | 3.48         | 3.55x   | 88.7%      |
| 8     | 2.01         | 6.14x   | 76.7%      |

Efficiency drops at higher rank counts due to `Allgather` communication overhead: every rank must receive O(N) position data at each timestep.

## Setup

```bash
# Install dependencies
pip install numpy mpi4py matplotlib

# Run tests
python tests/test_simulation.py

# Run benchmark
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh

# Single run with 4 MPI ranks
mpiexec -n 4 python src/mpi_runner.py --particles 1000 --steps 50
```

## Tech Stack

Python, NumPy, mpi4py, matplotlib, concurrent.futures, MPI (OpenMPI/MPICH)

## What I Learned

The hardest part of parallel computing is not the parallelism itself but the communication patterns. The brute-force N-body problem requires global communication (`Allgather`) at every timestep which creates a bandwidth bottleneck. Tree-based algorithms (like Barnes-Hut) reduce this to O(N log N) and would be the natural next step for scaling to millions of particles.

---

*Built as part of coursework at VIT Chennai. Optimized for documentation.*
