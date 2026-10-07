#!/bin/bash
# =============================================================
# Benchmark Script: Strong Scaling Analysis
# =============================================================
# Runs the MPI simulation with varying processor counts
# to measure parallel speedup and efficiency.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh
#
# Prerequisites:
#   - MPI implementation installed (OpenMPI or MPICH)
#   - mpi4py installed: pip install mpi4py
#   - numpy installed:  pip install numpy
# =============================================================

N_PARTICLES=1000
N_STEPS=50
PROC_COUNTS="1 2 4 8"
OUTPUT_DIR="results"

mkdir -p $OUTPUT_DIR

echo "MPI N-Body Scaling Benchmark"
echo "Particles: $N_PARTICLES | Steps: $N_STEPS"


for np in $PROC_COUNTS; do
    echo ""
    echo "Running with $np MPI rank(s)"
    mpiexec -n $np python src/mpi_runner.py \
        --particles $N_PARTICLES \
        --steps $N_STEPS \
        2>&1 | tee "$OUTPUT_DIR/run_np${np}.log"
done

echo ""
echo "All runs complete. Logs saved to $OUTPUT_DIR/"

