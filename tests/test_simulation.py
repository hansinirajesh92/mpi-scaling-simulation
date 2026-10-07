"""
Unit Tests for Simulation Accuracy
===================================
These tests verify that the physics is correct BEFORE scaling up.
A parallel code that scales perfectly but computes the wrong answer
is worse than useless.

Key testing strategy:
    1. Test known analytical solutions (two-body problem)
    2. Test conservation laws (energy conservation)
    3. Test symmetry (equal masses should produce symmetric forces)
    4. Test serial vs vectorized consistency
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.simulation import (
    initialize_particles,
    compute_forces,
    compute_forces_vectorized,
    leapfrog_step,
    G,
    EPSILON,
)


def test_two_body_force_direction():
    """
    Two particles along the x-axis: force on particle 0
    should point in the +x direction (toward particle 1).
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])

    forces = compute_forces(positions, masses)

    # Force on particle 0 should be positive in x
    assert forces[0, 0] > 0, f"Expected +x force, got {forces[0, 0]}"
    # Force on particle 1 should be negative in x (Newton's 3rd law)
    assert forces[1, 0] < 0, f"Expected -x force, got {forces[1, 0]}"
    # y and z forces should be ~zero
    assert abs(forces[0, 1]) < 1e-10
    assert abs(forces[0, 2]) < 1e-10

    print("PASS: two_body_force_direction")


def test_force_symmetry():
    """
    For equal masses, |F_01| should equal |F_10| (Newton's 3rd law).
    """
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]])
    masses = np.array([1.0, 1.0])

    forces = compute_forces(positions, masses)

    f01_mag = np.linalg.norm(forces[0])
    f10_mag = np.linalg.norm(forces[1])

    assert abs(f01_mag - f10_mag) < 1e-10, (
        f"Force magnitudes differ: {f01_mag} vs {f10_mag}"
    )
    print("PASS: force_symmetry")


def test_serial_vs_vectorized():
    """
    The loop-based and vectorized force computations should produce
    identical results (within floating-point tolerance).
    """
    positions, _, masses = initialize_particles(50, seed=123)

    forces_loop = compute_forces(positions, masses)
    forces_vec = compute_forces_vectorized(positions, masses)

    max_diff = np.max(np.abs(forces_loop - forces_vec))
    assert max_diff < 1e-8, f"Max difference: {max_diff}"

    print("PASS: serial_vs_vectorized")


def test_energy_conservation():
    """
    The leapfrog integrator should approximately conserve total energy
    (kinetic + potential) over short runs.

    Total energy = KE + PE
    KE = sum(0.5 * m_i * |v_i|^2)
    PE = sum over pairs: -G * m_i * m_j / |r_ij|
    """
    n = 20
    positions, velocities, masses = initialize_particles(n, seed=99)

    def total_energy(pos, vel, m):
        ke = 0.5 * np.sum(m[:, np.newaxis] * vel ** 2)
        pe = 0.0
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                r = np.linalg.norm(pos[j] - pos[i])
                pe -= G * m[i] * m[j] / np.sqrt(r ** 2 + EPSILON ** 2)
        return ke + pe

    e_initial = total_energy(positions, velocities, masses)

    # Run 100 steps with small dt
    for _ in range(100):
        positions, velocities = leapfrog_step(
            positions, velocities, masses, dt=0.001
        )

    e_final = total_energy(positions, velocities, masses)

    # Energy should be conserved within ~1% for small dt
    rel_error = abs(e_final - e_initial) / abs(e_initial)
    assert rel_error < 0.01, f"Energy drift: {rel_error:.4%}"

    print(f"PASS: energy_conservation (drift: {rel_error:.4%})")


def test_reproducibility():
    """
    Same seed should produce identical initial conditions.
    This is critical for benchmarking fairness.
    """
    p1, v1, m1 = initialize_particles(100, seed=42)
    p2, v2, m2 = initialize_particles(100, seed=42)

    assert np.array_equal(p1, p2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(m1, m2)

    print("PASS: reproducibility")


if __name__ == "__main__":
    test_two_body_force_direction()
    test_force_symmetry()
    test_serial_vs_vectorized()
    test_energy_conservation()
    test_reproducibility()
    print("\nAll tests passed.")
