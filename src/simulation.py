# N-Body Gravitational Simulation Core
# simulating gravitational dynamics for N particles in 3d space
# this module handles particle initialization and force computation without any parallelism
# for parallelization layer - check out mpi_runner.py

import numpy as np
from typing import Tuple

# gravitational constant (normalized to 1.0 for simulation units)
# in real astrophysics codes you would use G = 6.674e-11 m^3 kg^-1 s^-2
# but normalizing simplifies the math and avoids floating-point overflow
G = 1.0

# softening length: prevents force blow-up when r_ij approaches 0
# without this, if two particles happen to land very close together
# it would produce near-infinite forces and will crash the simulation
# we should typically try to use a small fraction of the mean inter-particle spacing
EPSILON = 0.01

# creating intial conditions for particles -
# this creates n particles scattered randomly inside a 3d box
# each particle gets a position: (x, y, z), a velocity: (vx, vy, vz) and a mass
def initialize_particles(
    n_particles: int,
    seed: int = 42,
    box_size: float = 10.0,
    mass_range: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # fixed seed - same seed = same particles every time
    rng = np.random.default_rng(seed) # for reproducibility (like, benchmarking or debugging)

    positions = rng.uniform(-box_size / 2, box_size / 2, size=(n_particles, 3))

    # small initial velocities so the system evolves slowly enough
    # to observe dynamics over many timesteps
    velocities = rng.uniform(-0.1, 0.1, size=(n_particles, 3))

    masses = rng.uniform(mass_range[0], mass_range[1], size=n_particles)

    return positions, velocities, masses

# in this function, we are computing 
# the gravitational forces on all the particles - serial version
def compute_forces(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    n = positions.shape[0]
    forces = np.zeros_like(positions)

    # we need to check if the particles are same
    # and if they are, we should skip it coz
    # we dont need to check force for the same particle
    # coz a particle does not exert force on itself
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # direction vector from particle i to particle j
            r_ij = positions[j] - positions[i]

            # this is the gravitational force formula
            # G * m_i * m_j * r_ij / dist_sq^(3/2) = forces[i]

            # squared distance with softening
            dist_sq = np.dot(r_ij, r_ij) + EPSILON ** 2

            # force magnitude: G * m_i * m_j / dist_sq
            # direction: r_ij / sqrt(dist_sq)
            inv_dist_cubed = dist_sq ** (-1.5)
            forces[i] += G * masses[j] * r_ij * inv_dist_cubed

            # after we include the direction verctor
            # the power becomes 3/2

        # multiply by particle's own mass at the end, to complete the formula
        forces[i] *= masses[i]
    return forces

# this is the same thing as above but uses numpy broadcasting 
# it avoids python loops and on a single core, this runs 50-100x faster
# there is a comaprison of the two present
# we cannot use this alone though coz there are memory limits
# suppose, we want to compute the forces for like 1 million particles
# we would need close to 24 TB of memory, which is just impractical
# it is NECESSARY to split across nodes
def compute_forces_vectorized(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    n = positions.shape[0]

    # r_ij[i, j, :] = positions[j] - positions[i]
    # shape: (N, N, 3)
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # squared distances: shape (N, N)
    dist_sq = np.sum(diff ** 2, axis=2) + EPSILON ** 2

    # inverse cube of distance: shape (N, N)
    inv_dist_cubed = dist_sq ** (-1.5)

    # zero out self-interaction (i == j diagonal)
    np.fill_diagonal(inv_dist_cubed, 0.0)

    # force contribution from j on i: G * m_j * r_ij / |r_ij|^3
    # broadcasting: (N, N, 1) * (N, N, 3) -> (N, N, 3)
    force_matrix = (
        G * masses[np.newaxis, :, np.newaxis]
        * diff
        * inv_dist_cubed[:, :, np.newaxis]
    )

    # sum over j (axis=1) to get net force on each particle i
    # then multiply by each particle's mass
    forces = np.sum(force_matrix, axis=1) * masses[:, np.newaxis]

    return forces

# this is the steo where we move the particles forward in time
# we are using the leapfrog integrator 
# even though it costs 2x the force calculation 
# it has more accuracy over time (becoz it conserves energy over long timescales)
# which means that, even if you run the simulation a million times
# with leap-frog, the total energy at step no. million will essentially be the same as step 0
# we choose this over a simpler integrator like euler coz
# using the same example - with euler, the smaller errors that come when we assume that
# the acceleration of each particle at each jump is constant
# these errors will accumulate and over a million timestamps (within one simulation), 
# the energy will be wildly different
def leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float = 0.01,
    use_vectorized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    
    force_fn = compute_forces_vectorized if use_vectorized else compute_forces

    # step 1: compute acceleration at current positions
    forces = force_fn(positions, masses)
    accelerations = forces / masses[:, np.newaxis]

    # step 2: half-kick velocities
    vel_half = velocities + 0.5 * dt * accelerations

    # step 3: drift positions
    new_positions = positions + dt * vel_half

    # step 4: recompute forces at new positions
    new_forces = force_fn(new_positions, masses)
    new_accelerations = new_forces / masses[:, np.newaxis]

    # step 5: second half-kick
    new_velocities = vel_half + 0.5 * dt * new_accelerations

    return new_positions, new_velocities
