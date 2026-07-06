"""Neutral-atom macroparticle handling for the discharge channel.

`particle_injection` is called once per time step and emits neutral
macroparticles through the anode plane (z = z_inj) so that the injected mass
per step exactly matches the requested propellant mass flow rate.

`deposit_moments` is the particle-to-grid (scatter) step: every time step it
distributes the macroparticle weights onto the grid nodes and turns them into
number density and mean-velocity/temperature fields. Axial shape factors are
linear (CIC); radial shape factors are the density-conserving ones of
Ruyten (1993), which compensate the r-growth of cylindrical cell volumes so
that a uniform particle cloud yields a uniform density.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.classes import ParticleArray, Grid2D

KB = np.float64(1.380649e-23)  # Boltzmann constant, J/K


def particle_injection(
        g: Grid2D,
        particles: ParticleArray,
        rng: np.random.Generator,
        mdot: np.float64,
        m_atom: np.float64,
        T_inj: np.float64,
        dt: np.float64,
        n_inject: int,
        r_in: np.float64 | None = None,
        r_out: np.float64 | None = None,
        z_inj: np.float64 | None = None,
        ) -> int:
    """Inject `n_inject` neutral macroparticles through the anode plane.

    The macroparticle statistical weight is chosen so that the mass injected
    per step matches the mass flow rate exactly:

        weight = mdot * dt / (m_atom * n_inject)

    Sampling (thermal emission from a wall at temperature `T_inj`):
      * r        -- uniform over the annulus *area* between `r_in` and `r_out`
                    (i.e. uniform in r^2, correct for an axisymmetric domain);
      * v_z      -- flux-weighted half-Maxwellian, f(v) ~ v*exp(-v^2/(2*vth^2)),
                    always directed into the domain (+z);
      * v_r, v_theta -- full Maxwellian, normal with sigma = vth;
      * z        -- `z_inj` plus a random fraction of the first step's flight
                    distance v_z*dt, which removes the artificial density
                    banding that appears if every particle starts exactly on
                    the plane.

    Returns the number of injected macroparticles.
    """
    if r_in is None:
        r_in = g.r_min
    if r_out is None:
        r_out = g.r_max
    if z_inj is None:
        z_inj = g.z_min

    weight = mdot * dt / (m_atom * n_inject)
    vth = np.sqrt(KB * T_inj / m_atom)  # thermal speed per component, m/s

    u = rng.random(n_inject)
    r_new = np.sqrt(r_in**2 + u * (r_out**2 - r_in**2))
    v_z_new = vth * np.sqrt(-2.0 * np.log(1.0 - rng.random(n_inject)))
    v_r_new = rng.normal(0.0, vth, n_inject)
    v_theta_new = rng.normal(0.0, vth, n_inject)
    z_new = z_inj + rng.random(n_inject) * v_z_new * dt

    particles.add(z_new, r_new, v_r_new, v_z_new, v_theta_new,
                  0, T_inj, weight)
    return n_inject


def node_volumes(g: Grid2D) -> np.ndarray:
    """Volume associated with every grid node, shape (N_r, N_z), m^3.

    The radial part is made consistent with the Ruyten radial shape factors
    used by `deposit_moments`: node j's area is the integral of its shape
    factor times 2*pi*r over the two adjacent cells, so a uniform particle
    cloud deposits into an exactly uniform density, boundary nodes included.
    The integrand is a cubic in r, hence 3-point Gauss-Legendre is exact.
    The axial part is h_z for interior nodes and h_z/2 on the z boundaries.
    Node volumes sum to the full domain volume pi*(r_max^2 - r_min^2)*L_z.
    """
    r = g.r_nodes()
    xq, wq = np.polynomial.legendre.leggauss(3)

    areas = np.zeros(g.N_r)
    for j in range(g.N_r - 1):
        ra, rb = r[j], r[j + 1]
        rq = 0.5 * (rb - ra) * xq + 0.5 * (ra + rb)
        jac = 0.5 * (rb - ra)
        S_lo = (rb - rq) * (2.0 * rb + 3.0 * ra - rq) / (2.0 * (rb**2 - ra**2))
        areas[j] += 2.0 * np.pi * jac * np.sum(wq * rq * S_lo)
        areas[j + 1] += 2.0 * np.pi * jac * np.sum(wq * rq * (1.0 - S_lo))

    dz = np.full(g.N_z, g.h_z)
    dz[0] *= 0.5
    dz[-1] *= 0.5
    return np.outer(areas, dz)


def deposit_moments(
        g: Grid2D,
        particles: ParticleArray,
        select: np.ndarray,
        volumes: np.ndarray,
        n: np.ndarray,
        v_r: np.ndarray,
        v_z: np.ndarray,
        v_theta: np.ndarray,
        T: np.ndarray,
        ) -> None:
    """Deposit macroparticle weights onto the grid (particle-to-grid step).

    `select` is a boolean mask over the first `particles.n` slots choosing
    which particles contribute (e.g. active neutrals). Each selected particle
    contributes weight * S_r * S_z to the four nodes of its cell: S_z are the
    linear (CIC) axial factors, S_r the radial density-conserving factors of
    Ruyten (1993). `volumes` must come from `node_volumes(g)` so that the two
    stay consistent.

    Results are written in place into arrays of shape (N_r, N_z):
      n                 -- number density, 1/m^3;
      v_r, v_z, v_theta -- weight-averaged mean velocities, m/s;
      T                 -- weight-averaged temperature, K.
    Nodes that received no particles are set to 0. Positions outside the grid
    are clamped to its edge, matching `interpolation_weights`.
    """
    for out in (n, v_r, v_z, v_theta, T):
        out.fill(0.0)

    idx = np.flatnonzero(select)
    if idx.size == 0:
        return

    pr = particles.r[idx]
    pz = particles.z[idx]
    w = particles.weight[idx]
    moment_vals = (particles.v_r[idx], particles.v_z[idx],
                   particles.v_theta[idx], particles.T[idx])
    moment_outs = (v_r, v_z, v_theta, T)

    r_grid = g.r_nodes()
    z_grid = g.z_nodes()
    jr = np.clip(np.floor((pr - g.r_min) / g.h_r).astype(int), 0, g.N_r - 2)
    iz = np.clip(np.floor((pz - g.z_min) / g.h_z).astype(int), 0, g.N_z - 2)

    # Sr_lo -- weight of the inner node j (eq. 2.19); Sz_hi -- weight of the
    # upper node i+1 of the linear axial pair.
    ra, rb = r_grid[jr], r_grid[jr + 1]
    rc = np.clip(pr, ra, rb)
    Sr_lo = (rb - rc) * (2.0 * rb + 3.0 * ra - rc) / (2.0 * (rb**2 - ra**2))
    zc = np.clip(pz, z_grid[iz], z_grid[iz + 1])
    Sz_hi = (zc - z_grid[iz]) / g.h_z

    # One flattened-node index / shape-factor pair per cell corner, so every
    # moment is accumulated with a single bincount over 4*N entries.
    flat = np.concatenate([
        (jr + dj) * g.N_z + (iz + di)
        for dj in (0, 1) for di in (0, 1)
    ])
    shape = np.concatenate([
        w * Sr * Sz
        for Sr in (Sr_lo, 1.0 - Sr_lo)
        for Sz in (1.0 - Sz_hi, Sz_hi)
    ])

    size = g.N_r * g.N_z
    W = np.bincount(flat, shape, minlength=size).reshape(g.N_r, g.N_z)

    np.divide(W, volumes, out=n)
    got = W > 0.0
    for vals, out in zip(moment_vals, moment_outs):
        acc = np.bincount(flat, shape * np.tile(vals, 4),
                          minlength=size).reshape(g.N_r, g.N_z)
        np.divide(acc, W, out=out, where=got)
