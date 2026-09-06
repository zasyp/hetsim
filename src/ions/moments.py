# Particle -> grid moments: turning macroparticle clouds into the density
# fields the electron fluid consumes.
#
# The deposition machinery already exists (deposition/deposit.py: CIC scatter).
# What this module adds is the two things that separate "a scatter call" from
# "a density the solver can eat":
#
#   1. division by the node control volume (neutrals.node_volume), which in
#      axisymmetric geometry is an annulus, not a box — a node at r = 1 mm and
#      one at r = 30 mm own volumes differing by thirty times;
#   2. noise control. A PIC density carries relative noise ~1/sqrt(N_cell).
#      At the particle counts a Python model can afford that is 10-20% per
#      cell, and the electron solver reacts badly to it: the cross-field
#      conductance goes like n_e, so speckle in the density becomes speckle in
#      the potential, which the Gummel loop then amplifies. Smoothing the
#      density once before handing it over costs a little resolution and buys
#      a solve that converges.

import numpy as np

from ..structs.classes import Grid2D, ParticleArray
from ..deposition.deposit import locate_particle, scatter
from ..neutrals.neutrals import node_volume


def number_density(part: ParticleArray, grid: Grid2D,
                   dV: np.ndarray | None = None) -> np.ndarray:
    """Number density [1/m^3] on the (N_z, N_r) nodes from macroparticles.

    CIC scatter of the weights, divided by the annular node volume. Pass dV
    if you already have it (node_volume is not free and the volumes never
    change) — the hybrid loop builds it once and reuses it every step.
    """
    if dV is None:
        dV = node_volume(grid)
    if len(part) == 0:
        return np.zeros((grid.N_z, grid.N_r))
    iz, ir, wz, wr = locate_particle(part, grid)
    S = scatter(part, grid, iz, ir, wz, wr)
    return S / dV


def mean_velocity(part: ParticleArray, grid: Grid2D,
                  passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Density-weighted mean velocity (v_z, v_r) on the nodes [m/s].

    The first velocity moment: scatter w*v and divide by scatter w, which is
    the flux divided by the density. Cells no particle reached come back zero
    rather than NaN, so the maps stay plottable; read them only where the
    density map is non-trivial.

    This is a MEAN, not a beam velocity: where ions move both ways (near the
    anode, or across the axis in the plume) the average cancels and the map
    reads low even though individual ions are fast. For the exit velocity use
    beam.BeamTally, which keeps the distribution instead of averaging it.
    """
    if len(part) == 0:
        z = np.zeros((grid.N_z, grid.N_r))
        return z, z.copy()
    iz, ir, wz, wr = locate_particle(part, grid)
    w = scatter(part, grid, iz, ir, wz, wr)
    pz = scatter(part, grid, iz, ir, wz, wr, q=part.v_z)
    pr = scatter(part, grid, iz, ir, wz, wr, q=part.v_r)
    w = smooth(w, passes)
    good = w > 0
    v_z = np.zeros_like(w); v_r = np.zeros_like(w)
    v_z[good] = smooth(pz, passes)[good] / w[good]
    v_r[good] = smooth(pr, passes)[good] / w[good]
    return v_z, v_r


def smooth(field: np.ndarray, passes: int = 1) -> np.ndarray:
    """Binomial [1 2 1] smoothing in both directions, `passes` times.

    The standard PIC digital filter: it removes the cell-to-cell speckle that
    finite particle counts produce while leaving anything resolved over more
    than a couple of cells essentially untouched. Edges use a reflecting
    stencil so the filter neither leaks density out of the domain nor pulls
    the boundary values toward zero — density is conserved to round-off.
    """
    out = field
    for _ in range(passes):
        for axis in (0, 1):
            a = np.moveaxis(out, axis, 0)
            padded = np.concatenate([a[1:2], a, a[-2:-1]], axis=0)
            a = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            out = np.moveaxis(a, 0, axis)
    return out


def seed_uniform(grid: Grid2D, thruster, n_target: float, weight: float,
                 temperature: float | None = None,
                 rng: np.random.Generator | None = None) -> ParticleArray:
    """Fill the channel with a population at roughly density n_target.

    Used twice at startup, for both heavy species:

    * neutrals, at the density the mass flow will settle at. Starting from an
      empty domain means waiting out the neutral residence time (channel
      length / thermal speed, ~100 us for xenon at 750 K) before anything
      physical happens; seeding skips that dead stretch.
    * ions, at a low density, to light the discharge. Without a seed the
      model is stuck: ionization needs electrons, n_e = n_i needs ions, and
      an empty domain stays empty forever. Physically the cathode supplies
      the first electrons; here a thin seed plasma stands in for that. What
      happens next — growth to a steady discharge, or decay — is the model's
      answer, not the seed's.

    Uniform in volume, Maxwellian in velocity. A starting guess, not a
    solution; the run still has to relax away from it.
    """
    if rng is None:
        rng = np.random.default_rng()
    if temperature is None:
        temperature = thruster.temperature_anode
    L = thruster.channel_length
    volume = np.pi * (thruster.r_max ** 2 - thruster.r_min ** 2) * L
    N = int(n_target * volume / weight)
    if N <= 0:
        return ParticleArray()

    # uniform in volume: r ~ sqrt(uniform between r_min^2 and r_max^2)
    r = np.sqrt(thruster.r_min ** 2
                + rng.random(N) * (thruster.r_max ** 2 - thruster.r_min ** 2))
    z = rng.random(N) * L
    v_th = thruster.propellant.thermal_speed(temperature)
    v = v_th * rng.standard_normal((3, N))
    return ParticleArray(z, r, v[0], v[1], v[2], np.full(N, weight))


def seed_plasma_profile(grid: Grid2D, thruster, n_peak: float, weight: float,
                        temperature: float = 1000.0,
                        rng: np.random.Generator | None = None) -> ParticleArray:
    """Seed ions over channel AND plume with a realistic axial profile.

    seed_uniform fills the channel only, and that turns out to matter a great
    deal. An empty plume means n_e drops to the floor at the exit plane, a
    hundredfold resistance jump right there; the potential solve puts the
    whole voltage across that jump, the ohmic heating follows, and the hot
    layer lands DOWNSTREAM of the exit — where there are no neutrals left to
    ionize. The discharge then cannot sustain itself, for a reason that is
    pure startup artefact.

    So the seed carries the shape the placeholder state used (rising from the
    anode to a peak at the exit plane, decaying into the plume), which is what
    a Hall thruster actually looks like. It is still only a starting guess:
    where the profile ends up is the model's answer.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = thruster.channel_length
    z_nodes = grid.z_nodes()
    profile = np.interp(z_nodes, [0.0, L, 2 * L],
                        [0.05 * n_peak, n_peak, 0.25 * n_peak])

    # radial band: the channel annulus inside, the full radius in the plume
    r_lo = np.where(z_nodes < L, thruster.r_min, grid.min_r)
    r_hi = np.where(z_nodes < L, thruster.r_max, grid.max_r)
    ring = np.pi * (r_hi ** 2 - r_lo ** 2)
    dz = z_nodes[1] - z_nodes[0]

    counts = rng.poisson(profile * ring * dz / weight)
    N = int(counts.sum())
    if N == 0:
        return ParticleArray()

    z = np.repeat(z_nodes, counts) + rng.random(N) * dz
    lo = np.repeat(r_lo, counts)
    hi = np.repeat(r_hi, counts)
    r = np.sqrt(lo ** 2 + rng.random(N) * (hi ** 2 - lo ** 2))
    v_th = thruster.propellant.thermal_speed(temperature)
    v = v_th * rng.standard_normal((3, N))
    return ParticleArray(z, r, v[0], v[1], v[2], np.full(N, weight))
