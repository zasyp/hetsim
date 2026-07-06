"""Heavy-particle transport for the 2D/3V axisymmetric PIC model.

`push_particles` advances neutrals and ions with the classical leapfrog
scheme (thesis eqs. 2.15-2.16): velocities live at half time steps, positions
at integer steps. Heavy particles are treated as unmagnetized (eq. 2.10 --
their Larmor radius exceeds the channel size), so the only force is qE on
ions; neutrals fly ballistically. Velocities are cartesian in the local
(r, theta, z) frame; after each position update the velocity vector is
rotated back into the (r, z, theta = 0) plane, which is what makes the
2D/3V treatment exact for axisymmetric motion.

`apply_boundaries` implements the boundary interaction of Sec. 2.3.4 for the
domain of Fig. 2.3 (channel + near-field):
  * any thruster surface (anode, channel walls, front face): neutrals are
    re-emitted with full thermal accommodation at the wall temperature; ions
    first recombine into neutrals, then are re-emitted the same way;
  * domain exits (top and right of the near field): the macroparticle is
    deactivated and its contribution recorded in an `ExitTally` for the
    performance formulas (2.108-2.115);
  * the centerline is a symmetry boundary (the axisymmetric position update
    can never produce r < 0, so nothing needs reflecting explicitly).

All stages are vectorised over the `ParticleArray` storage.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.classes import ParticleArray, Grid2D
from numerical.numerical_funcs import bilinear_gather

E_CHARGE = np.float64(1.602176634e-19)  # elementary charge, C
KB = np.float64(1.380649e-23)           # Boltzmann constant, J/K


@dataclass
class ChannelGeometry:
    """Hall thruster domain of Fig. 2.3 embedded in the full (r, z) grid.

    The acceleration channel occupies r in [r_in, r_out], z in [z_min, z_exit]
    (anode at z_min); everything downstream of z_exit is the near-field, where
    the plasma may fill r in [0, r_max]. The regions r < r_in and r > r_out at
    z < z_exit are solid thruster body.
    """
    r_in: float
    r_out: float
    z_exit: float

    def plasma_mask(self, g: Grid2D) -> np.ndarray:
        """Boolean (N_r, N_z) mask of nodes that belong to the plasma domain."""
        R, Z = g.mesh()
        in_channel = (Z <= self.z_exit) & (R >= self.r_in) & (R <= self.r_out)
        in_near_field = Z >= self.z_exit
        return in_channel | in_near_field


@dataclass
class ExitTally:
    """Per-species accumulators for macroparticles that left through a domain
    exit, feeding the performance formulas (2.108-2.115). Keys are the charge
    state (0, 1, 2). `weight` sums w, `momentum_z` sums m*w*v_z (thrust), and
    `charge_flow` sums q*e*w (ion current).
    """
    weight: dict = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0})
    momentum_z: dict = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0})
    charge_flow: float = 0.0

    def record(self, pa: ParticleArray, idx: np.ndarray) -> None:
        """Record the particles at slots `idx` as having left the domain."""
        charge = pa.charge[idx]
        w = pa.weight[idx]
        mwvz = pa.mass * w * pa.v_z[idx]
        for q in (0, 1, 2):
            m = charge == q
            self.weight[q] += float(w[m].sum())
            self.momentum_z[q] += float(mwvz[m].sum())
        self.charge_flow += float(E_CHARGE * np.sum(charge * w))


def push_particles(
        g: Grid2D,
        particles: ParticleArray,
        dt: float,
        Er: np.ndarray | None = None,
        Ez: np.ndarray | None = None,
        ) -> None:
    """One leapfrog step (2.15-2.16) for every active particle.

    Ions (charge > 0) receive the impulse q*E(x)*dt with E gathered bilinearly
    at the particle position; neutrals only drift. The position update runs in
    local cartesian coordinates (x along r_hat, y along theta_hat), after
    which the radius and the (v_r, v_theta) pair are rotated back to the
    theta = 0 plane:

        r' = sqrt((r + v_r dt)^2 + (v_theta dt)^2),
        v_r'     =  cos(a) v_r + sin(a) v_theta,
        v_theta' = -sin(a) v_r + cos(a) v_theta,   a = atan2(v_theta dt, r + v_r dt).
    """
    n = particles.n
    if n == 0:
        return
    act = particles.active[:n]
    pr = particles.r[:n]
    pz = particles.z[:n]
    vr = particles.v_r[:n]
    vz = particles.v_z[:n]
    vth = particles.v_theta[:n]

    if Er is not None and Ez is not None:
        ch = np.flatnonzero(act & (particles.charge[:n] > 0))
        if ch.size:
            r_nodes = g.r_nodes()
            z_nodes = g.z_nodes()
            Er_p = bilinear_gather(Er, r_nodes, z_nodes, pr[ch], pz[ch])
            Ez_p = bilinear_gather(Ez, r_nodes, z_nodes, pr[ch], pz[ch])
            accel = (particles.charge[ch] * E_CHARGE / particles.mass) * dt
            vr[ch] += accel * Er_p
            vz[ch] += accel * Ez_p

    idx = np.flatnonzero(act)
    x = pr[idx] + vr[idx] * dt
    y = vth[idx] * dt
    r_new = np.hypot(x, y)

    safe = r_new > 0.0
    denom = np.where(safe, r_new, 1.0)
    c = np.where(safe, x / denom, 1.0)
    s = np.where(safe, y / denom, 0.0)
    vr_i = vr[idx]
    vth_i = vth[idx]

    pr[idx] = r_new
    pz[idx] += vz[idx] * dt
    vr[idx] = c * vr_i + s * vth_i
    vth[idx] = -s * vr_i + c * vth_i
    # v_z is unchanged by the in-plane rotation.


def _reemit(
        pa: ParticleArray,
        idx: np.ndarray,
        rng: np.random.Generator,
        T_wall: float,
        normal: str,
        ) -> None:
    """Full thermal accommodation at a wall of temperature `T_wall` for the
    particles at slots `idx`.

    The velocity component along the inward wall normal is drawn from the
    flux-weighted half-Maxwellian, the two tangential components from the full
    Maxwellian -- the same sampling used at injection. `normal` is one of
    '+r', '-r', '+z' (the inward-pointing wall normal). Ions recombine: the
    particle continues as a neutral of the same weight (Sec. 2.3.4).
    """
    m = idx.size
    if m == 0:
        return
    vth = np.sqrt(KB * T_wall / pa.mass)
    v_n = vth * np.sqrt(-2.0 * np.log(1.0 - rng.random(m)))
    v_t1 = rng.normal(0.0, vth, m)
    v_t2 = rng.normal(0.0, vth, m)
    if normal == '+r':
        pa.v_r[idx], pa.v_z[idx], pa.v_theta[idx] = v_n, v_t1, v_t2
    elif normal == '-r':
        pa.v_r[idx], pa.v_z[idx], pa.v_theta[idx] = -v_n, v_t1, v_t2
    else:  # '+z'
        pa.v_z[idx], pa.v_r[idx], pa.v_theta[idx] = v_n, v_t1, v_t2
    pa.charge[idx] = 0
    pa.T[idx] = T_wall


def apply_boundaries(
        g: Grid2D,
        particles: ParticleArray,
        geom: ChannelGeometry,
        rng: np.random.Generator,
        dt: float,
        tally: ExitTally,
        T_wall: float = 1000.0,
        ) -> int:
    """Resolve every active particle that ended the step outside the plasma
    domain; returns the number that left through a domain exit.

    Thruster surfaces re-emit (see `_reemit`); domain exits deactivate the
    particle and record it in `tally`. Whether a particle at z < z_exit and
    r outside the channel hit a radial channel wall or the thruster front
    face is decided from where it started the step (z_old = z - v_z*dt, exact
    for the leapfrog update because v_z is unaffected by the rotation).
    """
    n = particles.n
    if n == 0:
        return 0
    eps_r = 1e-6 * g.h_r
    eps_z = 1e-6 * g.h_z

    act = particles.active[:n]
    r = particles.r[:n]
    z = particles.z[:n]

    # Domain exits (near-field top and right, Fig. 2.3).
    exited = act & ((z >= g.z_max) | ((z > geom.z_exit) & (r >= g.r_max)))
    exit_idx = np.flatnonzero(exited)
    if exit_idx.size:
        tally.record(particles, exit_idx)
        act[exit_idx] = False

    remaining = act & ~exited

    # Anode plane (a thruster surface).
    anode = remaining & (z <= g.z_min)
    anode_idx = np.flatnonzero(anode)
    if anode_idx.size:
        z[anode_idx] = g.z_min + eps_z
        r[anode_idx] = np.clip(r[anode_idx],
                               geom.r_in + eps_r, geom.r_out - eps_r)
        _reemit(particles, anode_idx, rng, T_wall, '+z')

    # Channel walls / thruster front face.
    wall = (remaining & ~anode & (z < geom.z_exit)
            & ((r <= geom.r_in) | (r >= geom.r_out)))
    wall_idx = np.flatnonzero(wall)
    if wall_idx.size:
        z_old = z[wall_idx] - particles.v_z[wall_idx] * dt
        front = z_old >= geom.z_exit
        inner = ~front & (r[wall_idx] <= geom.r_in)
        outer = ~front & ~inner

        front_idx = wall_idx[front]
        z[front_idx] = geom.z_exit + eps_z  # entered from the near field
        _reemit(particles, front_idx, rng, T_wall, '+z')

        inner_idx = wall_idx[inner]
        r[inner_idx] = geom.r_in + eps_r
        _reemit(particles, inner_idx, rng, T_wall, '+r')

        outer_idx = wall_idx[outer]
        r[outer_idx] = geom.r_out - eps_r
        _reemit(particles, outer_idx, rng, T_wall, '-r')

    return int(exit_idx.size)
