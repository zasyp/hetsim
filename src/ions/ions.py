# Ion species: electrostatic push and surface boundaries.
#
# Ions are the missing half of the discharge. Neutrals fly straight and are
# pushed by nothing (neutrals/neutrals.py); electrons are a fluid (block 1-6).
# Ions are in between: kinetic macroparticles like the neutrals, but they feel
# the electric field the electron fluid produces. That closes the loop —
# E from the electrons accelerates the ions, the ions deposit n_i, and n_e = n_i
# replaces PlasmaState.placeholder.
#
# What is deliberately NOT here:
#   * no magnetic force. The Xe+ Larmor radius r_L = M v / (e B) at 20 km/s is
#     2.7 m at 100 G and 1.8 m at 150 G -- fifty to eighty times the channel
#     radius. Over one 0.5 mm cell the field bends the trajectory by ~0.02-0.03%,
#     far below the statistical noise of a PIC push. (An earlier version of
#     this comment said ~20 cm and ~0.2%: those are the numbers for B = 0.15 T,
#     i.e. 1500 G -- a gauss/millitesla slip. The conclusion only gets safer.)
#   * no ion-ion or ion-neutral collisions. Charge exchange matters for plume
#     divergence, not for the discharge; add it later as an MCC step.
#   * no ionization. That lives in electron_liquid/ionization.py (apply_ionization),
#     which converts neutral macroparticles into ions. This module only moves
#     ions that already exist and decides what happens when they hit something.
#
# Boundary policy, stated once because everything below follows from it:
#
#   anode (z < 0)        ion recombines, comes back as a wall-thermal NEUTRAL
#   channel walls        same: recombine, return to the volume as a neutral
#   thruster front face  same
#   plume (z, r > grid)  gone for good — this is the beam, i.e. the thrust
#   axis (r -> 0)        nothing to do; free_flight already rotates through it
#
# The point of recombining rather than deleting: propellant must not leak out
# of the model. Every ion that reaches a wall gives its atom back to the gas,
# and that returned flux is a large part of the neutral density near the anode.
# Deleting them instead would quietly under-feed the ionization source.

import numpy as np

from ..structs.classes import Grid2D, Thruster, ParticleArray
from ..deposition.deposit import locate_particle, gather
from ..neutrals.neutrals import free_flight, sample_wall_flux


def push_ions(part: ParticleArray, grid: Grid2D,
              E_z: np.ndarray, E_r: np.ndarray,
              q_over_m: float, dt: float) -> None:
    """Kick-drift push of the ion macroparticles in the electron-fluid field.

    E_z, E_r are the (N_z, N_r) node fields from PlasmaState; they are gathered
    to each particle bilinearly, the velocity takes the kick, then free_flight
    does the drift (and the r-theta rotation back into the plane).

    First order in dt by construction: the kick uses the field at the OLD
    position. That is the standard Euler-Cromer form and it is stable here
    because the ion crosses a cell in many steps. If second order is ever
    needed, offset the velocity by half a step once at startup and this same
    call becomes leapfrog — no other change.

    q_over_m : ion charge-to-mass ratio [C/kg], e / m_Xe for singly charged.
    """
    if len(part) == 0:
        return
    iz, ir, wz, wr = locate_particle(part, grid)
    Ez_p = gather(E_z, iz, ir, wz, wr)
    Er_p = gather(E_r, iz, ir, wz, wr)

    part.v_z += q_over_m * Ez_p * dt
    part.v_r += q_over_m * Er_p * dt

    part.z, part.r, part.v_r, part.v_theta = free_flight(
        part.z, part.r, part.v_z, part.v_r, part.v_theta, dt
    )


def _born_neutrals(part: ParticleArray, hit: np.ndarray, v_th: float,
                   axis: str, sign: float) -> ParticleArray:
    """The wall-thermal neutrals left behind by the ions selected by `hit`.

    Positions are taken as-is (the caller has already snapped them onto the
    surface); velocities are drawn from the same diffuse re-emission law the
    neutrals use, with the normal component along `axis` in direction `sign`.
    Weights carry over one-for-one — one ion in, one atom out.
    """
    k = int(hit.sum())
    if k == 0:
        return ParticleArray()
    v_n, t1, t2 = sample_wall_flux(k, v_th)
    if axis == "z":
        v_z, v_r, v_theta = sign * v_n, t1, t2
    else:
        v_z, v_r, v_theta = t1, sign * v_n, t2
    return ParticleArray(part.z[hit].copy(), part.r[hit].copy(),
                         v_z, v_r, v_theta, part.weight[hit].copy())


def apply_ion_boundaries(part: ParticleArray, thruster: Thruster, grid: Grid2D,
                         z_prev: np.ndarray,
                         v_th_anode: float, v_th_wall: float):
    """Resolve every ion that left the plasma volume during the last push.

    Mirrors neutrals.apply_boundaries in shape and in the z_prev trick — the
    position BEFORE the push tells whether a particle was inside the channel
    annulus or out in the plume, which is what decides whether crossing
    r = r_min is a wall impact or just free flight past the pole piece.

    Returns (alive, neutrals, beam, tally):
      alive    boolean mask of ions that stay in the ion population. The caller
               applies it: part.keep(alive). Everything False either left
               through the plume or turned back into gas.
      neutrals ParticleArray of the atoms handed back to the neutral
               population; the caller does neutral_part.extend(neutrals).
      beam     ParticleArray of the ions that left through the plume, handed
               out whole rather than as a number. They ARE the thrust: their
               axial momentum flux is what the thruster pushes against, and
               their velocity and angle distributions are the only place an
               exit velocity or a divergence angle can come from. Summing them
               into a scalar here would throw that away (see ions/beam.py).
      tally    dict of summed weights (real particles, not macroparticles) per
               destination: 'anode', 'inner_wall', 'outer_wall', 'front_face',
               'beam'. Divide by dt and multiply by e to get currents — the
               anode and beam entries are what the discharge current balance
               needs, the wall entries feed the sheath power budget.

    The order of the tests matters: a particle is resolved once, by the first
    surface it satisfies. Escape is checked first so an ion that leaves through
    the plume corner is never also counted as a wall hit.
    """
    L = thruster.channel_length
    n = len(part)
    if n == 0:
        return np.zeros(0, dtype=bool), ParticleArray(), ParticleArray(), {}

    alive = np.ones(n, dtype=bool)
    tally = {}
    pieces = []
    z_edges = grid.z_nodes()

    def _resolve(hit, key, v_th=None, axis=None, sign=None):
        """Book-keep one destination; recombine into gas unless v_th is None.

        Surfaces also get an AXIAL HISTOGRAM of what landed on them, under
        '<key>_zhist'. A scalar wall current says the model is losing ions;
        only the profile says whether they are lost where they are born (poor
        confinement everywhere) or where they are accelerated (the field is
        pushing them into the wall), and those have opposite fixes. The caller
        sums these the same way it sums the scalars — numpy arrays add.
        """
        tally[key] = float(part.weight[hit].sum())
        if key != "beam":
            tally[key + "_zhist"] = np.histogram(
                part.z[hit], bins=z_edges, weights=part.weight[hit])[0]
        if v_th is not None:
            pieces.append(_born_neutrals(part, hit, v_th, axis, sign))
        alive[hit] = False

    # --- 1) open plume boundary: the beam. Gone, no atom returned.
    escaped = alive & ((part.z > grid.max_z) | (part.r > grid.max_r))
    beam = ParticleArray(part.z[escaped].copy(), part.r[escaped].copy(),
                         part.v_z[escaped].copy(), part.v_r[escaped].copy(),
                         part.v_theta[escaped].copy(),
                         part.weight[escaped].copy())
    _resolve(escaped, "beam")

    # --- 2) anode: snap to z = 0, re-emit along +z at anode temperature
    hit = alive & (part.z < 0.0)
    part.z[hit] = 0.0
    _resolve(hit, "anode", v_th_anode, "z", +1.0)

    was_in_plume = z_prev > L
    in_channel = alive & ~was_in_plume & (part.z <= L)

    # --- 3) inner channel wall: snap to r_min, re-emit along +r
    hit = in_channel & (part.r < thruster.r_min)
    part.r[hit] = thruster.r_min
    _resolve(hit, "inner_wall", v_th_wall, "r", +1.0)

    # --- 4) outer channel wall: snap to r_max, re-emit along -r
    hit = in_channel & (part.r > thruster.r_max)
    part.r[hit] = thruster.r_max
    _resolve(hit, "outer_wall", v_th_wall, "r", -1.0)

    # --- 5) thruster front face: a plume ion flying back onto the pole pieces
    hit = alive & was_in_plume & (part.z < L) & (
        (part.r < thruster.r_min) | (part.r > thruster.r_max))
    part.z[hit] = L
    _resolve(hit, "front_face", v_th_wall, "z", +1.0)

    neutrals = ParticleArray()
    for p in pieces:
        neutrals.extend(p)
    return alive, neutrals, beam, tally


def cfl_dt(grid: Grid2D, thruster: Thruster, courant: float = 0.4) -> float:
    """Largest ion time step that keeps the fastest ion inside `courant` cells.

    The fastest ion is one that fell through the whole discharge voltage:
    v_max = sqrt(2 q V_d / m). For SPT-70 (Xe, 300 V, 0.5 mm cells) this gives
    ~1e-8 s. Nothing here is stiff, so this is the only step constraint the
    ions impose — unlike a full PIC, where the electron plasma frequency would
    dominate. That is exactly the saving a hybrid model buys.
    """
    import scipy.constants as cst

    dz = (grid.max_z - grid.min_z) / (grid.N_z - 1)
    dr = (grid.max_r - grid.min_r) / (grid.N_r - 1)
    v_max = np.sqrt(2 * cst.elementary_charge * thruster.voltage / thruster.mass)
    return courant * min(dz, dr) / v_max
