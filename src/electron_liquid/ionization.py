# Ionization operator — the plasma source of the model. Everything else
# in the electron-fluid model consumes the plasma; this is where it is
# born. Electron-impact ionization of a neutral,
#
#     e + A  ->  2e + A+ ,
#
# couples the three species through the single volumetric rate
#
#     S_iz = n_e * n_n * k_iz(T_e)        [1/(m^3 s)]
#
# (k_iz the Maxwellian rate coefficient carried by the propellant). One
# event adds one electron-ion pair and removes one neutral, so S_iz is
# simultaneously the ion/electron continuity source (+S_iz), the neutral
# continuity sink (-S_iz), the ion source of the current-continuity solve
# (layer_potential.layer_ionization_current), and — times the effective
# cost E_c(T_e) — the electron energy sink (power_balance.
# ionization_power_density).
#
# Two faces are provided:
#   * an Eulerian field S_iz(z, r) for the fluid/continuity/potential/
#     energy solves (ionization_source and the continuity helpers);
#   * an MCC particle operator (apply_ionization) that converts a random
#     subset of neutral macroparticles into ions each step, for the PIC
#     heavy-species side.
#
# Unit convention: SI everywhere EXCEPT T_e, which is in eV.

import numpy as np

from ..deposition.deposit import locate_particle, gather
from ..structs.classes import Grid2D, ParticleArray, WorkingSubstance


# --------------------------------------------------------------------------
# Eulerian (fluid) face
# --------------------------------------------------------------------------

def ionization_source(Te: np.ndarray,
                      n_e: np.ndarray,
                      n_n: np.ndarray,
                      gas: WorkingSubstance,
                      ) -> np.ndarray:
    """Volumetric ionization rate S_iz = n_e n_n k_iz(Te) [1/(m^3 s)].

    The single quantity every other coupling is built from. Evaluate it on
    the grid from the current electron temperature, plasma density and
    neutral density, then feed the continuity helpers below and the
    electron energy sink (power_balance.ionization_power_density).
    """
    return gas.volumetric_ionization_rate_Te(n_e, n_n, Te)


def ion_continuity_source(S_iz: np.ndarray) -> np.ndarray:
    """dn_i/dt|_iz = +S_iz [1/(m^3 s)] — and, by quasineutrality of a
    singly-charged plasma, dn_e/dt|_iz too. Trivial, but names the sign
    and pairs with neutral_continuity_sink so callers can't mismatch them.
    """
    return S_iz


def neutral_continuity_sink(S_iz: np.ndarray) -> np.ndarray:
    """dn_n/dt|_iz = -S_iz [1/(m^3 s)]: every ionization removes one
    neutral. Add this to the neutral continuity equation (or deplete the
    neutral macroparticle weight by S_iz * dV * dt per cell in PIC).
    """
    return -S_iz


def ionization_length(Te: np.ndarray,
                      n_e: np.ndarray,
                      gas: WorkingSubstance,
                      neutral_speed: float | np.ndarray,
                      ) -> np.ndarray:
    """Mean distance a neutral of speed `neutral_speed` [m/s] travels
    before it is ionized, L_iz = v_n / (n_e k_iz(Te)) [m]. A diagnostic:
    for a well-designed thruster it should come out on the order of the
    channel length (mm-cm), i.e. most propellant is ionized inside the
    channel. Ionization frequency per neutral is nu = n_e k_iz(Te).
    """
    nu = np.maximum(n_e * gas.ionization_rate_Te(Te), 1e-30)
    return neutral_speed / nu


# --------------------------------------------------------------------------
# MCC (particle) face
# --------------------------------------------------------------------------

def ionization_probability(Te_at_particle: np.ndarray,
                           ne_at_particle: np.ndarray,
                           gas: WorkingSubstance,
                           dt: float,
                           ) -> np.ndarray:
    """Probability that a given neutral macroparticle ionizes during dt:

        P = 1 - exp(-nu dt),   nu = n_e k_iz(Te)   [per-neutral rate, 1/s],

    with n_e and Te sampled (gathered) at the particle's own position.
    -expm1 keeps it accurate for the usual small nu*dt. Note nu here is the
    ionization frequency seen by ONE neutral (proportional to n_e), which
    is the electron-collision nu_iz divided by n_n — do not confuse it with
    collisions.ionization_collision (the electron's frequency, ~n_n).
    """
    nu = ne_at_particle * gas.ionization_rate_Te(Te_at_particle)
    return -np.expm1(-nu * dt)


def apply_ionization_weighted(neutrals: ParticleArray,
                              ions: ParticleArray,
                              grid: Grid2D,
                              Te: np.ndarray,
                              n_e: np.ndarray,
                              gas: WorkingSubstance,
                              dt: float,
                              ion_weight: float,
                              rng: np.random.Generator | None = None,
                              ) -> int:
    """MCC ionization with SEPARATE ion and neutral macroparticle weights.

    Why this exists alongside apply_ionization: in a Hall thruster n_n is two
    orders of magnitude above n_i (1e19 vs 1e17). One shared weight therefore
    cannot serve both — sized for the neutrals it leaves under one ion
    macroparticle per cell, and sized for the ions it needs ten million
    neutrals. The fix is the standard one: give ions their own, smaller weight
    and transfer weight rather than whole particles.

    Each neutral macroparticle of weight w spawns

        N ~ Poisson(w * nu * dt / ion_weight),   nu = n_e k_iz(Te),

    ions of weight `ion_weight`, and loses N*ion_weight from its own weight.
    The expected transferred weight is w*nu*dt, exactly the physical rate, so
    the source is unbiased regardless of the weight ratio. Poisson rather than
    Bernoulli because with a ratio of ~50 the per-particle expectation is no
    longer small, and a coin flip would silently cap the rate at one event per
    macroparticle per step.

    A neutral worn down below one ion weight cannot transfer again, so it is
    resolved by Russian roulette: promoted to a full-weight ion with
    probability w/ion_weight, deleted otherwise. Unbiased in expectation, and
    it stops the population filling up with immortal near-zero-weight
    particles. Returns the number of newborn ion macroparticles.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(neutrals) == 0:
        return 0

    left_z, lower_r, wr, wu = locate_particle(neutrals, grid)
    Te_p = gather(Te, left_z, lower_r, wr, wu)
    ne_p = gather(n_e, left_z, lower_r, wr, wu)
    nu = ne_p * gas.ionization_rate_Te(Te_p)

    lam = neutrals.weight * nu * dt / ion_weight
    # a macroparticle cannot give away more weight than it has
    lam = np.minimum(lam, neutrals.weight / ion_weight)
    counts = rng.poisson(np.maximum(lam, 0.0))
    counts = np.minimum(counts, np.floor(neutrals.weight / ion_weight)).astype(np.int64)

    n_new = int(counts.sum())
    if n_new:
        born = ParticleArray(
            z=np.repeat(neutrals.z, counts),
            r=np.repeat(neutrals.r, counts),
            v_z=np.repeat(neutrals.v_z, counts),
            v_r=np.repeat(neutrals.v_r, counts),
            v_theta=np.repeat(neutrals.v_theta, counts),
            weight=np.full(n_new, ion_weight),
        )
        ions.extend(born)
        neutrals.weight -= counts * ion_weight

    # Russian roulette on the worn-down remainder
    low = neutrals.weight < ion_weight
    if low.any():
        promote = np.zeros(len(neutrals), dtype=bool)
        promote[low] = rng.random(int(low.sum())) < neutrals.weight[low] / ion_weight
        if promote.any():
            k = int(promote.sum())
            ions.extend(ParticleArray(
                neutrals.z[promote].copy(), neutrals.r[promote].copy(),
                neutrals.v_z[promote].copy(), neutrals.v_r[promote].copy(),
                neutrals.v_theta[promote].copy(), np.full(k, ion_weight)))
            n_new += k
        neutrals.keep(~low)
    return n_new


def apply_ionization(neutrals: ParticleArray,
                     ions: ParticleArray,
                     grid: Grid2D,
                     Te: np.ndarray,
                     n_e: np.ndarray,
                     gas: WorkingSubstance,
                     dt: float,
                     rng: np.random.Generator | None = None,
                     ) -> int:
    """One MCC ionization step. Each neutral macroparticle ionizes with
    probability ionization_probability evaluated at its local (bilinearly
    gathered) Te and n_e; every newborn ion inherits the neutral's
    position, velocity and weight — ionization barely changes the heavy
    particle's momentum — and the ionized neutrals are removed. Mutates
    both arrays in place (neutrals shrink, ions grow) and returns the
    number of events.

    Assumes ions and neutrals share one macroparticle weight (as the
    injector sets it); with unequal weights the per-particle 1:1 swap
    would break particle conservation and a weight-split scheme is needed
    instead.

    Te and n_e are full (N_z, N_r) grid fields; n_e is the value gather
    sees, so pass the sheath-edge / bulk density consistently with the
    rest of the model.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(neutrals) == 0:
        return 0

    left_z, lower_r, wr, wu = locate_particle(neutrals, grid)
    Te_p = gather(Te, left_z, lower_r, wr, wu)
    ne_p = gather(n_e, left_z, lower_r, wr, wu)

    P = ionization_probability(Te_p, ne_p, gas, dt)
    hit = rng.random(len(neutrals)) < P
    n_events = int(np.count_nonzero(hit))
    if n_events == 0:
        return 0

    born = ParticleArray(
        z=neutrals.z[hit].copy(),
        r=neutrals.r[hit].copy(),
        v_z=neutrals.v_z[hit].copy(),
        v_r=neutrals.v_r[hit].copy(),
        v_theta=neutrals.v_theta[hit].copy(),
        weight=neutrals.weight[hit].copy(),
    )
    ions.extend(born)
    neutrals.keep(~hit)
    return n_events
