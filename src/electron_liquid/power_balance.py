# Global discharge power balance — the terms of Goebel & Katz Eq. 7.3-27,
#
#     P_d = P_b + P_w + P_a + P_R + P_ion ,
#
# used to close the electron-fluid energy equation: the peak channel
# electron temperature is the value that makes the power fed into the
# plasma (plasma_power) equal the sum of the loss terms below
# (Fundamentals of Electric Propulsion, Ch. 7.3.4).
#
# The wall term P_w is built per-area in sheath_interaction; this module
# holds the remaining, mostly volume/current-integrated terms.
#
# Unit convention: SI everywhere EXCEPT Te and potentials/thresholds,
# which are in eV / volts. A power written as (energy in eV) * (current in
# A) comes out in watts directly, because eV / e = volts.

import numpy as np
import scipy.constants as const

from ..structs.classes import WorkingSubstance


def discharge_power(I_d: float, V_d: float) -> float:
    """Input power P_d = I_d * V_d [W] from the discharge supply
    (Goebel & Katz, text above Eq. 7.3-50)."""
    return I_d * V_d


def beam_power(eta_b: float, eta_v: float, I_d: float, V_d: float) -> float:
    """Useful beam (thrust) power P_b = eta_b * eta_v * I_d * V_d [W]
    (Goebel & Katz Eq. 7.3-50), eta_b the current-utilization and eta_v
    the voltage-utilization efficiency. Equivalently eta_v * I_b * V_d
    with the beam current I_b = eta_b * I_d.
    """
    return eta_b * eta_v * I_d * V_d


def plasma_power(eta_b: float, I_d: float, V_d: float) -> float:
    """Power remaining in the plasma to create/heat it and feed the
    losses, P_p = (1 - eta_b) * I_d * V_d [W] (Goebel & Katz Eq. 7.3-51).
    This is the left-hand side the loss terms are balanced against.
    """
    return (1.0 - eta_b) * I_d * V_d


def anode_power(Te_anode: float, I_d: float) -> float:
    """Electron power deposited on the anode P_a = 2 * Te_anode * I_d [W]
    (Goebel & Katz Eq. 7.3-47/7.3-53): the discharge electron current is
    collected at the anode, each electron dumping ~2*kTe. Te_anode in eV
    is evaluated near the anode. Dominant loss in metallic-wall TAL
    thrusters.
    """
    return 2.0 * Te_anode * I_d


def anode_power_from_beam(Te_anode: float, I_b: float, eta_b: float) -> float:
    """Anode power written via the beam current, P_a = 2*Te_anode*I_b/eta_b
    [W] (Goebel & Katz Eq. 7.3-49), using eta_b = I_b / I_d. For the
    typical eta_b = 0.6-0.8 this is 3-4x (Te_anode * I_b).
    """
    return 2.0 * Te_anode * I_b / eta_b


def radiated_power(Te: np.ndarray,
                   n_neutral: np.ndarray,
                   n_e: np.ndarray,
                   volume: float,
                   propellant: WorkingSubstance,
                   ) -> np.ndarray:
    """Excitation/radiation power loss P_R = n_o n_e <sigma* v_e> E_exc V
    [W] (Goebel & Katz Eq. 7.3-54): every electron-impact excitation of a
    neutral radiates away roughly the excitation threshold E_exc. Built
    from the propellant's Maxwellian excitation rate coefficient k_exc(Te)
    [m^3/s] and threshold E_exc [eV]. Negligible at the high channel Te
    where wall losses dominate, but included for completeness.
    """
    k_exc = propellant.k_exc(Te)
    E_exc_J = propellant.E_exc * const.elementary_charge
    return n_neutral * n_e * k_exc * E_exc_J * volume


def ionization_power_density(Te: np.ndarray,
                             n_neutral: np.ndarray,
                             n_e: np.ndarray,
                             propellant: WorkingSubstance,
                             ) -> np.ndarray:
    """Volumetric electron energy sink to ionization (+ the excitation
    losses bundled into the effective cost) [W/m^3]:

        p_ion = n_e * n_n * k_iz(Te) * E_c(Te) * e ,

    with k_iz the Maxwellian ionization rate coefficient and E_c the
    effective energy cost per ionization event (propellant.ionization_
    energy_cost, which already folds in excitation losses). Integrate over
    the plasma volume for the P_ion of Eq. 7.3-56.
    """
    k_iz = propellant.ionization_rate_Te(Te)
    E_c_J = propellant.ionization_energy_cost(Te) * const.elementary_charge
    return n_e * n_neutral * k_iz * E_c_J


def ionization_power(I_b: float, I_wall_ion: float, U_plus: float) -> float:
    """Power spent producing the ions, P_ion = (I_b + I_iw) * U+ [W]
    (Goebel & Katz Eq. 7.3-56): every ion that leaves as beam (I_b) or
    recombines at a wall (I_iw) cost one ionization potential U+ [V] to
    make. A current-based alternative to integrating ionization_power_
    density over the channel volume.
    """
    return (I_b + I_wall_ion) * U_plus
