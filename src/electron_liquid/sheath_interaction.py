# Sheath / wall interaction — block 5 of the electron-fluid model: the
# electron energy the plasma loses at the dielectric channel walls.
#
# The chain is: the wall floats to a negative sheath potential phi_s
# (sheath_potential) set by the secondary-electron-emission yield gamma
# (see_yield); that potential fixes how many plasma electrons reach the
# wall (wall_electron_flux) and how much energy each carries away
# (wall_energy_loss). For dielectric-wall Hall thrusters this electron
# term is the dominant power-loss mechanism (Goebel & Katz, Fundamentals
# of Electric Propulsion, Ch. 7.3).
#
# Sign convention: phi_s is the ACTUAL wall potential relative to the
# plasma, i.e. it is NEGATIVE. The book's Eqs. 7.3-29/41 quote |phi_s|;
# here we carry the sign so the Boltzmann factor exp(e*phi_s/kTe) comes
# out < 1 without extra bookkeeping.
#
# Unit convention: SI everywhere EXCEPT Te and potentials, which are in
# eV / volts (numerically k_B*T/e). With Te in eV the exponent
# e*phi_s/kTe is just phi_s / Te.

import numpy as np
import scipy.constants as const

from ..structs.classes import WallMaterial, WorkingSubstance

# Space-charge-limited floating potential of a xenon sheath with strong
# secondary emission (Hobbs & Wesson; Goebel & Katz Eq. 7.3-41): the
# most positive the wall sheath is allowed to get, in units of Te.
PHI_SPACE_CHARGE = -1.02
# Slope of the space-charge-limit yield gamma_o = 1 - SEE_SC_SLOPE*sqrt(m/M)
# (Goebel & Katz Eq. 7.3-42); ~0.983 for xenon.
SEE_SC_SLOPE = 8.3


def see_yield(Te, material: WallMaterial):
    """Secondary-electron-emission yield gamma(Te) [dimensionless] for a
    channel wall, Maxwellian-averaged over the incident electrons:

        gamma(Te) = Gamma(2+b) * a * Te^b          (Te in eV)

    (Goebel & Katz, Fundamentals of Electric Propulsion, Eq. 7.3-30). The
    material supplies a, b and the averaging factor Gamma(2+b). The yield
    is left uncapped here -- it may cross 1 for hot electrons, as the raw
    data do; the space-charge limit that clamps the *sheath potential*
    (Hobbs & Wesson) is applied in sheath_potential, not to gamma.
    """
    return material.gamma_2plusb * material.a * np.asarray(Te, dtype=float) ** material.b


def space_charge_limited_yield(gas: WorkingSubstance) -> float:
    """Critical SEE yield gamma_o at which the wall sheath becomes
    space-charge limited (Goebel & Katz Eq. 7.3-42):

        gamma_o = 1 - 8.3 * sqrt(m_e / M)

    Above this yield the emitted-electron space charge in the sheath caps
    the potential at PHI_SPACE_CHARGE*Te regardless of how much larger the
    raw yield gets. ~0.983 for xenon. A diagnostic threshold — the actual
    clamp in sheath_potential is enforced directly on the potential.
    """
    return 1.0 - SEE_SC_SLOPE * np.sqrt(const.electron_mass / gas.mass)


def sheath_potential(Te: np.ndarray, gamma: np.ndarray, gas: WorkingSubstance) -> np.ndarray:
    """Wall floating potential phi_s [V] relative to the plasma, NEGATIVE,
    including secondary emission and the Hobbs & Wesson space-charge limit.

    Classic floating value (Goebel & Katz Eq. 7.3-29, sign carried):

        phi_s = -Te * ln[ (1 - gamma) * sqrt(2M / (pi m_e)) ]

    With no emission (gamma=0) this is the ~-5.97*Te of Ch. 3; as gamma
    rises toward 1 the wall charges less negative and phi_s climbs toward
    the plasma potential. Physically it can never rise above the
    space-charge floor phi_o = PHI_SPACE_CHARGE*Te = -1.02*Te (Eq. 7.3-41,
    Hobbs & Wesson): once the emitted-electron space charge saturates, the
    sheath stops collapsing. We enforce that with a hard cap, which also
    makes the function well-defined for the uncapped gamma>1 that
    see_yield can return (where the bare log would be NaN).

    Te, gamma may be scalars or arrays (broadcast together).
    """
    Te = np.asarray(Te, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    # Keep 1 - gamma strictly positive so the log is finite; any gamma at
    # or above this is space-charge limited anyway and gets capped below.
    one_minus_gamma = np.maximum(1.0 - gamma, 1e-12)
    phi_classic = -Te * np.log(one_minus_gamma * np.sqrt(2 * gas.mass / (np.pi * const.electron_mass)))

    phi_floor = PHI_SPACE_CHARGE * Te                    # -1.02*Te, most positive allowed
    return np.minimum(phi_classic, phi_floor)


def bohm_velocity(Te: np.ndarray, gas: WorkingSubstance) -> np.ndarray:
    """Bohm speed u_B = sqrt(e*Te / M) [m/s] at which ions enter the
    sheath (Goebel & Katz Eq. 7.3-55, Te in eV). The directed ion speed
    the pre-sheath accelerates the cold ions up to before they cross into
    the wall sheath.
    """
    return np.sqrt(const.elementary_charge * np.asarray(Te, dtype=float) / gas.mass)


def electron_thermal_flux(Te: np.ndarray, ne: np.ndarray) -> np.ndarray:
    """One-sided random electron flux (1/4) n_e <v_e> [1/(m^2 s)] toward a
    surface, <v_e> = sqrt(8 e Te / (pi m_e)) the Maxwellian mean speed
    (Te in eV). This is the flux BEFORE the repelling sheath filters it —
    multiply by exp(phi_s/Te) for the flux that actually reaches the wall.
    """
    Te = np.asarray(Te, dtype=float)
    mean_speed = np.sqrt(8 * const.elementary_charge * Te / (np.pi * const.electron_mass))
    return 0.25 * ne * mean_speed


def wall_electron_flux(Te: np.ndarray, ne: np.ndarray, phi_s: np.ndarray) -> np.ndarray:
    """Plasma-electron flux that actually reaches the wall [1/(m^2 s)]:
    the random thermal flux thinned by the Boltzmann factor of the
    repelling sheath (Goebel & Katz Eq. 7.3-52, electron part),

        Gamma_e = (1/4) n_e <v_e> * exp(e*phi_s / kTe) .

    phi_s is negative, so the exponential is < 1. ne is the density at the
    sheath edge (roughly half the channel-centre value because of the
    radial pre-sheath).
    """
    return electron_thermal_flux(Te, ne) * np.exp(np.asarray(phi_s, dtype=float) / np.asarray(Te, dtype=float))


def wall_energy_loss(Te: np.ndarray, ne: np.ndarray, phi_s: np.ndarray) -> np.ndarray:
    """Electron energy flux to the channel wall [W/m^2] — the dominant
    electron-fluid energy sink in a dielectric-wall Hall thruster and the
    electron term of Goebel & Katz Eq. 7.3-45 / 7.3-52:

        q_e = Gamma_e_wall * 2 * Te ,

    i.e. the sheath-filtered electron flux (wall_electron_flux) times the
    mean energy 2*kTe each electron carries across the presheath+sheath.
    Multiply by a layer's wall area (lambda_layers.layer_wall_area) to get
    watts; the term vanishes automatically on plume layers, which own no
    wall area.

    Scales as n_e * Te^(3/2) * exp(phi_s/Te): because phi_s ~ -few*Te the
    exponential is roughly constant, leaving the strong Te^(3/2) growth
    that concentrates wall losses in the hottest part of the channel.

    Secondary-electron cooling of the wall (the cold electrons re-emitted
    from the surface) is neglected, following the book.
    """
    Te = np.asarray(Te, dtype=float)
    energy_per_electron = 2.0 * Te * const.elementary_charge          # 2 kTe [J]
    return wall_electron_flux(Te, ne, phi_s) * energy_per_electron


def wall_ion_energy_loss(Te: np.ndarray,
                         ne: np.ndarray,
                         phi_s: np.ndarray,
                         gas: WorkingSubstance,
                         ion_energy_eV: np.ndarray | None = None,
                         ) -> np.ndarray:
    """Ion energy flux to the channel wall [W/m^2] — the ion term of
    Goebel & Katz Eq. 7.3-45:

        q_i = n_e * u_B * e * (E_ion - phi_s) ,

    the Bohm-speed ion flux carrying its presheath energy E_ion plus the
    energy e|phi_s| it gains falling through the sheath. This is ion
    kinetic energy (drawn from the field, not the electron fluid); it is
    provided for the full wall power budget, not as an electron-fluid
    sink.

    ion_energy_eV : presheath ion energy E_ion [eV]; defaults to the
        space-charge-limited Bohm value 0.58*Te (Eq. 7.3-46). Pass
        0.5*Te for the classic Bohm condition.
    """
    Te = np.asarray(Te, dtype=float)
    phi_s = np.asarray(phi_s, dtype=float)
    if ion_energy_eV is None:
        ion_energy_eV = 0.58 * Te
    # phi_s < 0, so (E_ion - phi_s) = E_ion + |phi_s|, both in volts -> J via e.
    return ne * bohm_velocity(Te, gas) * const.elementary_charge * (ion_energy_eV - phi_s)


def wall_power_density(Te: np.ndarray,
                       ne: np.ndarray,
                       phi_s: np.ndarray,
                       gas: WorkingSubstance,
                       ion_energy_eV: np.ndarray | None = None,
                       ) -> np.ndarray:
    """Total wall power flux [W/m^2], electrons + ions (Goebel & Katz
    Eq. 7.3-45). Convenience sum of wall_energy_loss and
    wall_ion_energy_loss for heat-load estimates; the electron-fluid
    energy equation should use wall_energy_loss alone.
    """
    return (wall_energy_loss(Te, ne, phi_s)
            + wall_ion_energy_loss(Te, ne, phi_s, gas, ion_energy_eV))
