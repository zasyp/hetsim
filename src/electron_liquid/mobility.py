# Magnetized electron transport closures — block 2 of the electron-fluid
# model: turning the total collision frequency (block 1) into cross-field
# mobility and diffusion.
#
# Module convention: these functions know nothing about B itself, only
# about frequencies — omega_ce is computed once by the caller
# (default_plasm_params.omega_ce) and passed in.
#
# Unit convention: SI everywhere EXCEPT Te, which is in eV.

import numpy as np
import scipy.constants as const


def zeroB_mobility(nu_e:np.ndarray) -> np.ndarray:
    """Unmagnetized electron mobility mu_0 = e / (m_e * nu_e) [m^2/(V s)]:
    the more often the electron is scattered, the slower it drifts."""
    return const.elementary_charge / (const.electron_mass * nu_e)


def hall_parameter(omega_ce:np.ndarray, nu_e:np.ndarray) -> np.ndarray:
    """Omega = omega_ce / nu_e — gyrations per collision. Omega >> 1
    means the electron is tied to its field line and only steps across B
    when a collision kicks it."""
    return omega_ce / nu_e


def perp_mobility(mu0:np.ndarray, Omega:np.ndarray) -> np.ndarray:
    """Cross-field mobility mu_perp = mu_0 / (1 + Omega^2) [m^2/(V s)].

    Limits worth remembering: Omega -> 0 recovers mu_0; for Omega >> 1
    mu_perp ~ m_e nu / (e B^2) GROWS with the collision frequency —
    collisions are what moves a magnetized electron across B. With the
    Bohm closure nu = omega_ce/16 this collapses to mu_perp ~ 1/(16 B),
    independent of the gas.
    """
    return mu0 / (1 + Omega ** 2)


def perp_diffusion(mu_perp:np.ndarray, Te:np.ndarray) -> np.ndarray:
    """Einstein relation D_perp = mu_perp * Te [m^2/s]. With Te in eV
    (numerically k_B T / e in volts) the product needs no conversion."""
    return mu_perp * Te


def perp_thermal_conductivity(ne:np.ndarray,
                              Te:np.ndarray,
                              mu_perp:np.ndarray,
                              coeff:float = 2.5,
                              ) -> np.ndarray:
    """Cross-field electron thermal conductivity kappa_perp [A/m], the
    thermal counterpart of perp_mobility/perp_diffusion, defined so that
    the electron heat flux is

        q_e = -kappa_perp * grad(Te)        [W/m^2]   (Te in eV = volts)

    with the Einstein-style closure

        kappa_perp = coeff * e * n_e * Te * mu_perp .

    Because it inherits mu_perp = mu_0 / (1 + Omega^2), the conduction is
    quenched across the strong-field region exactly like the particle
    mobility, so heat (like current) crosses B only where collisions or
    anomalous transport allow it. coeff is the O(1) Braginskii-type number
    (~2.5 for a Maxwellian; tune against data). Units work out to amps per
    metre so that kappa_perp * grad(Te)[V/m] is W/m^2 and the layer
    machinery (layer_conductance) yields a thermal conductance in W/V.
    """
    return coeff * const.elementary_charge * ne * Te * mu_perp
