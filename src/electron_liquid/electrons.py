import numpy as np
import scipy.constants as cst


def hall_parameter(B:np.ndarray, nu_e:np.ndarray) -> np.ndarray:
    """Omega_e = omega_ce / nu_e, the electron magnetization parameter.
    nu_e should already be the total collision frequency (classical
    electron-neutral + electron-ion, plus anomalous/near-wall terms).
    """
    omega_ce = cst.e * B / cst.m_e
    return omega_ce / nu_e


def perp_mobility(B:np.ndarray, nu_e:np.ndarray) -> np.ndarray:
    """Electron mobility across the magnetic field:
    mu_e_perp = mu_e0 / (1 + Omega_e**2), with mu_e0 = e/(m_e*nu_e) the
    unmagnetized mobility and Omega_e the Hall parameter.
    """
    Omega_e = hall_parameter(B, nu_e)
    mu_e0 = cst.e / (cst.m_e * nu_e)
    return mu_e0 / (1 + Omega_e**2)


def perp_diffusion(mu_e_perp:np.ndarray, Te:np.ndarray) -> np.ndarray:
    """D_e_perp = mu_e_perp * Te / e (Einstein relation, Te in eV)."""
    return mu_e_perp * Te / cst.e