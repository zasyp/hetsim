# Basic plasma parameters shared across the electron-fluid module.
# Unit convention: SI everywhere EXCEPT the electron temperature Te,
# which is in eV (numerically equal to k_B*T/e in volts).

import numpy as np
import scipy.constants as const


def omega_ce(B:np.ndarray) -> np.ndarray:
    """Electron cyclotron frequency |e|B/m_e [1/s]; sign of B ignored."""
    return np.abs(const.elementary_charge * B / const.electron_mass)


def debye_radius(Te:np.ndarray, electron_concentration:np.ndarray) -> np.ndarray:
    """Electron Debye length [m], Te in eV:

        lambda_D = sqrt(eps0 * k_B T / (n_e e^2)) = sqrt(eps0 * Te / (n_e * e)),

    the second form because with Te in eV the thermal energy k_B T equals
    e * Te, cancelling one factor of e.
    """
    return np.sqrt(
        const.epsilon_0 * Te / (electron_concentration * const.elementary_charge)
    )
