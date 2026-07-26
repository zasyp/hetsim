# Electron energy equation — block 6, the closure that sets T_e. Electrons
# conduct heat freely along field lines and barely across them, so (like
# the potential in block 4) T_e is one unknown per lambda layer, and the
# balance is integrated over each layer:
#
#     d/dlambda[ G^T dTe/dlambda ]  +  P_ohmic  -  (W_wall + W_ion + W_rad) = 0
#
# i.e. cross-field heat conduction between neighbouring layers, fed by
# ohmic heating of the cross-field electron current and drained by the
# wall sheath (block 5) and the inelastic collisions (ionization +
# radiation). The discrete form is the same tridiagonal system as
# solve_potential, with a thermal conductance G^T in place of the
# electrical one and the loss terms linearized onto the diagonal for
# stability.
#
# Unit convention: SI everywhere EXCEPT Te, in eV. Powers are in watts;
# with Te in eV a "conductance" G^T [W/V] gives G^T*dTe in watts.

import numpy as np
import scipy.constants as const

from ..utils.utils import thomas_alg
from .lambda_layers import _bin


# --- heating source -------------------------------------------------------

def ohmic_heating(ne: np.ndarray,
                  mu_perp: np.ndarray,
                  E_z: np.ndarray,
                  E_r: np.ndarray,
                  ) -> np.ndarray:
    """Ohmic (Joule) heating density of the cross-field electron current
    [W/m^3]:

        p = j_e . E = sigma_perp |E|^2 = e n_e mu_perp (E_z^2 + E_r^2) ,

    sigma_perp = e n_e mu_perp the cross-field electron conductivity. This
    is the electrons being heated as they migrate down the field they are
    trying to hold back — the dominant electron heating in a Hall thruster,
    peaking in the low-mobility acceleration layer where |E| is largest.
    """
    return const.elementary_charge * ne * mu_perp * (E_z ** 2 + E_r ** 2)


# --- cross-field thermal conductance on the layers ------------------------

def node_thermal_weight(kappa_perp: np.ndarray,
                        B_r: np.ndarray,
                        B_z: np.ndarray,
                        r: np.ndarray,
                        dV: np.ndarray,
                        ) -> np.ndarray:
    """Per-node contribution to the cross-field thermal conductance,
    kappa_perp * |grad lambda|^2 * dV with |grad lambda| = r|B| — the exact
    thermal analogue of layer_potential.node_weight (which carries
    e mu_perp n_e instead of kappa_perp). Feed the result to
    layer_potential.layer_conductance to get the face conductances G^T.
    """
    grad_lambda_sq = (r * np.hypot(B_r, B_z)) ** 2
    return kappa_perp * grad_lambda_sq * dV


# --- loss terms integrated per layer --------------------------------------

def layer_power(idx: np.ndarray, p_density: np.ndarray, dV: np.ndarray,
                N_layers: int) -> np.ndarray:
    """Integrate a volumetric power density [W/m^3] over each layer -> [W].
    Used for the ohmic source and the volumetric inelastic sinks."""
    return _bin(idx, p_density * dV, N_layers)


def solve_electron_energy(G_T_face: np.ndarray,
                          source: np.ndarray,
                          loss_coeff: np.ndarray,
                          Te_anode: float,
                          Te_cathode: float,
                          ) -> np.ndarray:
    """Solve the layer energy balance for Te [eV], one value per layer.

    Interior layers balance conduction across their two faces against the
    net local power, with the loss terms linearized as loss = loss_coeff*Te
    (a conductance to a Te=0 sink) so they sit on the diagonal and keep the
    system diagonally dominant:

        G^T_{j-1/2}(Te_{j-1}-Te_j) + G^T_{j+1/2}(Te_{j+1}-Te_j)
            + source_j - loss_coeff_j * Te_j = 0 .

    source [W] is the per-layer ohmic heating (>=0); loss_coeff [W/V] is
    (wall + ionization + radiation power at the previous Te) / Te, so the
    solve is one Picard step of the nonlinear balance — iterate to
    convergence. The anode and cathode layers are Dirichlet-pinned to
    Te_anode / Te_cathode. Mirrors solve_potential.
    """
    N = len(source)
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    lower[1:-1] = G_T_face[:-1]
    upper[1:-1] = G_T_face[1:]
    diag[1:-1] = -(G_T_face[:-1] + G_T_face[1:] + loss_coeff[1:-1])
    rhs[1:-1] = -source[1:-1]

    diag[0] = 1.0
    rhs[0] = Te_anode
    diag[-1] = 1.0
    rhs[-1] = Te_cathode

    return thomas_alg(lower, diag, upper, rhs)
