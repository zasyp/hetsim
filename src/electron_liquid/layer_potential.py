import numpy as np
import scipy.constants as const

from .lambda_layers import _bin


def node_weight(mu_perp: np.ndarray,
                ne: np.ndarray,
                B_r: np.ndarray,
                B_z: np.ndarray,
                r: np.ndarray,
                dV: np.ndarray,
                ) -> np.ndarray:
    """Per-node contribution to the cross-field conductance integral:

        e * mu_perp * n_e * |grad lambda|^2 * dV,   with |grad lambda| = r|B|.

    r is the (N_r,) radial node array, broadcast along axis 1 of the
    (N_z, N_r) field arrays; the result is summed into layers by
    layer_conductance.
    """
    grad_lambda_sq = (r * np.hypot(B_r, B_z)) ** 2
    return const.elementary_charge * mu_perp * ne * grad_lambda_sq * dV


def layer_conductance(idx: np.ndarray,
                      N_layers: int,
                      d_lambda: float,
                      weight: np.ndarray,
                      ) -> np.ndarray:
    """Face conductances G_{j+1/2} between adjacent layers [A/V].

    Node weights are summed into per-layer conductances G_layer (divided
    by d_lambda^2 to close the finite-difference of grad lambda), then
    each face takes the harmonic mean of its two neighbouring layers --
    the series-resistor rule for current crossing the face. Faces with a
    non-conducting (empty) layer on either side are left at zero.
    """
    G_layer = _bin(idx, weight, N_layers) / d_lambda ** 2
    G_a, G_b = G_layer[:-1], G_layer[1:]

    G_face = np.zeros(N_layers - 1)
    both = np.minimum(G_a, G_b) > 0.0
    G_face[both] = 2.0 * G_a[both] * G_b[both] / (G_a[both] + G_b[both])
    return G_face


def layer_ionization_current(idx: np.ndarray,
                             ne: np.ndarray,
                             nn: np.ndarray,
                             k_iz: np.ndarray,
                             dV: np.ndarray,
                             N_layers: int,
                             ) -> np.ndarray:
    """Ion current born inside each layer [A]: e * n_e * n_n * k_iz * dV,
    summed over the nodes of the layer. This is the source term dI_iz in
    the current-continuity solve (solve_potential).
    """
    elementary_current = const.elementary_charge * ne * nn * k_iz * dV
    return _bin(idx, elementary_current, N_layers)
