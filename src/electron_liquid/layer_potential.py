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


def layer_conductance_profile(idx: np.ndarray,
                              N_layers: int,
                              d_lambda: float,
                              weight: np.ndarray,
                              ) -> np.ndarray:
    """Per-LAYER cross-field conductance [A/V] (before the faces take their
    harmonic means). Exposed because the thermal-force factor has to be
    averaged onto the faces with these same weights -- see
    face_thermal_force."""
    return _bin(idx, weight, N_layers) / d_lambda ** 2


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
    G_layer = layer_conductance_profile(idx, N_layers, d_lambda, weight)
    G_a, G_b = G_layer[:-1], G_layer[1:]

    G_face = np.zeros(N_layers - 1)
    both = np.minimum(G_a, G_b) > 0.0
    G_face[both] = 2.0 * G_a[both] * G_b[both] / (G_a[both] + G_b[both])
    return G_face


def layer_log_density(idx: np.ndarray,
                      ne: np.ndarray,
                      n_ref: float,
                      weight: np.ndarray,
                      N_layers: int,
                      clamp: float | None = None,
                      ) -> np.ndarray:
    """Per-layer value of ln(n_e / n_ref), averaged over the layer with the
    SAME weight that builds the conductance (node_weight).

    This is the factor in the thermal-force term of the cross-field electron
    current (Koo 2005, Eq. 2.62):

        u_e = mu_perp r B [ dphi*/dlambda + (ln(n_e/n_e*) - 1) dTe/dlambda ]

    and in Koo's Eq. (2.65) it lives INSIDE the same surface integral as the
    conductance -- numerator and denominator both carry e n mu_perp r B dS.
    So the honest per-layer value is the conductance-weighted mean of the
    logarithm, which is what this computes: a plain volume average would let
    cold, empty corners of a layer (where n_e sits on its floor and ln is
    hugely negative) drag the factor down even though they carry no current.

    The average is of the LOGARITHM, not the logarithm of the average, again
    because that is where it sits in the integral.

    n_ref must be the same gauge density that potential_on_grid uses to build
    phi from phi*, or the two halves of the model are using different
    definitions of phi*.

    clamp caps |ln(n_e/n_ref)| at that many e-foldings, and it must be the
    SAME cap potential_on_grid applies to its Boltzmann term, for the same
    reason: this logarithm is the Boltzmann relation, and the population that
    can sit k*Te below the reference is the exp(-k) tail, so past a few
    e-foldings the relation has invalidated itself.

    Without it the term is a startup bomb. In the first microseconds the
    anode layer -- the gauge -- can be nearly empty while the seed plasma
    downstream is not, so ln(n/n_ref) runs to 10-20 instead of ~1, the
    thermal force swamps the electrostatic one, the current it drives raises
    Te, and the two run away together: an unclamped run hit 82 A and 1.6M ion
    macroparticles at t = 5 us before the energy solve overflowed.
    """
    w = _bin(idx, weight, N_layers)
    ln_n = np.log(np.maximum(ne, 1e-300) / n_ref)
    if clamp is not None:
        ln_n = np.clip(ln_n, -clamp, clamp)
    num = _bin(idx, weight * ln_n, N_layers)
    return np.divide(num, w, out=np.zeros_like(num), where=w > 0.0)


def face_thermal_force(ln_n_layers: np.ndarray,
                       G_layer: np.ndarray,
                       G_face: np.ndarray,
                       Te_layers: np.ndarray,
                       ) -> np.ndarray:
    """Thermal-force EMF across every layer face [A]:

        E_{j+1/2} = G_{j+1/2} * f_{j+1/2} * (Te_j - Te_{j+1}) ,
        f = ln(n_e/n_ref) - 1 .

    It is the second term of Koo Eq. (2.62), carried by the same conductance
    as the electrostatic one -- k2/k1 = (ln(n_e/n_e*) - 1) * 2/3 and
    (2/3) d(eps)/dlambda = dTe/dlambda for eps = (3/2)Te, so nothing new has
    to be integrated over the grid; the factor just multiplies G.

    Where it comes from: the thermalized potential is derived assuming Te is
    constant ALONG a field line, which is the whole point of the lambda
    reduction. Nothing licenses treating it as constant ACROSS lines, and Te
    moves by an eV or more from layer to layer, so grad(Te ln n) is not
    Te grad(ln n). This term is the difference.

    f on the FACE is the mean of its two layers weighted by THEIR
    conductances, G_layer -- the same weight the face conductance itself is
    built from, so a nearly-empty neighbour contributes to the factor as
    little as it contributes to the current. The strictly correct object is
    one surface integral over the face (Koo Eq. 2.65), which the layer
    reduction does not keep; faces with two dead neighbours fall back to the
    plain mean, where the whole term is multiplied by G_face = 0 anyway.
    """
    f = ln_n_layers - 1.0
    Ga, Gb = G_layer[:-1], G_layer[1:]
    denom = Ga + Gb
    f_face = np.divide(Ga * f[:-1] + Gb * f[1:], denom,
                       out=0.5 * (f[:-1] + f[1:]), where=denom > 0.0)
    emf = G_face * f_face * (Te_layers[:-1] - Te_layers[1:])

    # Not on the two faces next to the Dirichlet layers. Te there is pinned
    # (Te_anode, Te_cathode ~ 3-4 eV) while the first interior layer runs at
    # 20-40 eV, so the difference across that one face is a jump in the
    # BOUNDARY CONDITION, not a resolved gradient of the profile, and the
    # thermal force would turn it into a current all the same. It was the
    # largest single-face contribution on the path at startup (emf/I = 0.72
    # on the anode face).
    #
    # This is NOT what makes the coupled startup unstable -- that was checked,
    # not assumed. With these two faces zeroed the seed transient still runs
    # away (65 A, 1.46M ions by 5.7 us, against 5.6 A without the term); the
    # growth is inside the channel, where emf/I goes -0.26 -> -0.35 -> -1.03
    # over eight macro-steps. See SolverSettings.thermal_force.
    emf[0] = 0.0
    emf[-1] = 0.0
    return emf


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


def layer_currents(G_face: np.ndarray, phi_star: np.ndarray,
                   emf_face: np.ndarray | None = None) -> np.ndarray:
    """Electron conduction current across every layer face [A], counted
    POSITIVE DOWNSTREAM (anode -> cathode):

        I_j = G_{j+1/2} (phi*_j - phi*_{j+1}) .

    It is conventional current, so it points along E while the electrons
    drift the other way, toward the anode. With the continuity equation
    solved correctly (see solve_potential) I is largest at the anode face
    and falls off downstream by the ion current born on the way:

        I_anode_face - I_cathode_face = sum(dI_iz) .

    emf_face, if given, is the thermal-force term of face_thermal_force,
    already in amps: the current is then

        I_j = G_{j+1/2}(phi*_j - phi*_{j+1}) + E_{j+1/2} .

    Pass it whenever solve_potential was given the same array, or the current
    reported here stops being the current the potential was solved for -- and
    both the energy equation's convection and the discharge-current balance
    read this number.

    Two things read this. The energy equation needs it to convect electron
    enthalpy (5/2)Te per electron across the same faces, and the discharge
    current balance needs I[0] -- the electron back-current the anode
    collects.
    """
    I = G_face * (phi_star[:-1] - phi_star[1:])
    return I if emf_face is None else I + emf_face
