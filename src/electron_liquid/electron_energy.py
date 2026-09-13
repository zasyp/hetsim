# Electron energy equation — block 6, the closure that sets T_e. Electrons
# conduct heat freely along field lines and barely across them, so (like
# the potential in block 4) T_e is one unknown per lambda layer, and the
# balance is integrated over each layer:
#
#     d/dlambda[ G^T dTe/dlambda ]  -  d/dlambda[ (5/2) Te I ]
#         +  P_ohmic  -  (W_wall + W_ion + W_rad) = 0
#
# i.e. cross-field heat CONDUCTION between neighbouring layers plus the
# enthalpy the cross-field electron flow CONVECTS with it, fed by ohmic
# heating and drained by the wall sheath (block 5) and the inelastic
# collisions (ionization + radiation). The discrete form is the same
# tridiagonal system as solve_potential, with a thermal conductance G^T in
# place of the electrical one, the convective term upwinded onto the same
# faces, and the loss terms linearized onto the diagonal for stability.
#
# The convective term used to be missing, and it is not a small correction:
# it is the only transport channel that knows which WAY the electrons are
# going. Electrons enter cold at the cathode line, are heated crossing the
# transport barrier, and carry that heat UPSTREAM toward the anode with
# them. Without it the plume had no energy sink at all -- the cathode
# Dirichlet was absorbing several hundred watts to stand in for one -- and
# the profile came out as a single broad hump smeared over the whole
# channel, peaking downstream of the exit. With it the shape matches the
# internal probe maps: Linnell & Gallimore, IEPC-2005-024, Figs. 10 and 14
# find "a region of high electron temperature that begins immediately
# upstream of the acceleration zone and continues into the acceleration
# zone", cooling downstream of it, plus a second warm region at the anode.
#
# Read those maps for SHAPE, not for scale, twice over. They are a 500 V and
# a 600 V operating point on the NASA-173Mv1, not a 300 V SPT-70; and their
# Te comes from a floating emissive probe, which the authors themselves call
# "the least robust aspect of this study", with the near-anode warm region
# flagged as possibly an artifact of the probe entering another regime.
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
                  E_z_phi: np.ndarray | None = None,
                  E_r_phi: np.ndarray | None = None,
                  ) -> np.ndarray:
    """Electron heating density [W/m^3], the source term of the energy
    equation above:

        p = j_e . E = sigma_perp (E*_z E_z + E*_r E_r) ,

    with sigma_perp = e n_e mu_perp, E* the CROSS-FIELD (thermalized-
    potential) field that drives the current, j_e = sigma_perp E*, and E the
    real ELECTROSTATIC field, passed as E_z_phi / E_r_phi. Leaving those out
    uses E = E* and recovers the pure resistive dissipation
    sigma_perp |E*|^2 = eta_perp j_e^2.

    --- which of the two belongs here ---

    They differ by the diamagnetic work, because the two fields differ by
    the Boltzmann term of the thermalized potential:

        E = E* - grad(Te ln n_e)   =>   j.E = eta j^2 + j.(-grad(Te ln n_e))
                                             = eta j^2 + v_e . grad p_e / (-e n_e) .

    Which one is correct depends on how the left-hand side is written, and
    only one combination is self-consistent. Start from the internal energy
    equation with the collisional (frictional) heating eta j^2:

        div[(3/2) p_e v_e] + p_e div v_e + div q = eta j^2 - losses ,

    and fold the compression work into the flux: (3/2)div(p v) + p div v =
    (5/2)div(p v) - v.grad p. The electron momentum balance turns that
    stray -v.grad p into exactly the missing field work,

        v_e . grad p_e = j_e . E - eta j_e^2 ,

    which leaves

        div[(5/2) p_e v_e + q] = j_e . E - losses .

    So a (5/2)Te convective flux goes with j.E, and eta j^2 goes with a
    (3/2) flux plus an explicit p div v term. This module convects (5/2)Te
    (see the header), so its source is j.E.

    Two primary sources write the same pairing. Brick, Roberts & Jorns,
    IEPC-2024-411 Eq. (4):

        (3/2) e n_e dTe/dt + div[(5/2) e n_e Te v_e + q] = E.j_e + Q - ...

    and -- closer to home, because it is this model's own formulation, the
    Fife-lineage hybrid on thermalized-potential lambda layers, run on the
    SPT-70 at 300 V -- Koo, PhD thesis, U. Michigan 2005, Eqs. (2.51)-(2.53)
    with S_h = j_e.E = -n_e e u_e.E. Koo then spells out exactly which field
    that is, Eq. (2.71):

        E_n = -dphi/dn = -dphi*/dn - (2/3) eps dln(n_e)/dn
                                   - (2/3) (deps/dn) ln(n_e/n_e*)

    i.e. E = E* - grad(Te ln n_e) written out by the product rule (their
    eps = (3/2)Te). The velocity multiplying it, their Eq. (2.62), comes from
    dphi*/dlambda. Two different fields in one product, which is what this
    function computes.

    (Their Eq. (2.62) carries one term we do not: a (ln(n_e/n_e*) - 1)
    (2/3) deps/dlambda contribution to u_e, from Te varying BETWEEN layers,
    which the thermalized-potential derivation assumes away along each line.
    Our j_e is the dphi*/dlambda part only. Unquantified.)

    Using eta j^2 with a (5/2) flux double-counts the compression work. It
    is not a rounding error: on the reference SPT-70 state it overstates the
    total heating by 12%, and it misplaces it badly, running 2-4x too hot on
    the anode side of the channel and about 2x too cold in the plume, where
    the density falls steeply and the diamagnetic term changes sign.

    p can be locally NEGATIVE where the pressure gradient does work against
    the current — physical expansion cooling, not a bug; the layer solve
    takes it as a negative source and the Te floor keeps it bounded.
    """
    if E_z_phi is None:
        E_z_phi = E_z
    if E_r_phi is None:
        E_r_phi = E_r
    return const.elementary_charge * ne * mu_perp * (E_z * E_z_phi
                                                    + E_r * E_r_phi)


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


# Enthalpy carried per electron, in units of Te: h = (5/2) k T for a
# Maxwellian (3/2 internal + 1 of flow work). With Te in eV and the face
# current I in amps, (5/2) Te I is already watts.
ENTHALPY_COEFF = 2.5


def solve_electron_energy(G_T_face: np.ndarray,
                          source: np.ndarray,
                          loss: np.ndarray,
                          dloss_dTe: np.ndarray,
                          Te_prev: np.ndarray,
                          Te_anode: float,
                          Te_cathode: float,
                          I_face: np.ndarray | None = None,
                          ) -> np.ndarray:
    """Solve the layer energy balance for Te [eV], one value per layer.

    Interior layers balance conduction and convection across their two
    faces against the net local power, with the loss terms linearized about
    the previous temperature so they sit on the diagonal and keep the
    system diagonally dominant:

        G^T_{j-1/2}(Te_{j-1}-Te_j) + G^T_{j+1/2}(Te_{j+1}-Te_j)
            + H_{j+1/2} - H_{j-1/2}
            + source_j - [ loss_j + L'_j (Te_j - Te_prev_j) ] = 0 ,

        H_{j+1/2} = (5/2) I_{j+1/2} Te_upwind        [W]

    where I is the face current from layer_potential.layer_currents, in
    amps and positive downstream. Positive I means conventional current
    flows downstream, i.e. the ELECTRONS cross that face upstream, from
    layer j+1 into layer j, so the upwind temperature is Te_{j+1} and layer
    j gains H_{j+1/2} while losing H_{j-1/2} to its upstream neighbour.
    Upwinding is what keeps the matrix diagonally dominant: with I > 0
    throughout (the normal case, electrons streaming from the cathode line
    to the anode) the convective terms only add to |diag|.

    source [W] is the per-layer ohmic heating (>=0); loss [W] and
    dloss_dTe = L' [W/V] are the wall + ionization + radiation power and
    its slope, both evaluated at Te_prev, so the solve is one Newton step in
    the loss and one Picard step in everything else — iterate to
    convergence. L' must be >= 0 (it physically is) or the diagonal loses
    its dominance. An earlier version passed loss/Te in place of L', the
    secant through the origin rather than the tangent; it understates the
    slope badly enough to make the loop limit-cycle.

    The anode and cathode layers are Dirichlet-pinned to Te_anode /
    Te_cathode. Mirrors solve_potential.

    I_face=None drops the convective term and recovers the
    conduction-only balance; it is there for the block checks that exercise
    the energy equation without a potential solution to draw currents from.
    """
    N = len(source)
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    lower[1:-1] = G_T_face[:-1]
    upper[1:-1] = G_T_face[1:]
    diag[1:-1] = -(G_T_face[:-1] + G_T_face[1:] + dloss_dTe[1:-1])
    rhs[1:-1] = -(source[1:-1] - loss[1:-1]
                  + dloss_dTe[1:-1] * Te_prev[1:-1])

    if I_face is not None:
        c = ENTHALPY_COEFF
        I_up = np.maximum(I_face, 0.0)      # electrons crossing upstream
        I_dn = np.minimum(I_face, 0.0)      # electrons crossing downstream
        upper[1:-1] += c * I_up[1:]
        diag[1:-1] += c * (I_dn[1:] - I_up[:-1])
        lower[1:-1] += -c * I_dn[:-1]

    diag[0] = 1.0
    rhs[0] = Te_anode
    diag[-1] = 1.0
    rhs[-1] = Te_cathode

    return thomas_alg(lower, diag, upper, rhs)
