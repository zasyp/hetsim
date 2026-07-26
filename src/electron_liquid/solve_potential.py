import numpy as np

from ..utils.utils import thomas_alg


def solve_potential(G_face, dI_iz, V_anode, V_cathode):
    """Solve current continuity on the lambda layers for phi* [V].

    Each interior layer balances the electron current across its two
    faces against the ion current born inside it (quasineutrality):

        G_{j-1/2} (phi*_{j-1} - phi*_j) + G_{j+1/2} (phi*_{j+1} - phi*_j)
            = -dI_iz_j

    which is a tridiagonal system in phi*. The anode (layer 0) and
    cathode (layer N-1) are Dirichlet-pinned, so V_anode - V_cathode is
    the discharge voltage.
    """
    N = len(dI_iz)
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    lower[1:-1] = G_face[:-1]
    upper[1:-1] = G_face[1:]
    diag[1:-1] = -(G_face[:-1] + G_face[1:])
    rhs[1:-1] = -dI_iz[1:-1]

    diag[0] = 1.0
    rhs[0] = V_anode                                 # phi*_0   = V_a
    diag[-1] = 1.0
    rhs[-1] = V_cathode                              # phi*_N-1 = V_c

    return thomas_alg(lower, diag, upper, rhs)


def potential_on_grid(phi_star, lam_layers, lam, Te_layers, n_e, n_ref=None):
    """Reconstruct the full 2-D potential from the layer solution via the
    thermalized-potential relation

        phi(z, r) = phi*(lambda) + Te(lambda) * ln(n_e / n_ref)     [V]

    (Te in eV = volts). phi* and Te are known per layer and interpolated
    to the local lambda of every node; the density term adds the
    Boltzmann variation ALONG each field line. lam_layers is the lambda
    value of each layer index (need not be sorted -- handled here). Nodes
    with lambda outside the layered range clamp to the nearest boundary
    layer, which puts the near-anode wedge at ~anode potential and the
    far plume at ~cathode potential, as intended. n_ref only shifts phi
    by a constant; it defaults to max(n_e).
    """
    if n_ref is None:
        n_ref = float(n_e.max())

    order = np.argsort(lam_layers)                   # np.interp needs xp ascending
    xp = lam_layers[order]
    flat = lam.ravel()                               # np.interp takes 1-D query points

    phi_star_g = np.interp(flat, xp, phi_star[order]).reshape(lam.shape)
    Te_g = np.interp(flat, xp, Te_layers[order]).reshape(lam.shape)

    return phi_star_g + Te_g * np.log(np.maximum(n_e, 1.0) / n_ref)


def electric_field(phi, z_nodes, r_nodes):
    """E = -grad(phi) on the (N_z, N_r) grid. Returns (E_z, E_r) [V/m]."""
    dphi_dz = np.gradient(phi, z_nodes, axis=0)
    dphi_dr = np.gradient(phi, r_nodes, axis=1)
    return -dphi_dz, -dphi_dr
