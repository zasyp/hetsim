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


def potential_on_grid(phi_star, lam_layers, lam, Te_layers, n_e,
                      n_ref=None, boltzmann_clamp=None):
    """Reconstruct the full 2-D potential from the layer solution via the
    thermalized-potential relation

        phi(z, r) = phi*(lambda) + Te(lambda) * ln(n_e / n_ref)     [V]

    (Te in eV = volts). phi* and Te are known per layer and interpolated to
    the local lambda of every node; the density term adds the Boltzmann
    variation ALONG each field line. Nodes with lambda outside the layered
    range clamp to the nearest boundary layer.

    --- the gauge, n_ref ---

    phi* is defined only up to the choice of n_ref: rescaling it by c moves
    phi* by -Te*ln(c). Along ONE line that is a constant and changes nothing,
    which is why it is often called a free gauge. It is not free here, for
    two separate reasons.

    First, n_ref must be a single GLOBAL number. The layer solve drives the
    cross-field current with differences of phi* between adjacent layers, and
    the physical driving force is -grad(phi) + Te grad(ln n_e). Writing that
    in terms of phi* leaves a residual -Te*grad(ln n_ref): a per-layer gauge
    injects a spurious driving force and the differences stop meaning
    anything. (A per-layer version was tried and reverted for exactly this.)

    Second, its VALUE is fixed by the boundary condition, not by taste.
    solve_potential pins phi*[0] = V_d at the anode layer, but the physical
    statement is phi = V_d there. The two agree only where the Boltzmann term
    vanishes, i.e. where n_e = n_ref. So the anchor is the anode density, and
    the caller passes it. Using max(n_e) instead -- the old default -- put the
    zero somewhere in mid-channel and left the anode potential off V_d.

    --- the clamp ---

    boltzmann_clamp caps |ln(n_e/n_ref)| at that many e-foldings, i.e. the
    Boltzmann term at boltzmann_clamp * Te volts.

    This is not a cosmetic limiter. The relation comes from balancing the
    parallel pressure gradient against the electric force, and it assumes the
    electrons are in equilibrium along the line. The population that can sit
    at a potential k*Te below the reference is the Maxwellian tail, a fraction
    exp(-k) of the whole: past a few Te there are no electrons left to
    maintain the equilibrium the formula assumes, so the formula invalidates
    itself. Beyond the clamp we do not know the potential and refuse to
    extrapolate.

    Where this bites is the near plume, and it is worth being clear about why
    the raw formula fails there rather than just capping it: plume electrons
    stream out along lines that leave the domain instead of equilibrating on
    them, and the density falls mostly by geometric expansion rather than
    against a potential barrier. The relation charges the whole density drop
    to the potential and digs a well -- measured at 120 V BELOW cathode
    potential, in a region where nothing physical produces one.

    Pass None to disable the clamp and recover the raw relation.
    """
    order = np.argsort(lam_layers)                   # np.interp needs xp ascending
    xp = lam_layers[order]
    flat = lam.ravel()                               # np.interp takes 1-D query points

    phi_star_g = np.interp(flat, xp, phi_star[order]).reshape(lam.shape)
    Te_g = np.interp(flat, xp, Te_layers[order]).reshape(lam.shape)

    if n_ref is None:
        n_ref = float(n_e.max())
    n_ref = max(float(n_ref), 1.0)

    ln_ratio = np.log(np.maximum(n_e, 1.0) / n_ref)
    if boltzmann_clamp is not None:
        ln_ratio = np.clip(ln_ratio, -abs(boltzmann_clamp), abs(boltzmann_clamp))
    return phi_star_g + Te_g * ln_ratio


def electric_field(phi, z_nodes, r_nodes):
    """E = -grad(phi) on the (N_z, N_r) grid. Returns (E_z, E_r) [V/m]."""
    dphi_dz = np.gradient(phi, z_nodes, axis=0)
    dphi_dr = np.gradient(phi, r_nodes, axis=1)
    return -dphi_dz, -dphi_dr
