# Axisymmetric magnetostatics for the thruster magnetic system,
# ported from the earlier branch (commit 3171bd4).
# Internal array layout here is (nr, nz) — r first — because the solver
# was written and validated that way; spt70_system.py transposes the
# results into the project's (N_z, N_r) layout at the interface.

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4.0e-7 * np.pi  # vacuum permeability, T*m/A


def solve_axisymmetric_flux(
        r: np.ndarray,
        z: np.ndarray,
        mu_r: np.ndarray,
        current_density: np.ndarray,
        ) -> np.ndarray:
    """Solve axisymmetric magnetostatics for the flux function psi = r*A_phi
    of a system of azimuthal current loops (coils) and permeable material
    (an iron magnetic circuit) on a uniform (r, z) grid.

    The single azimuthal component of the vector potential A_phi obeys
    Ampere's law curl(B/mu) = J with B = curl(A). Writing psi = r*A_phi (so
    that B_r = -(1/r) dpsi/dz and B_z = (1/r) dpsi/dr, and contours of psi are
    magnetic field lines) reduces this to the elliptic equation

        d/dr[(1/(mu*r)) dpsi/dr] + d/dz[(1/(mu*r)) dpsi/dz] = -J_phi ,

    where mu = MU0 * `mu_r` is the local permeability (large inside iron,
    unity in vacuum) and `current_density` is the azimuthal coil current
    density J_phi (A/m^2, signed).

    Material interfaces are handled with arithmetic face averaging of mu (the
    flux-conserving choice for this operator), and psi = 0 is imposed on the
    axis (where r*A_phi vanishes identically) and on the outer domain
    boundaries (the domain is taken large enough that the field has decayed
    there). The resulting sparse linear system is solved directly.

    Returns psi on the full (nr, nz) grid.
    """
    nr, nz = len(r), len(z)
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    mu = MU0 * mu_r

    def idx(i: int, j: int) -> int:
        return i * nz + j

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nr * nz)

    boundary = np.zeros((nr, nz), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True

    for i in range(nr):
        for j in range(nz):
            n = idx(i, j)
            if boundary[i, j]:
                rows.append(n); cols.append(n); data.append(1.0)
                rhs[n] = 0.0
                continue

            r_ip = 0.5 * (r[i] + r[i + 1])   # radial face above
            r_im = 0.5 * (r[i] + r[i - 1])   # radial face below
            mu_e = 0.5 * (mu[i, j] + mu[i, j + 1])
            mu_w = 0.5 * (mu[i, j] + mu[i, j - 1])
            mu_n = 0.5 * (mu[i, j] + mu[i + 1, j])
            mu_s = 0.5 * (mu[i, j] + mu[i - 1, j])

            a_e = 1.0 / (mu_e * r[i]) / dz**2
            a_w = 1.0 / (mu_w * r[i]) / dz**2
            a_n = 1.0 / (mu_n * r_ip) / dr**2
            a_s = 1.0 / (mu_s * r_im) / dr**2

            rows.append(n); cols.append(idx(i, j + 1)); data.append(a_e)
            rows.append(n); cols.append(idx(i, j - 1)); data.append(a_w)
            rows.append(n); cols.append(idx(i + 1, j)); data.append(a_n)
            rows.append(n); cols.append(idx(i - 1, j)); data.append(a_s)
            rows.append(n); cols.append(n); data.append(-(a_e + a_w + a_n + a_s))
            rhs[n] = -current_density[i, j]

    A = csr_matrix((data, (rows, cols)), shape=(nr * nz, nr * nz))
    psi = spsolve(A, rhs).reshape(nr, nz)
    return psi


def field_from_flux(
        psi: np.ndarray,
        r: np.ndarray,
        z: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Magnetic field from the flux function psi = r*A_phi:
    B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr.

    On the axis (r = 0) B_r = 0 by symmetry and B_z is recovered from the
    near-axis behaviour psi ~ (1/2) B_z r^2, i.e. B_z(0) = 2*psi(dr)/dr^2.
    """
    dpsi_dr = np.gradient(psi, r, axis=0)
    dpsi_dz = np.gradient(psi, z, axis=1)

    r_safe = np.where(r == 0.0, 1.0, r)[:, None]
    Br = -dpsi_dz / r_safe
    Bz = dpsi_dr / r_safe

    Br[0, :] = 0.0
    Bz[0, :] = 2.0 * psi[1, :] / r[1] ** 2
    return Br, Bz


def magnetic_streamfunction(
        Br: np.ndarray,
        Bz: np.ndarray,
        r: np.ndarray,
        z: np.ndarray,
        ) -> np.ndarray:
    """Magnetic field streamfunction lambda, with lambda = 0 on the axis
    (r = 0) at z = z[0]. Contours of lambda are magnetic field lines; the
    electron fluid model treats each lambda layer as one unknown.
    """
    lambda_z0 = cumulative_trapezoid(r * Bz[:, 0], r, initial=0.0)
    lam = cumulative_trapezoid(-r[:, None] * Br, z, axis=1, initial=0.0)
    lam += lambda_z0[:, None]
    return lam


def lambda_gradient(
        Br: np.ndarray,
        Bz: np.ndarray,
        r: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """grad(lambda) = r*Bz r_hat - r*Br z_hat, computed directly from B so
    no numerical differentiation of lambda is needed.
    """
    dlambda_dr = r[:, None] * Bz
    dlambda_dz = -r[:, None] * Br
    return dlambda_dr, dlambda_dz
