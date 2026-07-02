from typing import Any
import numpy as np
from scipy import ndimage
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4.0e-7 * np.pi  # vacuum permeability, T*m/A


def solve_vacuum_potential(
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        dirichlet_mask: np.ndarray[Any, np.dtype[np.bool_]],
        dirichlet_values: np.ndarray[Any, np.dtype[np.float64]],
        excluded_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
        tol: float = 1e-8,
        max_iter: int = 20000,
        omega: float = 1.0,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Solve the axisymmetric Laplace equation for the magnetic scalar
    potential phi (curl B = 0 => B = -grad(phi), div B = 0 => Laplacian(phi) = 0),
    on a uniform (r, z) grid, given Dirichlet boundary values at the masked nodes.
    Nodes outside the mask default to a zero-gradient (Neumann) boundary, and
    r = 0 uses the axisymmetric on-axis stencil.

    `excluded_mask` marks nodes that are not part of the solved domain at all
    (e.g. solid magnetic material behind a pole face, away from its exposed
    tip). They are never updated, and any *other* node's stencil treats an
    excluded neighbor as a zero-flux (Neumann) wall -- mirroring the node's
    own value back at itself, the same trick already used for the domain's
    outer edges -- so no field leaks into or out of an excluded region except
    through whatever Dirichlet nodes are explicitly carved out of it (e.g. a
    single exposed face). After the solve, excluded cells are filled in from
    their nearest solved neighbor purely so that a later `np.gradient` call
    over the whole array doesn't see a spurious jump at that boundary; their
    values carry no physical meaning.

    This is a Jacobi relaxation (all nodes updated simultaneously from the
    previous sweep), so omega must stay at or below 1 for stability -- unlike
    Gauss-Seidel, over-relaxation (omega > 1) here diverges.
    """
    nr, nz = len(r), len(z)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    if excluded_mask is None:
        excluded_mask = np.zeros_like(dirichlet_mask, dtype=bool)

    phi = np.where(dirichlet_mask, dirichlet_values, 0.0)

    r_plus = r + 0.5 * dr
    r_minus = r - 0.5 * dr
    r_safe = np.where(r == 0.0, 1.0, r)

    interior_denom = (r_plus + r_minus) / (r_safe * dr**2) + 2.0 / dz**2
    axis_denom = 2.0 / dr**2 + 2.0 / dz**2

    padded_excluded = np.pad(excluded_mask, 1, mode='constant', constant_values=False)
    excluded_ip1 = padded_excluded[2:, 1:-1]
    excluded_im1 = padded_excluded[:-2, 1:-1]
    excluded_jp1 = padded_excluded[1:-1, 2:]
    excluded_jm1 = padded_excluded[1:-1, :-2]

    for _ in range(max_iter):
        padded = np.pad(phi, 1, mode='edge')
        phi_ip1 = np.where(excluded_ip1, phi, padded[2:, 1:-1])
        phi_im1 = np.where(excluded_im1, phi, padded[:-2, 1:-1])
        phi_jp1 = np.where(excluded_jp1, phi, padded[1:-1, 2:])
        phi_jm1 = np.where(excluded_jm1, phi, padded[1:-1, :-2])

        numerator = (
            (r_plus[:, None] * phi_ip1 + r_minus[:, None] * phi_im1) / (r_safe[:, None] * dr**2)
            + (phi_jp1 + phi_jm1) / dz**2
        )
        phi_new = numerator / interior_denom[:, None]

        axis_numerator = 2.0 * phi_ip1[0, :] / dr**2 + (phi_jp1[0, :] + phi_jm1[0, :]) / dz**2
        phi_new[0, :] = axis_numerator / axis_denom

        phi_new = phi + omega * (phi_new - phi)
        phi_new = np.where(dirichlet_mask, dirichlet_values, phi_new)
        phi_new = np.where(excluded_mask, phi, phi_new)

        delta = np.max(np.abs(phi_new - phi))
        phi = phi_new
        if delta < tol:
            break

    if excluded_mask.any():
        nearest_index = ndimage.distance_transform_edt(
            excluded_mask, return_distances=False, return_indices=True,
        )
        phi = np.where(excluded_mask, phi[tuple(nearest_index)], phi)

    return phi


def magnetic_field_from_potential(
        phi: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """B = -grad(phi) on the (r, z) grid."""
    Br = -np.gradient(phi, r, axis=0)
    Bz = -np.gradient(phi, z, axis=1)
    return Br, Bz


def magnetic_streamfunction(
        Br: np.ndarray[Any, np.dtype[np.float64]],
        Bz: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Magnetic field streamfunction lambda (Eqn. 2.2-2.3), with lambda = 0 on
    the axis (r = 0) at z = z[0].
    """
    lambda_z0 = cumulative_trapezoid(r * Bz[:, 0], r, initial=0.0)
    lam = cumulative_trapezoid(-r[:, None] * Br, z, axis=1, initial=0.0)
    lam += lambda_z0[:, None]
    return lam


def lambda_gradient(
        Br: np.ndarray[Any, np.dtype[np.float64]],
        Bz: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """grad(lambda) = r*Bz r_hat - r*Br z_hat (Eqn. 2.7), computed directly
    from B so no numerical differentiation of lambda is needed.
    """
    dlambda_dr = r[:, None] * Bz
    dlambda_dz = -r[:, None] * Br
    return dlambda_dr, dlambda_dz


def solve_axisymmetric_flux(
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        mu_r: np.ndarray[Any, np.dtype[np.float64]],
        current_density: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
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
    density J_phi (A/m^2, signed). Unlike the scalar-potential solver above,
    this handles the coils that actually excite the field and the graded
    permeability of the magnetic circuit directly, instead of imposing a fixed
    potential on a pole face.

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
        psi: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
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


def vacuum_magnetic_field(
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        dirichlet_mask: np.ndarray[Any, np.dtype[np.bool_]],
        dirichlet_values: np.ndarray[Any, np.dtype[np.float64]],
        excluded_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
        tol: float = 1e-8,
        max_iter: int = 20000,
        omega: float = 1.0,
        ):
    """Solve for the static vacuum magnetic field from Dirichlet boundary
    values of the scalar potential, returning (phi, Br, Bz, lambda, grad_lambda).

    See `solve_vacuum_potential` for what `excluded_mask` does.
    """
    phi = solve_vacuum_potential(r, z, dirichlet_mask, dirichlet_values, excluded_mask, tol, max_iter, omega)
    Br, Bz = magnetic_field_from_potential(phi, r, z)
    lam = magnetic_streamfunction(Br, Bz, r, z)
    grad_lambda = lambda_gradient(Br, Bz, r)
    return phi, Br, Bz, lam, grad_lambda
