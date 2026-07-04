import sys
from pathlib import Path
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src (numerical.*)
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # physics (classes)

from numerical.numerical_funcs import interpolation_weights
from classes import Particle, Grid2D

N_FLOOR = 1e-10      # nodal weight below this -> treat the cell as empty
T_FLOOR = 0.0        # temperature assigned to empty cells


def deposit(
        grid: Grid2D,
        particles: np.ndarray[Any, np.dtype[np.object_]],
        n: np.ndarray[Any, np.dtype[np.float64]],
        v_r: np.ndarray[Any, np.dtype[np.float64]],
        v_z: np.ndarray[Any, np.dtype[np.float64]],
        T: np.ndarray[Any, np.dtype[np.float64]],
        nvr: np.ndarray[Any, np.dtype[np.float64]],
        nvz: np.ndarray[Any, np.dtype[np.float64]],
        nT: np.ndarray[Any, np.dtype[np.float64]],
        ):
    """Scatter macroparticles onto the (r, z) grid -- the inverse of
    `bilinear_interp`.

    Each particle is spread over the four surrounding nodes with the same
    bilinear weights used for gathering. The density-weighted sums accumulate
    into the scratch arrays `nvr`/`nvz`/`nT` and into `n`; a second pass
    normalises them to the mean radial/axial velocity and temperature at every
    node and divides `n` by the axisymmetric nodal control volume so it becomes
    a true number density (particles per m^3). Empty cells (nodal weight below
    `N_FLOOR`) get zero velocity and `T_FLOOR`.

    All field arrays have shape `(grid.N_r, grid.N_z)`. `weight`, `T` and
    `active` are read off each particle via `getattr` with sensible defaults,
    so the function already works with the current minimal `Particle`.
    """
    r_grid = grid.r_nodes()
    z_grid = grid.z_nodes()
    h_r = grid.h_r
    h_z = grid.h_z

    n.fill(0.0)
    nvr.fill(0.0)
    nvz.fill(0.0)
    nT.fill(0.0)

    # --- pass 1: accumulate density-weighted sums over the 4 corner nodes ---
    for p in particles:
        if not getattr(p, "active", True):
            continue
        w = getattr(p, "weight", 1.0)          # macroparticle statistical weight
        p_T = getattr(p, "T", 0.0)
        ir0, ir1, wr0, wr1 = interpolation_weights(p.r, r_grid)
        iz0, iz1, wz0, wz1 = interpolation_weights(p.z, z_grid)
        for ir, wr in ((ir0, wr0), (ir1, wr1)):
            for iz, wz in ((iz0, wz0), (iz1, wz1)):
                shape = w * wr * wz
                n[ir, iz] += shape
                nvr[ir, iz] += shape * p.v_r
                nvz[ir, iz] += shape * p.v_z
                nT[ir, iz] += shape * p_T

    # --- pass 2: normalise moments and convert n to a per-volume density ---
    for i in range(grid.N_r):
        # radial control cell of node i, clipped to the grid; the ring volume
        # pi*(r_hi^2 - r_lo^2) == 2*pi*r_i*dr for interior nodes and stays
        # finite on the axis (r=0), unlike a bare 2*pi*r_i*dr factor.
        r_lo = max(r_grid[i] - 0.5 * h_r, r_grid[0])
        r_hi = min(r_grid[i] + 0.5 * h_r, r_grid[-1])
        ring = np.pi * (r_hi**2 - r_lo**2)
        for j in range(grid.N_z):
            dz = h_z if 0 < j < grid.N_z - 1 else 0.5 * h_z
            vol = ring * dz
            if n[i, j] > N_FLOOR:
                v_r[i, j] = nvr[i, j] / n[i, j]
                v_z[i, j] = nvz[i, j] / n[i, j]
                T[i, j] = nT[i, j] / n[i, j]
            else:
                v_r[i, j] = 0.0
                v_z[i, j] = 0.0
                T[i, j] = T_FLOOR
            n[i, j] /= vol

    return n, v_r, v_z, T


def _cell_geometry(grid: Grid2D):
    """Axisymmetric control-volume geometry shared by the continuity routines.

    Returns, for the (r, z) grid:
      * `ring[i]`     -- area of the axial (z-normal) face of node i's control
                         cell, pi*(r_hi^2 - r_lo^2); identical to the ring used
                         in `deposit`, and finite on the axis.
      * `r_lo`,`r_hi` -- radii of the lower/upper radial faces of each node's
                         control cell (clamped to the domain), used for the
                         2*pi*r*dz side-wall areas; r_lo[0] = 0 on the axis.
      * `dz[j]`       -- axial extent of node j's control cell (half at the ends).
      * `vol[i, j]`   -- cell volume ring[i]*dz[j].
    """
    r = grid.r_nodes()
    z = grid.z_nodes()
    h_r, h_z = grid.h_r, grid.h_z

    r_lo = np.maximum(r - 0.5 * h_r, r[0])
    r_hi = np.minimum(r + 0.5 * h_r, r[-1])
    ring = np.pi * (r_hi**2 - r_lo**2)

    dz = np.full(grid.N_z, h_z)
    dz[0] = dz[-1] = 0.5 * h_z

    vol = ring[:, None] * dz[None, :]
    return r, z, ring, r_lo, r_hi, dz, vol


def neutral_flux_divergence(
        grid: Grid2D,
        n: np.ndarray[Any, np.dtype[np.float64]],
        v_r: np.ndarray[Any, np.dtype[np.float64]],
        v_z: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Divergence of the neutral number flux F = n*u in cylindrical (r, z)
    coordinates (axisymmetric, no azimuthal dependence):

        div F = (1/r) d(r * n * v_r)/dr + d(n * v_z)/dz .

    This is the left-hand side of the steady continuity equation, so it is the
    natural diagnostic for how well a deposited (n, v) field conserves mass:
    in a source-free region it should vanish, and where it is positive neutrals
    are being depleted (e.g. by ionization). On the axis (r = 0) the singular
    term is replaced by its finite limit 2 * d(n*v_r)/dr. Returned array has
    the field shape (N_r, N_z), in units of [n]/m.
    """
    r, z, _, _, _, _, _ = _cell_geometry(grid)

    F_r = n * v_r
    F_z = n * v_z

    r_safe = np.where(r == 0.0, 1.0, r)[:, None]
    div = np.gradient(r[:, None] * F_r, r, axis=0) / r_safe
    div += np.gradient(F_z, z, axis=1)

    if r[0] == 0.0:
        div[0, :] = 2.0 * np.gradient(F_r, r, axis=0)[0, :] + np.gradient(F_z, z, axis=1)[0, :]

    return div


def advance_neutral_density(
        grid: Grid2D,
        n: np.ndarray[Any, np.dtype[np.float64]],
        v_r: np.ndarray[Any, np.dtype[np.float64]],
        v_z: np.ndarray[Any, np.dtype[np.float64]],
        dt: float,
        sink: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Explicit (forward-Euler) update of the neutral continuity equation

        dn/dt = -div(n*u) - S ,

    over one step `dt`, given a deposited density/velocity field and an
    optional volumetric sink `S` (e.g. the ionization rate from
    `ionization.ionization_rate`, events per m^3 per s). Useful for marching a
    fluid neutral density alongside the particle push. The result is clamped to
    be non-negative. `dt` must respect the CFL condition
    dt < min(h_r, h_z)/max|u| for stability.
    """
    n_new = n - dt * neutral_flux_divergence(grid, n, v_r, v_z)
    if sink is not None:
        n_new = n_new - dt * sink
    np.clip(n_new, 0.0, None, out=n_new)
    return n_new


def solve_neutral_continuity(
        grid: Grid2D,
        v_r: np.ndarray[Any, np.dtype[np.float64]],
        v_z: np.ndarray[Any, np.dtype[np.float64]],
        inlet_mask: np.ndarray[Any, np.dtype[np.bool_]],
        inlet_density: np.ndarray[Any, np.dtype[np.float64]],
        ionization_freq: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        source: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Steady-state neutral density from the continuity equation

        div(n * u) + nu * n = q ,

    solved conservatively on the axisymmetric grid with a first-order upwind
    finite-volume scheme, given a *fixed* velocity field `(v_r, v_z)` (e.g.
    from `deposit`), a linear loss frequency `nu = ionization_freq`
    (1/s, from `ionization.ionization_frequency`) and an optional volumetric
    source `q` (m^-3 s^-1). Because the ionization loss is linear in n, the
    whole thing is a single sparse linear solve -- no time marching -- so it
    directly returns the neutral distribution depleted by ionization.

    Boundary conditions:
      * `inlet_mask` nodes are held at the prescribed `inlet_density`
        (Dirichlet), e.g. the anode injection plane.
      * every other domain boundary is a passive open boundary: neutrals may
        convect out where the flow points outward, and no neutrals convect in
        (the flux across walls where the flow is tangential is ~0, so solid
        radial walls and the r = 0 symmetry axis are handled by the same rule,
        the axis additionally having zero face area).

    Face velocities are arithmetic averages of the two straddling nodes; the
    upwind node supplies the density on each face. Returns n on the full
    (N_r, N_z) grid.
    """
    nr, nz = grid.N_r, grid.N_z
    _, _, ring, r_lo, r_hi, dz, vol = _cell_geometry(grid)

    if ionization_freq is None:
        ionization_freq = np.zeros((nr, nz))
    if source is None:
        source = np.zeros((nr, nz))

    def idx(i: int, j: int) -> int:
        return i * nz + j

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nr * nz)

    for i in range(nr):
        for j in range(nz):
            n_id = idx(i, j)

            if inlet_mask[i, j]:
                rows.append(n_id); cols.append(n_id); data.append(1.0)
                rhs[n_id] = inlet_density[i, j]
                continue

            diag = ionization_freq[i, j] * vol[i, j]

            # Every cell has all four geometric faces. `sign` orients the node
            # velocity along the outward normal; `nb`/`nb_v` are the neighbour
            # index and velocity along that axis (None on a domain boundary).
            #   (di, dj, velocity field, sign, outward face area)
            faces = (
                (1, 0, v_r, 1.0, 2.0 * np.pi * r_hi[i] * dz[j]),   # north / +r
                (-1, 0, v_r, -1.0, 2.0 * np.pi * r_lo[i] * dz[j]),  # south / -r
                (0, 1, v_z, 1.0, ring[i]),                          # east  / +z
                (0, -1, v_z, -1.0, ring[i]),                        # west  / -z
            )

            for di, dj, vel, sign, area in faces:
                if area == 0.0:                       # r = 0 axis face
                    continue
                has_nb = 0 <= i + di < nr and 0 <= j + dj < nz
                if has_nb:
                    u_face = 0.5 * (vel[i, j] + vel[i + di, j + dj])
                else:
                    u_face = vel[i, j]                 # one-sided at the boundary
                u_out = sign * u_face

                diag += area * max(u_out, 0.0)         # outflow: upwind is (i, j)
                if has_nb:
                    # inflow (u_out < 0): upwind is the neighbour
                    rows.append(n_id); cols.append(idx(i + di, j + dj))
                    data.append(area * min(u_out, 0.0))
                # else: open boundary -> outflow only, no incoming neutrals

            rows.append(n_id); cols.append(n_id); data.append(diag)
            rhs[n_id] = source[i, j] * vol[i, j]

    A = csr_matrix((data, (rows, cols)), shape=(nr * nz, nr * nz))
    n = spsolve(A, rhs).reshape(nr, nz)
    return n
