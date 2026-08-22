# Nodal-P1 finite-element axisymmetric magnetostatics — the FEMM-class
# replacement for the finite-difference solver in magnetic.py.
#
# Why FEM here: the FD field recovery straddles the iron/vacuum interface
# with a centered stencil and leaks the in-iron field into the adjacent
# vacuum node. FEM removes that at the root: the mesh is CONFORMAL (every
# element lies wholly in one material, the mu jump lives on element edges),
# the weak form enforces the B_n / H_t interface conditions variationally,
# and the field is recovered PER ELEMENT with that element's material — so B
# is honestly two-valued at the interface, never averaged across it.
#
# Formulation: axisymmetric, azimuthal potential only. With psi = r*A_phi
# (contours of psi are field lines, B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr)
# the magnetostatic equation div( (1/(mu r)) grad psi ) = -J_phi has the weak
# form, for test function w vanishing on Dirichlet boundaries,
#
#     integral (1/(mu r)) grad(psi) . grad(w) dr dz = integral J_phi w dr dz .
#
# This is a SCALAR elliptic problem, so nodal Lagrange (P1) elements are the
# correct and standard choice — NOT vector Nedelec edge elements, which are
# for in-plane vector potentials / H(curl) problems. Per-element mu (constant
# on each triangle) makes the interface conditions natural.
#
# Internal layout: points are (r, z) rows; everything is SI, mu = mu0*mu_r.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.constants import mu_0


def triangulate_structured(r: np.ndarray, z: np.ndarray):
    """Split a structured (r, z) grid into a conformal P1 triangulation.

    Each rectangular cell becomes two triangles. Because the material boxes
    in spt70_system are aligned to these grid lines, no triangle straddles a
    material boundary — the mesh is conformal by construction. Returns
    (points [N,2] as (r,z), tris [M,3] node indices, ccw).
    """
    nr, nz = len(r), len(z)
    R, Z = np.meshgrid(r, z, indexing="ij")
    points = np.column_stack([R.ravel(), Z.ravel()])

    def nid(i, j):
        return i * nz + j

    tris = []
    for i in range(nr - 1):
        for j in range(nz - 1):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            # two triangles per cell, split along the (n00, n11) diagonal
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    return points, np.asarray(tris, dtype=np.int64)


def _p1_geometry(points: np.ndarray, tris: np.ndarray):
    """Per-element area, gradient coefficients and centroid radius.

    For linear shape functions phi_i = (a_i + b_i r + c_i z)/(2A),
    grad(phi_i) = (b_i, c_i)/(2A) is constant on the element. Returns
    (area [M], grad [M,3,2], r_centroid [M])."""
    p = points[tris]                         # (M, 3, 2)
    r1, z1 = p[:, 0, 0], p[:, 0, 1]
    r2, z2 = p[:, 1, 0], p[:, 1, 1]
    r3, z3 = p[:, 2, 0], p[:, 2, 1]

    # b_i, c_i (see any P1 FEM text); signed area from the cross product
    b = np.stack([z2 - z3, z3 - z1, z1 - z2], axis=1)   # (M,3) = d/dr
    c = np.stack([r3 - r2, r1 - r3, r2 - r1], axis=1)   # (M,3) = d/dz
    area2 = (r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1)  # = 2A signed
    area = 0.5 * np.abs(area2)

    grad = np.stack([b, c], axis=2) / area2[:, None, None]  # (M,3,2)
    r_c = (r1 + r2 + r3) / 3.0
    return area, grad, r_c


def assemble(points, tris, mu_r_tri, Jphi_tri, dirichlet):
    """Assemble and solve the axisymmetric P1 system for psi = r*A_phi.

    mu_r_tri, Jphi_tri : per-element relative permeability and azimuthal
        current density [A/m^2]. dirichlet : boolean per node, psi = 0 there
        (axis r=0 and the far outer boundary). Returns psi at the nodes.
    """
    area, grad, r_c = _p1_geometry(points, tris)
    nu_over_r = 1.0 / (mu_0 * mu_r_tri * r_c)   # (1/(mu r)) per element

    # element stiffness K^e_ij = (1/(mu r_c)) (grad_i . grad_j) * area
    gg = np.einsum("mik,mjk->mij", grad, grad)          # (M,3,3)
    Ke = nu_over_r[:, None, None] * gg * area[:, None, None]
    # element load f^e_i = J * area / 3 (P1 lumped source)
    fe = (Jphi_tri * area / 3.0)[:, None] * np.ones((1, 3))

    M = len(tris)
    rows = np.repeat(tris, 3, axis=1).reshape(M, 3, 3)
    cols = np.repeat(tris[:, None, :], 3, axis=1)
    K = csr_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())),
                   shape=(len(points), len(points)))
    b = np.bincount(tris.ravel(), weights=fe.ravel(), minlength=len(points))

    # Dirichlet psi = 0: zero the row/col, put 1 on the diagonal
    free = ~dirichlet
    K = K.tolil()
    for n in np.where(dirichlet)[0]:
        K.rows[n] = [n]
        K.data[n] = [1.0]
        b[n] = 0.0
    K = K.tocsr()
    # also drop Dirichlet columns from free rows (they contribute 0 anyway
    # since psi=0 there, so the RHS is unchanged)
    psi = np.zeros(len(points))
    psi[free] = spsolve(K[free][:, free], b[free])
    return psi


def recover_B_elements(points, tris, psi):
    """Per-element magnetic field (constant on each triangle).

    B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr, evaluated with the element's
    own constant gradient — the honest one-sided (per-material) field. At an
    interface the two adjacent elements give different B by construction, so
    nothing is averaged across the mu jump. Returns (Br [M], Bz [M])."""
    _, grad, r_c = _p1_geometry(points, tris)
    dpsi = np.einsum("mik,mi->mk", grad, psi[tris])   # (M,2) = (dpsi/dr, dpsi/dz)
    Br = -dpsi[:, 1] / r_c
    Bz = dpsi[:, 0] / r_c
    return Br, Bz


def recover_B_nodal(points, tris, psi, mu_r_tri):
    """Volume-weighted nodal field, averaged only within one material.

    A poor man's Zienkiewicz-Zhu recovery: smooth the per-element field to
    the nodes, but keep separate averages per material class so the field
    stays two-valued across the iron surface (no smearing of the interface).
    Weighted by element volume (area * r_centroid, dV = 2*pi*r dr dz), not
    plain area — near the axis r varies a lot across an element, and an
    area-only average would bias the mean since B itself scales like 1/r.
    Returns (Br_vac, Bz_vac) — the vacuum-side nodal field, which is what the
    plasma domain and the discharge-grid interpolation need. Iron-only nodes
    stay NaN.
    """
    Br_e, Bz_e = recover_B_elements(points, tris, psi)
    area, _, r_c = _p1_geometry(points, tris)
    vac = mu_r_tri <= 1.0                      # vacuum elements
    w = (area * r_c)[vac]
    idx = tris[vac].ravel()

    N = len(points)
    wsum = np.zeros(N)
    Br_n = np.zeros(N)
    Bz_n = np.zeros(N)
    np.add.at(wsum, idx, np.repeat(w, 3))
    np.add.at(Br_n, idx, np.repeat(w * Br_e[vac], 3))
    np.add.at(Bz_n, idx, np.repeat(w * Bz_e[vac], 3))
    good = wsum > 0
    Br_n[good] /= wsum[good]
    Bz_n[good] /= wsum[good]
    Br_n[~good] = np.nan
    Bz_n[~good] = np.nan
    return Br_n, Bz_n


def _edge_key(a: int, b: int) -> tuple:
    return (a, b) if a < b else (b, a)


def _longest_edge(points, tri):
    """The (u, v) node pair of tri's longest edge; ties broken by edge key so
    two triangles sharing an edge always agree on which is 'longest'."""
    a, b, c = tri
    edges = [(a, b), (b, c), (c, a)]
    lens = [float(np.sum((points[u] - points[v]) ** 2)) for u, v in edges]
    k = max(range(3), key=lambda i: (lens[i], _edge_key(*edges[i])))
    return _edge_key(*edges[k])


def refine_mesh(points, tris, marked):
    """Conformal refinement by Rivara longest-edge bisection.

    Bisecting a triangle's longest edge would leave a hanging node on the
    neighbour across that edge; Rivara's rule fixes this by first refining the
    neighbour until the shared edge is longest for both, then splitting the
    pair together. The result has no hanging nodes and bounded smallest angle.

    points : (N,2) array. tris : (M,3) int array. marked : indices of triangles
    to refine. Returns (points2, tris2) as fresh arrays.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    tri_of = {i: tuple(t) for i, t in enumerate(map(tuple, tris))}
    next_id = len(tris)
    edge2tri: dict[tuple, set] = {}

    def _add(tid, tri):
        tri_of[tid] = tri
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            edge2tri.setdefault(_edge_key(u, v), set()).add(tid)

    def _remove(tid):
        a, b, c = tri_of.pop(tid)
        for u, v in ((a, b), (b, c), (c, a)):
            edge2tri[_edge_key(u, v)].discard(tid)

    for i, t in tri_of.items():
        a, b, c = t
        for u, v in ((a, b), (b, c), (c, a)):
            edge2tri.setdefault(_edge_key(u, v), set()).add(i)

    mid_cache: dict[tuple, int] = {}

    def _midpoint(u, v):
        key = _edge_key(u, v)
        if key not in mid_cache:
            pts.append(0.5 * (pts[u] + pts[v]))
            mid_cache[key] = len(pts) - 1
        return mid_cache[key]

    def _third(tri, e):
        return next(n for n in tri if n not in e)

    def _bisect_pair(tid):
        nonlocal next_id
        tri = tri_of[tid]
        u, v = _longest_edge(pts, tri)
        share = edge2tri[_edge_key(u, v)] - {tid}
        if share:                                   # interior edge
            nid = next(iter(share))
            if _longest_edge(pts, tri_of[nid]) != _edge_key(u, v):
                _bisect_pair(nid)                   # refine neighbour first
                return _bisect_pair(tid)            # then retry
        m = _midpoint(u, v)
        c = _third(tri, (u, v))
        _remove(tid)
        _add(tid, (u, m, c)); _add(next_id, (m, v, c)); next_id += 1
        if share:
            nid = next(iter(share))
            d = _third(tri_of[nid], (u, v))
            _remove(nid)
            _add(nid, (u, m, d)); _add(next_id, (m, v, d)); next_id += 1

    for tid in list(marked):
        if tid in tri_of:
            _bisect_pair(tid)

    points2 = np.array(pts)
    tris2 = np.array(list(tri_of.values()), dtype=np.int64)
    return points2, tris2


def zz_error(points, tris, psi):
    """Per-element Zienkiewicz-Zhu error indicator [T*m].

    The recovered (smoothed) field B* minus the raw per-element field B is a
    cheap a-posteriori error estimate: it is largest where the discrete field
    is least able to represent the true one — the pole-tip corners, where the
    field is singular. err_t = |B*_t - B_t| * sqrt(area_t). Used by the AMR
    driver to pick which elements to refine.
    """
    Br_e, Bz_e = recover_B_elements(points, tris, psi)
    area, _, _ = _p1_geometry(points, tris)
    N = len(points)
    wsum = np.zeros(N); Br_n = np.zeros(N); Bz_n = np.zeros(N)
    idx = tris.ravel()
    np.add.at(wsum, idx, np.repeat(area, 3))
    np.add.at(Br_n, idx, np.repeat(Br_e * area, 3))
    np.add.at(Bz_n, idx, np.repeat(Bz_e * area, 3))
    good = wsum > 0
    Br_n[good] /= wsum[good]; Bz_n[good] /= wsum[good]
    Br_c = Br_n[tris].mean(axis=1)
    Bz_c = Bz_n[tris].mean(axis=1)
    return np.hypot(Br_e - Br_c, Bz_e - Bz_c) * np.sqrt(area)


def _spt70_element_data(points, tris):
    """(mu_r, Jphi) per element for the SPT-70 circuit, from box membership of
    each triangle centroid. Conformity guarantees a centroid unambiguously
    identifies its material."""
    from .spt70_system import (
        IRON_PIECES, INNER_COIL, OUTER_COIL, IRON_MU_R, COIL_AMPERE_TURNS,
    )
    _, _, r_c = _p1_geometry(points, tris)
    z_c = points[tris][:, :, 1].mean(axis=1)

    def in_box(box):
        r0, r1, z0, z1 = box
        return (r_c >= r0) & (r_c <= r1) & (z_c >= z0) & (z_c <= z1)

    mu_r_tri = np.ones(len(tris))
    for box in IRON_PIECES:
        mu_r_tri[in_box(box)] = IRON_MU_R
    Jphi_tri = np.zeros(len(tris))
    for coil in (INNER_COIL, OUTER_COIL):
        r0, r1, z0, z1 = coil
        Jphi_tri[in_box(coil)] = COIL_AMPERE_TURNS / ((r1 - r0) * (z1 - z0))
    return mu_r_tri, Jphi_tri


def _spt70_dirichlet(points, r, z):
    """psi = 0 on the axis (r=0) and the outer domain boundary. Coordinate-
    based so it stays correct for refinement midpoints on boundary edges."""
    rr, zz = points[:, 0], points[:, 1]
    return (rr == r[0]) | (rr == r[-1]) | (zz == z[0]) | (zz == z[-1])


def solve_spt70_fem(B_target: float, nr: int = 121, nz: int = 221,
                    amr_passes: int = 0, amr_frac: float = 0.02):
    """Solve the SPT-70 magnetic system with nodal P1 FEM, calibrated so |B|
    at the mid-channel exit equals B_target.

    With amr_passes=0 the field is returned on the structured (nr, nz) grid as
    (r, z, psi, Br, Bz) — the same signature as
    spt70_system.solve_spt70_field, ready to swap in. With amr_passes>0 the
    mesh is adaptively refined at the pole-tip corners (see zz_error) and the
    result is unstructured: (points, tris, psi, Br, Bz), where Br/Bz are the
    per-material vacuum-side nodal field. Use field_from_fem to sample either
    form onto a target grid.
    """
    from .spt70_system import (
        build_spt70_system, REFERENCE_POINT,
        CHANNEL_Z0, CHANNEL_Z_EXIT, CHANNEL_R_IN, CHANNEL_R_OUT,
    )
    r, z, _, _ = build_spt70_system(nr, nz)
    points, tris = triangulate_structured(r, z)

    for _pass in range(amr_passes + 1):
        mu_r_tri, Jphi_tri = _spt70_element_data(points, tris)
        dirichlet = _spt70_dirichlet(points, r, z)
        psi = assemble(points, tris, mu_r_tri, Jphi_tri, dirichlet)
        if _pass == amr_passes:
            break
        err = zz_error(points, tris, psi)
        # Restrict refinement to the plasma-facing region — the channel annulus
        # plus the plume behind it. A global ZZ estimator would instead pour
        # elements into the high-field iron circuit interior (the core/coil
        # surfaces), which the plasma never sees. This targets the exit-plane
        # pole-tip corners we actually care about.
        r_c = points[tris][:, :, 0].mean(axis=1)
        z_c = points[tris][:, :, 1].mean(axis=1)
        in_channel = ((z_c >= CHANNEL_Z0) & (z_c < CHANNEL_Z_EXIT)
                      & (r_c >= CHANNEL_R_IN) & (r_c <= CHANNEL_R_OUT))
        in_plume = z_c >= CHANNEL_Z_EXIT
        roi = (mu_r_tri <= 1.0) & (in_channel | in_plume)
        err = np.where(roi, err, 0.0)
        k = max(1, int(amr_frac * roi.sum()))
        marked = np.argsort(err)[-k:]
        points, tris = refine_mesh(points, tris, marked)

    Br, Bz = recover_B_nodal(points, tris, psi, mu_r_tri)
    # calibrate on |B| at the mid-channel exit reference point
    iref = int(np.argmin(np.hypot(points[:, 0] - REFERENCE_POINT[0],
                                  points[:, 1] - REFERENCE_POINT[1])))
    scale = B_target / np.hypot(np.nan_to_num(Br[iref]), np.nan_to_num(Bz[iref]))
    Br = np.nan_to_num(Br) * scale
    Bz = np.nan_to_num(Bz) * scale
    psi = psi * scale

    if amr_passes == 0:                      # structured: reshape to (nr, nz)
        return r, z, psi.reshape(nr, nz), Br.reshape(nr, nz), Bz.reshape(nr, nz)
    return points, tris, psi, Br, Bz


_FIELD_CACHE: dict = {}


def field_on_grid_fem(grid, B_target: float, amr_passes: int = 4,
                      amr_frac: float = 0.05):
    """FEM magnetic field and streamfunction on the discharge grid.

    Drop-in for spt70_system.field_on_grid: returns (Br, Bz, lam), each
    (N_z, N_r), sampled from the (AMR-refined) FEM solution by piecewise-
    linear interpolation over the triangulation. Iron elements are masked, so
    the interpolation only ever uses the vacuum-side field — no in-iron leak
    crosses onto the grid. lam is the flux function psi itself (its contours
    are the field lines, d(psi) = r*B_z dr - r*B_r dz), which is exactly the
    magnetic streamfunction the lambda-layer machinery expects.

    The (B_target, grid, AMR) result is cached: the field is a fixed input
    that does not change across e.g. a discharge-voltage sweep, so the ~5 s
    FEM+AMR solve runs once per configuration, not once per call.
    """
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    from .spt70_system import CHANNEL_Z0

    key = (round(B_target, 12), grid.N_z, grid.N_r,
           round(grid.min_z, 9), round(grid.max_z, 9),
           round(grid.min_r, 9), round(grid.max_r, 9), amr_passes, amr_frac)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    points, tris, psi, Br, Bz = solve_spt70_fem(
        B_target, amr_passes=max(amr_passes, 1), amr_frac=amr_frac)
    mu_r_tri, _ = _spt70_element_data(points, tris)

    triang = Triangulation(points[:, 0], points[:, 1], tris)
    triang.set_mask(mu_r_tri > 1.0)          # never interpolate through iron

    Z, R = np.meshgrid(grid.z_nodes() + CHANNEL_Z0, grid.r_nodes(), indexing="ij")
    rq, zq = R.ravel(), Z.ravel()
    shape = (grid.N_z, grid.N_r)

    def _interp(vals):
        out = LinearTriInterpolator(triang, vals)(rq, zq)
        return np.nan_to_num(np.asarray(out)).reshape(shape)

    result = (_interp(Br), _interp(Bz), _interp(psi))
    _FIELD_CACHE[key] = result
    return result
