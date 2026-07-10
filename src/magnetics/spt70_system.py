# SPT-70-like magnetic system: iron circuit + two excitation coils,
# adapted from the earlier branch to this project's channel geometry
# (r 17.5-35 mm, L = 30 mm) and grid conventions.
#
# The solver works in the "hardware frame" where z runs from the back of
# the magnetic circuit; the discharge frame used everywhere else in the
# project has the anode at z = 0. CHANNEL_Z0 is the offset between them.
# Geometry is an illustrative approximation of an SPT-70-class thruster —
# representative topology and scale, not manufacturer data.

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

from .magnetic import solve_axisymmetric_flux, field_from_flux, magnetic_streamfunction
from ..structs.classes import Grid2D

# --- geometry (meters), hardware frame ---------------------------------
CHANNEL_R_IN = 0.0175      # discharge channel inner wall radius
CHANNEL_R_OUT = 0.035      # discharge channel outer wall radius
CHANNEL_Z0 = 0.015         # anode plane (discharge frame z = 0)
CHANNEL_Z_EXIT = 0.045     # exit plane (discharge frame z = L = 30 mm)

DOMAIN_R = 0.060           # solver domain radial extent
DOMAIN_Z = 0.110           # solver domain axial extent

IRON_MU_R = 1000.0         # relative permeability of the circuit iron

# Iron magnetic circuit pieces, as (r_min, r_max, z_min, z_max) boxes:
# back yoke joins the inner core and the outer screen, the two pole
# pieces face each other across the channel exit.
YOKE = (0.000, 0.054, 0.004, 0.012)
INNER_CORE = (0.000, 0.010, 0.012, 0.039)
INNER_POLE = (0.000, 0.017, 0.039, 0.045)
OUTER_SCREEN = (0.050, 0.054, 0.012, 0.045)
OUTER_POLE = (0.0355, 0.054, 0.039, 0.045)
IRON_PIECES = (YOKE, INNER_CORE, INNER_POLE, OUTER_SCREEN, OUTER_POLE)

# Excitation coils in the pockets between the iron and the channel; the
# flux loop encloses both, so equal-sign ampere-turns reinforce.
INNER_COIL = (0.011, 0.016, 0.014, 0.037)
OUTER_COIL = (0.041, 0.049, 0.014, 0.037)
COIL_AMPERE_TURNS = 1.0    # pre-calibration; the solve is linear in this

REFERENCE_POINT = ((CHANNEL_R_IN + CHANNEL_R_OUT) / 2.0, CHANNEL_Z_EXIT)


def _box_mask(r: np.ndarray, z: np.ndarray, box: tuple) -> np.ndarray:
    r_min, r_max, z_min, z_max = box
    return (
        (r[:, None] >= r_min) & (r[:, None] <= r_max)
        & (z[None, :] >= z_min) & (z[None, :] <= z_max)
    )


def iron_mask(r: np.ndarray, z: np.ndarray, dilate: int = 0) -> np.ndarray:
    """Boolean (nr, nz) mask of the iron circuit. With dilate=N the mask
    grows by N nodes: np.gradient uses centered differences, so the first
    vacuum node next to iron picks up half of the huge in-iron field —
    dilate by 1 to hide that contaminated ring when displaying |B|.
    """
    mask = np.zeros((len(r), len(z)), dtype=bool)
    for piece in IRON_PIECES:
        mask |= _box_mask(r, z, piece)
    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    return mask


def build_spt70_system(nr: int, nz: int):
    """(r, z) solver grid, permeability map and coil current density."""
    r = np.linspace(0.0, DOMAIN_R, nr)
    z = np.linspace(0.0, DOMAIN_Z, nz)

    mu_r = np.ones((nr, nz))
    for piece in IRON_PIECES:
        mu_r[_box_mask(r, z, piece)] = IRON_MU_R

    current_density = np.zeros((nr, nz))
    for coil in (INNER_COIL, OUTER_COIL):
        r_min, r_max, z_min, z_max = coil
        area = (r_max - r_min) * (z_max - z_min)
        current_density[_box_mask(r, z, coil)] = COIL_AMPERE_TURNS / area

    return r, z, mu_r, current_density


def solve_spt70_field(B_target: float, nr: int = 121, nz: int = 221):
    """Solve the magnetic system and calibrate the coil current so that
    |B| at mid-channel on the exit plane equals B_target.

    Returns (r, z, psi, Br, Bz) in the hardware frame, (nr, nz) layout.
    """
    r, z, mu_r, current_density = build_spt70_system(nr, nz)
    psi = solve_axisymmetric_flux(r, z, mu_r, current_density)
    Br, Bz = field_from_flux(psi, r, z)

    r_ref, z_ref = REFERENCE_POINT
    interp_Br = RegularGridInterpolator((r, z), Br)
    interp_Bz = RegularGridInterpolator((r, z), Bz)
    B_ref = np.hypot(interp_Br((r_ref, z_ref)), interp_Bz((r_ref, z_ref)))
    scale = B_target / B_ref

    return r, z, psi * scale, Br * scale, Bz * scale


def field_on_grid(grid: Grid2D, B_target: float):
    """Magnetic field and streamfunction on the discharge grid.

    Solves the magnetic system in the hardware frame, then interpolates
    onto the Grid2D nodes (anode at z = 0). Returns (Br, Bz, lam), each
    of shape (N_z, N_r) — the project's field layout, ready for gather().
    """
    r, z, psi, Br, Bz = solve_spt70_field(B_target)
    lam = magnetic_streamfunction(Br, Bz, r, z)

    Z, R = np.meshgrid(grid.z_nodes() + CHANNEL_Z0, grid.r_nodes(), indexing="ij")
    points = np.stack([R.ravel(), Z.ravel()], axis=-1)

    shape = (grid.N_z, grid.N_r)
    Br_g = RegularGridInterpolator((r, z), Br)(points).reshape(shape)
    Bz_g = RegularGridInterpolator((r, z), Bz)(points).reshape(shape)
    lam_g = RegularGridInterpolator((r, z), lam)(points).reshape(shape)
    return Br_g, Bz_g, lam_g
