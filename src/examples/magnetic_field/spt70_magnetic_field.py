"""Magnetic field of an SPT-70-class Hall thruster magnetic system.

Unlike the earlier version of this file -- which faked the magnetic circuit
by pinning the scalar potential to +/-phi0 on two thin "pole tip" rows and
walling off everything behind them -- this models the actual magnetic system:
an iron magnetic circuit (the "magnetopровod": back yoke, inner core, outer
screen, and the two pole pieces) as a high-permeability material, energised by
two azimuthal excitation coils (inner + outer). The field is obtained from the
axisymmetric vector-potential / flux-function formulation, so the iron guides
the flux and the coils drive it, exactly as in a real thruster.

The flux path is a closed loop through the iron: up the inner core, out across
the discharge channel through the radial gap between the inner and outer pole
tips (this is the working, mostly-radial field the discharge sees), into the
outer screen, and back through the back yoke -- both coils sit inside this loop
so their magnetomotive forces add. Because the field lines are contours of the
flux function psi = r*A_phi, they close cleanly through the circuit with no
spurious corner singularities.

Geometry (channel radii, iron/coil footprints, domain size) is an illustrative
approximation of an SPT-70-class thruster -- representative of the topology and
scale, not certified manufacturer data. Use it as a template: swap in your own
measured/CAD geometry and coil ampere-turns for real analysis.

The excitation is set in ampere-turns and then rescaled once so that |B| at a
representative point (mid-channel, near the exit plane) matches TARGET_B_TESLA;
the problem is linear in the coil current, so a single scalar does this exactly
without a second solve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from physics.magnetic import solve_axisymmetric_flux, field_from_flux
from numerical.numerical_funcs import bilinear_interp

# --- SPT-70-class geometry (meters), illustrative ---------------------------
# Axial coordinate z runs from the back of the magnetic circuit; the discharge
# channel spans CHANNEL_Z0 (anode) to CHANNEL_Z_EXIT (exit plane).
CHANNEL_R_IN = 0.021       # discharge channel inner wall radius
CHANNEL_R_OUT = 0.035      # discharge channel outer wall radius
CHANNEL_Z0 = 0.020         # anode plane
CHANNEL_Z_EXIT = 0.045     # exit plane

DOMAIN_R = 0.060           # simulation domain radial extent
DOMAIN_Z = 0.100           # simulation domain axial extent

IRON_MU_R = 1000.0         # relative permeability of the magnetic circuit iron

# Iron magnetic circuit pieces, as (r_min, r_max, z_min, z_max) boxes.
YOKE = (0.000, 0.054, 0.004, 0.012)        # back plate joining core and screen
INNER_CORE = (0.000, 0.010, 0.012, 0.039)  # central magnetic core
INNER_POLE = (0.000, 0.0205, 0.039, 0.045) # inner pole piece (tip at r~R_IN)
OUTER_SCREEN = (0.050, 0.054, 0.012, 0.045)# outer magnetic screen
OUTER_POLE = (0.0355, 0.054, 0.039, 0.045) # outer pole piece (tip at r~R_OUT)
IRON_PIECES = (YOKE, INNER_CORE, INNER_POLE, OUTER_SCREEN, OUTER_POLE)

# Excitation coils, as (r_min, r_max, z_min, z_max) boxes. They sit in the
# annular pockets between the iron and the channel; the flux loop encloses
# both, so equal-sign ampere-turns reinforce each other.
INNER_COIL = (0.011, 0.019, 0.014, 0.037)
OUTER_COIL = (0.041, 0.049, 0.014, 0.037)
COIL_AMPERE_TURNS = 1.0  # N*I per coil (pre-calibration; scale is arbitrary)

# Representative calibration target: SPT-class thrusters peak in the
# ~100-300 G range near the channel exit. Order-of-magnitude target only.
TARGET_B_TESLA = 0.02
REFERENCE_POINT = ((CHANNEL_R_IN + CHANNEL_R_OUT) / 2.0, CHANNEL_Z_EXIT)


def _box_mask(r: np.ndarray, z: np.ndarray, box: tuple) -> np.ndarray:
    """Boolean (nr, nz) mask for grid nodes inside an (r_min, r_max, z_min,
    z_max) box."""
    r_min, r_max, z_min, z_max = box
    return (
        (r[:, None] >= r_min) & (r[:, None] <= r_max)
        & (z[None, :] >= z_min) & (z[None, :] <= z_max)
    )


def build_spt70_system(
        nr: int, nz: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the (r, z) grid, the relative-permeability map for the iron
    magnetic circuit, and the azimuthal coil current-density source.

    Returns (r, z, mu_r, current_density).
    """
    r = np.linspace(0.0, DOMAIN_R, nr)
    z = np.linspace(0.0, DOMAIN_Z, nz)

    mu_r = np.ones((nr, nz))
    for piece in IRON_PIECES:
        mu_r[_box_mask(r, z, piece)] = IRON_MU_R

    current_density = np.zeros((nr, nz))
    for coil in (INNER_COIL, OUTER_COIL):
        mask = _box_mask(r, z, coil)
        r_min, r_max, z_min, z_max = coil
        area = (r_max - r_min) * (z_max - z_min)
        current_density[mask] = COIL_AMPERE_TURNS / area

    return r, z, mu_r, current_density


def calibrate_field_scale(
        Br: np.ndarray, Bz: np.ndarray, r: np.ndarray, z: np.ndarray,
        ) -> float:
    """Scale factor mapping the field magnitude at REFERENCE_POINT onto
    TARGET_B_TESLA. The whole solve is linear in the coil current, so every
    field quantity can be multiplied by this one scalar.
    """
    r_ref, z_ref = REFERENCE_POINT
    Br_ref = bilinear_interp(Br, r, z, r_ref, z_ref)
    Bz_ref = bilinear_interp(Bz, r, z, r_ref, z_ref)
    return TARGET_B_TESLA / np.hypot(Br_ref, Bz_ref)


def _draw_hardware(ax: plt.Axes) -> None:
    """Overlay the iron circuit (grey), coils (orange), and channel walls
    (black) on a (z, r) axes, for orientation only -- purely cosmetic."""
    for piece in IRON_PIECES:
        r_min, r_max, z_min, z_max = piece
        ax.add_patch(Rectangle(
            (z_min, r_min), z_max - z_min, r_max - r_min,
            facecolor='0.55', edgecolor='0.2', lw=0.8, alpha=0.9, zorder=3,
        ))
    for coil in (INNER_COIL, OUTER_COIL):
        r_min, r_max, z_min, z_max = coil
        ax.add_patch(Rectangle(
            (z_min, r_min), z_max - z_min, r_max - r_min,
            facecolor='darkorange', edgecolor='saddlebrown', lw=0.8,
            alpha=0.85, zorder=3,
        ))
    ax.plot([CHANNEL_Z0, CHANNEL_Z_EXIT], [CHANNEL_R_IN, CHANNEL_R_IN],
            color='black', lw=2, zorder=4)
    ax.plot([CHANNEL_Z0, CHANNEL_Z_EXIT], [CHANNEL_R_OUT, CHANNEL_R_OUT],
            color='black', lw=2, zorder=4)
    ax.plot([CHANNEL_Z0, CHANNEL_Z0], [CHANNEL_R_IN, CHANNEL_R_OUT],
            color='black', lw=2, zorder=4)


def plot_results(
        r: np.ndarray, z: np.ndarray,
        Br: np.ndarray, Bz: np.ndarray, psi: np.ndarray,
        ) -> plt.Figure:
    """Two-panel figure: |B| with field lines (psi contours) on the left, the
    B vector field on the right, both on shared (z, r) axes with the hardware
    outline. |B| is shown only outside the iron (the field of interest lives
    in the channel and plume; |B| inside the iron is not meaningful here) and
    the colour scale is capped at the 99th percentile of the free-space field.
    """
    B_mag = np.hypot(Br, Bz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    free = np.ones_like(B_mag, dtype=bool)
    for piece in IRON_PIECES:
        free &= ~_box_mask(r, z, piece)
    B_show = np.where(free, B_mag, np.nan) * 1e4

    # Scale the colour map to the working field the discharge sees (channel +
    # near plume), so the physically interesting gradient is readable; the
    # much stronger flux concentration at the pole tips simply saturates.
    channel = _box_mask(r, z, (CHANNEL_R_IN, CHANNEL_R_OUT, CHANNEL_Z0, DOMAIN_Z))
    vmax = np.percentile(B_mag[channel], 98) * 1e4

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    ax = axes[0]
    cf = ax.contourf(Z, R, B_show, levels=np.linspace(0, vmax, 40),
                     cmap='viridis', extend='max')
    levels = np.linspace(psi.min(), psi.max(), 45)
    ax.contour(Z, R, psi, levels=levels, colors='white', linewidths=0.6)
    fig.colorbar(cf, ax=ax, label='|B| (Gauss)')
    _draw_hardware(ax)
    ax.set_xlabel('z, m (distance from circuit back)')
    ax.set_ylabel('r, m (distance from centerline)')
    ax.set_title('|B| with field lines (contours of psi)')

    # Right panel: unit-length arrows (direction only) coloured by |B|, drawn
    # in free space only, so the field the plasma sees is legible without the
    # very long arrows the strong in-iron flux would otherwise produce.
    ax = axes[1]
    stride = 4
    Bn = np.where(free & (B_mag > 0), B_mag, np.nan)
    U = np.where(free, Bz / Bn, np.nan)
    V = np.where(free, Br / Bn, np.nan)
    ax.quiver(
        Z[::stride, ::stride], R[::stride, ::stride],
        U[::stride, ::stride], V[::stride, ::stride],
        np.clip(B_mag[::stride, ::stride] * 1e4, 0, vmax), cmap='viridis',
        pivot='mid', scale=40, width=0.004,
    )
    _draw_hardware(ax)
    ax.set_xlabel('z, m (distance from circuit back)')
    ax.set_title('B direction (arrows) and magnitude (colour)')

    fig.suptitle('SPT-70-like magnetic system: iron circuit + coils '
                 '(illustrative geometry)')
    fig.tight_layout()
    return fig


def main() -> None:
    """Solve the SPT-70-class magnetic system, calibrate the field to a
    representative magnitude, print a short summary of the channel field, and
    save/show a two-panel plot.
    """
    nr, nz = 121, 201
    r, z, mu_r, current_density = build_spt70_system(nr, nz)

    psi = solve_axisymmetric_flux(r, z, mu_r, current_density)
    Br, Bz = field_from_flux(psi, r, z)

    scale = calibrate_field_scale(Br, Bz, r, z)
    psi, Br, Bz = psi * scale, Br * scale, Bz * scale
    B_mag = np.hypot(Br, Bz)

    r_ref, z_ref = REFERENCE_POINT
    Br_ref = bilinear_interp(Br, r, z, r_ref, z_ref)
    Bz_ref = bilinear_interp(Bz, r, z, r_ref, z_ref)
    print(f"calibration scale applied to coil current: {scale:.4g}")
    print(f"field at reference point (r={r_ref*1e3:.1f} mm, z={z_ref*1e3:.1f} mm): "
          f"|B| = {np.hypot(Br_ref, Bz_ref)*1e4:.1f} G "
          f"(Br = {Br_ref*1e4:.1f} G, Bz = {Bz_ref*1e4:.1f} G)")
    print(f"  -> radial fraction |Br|/|B| at exit mid-channel: "
          f"{abs(Br_ref)/np.hypot(Br_ref, Bz_ref):.2f} "
          f"(SPT fields are predominantly radial in the channel)")

    # Field magnitude sampled along the channel centreline vs axial position.
    r_mid = REFERENCE_POINT[0]
    print("axial profile of |B| along channel mid-radius:")
    for zc in np.linspace(CHANNEL_Z0, CHANNEL_Z_EXIT, 6):
        b = np.hypot(bilinear_interp(Br, r, z, r_mid, zc),
                     bilinear_interp(Bz, r, z, r_mid, zc))
        print(f"  z = {zc*1e3:5.1f} mm : |B| = {b*1e4:6.1f} G")

    fig = plot_results(r, z, Br, Bz, psi)
    fig.savefig(Path(__file__).with_name('spt70_magnetic_field.png'), dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
