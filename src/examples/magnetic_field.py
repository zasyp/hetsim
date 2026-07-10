# SPT-70-like magnetic system: solve, check, and draw.
# Left: |B| map with field lines and the hardware (iron, coils, channel).
# Right: the radial field profile B_r(z) along mid-channel — the curve
# the electron model will live on (peak at the exit plane).
# Run either way:
#   python -m src.examples.magnetic_field        (from repo root)
#   python src/examples/magnetic_field.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.structs.classes import Grid2D, Thruster
from src.magnetics.spt70_system import (
    solve_spt70_field, field_on_grid, iron_mask, _box_mask,
    IRON_PIECES, INNER_COIL, OUTER_COIL,
    CHANNEL_R_IN, CHANNEL_R_OUT, CHANNEL_Z0, CHANNEL_Z_EXIT,
)
import scipy.constants as cst


def draw_hardware(ax):
    """Iron (grey), coils (orange), channel walls (black); hardware frame."""
    for piece in IRON_PIECES:
        r_min, r_max, z_min, z_max = piece
        ax.add_patch(Rectangle(
            (z_min * 1e3, r_min * 1e3), (z_max - z_min) * 1e3, (r_max - r_min) * 1e3,
            facecolor="0.55", edgecolor="0.2", lw=0.8, alpha=0.9, zorder=3,
        ))
    for coil in (INNER_COIL, OUTER_COIL):
        r_min, r_max, z_min, z_max = coil
        ax.add_patch(Rectangle(
            (z_min * 1e3, r_min * 1e3), (z_max - z_min) * 1e3, (r_max - r_min) * 1e3,
            facecolor="darkorange", edgecolor="saddlebrown", lw=0.8,
            alpha=0.85, zorder=3,
        ))
    for r_wall in (CHANNEL_R_IN, CHANNEL_R_OUT):
        ax.plot([CHANNEL_Z0 * 1e3, CHANNEL_Z_EXIT * 1e3],
                [r_wall * 1e3, r_wall * 1e3], "k-", lw=2, zorder=4)
    ax.plot([CHANNEL_Z0 * 1e3, CHANNEL_Z0 * 1e3],
            [CHANNEL_R_IN * 1e3, CHANNEL_R_OUT * 1e3], "k-", lw=2, zorder=4)


def main():
    thruster = Thruster(
        r_min=0.0175, r_max=0.035, channel_length=0.03,
        mdot=2.5e-6, B_r_max=0.015, voltage=300,
        mass=131.293 * cst.atomic_mass, temperature_anode=750.0,
    )
    grid = Grid2D(max_z=0.06, max_r=0.05, N_r=101, N_z=121)

    # hardware-frame solution for the map
    r, z, psi, Br, Bz = solve_spt70_field(thruster.B_r_max)
    B_mag = np.hypot(Br, Bz)

    # discharge-frame field on the simulation grid
    Br_g, Bz_g, lam_g = field_on_grid(grid, thruster.B_r_max)

    # --- diagnostics ---
    z_nodes = grid.z_nodes()
    r_nodes = grid.r_nodes()
    j_mid = np.argmin(np.abs(r_nodes - (thruster.r_min + thruster.r_max) / 2))
    i_exit = np.argmin(np.abs(z_nodes - thruster.channel_length))
    Br_mid = Br_g[:, j_mid]

    B_exit = np.hypot(Br_g[i_exit, j_mid], Bz_g[i_exit, j_mid])
    print(f"|B| at exit mid-channel: {B_exit*1e4:.1f} G (target {thruster.B_r_max*1e4:.0f} G)")
    print(f"radial fraction |Br|/|B| there: {abs(Br_g[i_exit, j_mid])/B_exit:.2f}")
    i_peak = np.argmax(np.abs(Br_mid))
    print(f"Br peak on mid-channel: {abs(Br_mid[i_peak])*1e4:.1f} G at z = {z_nodes[i_peak]*1e3:.1f} mm"
          f" (exit at {thruster.channel_length*1e3:.0f} mm)")
    print(f"Br at anode: {abs(Br_mid[0])*1e4:.1f} G"
          f"  ({abs(Br_mid[0]/Br_mid[i_peak])*100:.0f}% of peak)")

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), width_ratios=[1.6, 1], layout="constrained")

    # left: |B| with field lines, hardware frame (mm); the iron mask is
    # dilated by one node to hide the ring contaminated by centered
    # differences across the vacuum/iron interface
    free = ~iron_mask(r, z, dilate=1)
    B_show = np.where(free, B_mag, np.nan) * 1e4

    channel = _box_mask(r, z, (CHANNEL_R_IN, CHANNEL_R_OUT, CHANNEL_Z0, z[-1]))
    vmax = np.percentile(B_mag[channel], 98) * 1e4

    Z, R = np.meshgrid(z * 1e3, r * 1e3, indexing="xy")
    cf = ax1.contourf(z * 1e3, r * 1e3, B_show, levels=np.linspace(0, vmax, 40),
                      cmap="viridis", extend="max")
    ax1.contour(z * 1e3, r * 1e3, psi, levels=45, colors="white", linewidths=0.5)
    fig.colorbar(cf, ax=ax1, label="|B|, G")
    draw_hardware(ax1)
    ax1.axvline(CHANNEL_Z0 * 1e3, color="w", ls=":", lw=1)
    ax1.set_xlabel("z (hardware frame), mm")
    ax1.set_ylabel("r, mm")
    ax1.set_title("|B| and field lines ($\\psi$ contours)")

    # right: Br along mid-channel, discharge frame
    ax2.plot(z_nodes * 1e3, np.abs(Br_mid) * 1e4)
    ax2.axvline(thruster.channel_length * 1e3, color="k", ls="--", lw=1,
                label="exit plane")
    ax2.set_xlabel("z (anode at 0), mm")
    ax2.set_ylabel("|B$_r$|, G")
    ax2.set_title("radial field along mid-channel")
    ax2.grid(alpha=0.3)
    ax2.legend()

    out = "magnetic_field.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
