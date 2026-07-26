# Sanity check for electron_liquid block 4: solve current continuity on
# the lambda layers for the thermalized potential phi*(lambda), on the
# real SPT-70 field with the full block 1->2->3->4 pipeline.
#
# The payoff: phi* falls monotonically from V_anode (300 V) to V_cathode
# (0 V), and MOST of that 300 V drop sits on the low-conductance faces
# near the exit plane -- the acceleration region. The per-face voltage
# step is |dphi*_k| = dI_k / G_k, so it peaks where G_face bottoms out
# (see conductance_check.py). If phi* is non-monotonic or the drop is
# spread evenly, the bug is in the assembly (solve_potential), not here.
#
# Te / n_e / n_n are prescribed placeholder profiles (no energy or
# ionization solve yet, as in hall_parameter.py / conductance_check.py);
# only their rough magnitude and z-shape matter.
#
# Run either way:
#   python -m src.examples.potential_check        (from repo root)
#   python src/examples/potential_check.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.magnetics.spt70_system import field_on_grid
from src.neutrals.neutrals import node_volume
from src.electron_liquid.default_plasm_params import omega_ce
from src.electron_liquid.collisions import (
    neutral_collision, ionization_collision, coulomb_collision,
    anomaly_collision, anomalous_alpha, electron_collision,
    ANOMALY_PROFILE_PRESETS,
)
from src.electron_liquid.mobility import zeroB_mobility, hall_parameter, perp_mobility
from src.electron_liquid.lambda_layers import (
    lambda_range, build_layers, thruster_body_mask, layer_average,
)
from src.electron_liquid.layer_potential import (
    node_weight, layer_conductance, layer_ionization_current,
)
from src.electron_liquid.solve_potential import (
    solve_potential, potential_on_grid, electric_field,
)

N_LAYERS = 40
PRESET_NAME = "SPT-100"


def plot_potential(z_layer_mm, phi_star, z_face_mm, G_face, L_mm, V_a, out="potential.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")

    ax1.plot(z_layer_mm, phi_star, "o-", ms=3)
    ax1.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax1.axhline(0, color="0.7", lw=0.8)
    ax1.set_ylabel("$\\varphi^*$, V")
    ax1.set_title(f"thermalized potential on the layers ({V_a:.0f} V discharge)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(z_face_mm, G_face, "o-", ms=3, color="tab:red")
    ax2.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("z, mm (anode at 0; layers/faces placed at mid-channel z)")
    ax2.set_ylabel("$G_{j+1/2}$, A/V")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_field_2d(phi, E_z, z, r, thruster, out="potential_field.png"):
    """2-D reconstructed potential (contours + field) and mid-channel E_z."""
    z_mm, r_mm = z * 1e3, r * 1e3
    L_mm = thruster.channel_length * 1e3

    # mask the thruster body so it doesn't dominate the color scale
    Z, R = np.meshgrid(z, r, indexing="ij")
    body = (Z < thruster.channel_length) & (
        (R < thruster.r_min) | (R > thruster.r_max))
    phi_show = np.where(body, np.nan, phi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={"height_ratios": [2, 1]},
                                   layout="constrained")

    pcm = ax1.pcolormesh(z_mm, r_mm, phi_show.T, cmap="viridis", shading="gouraud")
    ax1.contour(z_mm, r_mm, phi_show.T, levels=np.arange(0, 301, 25),
                colors="w", linewidths=0.4)
    fig.colorbar(pcm, ax=ax1, label="$\\varphi$, V")
    for r_wall in (thruster.r_min * 1e3, thruster.r_max * 1e3):
        ax1.plot([0, L_mm], [r_wall, r_wall], "k-", lw=1.5)
    ax1.axvline(L_mm, color="w", ls="--", lw=1)
    ax1.set_ylabel("r, mm")
    ax1.set_title("reconstructed potential $\\varphi(z,r)$ (equipotentials ~ field lines)")

    j_mid = np.argmin(np.abs(r - (thruster.r_min + thruster.r_max) / 2))
    ax2.plot(z_mm, E_z[:, j_mid] * 1e-3)
    ax2.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax2.set_xlabel("z, mm (anode at 0)")
    ax2.set_ylabel("$E_z$ mid-channel, kV/m")
    ax2.set_title("axial accelerating field")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    gas = xenon()
    z = grid.z_nodes()
    r = grid.r_nodes()
    L = thruster.channel_length
    preset = ANOMALY_PROFILE_PRESETS[PRESET_NAME]

    Br, Bz, lam = field_on_grid(grid, thruster.B_r_max)
    B = np.hypot(Br, Bz)

    # placeholder plasma state (z-profiles broadcast over r)
    Te = np.interp(z, [0.0, L, 2 * L], [3.0, 25.0, 10.0])[:, None]
    n_n = (1e20 * np.exp(-z / (0.4 * L)))[:, None]
    n_e = np.interp(z, [0.0, L, 2 * L], [1e16, 2e17, 5e16])[:, None]
    alpha = anomalous_alpha(z, L, background_pressure_torr=100.0, **preset)[:, None]

    nu = electron_collision(
        neutral_collision(Te, n_n, gas),
        ionization_collision(Te, n_n, gas),
        coulomb_collision(n_e, Te),
        anomaly_collision(B, alpha),
    )
    mu = perp_mobility(zeroB_mobility(nu), hall_parameter(omega_ce(B), nu))

    body = thruster_body_mask(grid, thruster)
    lam_a, lam_c = lambda_range(lam, z, r, thruster)
    edges, centers, idx = build_layers(lam, N_LAYERS, lam_a, lam_c, body)
    d_lambda = edges[1] - edges[0]

    dV = node_volume(grid)
    ne2, nn2 = np.broadcast_to(n_e, B.shape), np.broadcast_to(n_n, B.shape)
    k_iz = gas.ionization_rate_Te(np.broadcast_to(Te, B.shape))

    w = node_weight(mu, ne2, Br, Bz, r, dV)
    G_face = layer_conductance(idx, N_LAYERS, d_lambda, w)
    dI_iz = layer_ionization_current(idx, ne2, nn2, k_iz, dV, N_LAYERS)

    phi = solve_potential(G_face, dI_iz, thruster.voltage, 0.0)

    # place each layer at its mean mid-channel z for a spatial x-axis
    j_mid = np.argmin(np.abs(r - (thruster.r_min + thruster.r_max) / 2))
    idx_mid = idx[:, j_mid]
    z_layer = np.array([z[idx_mid == l].mean() if np.any(idx_mid == l) else np.nan
                        for l in range(N_LAYERS)])
    z_face = 0.5 * (z_layer[:-1] + z_layer[1:])

    # --- diagnostics ---
    # phi* is expected to fall from anode to cathode, EXCEPT for a small
    # hump just downstream of the anode: ionization born in those first
    # layers injects ion current that must be driven back upstream, which
    # needs phi* slightly ABOVE the anode. That is the physical near-anode
    # potential hump (Hara 2019 review, Fig. 12), not an assembly bug — so
    # report its size instead of failing on it. A hump of more than a few
    # percent of V_d, or non-monotonicity anywhere downstream of it, IS a bug.
    dphi = np.diff(phi)
    hump = float(phi.max() - phi[0])
    k_peak = int(np.argmax(phi))
    monotonic_after = np.all(dphi[k_peak:] <= 1e-9)
    k_drop = int(np.argmax(np.abs(dphi)))
    drop_channel = np.abs(phi[0] - np.interp(L, z_layer, phi))

    print(f"phi*: anode {phi[0]:.1f} V -> cathode {phi[-1]:.1f} V, "
          f"monotonic below the near-anode hump: "
          f"{'OK' if monotonic_after else 'FAIL'}")
    print(f"near-anode hump: +{hump:.1f} V "
          f"({hump/thruster.voltage*100:.1f}% of V_d) peaking at "
          f"z = {z_layer[k_peak]*1e3:.1f} mm")
    print(f"largest single-face drop: {abs(dphi[k_drop]):.1f} V at "
          f"z = {z_face[k_drop]*1e3:.1f} mm  (exit at {L*1e3:.0f} mm)")
    print(f"drop anode->exit plane: {drop_channel:.0f} V of {thruster.voltage} V "
          f"({drop_channel/thruster.voltage*100:.0f}% inside the channel)")
    print(f"G_face minimum at z = {z_face[int(np.argmin(G_face))]*1e3:.1f} mm "
          f"(drop should peak near here)")

    plot_potential(z_layer * 1e3, phi, z_face * 1e3, G_face,
                   L * 1e3, thruster.voltage)

    # --- reconstruct the 2-D potential and the accelerating field ---
    Te_layers = layer_average(idx, np.broadcast_to(Te, B.shape), dV, N_LAYERS)
    lam_layer = layer_average(idx, lam, dV, N_LAYERS)
    phi_grid = potential_on_grid(phi, lam_layer, lam, Te_layers, ne2)
    E_z, E_r = electric_field(phi_grid, z, r)

    i_Ez = int(np.argmax(np.abs(E_z[:, j_mid])))
    print(f"peak axial field E_z: {abs(E_z[i_Ez, j_mid])*1e-3:.1f} kV/m at "
          f"z = {z[i_Ez]*1e3:.1f} mm  (exit at {L*1e3:.0f} mm)")

    plot_field_2d(phi_grid, E_z, z, r, thruster)


if __name__ == "__main__":
    main()
