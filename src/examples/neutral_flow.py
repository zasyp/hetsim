# Checkpoint 1: pure neutral flow, SPT-70 channel + plume region behind it.
# Checks: (1) steady-state outflow through the open boundaries == mdot,
#         (2) mass flow through every channel cross-section == mdot,
#         (3) density near the anode vs the kinetic estimate 4*mdot/(m*v_bar*S).
# Also saves the time-averaged density map (neutral_density.png).
# Run either way:
#   python -m src.examples.neutral_flow        (from repo root)
#   python src/examples/neutral_flow.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.classes import ParticleArray
from src.injection.inject import inject_on_grid
from src.deposition.deposit import locate_particle, scatter
from src.neutrals.neutrals import node_volume, density, push_neutrals, apply_boundaries


def plot_density(n_grid, thruster, grid, out="neutral_density.png"):
    """2D density map + axial profile averaged over the channel annulus."""
    z_nodes = grid.z_nodes()
    r_nodes = grid.r_nodes()
    z_mm = z_nodes * 1e3
    r_mm = r_nodes * 1e3
    L_mm = thruster.channel_length * 1e3

    # mask the thruster body (z < L outside the channel annulus)
    Z, R = np.meshgrid(z_nodes, r_nodes, indexing="ij")
    body = (Z < thruster.channel_length) & (
        (R < thruster.r_min) | (R > thruster.r_max)
    )
    n_plot = np.where(body, np.nan, n_grid)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )

    pcm = ax1.pcolormesh(z_mm, r_mm, n_plot.T, cmap="viridis", shading="gouraud")
    fig.colorbar(pcm, ax=[ax1, ax2], label="n, m$^{-3}$")
    for r_wall in (thruster.r_min * 1e3, thruster.r_max * 1e3):
        ax1.plot([0, L_mm], [r_wall, r_wall], "w-", lw=1.5)
    ax1.axvline(L_mm, color="w", ls="--", lw=1)
    ax1.set_ylabel("r, mm")
    ax1.set_title("Neutral Xe density: SPT-70 channel + plume (no plasma)")

    annulus = (r_nodes >= thruster.r_min) & (r_nodes <= thruster.r_max)
    ax2.plot(z_mm, n_grid[:, annulus].mean(axis=1) * 1e-19)
    ax2.axvline(L_mm, color="k", ls="--", lw=1, label="exit plane")
    ax2.set_xlabel("z, mm")
    ax2.set_ylabel("n, 10$^{19}$ m$^{-3}$")
    ax2.set_title("profile averaged over the channel annulus")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    gas = thruster.propellant

    dt = 5e-7
    n_steps = 4000
    weight = 2e10
    sample_from = 3 * n_steps // 4  # average over the last quarter

    v_th = gas.thermal_speed(thruster.temperature_anode)
    V = node_volume(grid)
    dz = (grid.max_z - grid.min_z) / (grid.N_z - 1)

    part = ParticleArray()

    S0_acc = np.zeros((grid.N_z, grid.N_r))   # time-averaged sum(w)
    Svz_acc = np.zeros((grid.N_z, grid.N_r))  # time-averaged sum(w*v_z)
    exited_weight = 0.0
    n_samples = 0

    for step in range(n_steps):
        part.extend(inject_on_grid(dt, thruster, weight))
        z_prev = part.z.copy()
        push_neutrals(part, dt)
        alive = apply_boundaries(part, thruster, grid, v_th, z_prev)

        if step >= sample_from:
            exited_weight += part.weight[~alive].sum()

        part.keep(alive)

        if step >= sample_from:
            i, j, fz, fr = locate_particle(part, grid)
            S0_acc += scatter(part, grid, i, j, fz, fr)
            Svz_acc += scatter(part, grid, i, j, fz, fr, q=part.v_z)
            n_samples += 1

        if (step + 1) % 1000 == 0:
            print(f"step {step+1:5d}: {len(part):7d} particles")

    # --- checks ---
    m = gas.mass
    t_sampled = n_samples * dt

    # 1) total outflow through the open plume boundaries
    mdot_exit = m * exited_weight / t_sampled
    print(f"\noutflow:     {mdot_exit*1e6:.3f} mg/s  (mdot = {thruster.mdot*1e6:.3f}),"
          f"  ratio {mdot_exit/thruster.mdot:.3f}")

    # 2) mass flow through each cross-section INSIDE the channel:
    #    m/dz * sum_r <w*v_z>; in the plume mass also leaves radially,
    #    so the axial flow there is legitimately below mdot
    i_exit = int(round(thruster.channel_length / dz))
    mdot_z = m * (Svz_acc / n_samples).sum(axis=1) / dz
    interior = mdot_z[1:i_exit] / thruster.mdot
    print(f"flow vs z:   mean {interior.mean():.3f}, min {interior.min():.3f},"
          f" max {interior.max():.3f}  (channel interior, want ~1)")

    # 3) density near the anode (channel annulus only) vs kinetic estimate
    r_nodes = grid.r_nodes()
    annulus = (r_nodes >= thruster.r_min) & (r_nodes <= thruster.r_max)
    n_grid = density(S0_acc / n_samples, V)
    n_anode = n_grid[0:3, annulus].mean()
    v_bar = gas.mean_speed(thruster.temperature_anode)
    S_exit = thruster.exit_area()
    n_est = 4 * thruster.mdot / (m * v_bar * S_exit)
    print(f"anode density: {n_anode:.3e} m^-3, estimate {n_est:.3e} m^-3,"
          f"  ratio {n_anode/n_est:.2f}")

    plot_density(n_grid, thruster, grid)


if __name__ == "__main__":
    main()
