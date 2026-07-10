# Demo: everything built so far in one script.
# Neutral xenon flow, SPT-70 channel + plume region behind the exit:
# inject -> free flight -> walls/exit -> deposit -> density field -> figure.
# Run either way:
#   python -m src.examples.neutral_demo        (from repo root)
#   python src/examples/neutral_demo.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

from src.structs.classes import Grid2D, Thruster, ParticleArray
from src.injection.inject import inject_on_grid
from src.deposition.deposit import locate_particle, scatter
from src.neutrals.neutrals import node_volume, density, push_neutrals, apply_boundaries


def main():
    thruster = Thruster(
        r_min=0.0175, r_max=0.035, channel_length=0.03,
        mdot=2.5e-6, B_r_max=0.015, voltage=300,
        mass=131.293 * cst.atomic_mass, temperature_anode=750.0,
    )
    # domain: channel (30 mm) + plume region behind the exit (30 mm more,
    # radially from the axis out to 50 mm); 0.5 mm cells
    grid = Grid2D(max_z=0.06, max_r=0.05, N_r=101, N_z=121)

    dt = 5e-7
    n_steps = 3000
    weight = 4e10
    sample_from = 2500  # steady state by then (checked in neutral_flow.py)

    v_th = np.sqrt(thruster.temperature_anode * cst.Boltzmann / thruster.mass)
    V = node_volume(grid)

    part = ParticleArray()
    S0_acc = np.zeros((grid.N_z, grid.N_r))
    n_samples = 0

    for step in range(n_steps):
        part.extend(inject_on_grid(dt, thruster, weight))
        z_prev = part.z.copy()
        push_neutrals(part, dt)
        part.keep(apply_boundaries(part, thruster, grid, v_th, z_prev))

        if step >= sample_from:
            i, j, fz, fr = locate_particle(part, grid)
            S0_acc += scatter(part, grid, i, j, fz, fr)
            n_samples += 1

        if (step + 1) % 500 == 0:
            print(f"step {step+1:5d}: {len(part):7d} particles")

    n_grid = density(S0_acc / n_samples, V)  # (N_z, N_r), time-averaged

    # --- figure: 2D density map + axial profile in the channel annulus ---
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
    # channel outline
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

    out = "neutral_density.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
