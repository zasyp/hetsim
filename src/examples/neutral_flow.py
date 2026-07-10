# Checkpoint 1: pure neutral flow, SPT-70 channel + plume region behind it.
# Checks: (1) steady-state outflow through the open boundaries == mdot,
#         (2) mass flow through every channel cross-section == mdot,
#         (3) density near the anode vs the kinetic estimate 4*mdot/(m*v_bar*S).
# Run either way:
#   python -m src.examples.neutral_flow        (from repo root)
#   python src/examples/neutral_flow.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
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
    n_steps = 4000
    weight = 2e10
    sample_from = 3 * n_steps // 4  # average over the last quarter

    v_th = np.sqrt(thruster.temperature_anode * cst.Boltzmann / thruster.mass)
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
    m = thruster.mass
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
    v_bar = np.sqrt(8 * cst.Boltzmann * thruster.temperature_anode / (np.pi * m))
    S_exit = np.pi * (thruster.r_max**2 - thruster.r_min**2)
    n_est = 4 * thruster.mdot / (m * v_bar * S_exit)
    print(f"anode density: {n_anode:.3e} m^-3, estimate {n_est:.3e} m^-3,"
          f"  ratio {n_anode/n_est:.2f}")


if __name__ == "__main__":
    main()
