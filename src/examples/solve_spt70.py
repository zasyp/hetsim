# Headline example: the full self-consistent electron solve on the SPT-70.
# Assembles blocks 1-6 through the solver package and plots the electron
# temperature, plasma potential and accelerating field along the channel.
#
# The physical result to look for: Te is cold at the anode, climbs to a
# peak of ~25 eV (about 0.08 * V_d, the usual Hall-thruster rule of thumb)
# in the acceleration layer near the exit plane, then cools into the
# plume; the potential holds near the anode value through the channel and
# drops steeply across that same layer, so E_z peaks where Te does.
#
# Densities here are the placeholder profiles of PlasmaState.placeholder
# (stand-ins until PIC ions are pushed), so treat the absolute Te scale as
# calibratable (SolverSettings.kappa_coeff) and read the SHAPE as the
# result.
#
# Run:
#   python -m src.examples.solve_spt70
#   python src/examples/solve_spt70.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.solver import PlasmaState, FluidElectronSolver
from src.examples.field_maps import save_field_maps


def main(out="electron_solution.png", maps_dir="maps"):
    thruster, grid = spt70()
    gas = xenon()

    state = PlasmaState.placeholder(grid, thruster, gas)
    solver = FluidElectronSolver(state)
    state = solver.solve(verbose=True)
    d = state.diagnostics

    print(f"\nconverged={d['converged']} in {d['iterations']} iters "
          f"(residual {d['residual_eV']:.1e} eV)")
    print(f"peak Te = {d['Te_peak']:.1f} eV  (~{d['Te_peak']/thruster.voltage:.2f} V_d)")
    print(f"power budget [W]: ohmic in={d['P_ohmic']:.0f}  "
          f"wall={d['W_wall']:.0f}  ioniz={d['W_ion']:.0f}  rad={d['W_rad']:.0f}")

    z = grid.z_nodes() * 1e3
    L = thruster.channel_length * 1e3
    j_mid = np.argmin(np.abs(grid.r_nodes() - (thruster.r_min + thruster.r_max) / 2))
    z_layer = d["z_layer"] * 1e3
    order = np.argsort(z_layer)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                                        layout="constrained")

    ax1.plot(z_layer[order], d["Te_layers"][order], "o-", ms=3, color="tab:red")
    ax1.axvline(L, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax1.set_ylabel("$T_e$, eV")
    ax1.set_title("self-consistent electron temperature")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(z, state.phi[:, j_mid], color="tab:blue")
    ax2.axvline(L, color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_ylabel("$\\varphi$ mid-channel, V")
    ax2.set_title("plasma potential")
    ax2.grid(alpha=0.3)

    ax3.plot(z, state.E_z[:, j_mid] * 1e-3, color="tab:green")
    ax3.axvline(L, color="k", ls="--", lw=1, alpha=0.6)
    ax3.set_xlabel("z, mm (anode at 0)")
    ax3.set_ylabel("$E_z$ mid-channel, kV/m")
    ax3.set_title("accelerating field")
    ax3.grid(alpha=0.3)

    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    # full 2D solution: one colour map per field into maps_dir/
    save_field_maps(state, maps_dir)


if __name__ == "__main__":
    main()
