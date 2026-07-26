# Example: where the electron energy goes. Runs the full solver and shows
# the converged power budget of the electron fluid — the ohmic heating
# that comes in against the wall-sheath, ionization and radiation sinks
# that (with conduction to the boundaries) take it back out — and the
# axial profile of the wall heat flux, which peaks with Te in the
# acceleration layer.
#
# This is the electron side of the discharge power balance of Goebel &
# Katz Eq. 7.3-27; the loss terms are exactly the functions in
# electron_liquid.sheath_interaction / power_balance, assembled per layer.
#
# Run:
#   python -m src.examples.power_budget
#   python src/examples/power_budget.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.solver import PlasmaState, FluidElectronSolver
from src.electron_liquid import sheath_interaction as sheath


def main(out="power_budget.png"):
    thruster, grid = spt70()
    gas = xenon()

    state = PlasmaState.placeholder(grid, thruster, gas)
    solver = FluidElectronSolver(state)
    state = solver.solve()
    d = state.diagnostics

    # re-evaluate the wall heat-flux field at the converged Te for a profile
    Te = state.Te
    gamma = sheath.see_yield(Te, thruster.wall_material)
    phi_s = sheath.sheath_potential(Te, gamma, gas)
    q_wall = sheath.wall_energy_loss(Te, 0.5 * state.n_e, phi_s)  # W/m^2

    labels = ["ohmic in", "wall", "ionization", "radiation",
              "conduction to\nboundaries"]
    ohmic = d["P_ohmic"]
    sinks = [d["W_wall"], d["W_ion"], d["W_rad"]]
    to_bnd = ohmic - sum(sinks)          # remainder balanced by conduction
    values = [ohmic, d["W_wall"], d["W_ion"], d["W_rad"], to_bnd]

    print("electron power budget [W]:")
    for lab, v in zip(labels, values):
        print(f"  {lab.replace(chr(10),' '):24s} {v:8.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green", "0.6"]
    ax1.bar(labels, values, color=colors)
    ax1.set_ylabel("power, W")
    ax1.set_title("converged electron power budget")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # wall heat-flux profile along the outer channel wall
    z = grid.z_nodes() * 1e3
    L = thruster.channel_length * 1e3
    j_out = np.argmin(np.abs(grid.r_nodes() - thruster.r_max))
    in_channel = grid.z_nodes() <= thruster.channel_length
    ax2.plot(z[in_channel], q_wall[in_channel, j_out] * 1e-4, color="tab:blue")
    ax2.axvline(L, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax2.set_xlabel("z, mm (anode at 0)")
    ax2.set_ylabel("wall heat flux, W/cm$^2$")
    ax2.set_title("outer-wall heat flux")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
