# Example: how the peak electron temperature scales with discharge
# voltage. Runs the full solver at a series of V_d and plots Te_peak and
# the Te_peak / V_d ratio — the latter should land near the ~0.1
# rule-of-thumb Hall thrusters obey across a wide voltage range (Goebel &
# Katz, Ch. 7.3.4).
#
# Run:
#   python -m src.examples.voltage_sweep
#   python src/examples/voltage_sweep.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.solver import PlasmaState, FluidElectronSolver

VOLTAGES = [200, 250, 300, 350, 400]


def main(out="voltage_sweep.png"):
    thruster, grid = spt70()
    gas = xenon()

    Te_peak = []
    for V in VOLTAGES:
        thruster.voltage = V
        state = PlasmaState.placeholder(grid, thruster, gas)
        state = FluidElectronSolver(state).solve()
        Te_peak.append(state.diagnostics["Te_peak"])
        print(f"V_d = {V:3d} V  ->  Te_peak = {Te_peak[-1]:5.1f} eV  "
              f"({Te_peak[-1]/V:.3f} V_d)")

    Te_peak = np.array(Te_peak)
    V = np.array(VOLTAGES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    ax1.plot(V, Te_peak, "o-", color="tab:red")
    ax1.set_xlabel("discharge voltage $V_d$, V")
    ax1.set_ylabel("peak $T_e$, eV")
    ax1.set_title("peak electron temperature vs voltage")
    ax1.grid(alpha=0.3)

    ax2.plot(V, Te_peak / V, "s-", color="tab:purple")
    ax2.axhline(0.1, color="k", ls="--", lw=1, alpha=0.6, label="0.1 rule of thumb")
    ax2.set_xlabel("discharge voltage $V_d$, V")
    ax2.set_ylabel("$T_e^{peak} / V_d$")
    ax2.set_title("temperature-to-voltage ratio")
    ax2.set_ylim(0, 0.15)
    ax2.grid(alpha=0.3); ax2.legend()

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
