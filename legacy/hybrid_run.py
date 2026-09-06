# Checkpoint: the coupled hybrid run — kinetic ions and neutrals driving the
# fluid electrons, and the fluid electrons driving them back.
#
# This is the first example in the project whose plasma density is not
# prescribed. Everything before it (solve_spt70, voltage_sweep, power_budget)
# ran on PlasmaState.placeholder and could only be read for shape.
#
# What it prints and plots:
#   * current traces vs time — ionization, beam, wall and anode currents. This
#     is where the breathing mode shows up, if it shows up.
#   * the axial profiles at the end of the run: n_e, n_n, Te, phi.
#   * the mass balance: injected + seeded == in the domain + through the plume.
#     Nothing else is a sink, because walls and anode hand the atom back.
#
# Run:
#   python -m legacy.hybrid_run            # short demo, ~1 min
#   python -m legacy.hybrid_run 700        # long enough to matter
#
# TIMESCALES — read this before believing any output. The neutral residence
# time is L / v_thermal ~ 125 us, and nothing about the discharge equilibrates
# faster than that. A run of 100 macro-steps covers 48 us: less than half of
# one refill, i.e. pure startup transient. Judging the model there is like
# judging a thruster in its first microseconds after ignition. Several hundred
# microseconds (>= 700 macro-steps) is the minimum for the breathing cycle to
# be visible at all.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.config import spt70
from src.solver import HybridSolver, HybridSettings


def main(macro_steps: int = 120, out: str = "hybrid_run.png"):
    thruster, grid = spt70()
    hybrid = HybridSolver(thruster, grid, HybridSettings(seed_n_i=3e17))

    print(f"dt = {hybrid.dt:.2e} s (ion push), dt_slow = {hybrid.dt_slow:.2e} s "
          f"(everything else)")
    print(f"weights: neutrals {hybrid.s.weight:.2e}, ions {hybrid.ion_weight:.2e} "
          f"real particles per macroparticle")
    print(f"seeded: {len(hybrid.neutrals)} neutral, {len(hybrid.ions)} ion "
          f"macroparticles")
    print(f"running {macro_steps} macro-steps = "
          f"{macro_steps * hybrid.dt_slow * 1e6:.0f} us\n")

    hist = hybrid.run(macro_steps, verbose=True, every=max(1, macro_steps // 15))

    mb = hybrid.mass_balance()
    print(f"\nmass balance [particles]: injected {mb['injected']:.3e} + seeded "
          f"{mb['seeded']:.3e}")
    print(f"                          in domain {mb['in_domain']:.3e} + beam "
          f"{mb['escaped']:.3e}")
    print(f"                          residual {mb['residual']:.3e} "
          f"(rel {mb['rel_error']:.1e})")
    print("The residual is the stochastic injector: inject_on_grid emits a whole\n"
          "number of macroparticles per step and rounds the remainder randomly,\n"
          "so the realised flow random-walks around mdot. It is not a leak.")

    _plot(hybrid, hist, out)
    return hybrid


def _plot(hybrid, hist, out):
    t = np.array([h["time"] for h in hist]) * 1e6
    g = hybrid.grid
    z = g.z_nodes() * 1e3
    jm = int(np.argmin(np.abs(g.r_nodes()
                             - 0.5 * (hybrid.thruster.r_min + hybrid.thruster.r_max))))
    st = hybrid.state

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")

    a = ax[0, 0]
    for key, label in [("I_iz", "ionization"), ("I_beam", "beam"),
                       ("I_wall", "walls"), ("I_anode", "anode")]:
        a.plot(t, [h[key] for h in hist], label=label, lw=1.2)
    a.set_xlabel("t, us"); a.set_ylabel("current, A")
    a.set_title("currents"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(t, [h["n_ion_macro"] for h in hist], label="ions")
    a.plot(t, [h["n_neutral_macro"] for h in hist], label="neutrals")
    a.set_xlabel("t, us"); a.set_ylabel("macroparticles")
    a.set_title("populations"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(t, [h["Te_peak"] for h in hist], color="crimson")
    a.set_xlabel("t, us"); a.set_ylabel("peak Te, eV")
    a.set_title("peak electron temperature"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.semilogy(z, st.n_e[:, jm], label="n_e = n_i")
    a.semilogy(z, st.n_n[:, jm], label="n_n")
    a.axvline(hybrid.thruster.channel_length * 1e3, color="0.5", ls="--", lw=1)
    a.set_xlabel("z, mm"); a.set_ylabel("density, 1/m^3")
    a.set_title("densities, mid-channel (end of run)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(z, st.Te[:, jm], color="crimson")
    a.axvline(hybrid.thruster.channel_length * 1e3, color="0.5", ls="--", lw=1)
    a.set_xlabel("z, mm"); a.set_ylabel("Te, eV")
    a.set_title("electron temperature, mid-channel"); a.grid(alpha=0.3)

    a = ax[1, 2]
    a.plot(z, st.phi[:, jm])
    a.axvline(hybrid.thruster.channel_length * 1e3, color="0.5", ls="--", lw=1)
    a.set_xlabel("z, mm"); a.set_ylabel("phi, V")
    a.set_title("plasma potential, mid-channel"); a.grid(alpha=0.3)

    fig.suptitle(f"hybrid run: {len(hist)} macro-steps = {t[-1]:.0f} us")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    main(steps)
