# Hall parameter, cross-field mobility and diffusion along the real
# SPT-70 channel + near-field plume -- the full electron_liquid pipeline
# (collisions.py block 1 -> mobility.py block 2) evaluated on the actual
# B(z) from the magnetic solve (magnetics/spt70_system.py), using the
# calibrated anomalous-transport model of Marks & Jorns (arXiv:2507.08113,
# Eqs. 3-4; SPT-100 preset -- closest literature-calibrated analog for
# this thruster class, not a fit to this exact geometry).
#
# nu_e here is the TOTAL collision frequency nu_en+nu_iz+nu_ei+nu_anom
# (collisions.py: electron_collision), not nu_anom alone. That distinction
# matters a lot right at the bottom of the transport barrier: taking
# Eq. (3) literally there gives an "Omega_anom" in the ~1600-1700 range
# (alpha_anom*(1-beta_anom) is tiny), but electron-neutral collisions
# don't switch off just because the turbulence-driven term does -- they
# set a floor nu_anom alone can't beat. Once that floor is included, the
# peak Omega collapses to ~300-400, in line with the ~100-300 range
# reported for the *effective* Hall parameter elsewhere in the Hall
# thruster literature. See examples/anomalous_transport.py for the bare
# (un-floored) Omega_anom = 1/alpha across background pressures instead.
#
# Te(z)/n_n(z)/n_e(z) below are illustrative placeholders (this project
# has no electron energy or ionization solve yet -- only the neutral-only
# PIC loop in examples/neutral_flow.py), but the floor mechanism itself
# is real and not sensitive to their exact values, only to their rough
# order of magnitude.
#
# Run either way:
#   python -m src.examples.hall_parameter        (from repo root)
#   python src/examples/hall_parameter.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.magnetics.spt70_system import mid_channel_B
from src.electron_liquid.default_plasm_params import omega_ce
from src.electron_liquid.collisions import (
    neutral_collision, ionization_collision, coulomb_collision, electron_collision,
    anomalous_alpha, anomaly_collision, ANOMALY_PROFILE_PRESETS,
)
from src.electron_liquid.mobility import zeroB_mobility, hall_parameter, perp_mobility, perp_diffusion

PRESET_NAME = "SPT-100"


def plot_profiles(z_mm, L_mm, Omega, mu, D, out="hall_parameter.png"):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True, layout="constrained")
    for ax, (y, ylabel, title) in zip(axes, [
        (Omega, "$\\Omega = \\omega_{ce}/\\nu_e$", "Hall parameter"),
        (mu, "$\\mu_\\perp$, m$^2$/(V s)", "Cross-field mobility"),
        (D, "$D_\\perp$, m$^2$/s", "Cross-field diffusion"),
    ]):
        ax.plot(z_mm, y)
        ax.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("z, mm (anode at 0, exit plane dashed)")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    gas = xenon()
    preset = ANOMALY_PROFILE_PRESETS[PRESET_NAME]

    z = grid.z_nodes()
    L = thruster.channel_length
    B_mid = mid_channel_B(grid, thruster)  # real field, not a placeholder

    # placeholders, see module docstring: neutrals depleted from a
    # kinetic anode estimate, Te/n_e ramping toward the barrier/near-field
    n_n = 1e20 * np.exp(-z / (0.4 * L))
    Te_z = np.interp(z, [0.0, L, 2 * L], [3.0, 25.0, 10.0])
    n_e_z = np.interp(z, [0.0, L, 2 * L], [1e15, 2e17, 1e17])
    nu_en = neutral_collision(Te_z, n_n, gas)
    nu_iz = ionization_collision(Te_z, n_n, gas)
    nu_ei = coulomb_collision(n_e_z, Te_z)

    alpha_z = anomalous_alpha(z, L, background_pressure_torr=0.0, **preset)
    nu_anom = anomaly_collision(B_mid, alpha_z)
    nu_tot = electron_collision(nu_en, nu_iz, nu_ei, nu_anom)

    Omega = hall_parameter(omega_ce(B_mid), nu_tot)
    mu = perp_mobility(zeroB_mobility(nu_tot), Omega)
    D = perp_diffusion(mu, Te_z)

    i_exit = int(np.argmin(np.abs(z - L)))
    i_barrier = int(np.argmax(Omega))
    print(f"{PRESET_NAME} preset, P_B=0 (in space):")
    print(f"  barrier at z={z[i_barrier]*1e3:5.1f} mm: Omega={Omega[i_barrier]:6.1f}"
          f"  mu_perp={mu[i_barrier]:6.3f}  D_perp={D[i_barrier]:7.2f}")
    print(f"  exit    at z={z[i_exit]*1e3:5.1f} mm: Omega={Omega[i_exit]:6.1f}"
          f"  mu_perp={mu[i_exit]:6.3f}  D_perp={D[i_exit]:7.2f}")

    plot_profiles(z * 1e3, L * 1e3, Omega, mu, D)


if __name__ == "__main__":
    main()
