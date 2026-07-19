# Anomalous electron transport along the SPT-70 channel + plume, using
# the calibrated 4+2-parameter model of Marks & Jorns, "Uncertainty
# quantification of a multi-component Hall thruster model at varying
# facility pressures" (arXiv:2507.08113, 2025), Eqs. (3)-(4):
#
#   alpha(z) = alpha_anom * (1 - beta_anom * exp(-((z_hat-z_anom)/L_anom)^2))
#   z_hat = (z + Delta_z(P_B)) / channel_length
#
# nu_anom = alpha(z) * omega_ce(B(z)) is the anomalous collision
# frequency and Omega_anom = 1/alpha(z) the anomalous (inverse) Hall
# parameter. We use the SPT-100 preset (closest literature-calibrated
# analog for an unshielded, 300 V thruster of this class -- NOT a fit to
# this exact SPT-70 geometry) and sweep background pressure to reproduce
# the paper's qualitative result (Fig. 17): the low-mobility barrier
# moves upstream (toward the anode) as facility pressure rises.
#
# Run either way:
#   python -m src.examples.anomalous_transport        (from repo root)
#   python src/examples/anomalous_transport.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.magnetics.spt70_system import mid_channel_B
from src.electron_liquid.default_plasm_params import omega_ce
from src.electron_liquid.collisions import (
    anomalous_alpha, anomaly_collision, anomaly_axial_shift,
    ANOMALY_PROFILE_PRESETS, P0_DEFAULT,
)
from src.electron_liquid.mobility import hall_parameter

# background pressures to compare, Torr-Xe (0 = space / no facility effect)
PRESSURES_TORR = (0.0, 20e-6, 60e-6)
PRESET_NAME = "SPT-100"


def plot_profiles(z_mm, L_mm, nu_by_P, Omega_by_P, out="anomalous_transport.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")

    for P_B, nu in nu_by_P.items():
        ax1.plot(z_mm, nu, label=f"P$_B$ = {P_B*1e6:.0f} µTorr")
    ax1.axhline(1.0, color="k", ls=":", lw=1, label="Bohm ref. ($\\alpha$=1/16)")
    ax1.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel("$\\nu_{anom}$ / $\\nu_{Bohm}$")
    ax1.set_yscale("log")
    ax1.set_title(f"Anomalous collision frequency, {PRESET_NAME} preset")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for P_B, Omega in Omega_by_P.items():
        ax2.plot(z_mm, Omega, label=f"P$_B$ = {P_B*1e6:.0f} µTorr")
    ax2.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax2.set_xlabel("z, mm (anode at 0)")
    ax2.set_ylabel("$\\Omega_{anom} = 1/\\alpha$")
    ax2.set_yscale("log")
    ax2.set_title("Anomalous Hall parameter")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    L = thruster.channel_length
    preset = ANOMALY_PROFILE_PRESETS[PRESET_NAME]

    z_nodes = grid.z_nodes()
    B_mid = mid_channel_B(grid, thruster)

    nu_bohm_anode = (1 / 16) * omega_ce(thruster.B_r_max)  # normalization reference

    nu_by_P, Omega_by_P = {}, {}
    print(f"{PRESET_NAME} preset: {preset}")
    print(f"P0 = {P0_DEFAULT*1e6:.1f} uTorr, channel length = {L*1e3:.0f} mm\n")

    for P_B in PRESSURES_TORR:
        alpha_z = anomalous_alpha(z_nodes, L, background_pressure_torr=P_B, **preset)
        nu_anom = anomaly_collision(B_mid, alpha_z)
        Omega = hall_parameter(omega_ce(B_mid), nu_anom)

        # consistency check: Omega should equal 1/alpha exactly, since
        # nu_anom = alpha*omega_ce by construction (mobility.py and
        # collisions.py must agree on this definition)
        assert np.allclose(Omega, 1 / alpha_z, rtol=1e-10), "Omega != 1/alpha -- definition mismatch"

        shift = anomaly_axial_shift(P_B, L, preset["delta_z_anom"])
        i_min = int(np.argmin(alpha_z))
        print(f"P_B={P_B*1e6:5.1f} uTorr: shift={shift*1e3:+5.2f} mm, "
              f"barrier at z={z_nodes[i_min]*1e3:5.1f} mm, "
              f"Omega_anom there={Omega[i_min]:7.1f}, "
              f"nu_anom(anode)={nu_anom[0]:.2e} (nu_Bohm={nu_bohm_anode:.2e})")

        nu_by_P[P_B] = nu_anom / nu_bohm_anode
        Omega_by_P[P_B] = Omega

    plot_profiles(z_nodes * 1e3, L * 1e3, nu_by_P, Omega_by_P)


if __name__ == "__main__":
    main()
