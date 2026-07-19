# Sanity checks for electron_liquid/collisions.py (block 1: collision
# frequencies). Two independent, geometry-free checks:
#   1. a single reference point (values from a HallThruster.jl-like exit
#      condition) against known-good order-of-magnitude numbers;
#   2. the shape of the Eqs. (3)-(4) anomalous-transport profile
#      (Marks & Jorns, arXiv:2507.08113) against its own definition,
#      independent of any specific thruster's B(z).
#
# For the calibrated profile plotted against a *real* magnetic field
# profile (SPT-70 geometry), see examples/anomalous_transport.py and
# examples/hall_parameter.py instead -- this file only checks the bare
# formulas.
#
# Run either way:
#   python -m src.examples.collisions_check        (from repo root)
#   python src/examples/collisions_check.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from src.structs.propellants import xenon
from src.electron_liquid.collisions import (
    neutral_collision, ionization_collision, coulomb_logarithm, coulomb_collision,
    anomaly_collision, anomalous_transport_profile, electron_collision,
    ANOMALY_PROFILE_PRESETS,
)


def main():
    gas = xenon()
    n_n, n_e, Te, B = 1e19, 2e17, 20.0, 0.015

    nu_en = neutral_collision(Te, n_n, gas)
    nu_iz = ionization_collision(Te, n_n, gas)
    nu_ei = coulomb_collision(n_e, Te)
    nu_anom = anomaly_collision(B)
    print(f"lnL     = {coulomb_logarithm(Te, n_e):.1f}   (want ~14)")
    print(f"nu_en   = {nu_en:.2e}  (want ~7e6)")
    print(f"nu_iz   = {nu_iz:.2e}  (want ~6e5)")
    print(f"nu_ei   = {nu_ei:.2e}  (want ~1e5)")
    print(f"nu_anom = {nu_anom:.2e}  (want ~1.6e8, constant alpha=1/16)")
    print(f"total   = {electron_collision(nu_en, nu_iz, nu_ei, nu_anom):.2e}")

    # axial profile check: nu_anom should sit near the alpha_anom ceiling
    # at the anode (z_hat=0) and drop by ~(1-beta_anom) in the trough
    # around z_anom, per the calibrated presets in collisions.py.
    print("\naxial profile shape (GENERIC_300V preset, no facility pressure):")
    z_hat = np.array([0.0, 0.5, 1.0, 1.05, 1.5, 2.0])
    preset = {k: v for k, v in ANOMALY_PROFILE_PRESETS["GENERIC_300V"].items()
              if k != "delta_z_anom"}
    alpha_z = anomalous_transport_profile(z_hat, **preset)
    nu_anom_z = anomaly_collision(B, alpha_z)
    for zh, a, nu in zip(z_hat, alpha_z, nu_anom_z):
        print(f"  z_hat={zh:4.2f}  alpha={a:.4f}  nu_anom={nu:.2e}")


if __name__ == "__main__":
    main()
