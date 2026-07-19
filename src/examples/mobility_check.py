# Sanity check for electron_liquid/mobility.py (block 2: mobility and
# diffusion closures) at two fixed reference points -- values taken from
# the collisions.py self-check (see examples/collisions_check.py) so the
# two blocks stay consistent with each other.
#
# For the same quantities computed along the *real* SPT-70 magnetic
# field profile with the calibrated anomalous-transport model, see
# examples/hall_parameter.py instead -- this file only checks the bare
# block-2 formulas at two hand-picked points.
#
# Run either way:
#   python -m src.examples.mobility_check        (from repo root)
#   python src/examples/mobility_check.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from src.electron_liquid.default_plasm_params import omega_ce
from src.electron_liquid.mobility import zeroB_mobility, hall_parameter, perp_mobility, perp_diffusion


def main():
    # (nu_e from the collisions.py self-check, B, Te)
    points = {
        "exit  (B=150G)": (1.72e8, 0.015, 20.0),
        "anode (B=9G)  ": (1.72e7, 9e-4, 3.0),
    }
    for label, (nu, B, Te) in points.items():
        Omega = hall_parameter(omega_ce(B), nu)
        mu = perp_mobility(zeroB_mobility(nu), Omega)
        print(f"{label}: Omega={Omega:6.2f}  mu_perp={mu:7.2f}"
              f"  D_perp={perp_diffusion(mu, Te):7.1f}")
    print("want: exit Omega~15, mu~4.3, D~87; anode mu bigger by ~30x")


if __name__ == "__main__":
    main()
