# Sanity check for electron_liquid/potential.py block 4a: the cross-field
# layer conductances G_{j+1/2} (node_weight -> layer_conductance),
# evaluated on the real SPT-70 field with the full block 1->2->3 pipeline.
#
# The payoff to look for: G_face is MINIMAL where the perpendicular
# mobility is in its trough (near the exit plane / anomalous barrier).
# Low conductance = high resistance = that is where the potential drop
# will concentrate in the full solve -- the physical signature of a Hall
# thruster. If G_face's minimum is not near the exit, the bug is upstream
# (mobility or the field), not in the conductance assembly.
#
# Te / n_e / n_n are prescribed placeholder profiles (no energy or
# ionization solve yet, exactly as in examples/hall_parameter.py); only
# their rough magnitude and z-shape matter here.
#
# Run either way:
#   python -m src.examples.conductance_check        (from repo root)
#   python src/examples/conductance_check.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.structs.propellants import xenon
from src.magnetics.spt70_system import field_on_grid
from src.neutrals.neutrals import node_volume
from src.electron_liquid.default_plasm_params import omega_ce
from src.electron_liquid.collisions import (
    neutral_collision, ionization_collision, coulomb_collision,
    anomaly_collision, anomalous_alpha, electron_collision,
    ANOMALY_PROFILE_PRESETS,
)
from src.electron_liquid.mobility import zeroB_mobility, hall_parameter, perp_mobility
from src.electron_liquid.lambda_layers import lambda_range, build_layers, thruster_body_mask
from src.electron_liquid.layer_potential import node_weight, layer_conductance

N_LAYERS = 40
PRESET_NAME = "SPT-100"


def plot_conductance(z_face_mm, G_face, mu_mid, z_mm, L_mm, out="conductance.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")

    ax1.plot(z_mm, mu_mid)
    ax1.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6, label="exit plane")
    ax1.set_ylabel("$\\mu_\\perp$ mid-channel, m$^2$/(V s)")
    ax1.set_yscale("log")
    ax1.set_title("cross-field mobility (block 2) and layer conductance (block 4a)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(z_face_mm, G_face, "o-", ms=3)
    ax2.axvline(L_mm, color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("z, mm (anode at 0; each face placed at its mid-channel z)")
    ax2.set_ylabel("$G_{j+1/2}$, A/V")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    gas = xenon()
    z = grid.z_nodes()
    r = grid.r_nodes()
    L = thruster.channel_length
    preset = ANOMALY_PROFILE_PRESETS[PRESET_NAME]

    # real magnetic field on the grid
    Br, Bz, lam = field_on_grid(grid, thruster.B_r_max)
    B = np.hypot(Br, Bz)

    # placeholder plasma state (z-profiles broadcast over r), see docstring
    Te = np.interp(z, [0.0, L, 2 * L], [3.0, 25.0, 10.0])[:, None]
    n_n = (1e20 * np.exp(-z / (0.4 * L)))[:, None]
    n_e = np.interp(z, [0.0, L, 2 * L], [1e16, 2e17, 5e16])[:, None]
    alpha = anomalous_alpha(z, L, background_pressure_torr=0.0, **preset)[:, None]

    # block 1 -> 2: total collision frequency and cross-field mobility
    nu = electron_collision(
        neutral_collision(Te, n_n, gas),
        ionization_collision(Te, n_n, gas),
        coulomb_collision(n_e, Te),
        anomaly_collision(B, alpha),
    )
    mu = perp_mobility(zeroB_mobility(nu), hall_parameter(omega_ce(B), nu))

    # block 3: layers
    body = thruster_body_mask(grid, thruster)
    lam_a, lam_c = lambda_range(lam, z, r, thruster)
    edges, centers, idx = build_layers(lam, N_LAYERS, lam_a, lam_c, body)
    d_lambda = edges[1] - edges[0]

    # block 4a: node weights -> face conductances
    dV = node_volume(grid)
    w = node_weight(mu, np.broadcast_to(n_e, B.shape), Br, Bz, r, dV)
    G_face = layer_conductance(idx, N_LAYERS, d_lambda, w)

    # place each layer at its mean mid-channel z, for a spatial x-axis
    j_mid = np.argmin(np.abs(r - (thruster.r_min + thruster.r_max) / 2))
    idx_mid = idx[:, j_mid]
    z_layer = np.array([z[idx_mid == l].mean() if np.any(idx_mid == l) else np.nan
                        for l in range(N_LAYERS)])
    z_face = 0.5 * (z_layer[:-1] + z_layer[1:])

    # --- diagnostics ---
    mu_mid = mu[:, j_mid]
    in_range = z <= thruster.z_cathode
    i_mu_min = np.argmin(np.where(in_range, mu_mid, np.inf))
    k_G_min = int(np.nanargmin(np.where(G_face > 0, G_face, np.nan)))

    nonzero = G_face > 0
    print(f"mobility trough (mid-channel): z = {z[i_mu_min]*1e3:.1f} mm, "
          f"mu_perp = {mu_mid[i_mu_min]:.3f}  (exit at {L*1e3:.0f} mm)")
    print(f"G_face minimum:                z = {z_face[k_G_min]*1e3:.1f} mm, "
          f"G = {G_face[k_G_min]:.2e} A/V")
    print(f"G_face range: {G_face[nonzero].min():.2e} .. {G_face[nonzero].max():.2e} A/V, "
          f"{nonzero.sum()}/{N_LAYERS-1} faces conducting")

    # order-of-magnitude series check (placeholder fields!): the anode->
    # cathode chain is a series of the G_face, so the through conductance
    # is the harmonic sum; times the discharge voltage gives a rough
    # electron current scale (NOT a calibrated I_d -- Te/n_e are guessed)
    G_series = 1.0 / np.sum(1.0 / G_face[nonzero])
    print(f"series conductance ~ {G_series:.2e} A/V  ->  "
          f"I_e ~ {thruster.voltage * G_series:.2f} A at {thruster.voltage} V "
          f"(order-of-magnitude, placeholder plasma)")

    plot_conductance(z_face * 1e3, G_face, mu_mid, z * 1e3, L * 1e3)


if __name__ == "__main__":
    main()
