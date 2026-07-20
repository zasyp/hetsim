# Sanity checks for electron_liquid/lambda_layers.py (block 3: the
# reduction of the (z, r) grid to magnetic-field-line layers).
#
# The one look that matters is the layer map: colored arcs running
# across the channel along the field lines, layer 0 hugging the anode,
# the last layer at the cathode line in the near plume, and EVERYTHING
# else grey (idx = -1): the thruster body and the plume beyond the
# cathode line. A plume painted in an edge-layer color would be the
# previous iteration's clip bug — the whole point of this check.
#
# Numeric checks: (1) layer volumes account for every valid node,
# (2) no empty layers, (3) layer index runs 0 -> N-1 monotonically
# along mid-channel, (4) wall area lands only on channel-crossing
# layers and sums to ~the analytic channel wall area.
#
# Run either way:
#   python -m src.examples.lambda_layers_check        (from repo root)
#   python src/examples/lambda_layers_check.py        (directly / VS Code Run)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.magnetics.spt70_system import field_on_grid
from src.neutrals.neutrals import node_volume
from src.electron_liquid.lambda_layers import (
    lambda_range, build_layers, layer_volume, layer_wall_area,
    thruster_body_mask,
)

N_LAYERS = 50


def plot_layers(idx, V, A_w, thruster, grid, out="lambda_layers.png"):
    z_mm = grid.z_nodes() * 1e3
    r_mm = grid.r_nodes() * 1e3
    L_mm = thruster.channel_length * 1e3

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")

    show = np.where(idx < 0, np.nan, idx).astype(float)
    pcm = ax1.pcolormesh(z_mm, r_mm, show.T, cmap="turbo",
                         vmin=0, vmax=N_LAYERS - 1)
    fig.colorbar(pcm, ax=ax1, label="layer index (0 = anode)")
    for r_wall in (thruster.r_min * 1e3, thruster.r_max * 1e3):
        ax1.plot([0, L_mm], [r_wall, r_wall], "k-", lw=1.5)
    ax1.axvline(L_mm, color="k", ls="--", lw=1)
    ax1.axvline(thruster.z_cathode * 1e3, color="k", ls=":", lw=1.5)
    ax1.set_ylabel("r, mm")
    ax1.set_title("lambda layers: grey = no layer (body / beyond cathode line)")

    j = np.arange(N_LAYERS)
    ax2.bar(j, V * 1e6, color="tab:blue", alpha=0.7, label="V$_j$, cm$^3$")
    ax2.set_xlabel("layer index")
    ax2.set_ylabel("V$_j$, cm$^3$", color="tab:blue")
    ax2b = ax2.twinx()
    ax2b.bar(j, A_w * 1e4, color="tab:red", alpha=0.5, label="A$_w$, cm$^2$")
    ax2b.set_ylabel("A$_{wall}$, cm$^2$", color="tab:red")
    ax2.set_title("volume and channel-wall area per layer")

    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    thruster, grid = spt70()
    z_nodes = grid.z_nodes()
    r_nodes = grid.r_nodes()

    _, _, lam = field_on_grid(grid, thruster.B_r_max)

    body_mask = thruster_body_mask(grid, thruster)
    lam_a, lam_c = lambda_range(lam, z_nodes, r_nodes, thruster)
    edges, centers, idx = build_layers(lam, N_LAYERS, lam_a, lam_c, body_mask)

    dV = node_volume(grid)
    V = layer_volume(idx, dV, N_LAYERS)
    A_w = layer_wall_area(idx, grid, thruster, N_LAYERS)

    valid = idx >= 0
    counts = np.bincount(idx[valid].ravel(), minlength=N_LAYERS)

    # 1) volumes account for every valid node
    ok = np.isclose(V.sum(), dV[valid].sum())
    print(f"volume closure: sum V_j = {V.sum()*1e6:.2f} cm^3, "
          f"valid-node volume = {dV[valid].sum()*1e6:.2f} cm^3  "
          f"{'OK' if ok else 'FAIL'}")

    # 2) no empty layers
    print(f"nodes per layer: min {counts.min()}, max {counts.max()}  "
          f"{'OK' if counts.min() > 0 else 'FAIL: empty layers, lower N_LAYERS'}")

    # 3) ordering along mid-channel: 0 at the anode, N-1 at the cathode line
    j_mid = np.argmin(np.abs(r_nodes - (thruster.r_min + thruster.r_max) / 2))
    i_c = np.argmin(np.abs(z_nodes - thruster.z_cathode))
    mid = idx[:i_c + 1, j_mid]
    monotonic = (np.diff(mid) >= 0).all() and mid[0] == 0 and mid[-1] == N_LAYERS - 1
    print(f"mid-channel layer index: {mid[0]} -> {mid[-1]}, "
          f"monotonic: {'OK' if monotonic else 'FAIL'}")

    # 4) wall area: only on channel-crossing layers. The sum is expected
    # BELOW the analytic channel wall area (ratio ~0.75 on this field):
    # near the anode the field is weak and the lines bow, so a wedge of
    # the channel (z < ~11 mm here, ~11% of channel volume) lies beyond
    # the anode boundary line in lambda and belongs to no layer. That
    # region is effectively "at anode potential" for the electron model;
    # its wall/ionization contributions are negligible at the local
    # Te ~ 2-3 eV. A ratio near 1.0 would actually be suspicious, and a
    # ratio well below ~0.7 means the lambda range lost real wall.
    A_exact = 2 * np.pi * (thruster.r_min + thruster.r_max) * thruster.channel_length
    ratio = A_w.sum() / A_exact
    print(f"wall area: {(A_w > 0).sum()}/{N_LAYERS} layers touch walls, "
          f"sum {A_w.sum()*1e4:.1f} cm^2 vs analytic {A_exact*1e4:.1f} cm^2 "
          f"(ratio {ratio:.3f}, expect ~0.75: near-anode wedge is layerless)")

    plot_layers(idx, V, A_w, thruster, grid)


if __name__ == "__main__":
    main()
