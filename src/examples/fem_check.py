# Checkpoint: nodal-P1 FEM magnetostatics vs the legacy finite-difference
# solver. Validates that the FEM field (the new default in field_on_grid)
# reproduces the bulk field, removes the iron-interface leak, and resolves
# the pole-tip corner instead of smearing it.
#
# Checks (printed):
#   1) mid-channel B_r(z) agrees with FD to ~1% (physics unchanged),
#   2) lambda (streamfunction) correlates with the FD one (layers unchanged),
#   3) the global |B_r| max on the grid drops sharply (leak/corner tamed).
# Also saves fem_field.png: FD vs FEM B_r maps + the AMR mesh at a pole tip.
# Run:
#   python -m src.examples.fem_check

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.examples.common import spt70
from src.magnetics.spt70_system import field_on_grid
from src.magnetics import fem


def main(out="fem_field.png"):
    thruster, grid = spt70()
    B0 = thruster.B_r_max
    r_n, z_n = grid.r_nodes(), grid.z_nodes()

    Br_fd, Bz_fd, lam_fd = field_on_grid(grid, B0, method="fd")
    Br_fe, Bz_fe, lam_fe = field_on_grid(grid, B0, method="fem")

    # 1) mid-channel profile
    jm = np.argmin(np.abs(r_n - 0.5 * (thruster.r_min + thruster.r_max)))
    mid_fd = np.abs(Br_fd[:, jm]) * 1e4
    mid_fe = np.abs(Br_fe[:, jm]) * 1e4
    peak_err = abs(mid_fe.max() - mid_fd.max()) / mid_fd.max()
    print(f"1) mid-channel |Br| peak: FD {mid_fd.max():.0f} G, "
          f"FEM {mid_fe.max():.0f} G  (diff {peak_err*100:.1f}%)")

    # 2) lambda correlation over the channel annulus
    ch = ((z_n[:, None] < thruster.channel_length)
          & (r_n[None, :] >= thruster.r_min) & (r_n[None, :] <= thruster.r_max))
    corr = np.corrcoef(lam_fd[ch], lam_fe[ch])[0, 1]
    print(f"2) lambda FD-vs-FEM correlation over channel: {corr:.4f}")

    # 3) global max (the leak/corner indicator)
    print(f"3) global |Br| max on grid: FD {np.abs(Br_fd).max()*1e4:.0f} G, "
          f"FEM {np.abs(Br_fe).max()*1e4:.0f} G")

    # --- figure: FD vs FEM B_r maps + AMR mesh near the inner pole tip ---
    body = (z_n[:, None] < thruster.channel_length) & (
        (r_n[None, :] < thruster.r_min) | (r_n[None, :] > thruster.r_max))
    z_mm, r_mm = z_n * 1e3, r_n * 1e3
    L_mm = thruster.channel_length * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), layout="constrained")
    for ax, dat, ttl in [(axes[0], Br_fd, "FD (one-sided) B_r"),
                         (axes[1], Br_fe, "FEM P1 + AMR B_r")]:
        d = np.where(body, np.nan, np.abs(dat) * 1e4)
        vmax = np.nanpercentile(d, 99)
        pcm = ax.pcolormesh(z_mm, r_mm, d.T, cmap="turbo",
                            shading="gouraud", vmin=0, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label="B_r, G", extend="max")
        for rw in (thruster.r_min * 1e3, thruster.r_max * 1e3):
            ax.plot([0, L_mm], [rw, rw], "w-", lw=1.5)
        ax.axvline(L_mm, color="w", ls="--", lw=1)
        ax.set_ylabel("r, mm")
        ax.set_title(f"{ttl}   (grid max {np.nanmax(d):.0f} G)")

    # AMR mesh, zoomed on the inner pole tip (r~17.5, z~30 mm discharge)
    pts, tris, _, _, _ = fem.solve_spt70_fem(B0, amr_passes=4, amr_frac=0.05)
    from src.magnetics.spt70_system import CHANNEL_Z0
    ax = axes[2]
    ax.triplot((pts[:, 1] - CHANNEL_Z0) * 1e3, pts[:, 0] * 1e3, tris,
               lw=0.3, color="0.4")
    ax.set_xlim(20, 34); ax.set_ylim(14, 24)
    ax.set_xlabel("z, mm (anode at 0)"); ax.set_ylabel("r, mm")
    ax.set_title("AMR mesh at the inner pole tip (refined toward the corner)")

    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
