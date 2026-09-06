# Where does AMR actually spend its 5%-per-pass budget?
#
# Reimplements the solve_spt70_fem loop but records the (r,z) centroids of
# the marked elements each pass, so we can see whether refinement stays
# pinned to the pole-tip corner or spreads to other ROI hotspots.
#
# Run:
#   python -m src.examples.amr_marked_positions

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from src.magnetics import fem
from src.magnetics.spt70_system import (
    build_spt70_system, CHANNEL_Z0, CHANNEL_Z_EXIT, CHANNEL_R_IN, CHANNEL_R_OUT,
)

B_TARGET = 0.015
NR, NZ, AMR_PASSES, AMR_FRAC = 121, 221, 8, 0.05


def run():
    r, z, _, _ = build_spt70_system(NR, NZ)
    points, tris = fem.triangulate_structured(r, z)
    marks = []  # (pass, r_c, z_c) for every marked element

    for p in range(AMR_PASSES + 1):
        mu_r_tri, Jphi_tri = fem._spt70_element_data(points, tris)
        dirichlet = fem._spt70_dirichlet(points, r, z)
        psi = fem.assemble(points, tris, mu_r_tri, Jphi_tri, dirichlet)
        if p == AMR_PASSES:
            break
        err = fem.zz_error(points, tris, psi)
        # the library's own marking rule, not a copy of it — a second
        # implementation here would quietly drift from what the solver does
        marked = fem.mark_for_refinement(points, tris, mu_r_tri, err, AMR_FRAC)
        if marked.size == 0:
            break
        r_c = points[tris][:, :, 0].mean(axis=1)
        z_c = points[tris][:, :, 1].mean(axis=1)
        marks.append((p, r_c[marked].copy(), z_c[marked].copy()))
        points, tris = fem.refine_mesh(points, tris, marked)

    return marks


def plot(marks, out="amr_marked_positions.png"):
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    cmap = plt.get_cmap("viridis", len(marks))
    for p, r_c, z_c in marks:
        ax.scatter((z_c - CHANNEL_Z0) * 1e3, r_c * 1e3, s=6,
                   color=cmap(p), label=f"pass {p}", alpha=0.7)
    ax.axhline(CHANNEL_R_IN * 1e3, color="k", lw=0.5, ls="--")
    ax.axhline(CHANNEL_R_OUT * 1e3, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("z - CHANNEL_Z0, mm")
    ax.set_ylabel("r, mm")
    ax.set_title("marked-element centroids per AMR pass")
    ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    marks = run()
    for p, r_c, z_c in marks:
        z_mm = (z_c - CHANNEL_Z0) * 1e3
        print(f"pass {p}: {len(r_c):4d} marked  "
              f"z range [{z_mm.min():6.3f}, {z_mm.max():6.3f}] mm  "
              f"r range [{r_c.min()*1e3:6.3f}, {r_c.max()*1e3:6.3f}] mm")
    plot(marks)
