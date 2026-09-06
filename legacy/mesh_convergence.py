# Mesh-convergence study for the FEM magnetostatics (roadmap item 3).
#
# `amr_passes=4` in field_on_grid_fem was picked by eye. This example asks
# whether that is enough, and — more importantly — WHICH quantities are
# allowed to be asked the question at all.
#
# The pole-tip corner is a re-entrant corner of the iron: the exact solution
# has |B| ~ rho^(-1/3) there, so the POINTWISE maximum of |B| does not
# converge under refinement, by definition. It grows without bound, and any
# "convergence check" built on it reports failure forever. What does converge
# are integral (and off-singularity) functionals:
#
#   flux_exit   magnetic flux through the exit-plane annulus, 2*pi*d(psi).
#               Comes straight from the nodal potential, no differentiation
#               and no interface averaging -> the best-behaved metric here.
#   Br_int      integral of |B_r| along the mid-channel line over the grid's
#               axial span. The 1-D electron model lives on this profile.
#   Br_peak     peak of that profile, and z_peak its axial position (parabolic
#               fit around the discrete max). Off the singularity, so fair.
#   W_roi       field energy in the plasma-facing vacuum region.
#   Br_max_grid pointwise max of |B_r| on the discharge grid — the CONTROL.
#               Expected NOT to converge; printed to make the point.
#
# Everything is measured on the calibrated field (|B| at the mid-channel exit
# reference point pinned to B_target), because that is the field the plasma
# model actually sees. So the metrics are really shape ratios relative to
# B_ref, which is what we want: a global current rescale must not move them.
#
# Run:
#   python -m legacy.mesh_convergence

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
from scipy.constants import mu_0
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

from src.magnetics import fem
from src.magnetics.spt70_system import (
    CHANNEL_R_IN, CHANNEL_R_OUT, CHANNEL_Z0, CHANNEL_Z_EXIT, DOMAIN_R,
    check_conformal,
)

# reference discharge (see examples/common.spt70) — kept literal so this
# study does not drag in the propellant/wall tables it has no use for
B_TARGET = 0.015
GRID_MAX_Z, GRID_MAX_R, GRID_N_Z, GRID_N_R = 0.06, 0.05, 121, 101

R_MID = 0.5 * (CHANNEL_R_IN + CHANNEL_R_OUT)
N_LINE = 801                       # fixed query points, identical for all meshes


def solve_any(amr_passes, amr_frac=0.05, nr=121, nz=221, B_target=B_TARGET):
    """solve_spt70_fem with a single (points, tris, psi, Br, Bz) signature.

    amr_passes=0 returns the structured form; the base triangulation is
    rebuilt and the fields raveled so the metrics below see one layout.
    """
    t0 = time.perf_counter()
    out = fem.solve_spt70_fem(B_target, nr=nr, nz=nz,
                              amr_passes=amr_passes, amr_frac=amr_frac)
    dt = time.perf_counter() - t0
    if amr_passes == 0:
        r, z, psi, Br, Bz = out
        points, tris = fem.triangulate_structured(r, z)
        return points, tris, psi.ravel(), Br.ravel(), Bz.ravel(), dt
    points, tris, psi, Br, Bz = out
    return points, tris, psi, Br, Bz, dt


def _interp(points, tris, values, rq, zq, mask=None):
    """Piecewise-linear sample of a nodal field at (rq, zq)."""
    triang = Triangulation(points[:, 0], points[:, 1], tris)
    if mask is not None:
        triang.set_mask(mask)
    out = LinearTriInterpolator(triang, values)(np.ravel(rq), np.ravel(zq))
    return np.asarray(out.filled(np.nan))


def metrics(points, tris, psi, Br, Bz):
    """The five convergence metrics plus the non-converging control."""
    mu_r_tri, _ = fem._spt70_element_data(points, tris)
    iron = mu_r_tri > 1.0

    # --- 1) flux through the exit-plane annulus: Phi = 2*pi*(psi_out - psi_in)
    # psi is continuous through iron, so this needs no material mask.
    psi_in, psi_out = _interp(points, tris, psi,
                              [CHANNEL_R_IN, CHANNEL_R_OUT],
                              [CHANNEL_Z_EXIT, CHANNEL_Z_EXIT])
    flux_exit = 2 * np.pi * (psi_out - psi_in)

    # --- 2,3,4) mid-channel |B_r| profile, sampled on the vacuum side only
    z_line = np.linspace(CHANNEL_Z0, CHANNEL_Z0 + GRID_MAX_Z, N_LINE)
    r_line = np.full_like(z_line, R_MID)
    Br_line = np.abs(_interp(points, tris, Br, r_line, z_line, mask=iron))
    Br_int = np.trapezoid(np.nan_to_num(Br_line), z_line)
    k = int(np.nanargmax(Br_line))
    Br_peak = float(Br_line[k])
    if 0 < k < N_LINE - 1:                        # parabolic vertex refinement
        y0, y1, y2 = Br_line[k - 1], Br_line[k], Br_line[k + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        z_peak = z_line[k] + shift * (z_line[1] - z_line[0])
    else:
        z_peak = z_line[k]
    z_peak -= CHANNEL_Z0                          # report in the discharge frame

    # --- 5) field energy in the plasma-facing vacuum region
    Br_e, Bz_e = fem.recover_B_elements(points, tris, psi)
    area, _, r_c = fem._p1_geometry(points, tris)
    z_c = points[tris][:, :, 1].mean(axis=1)
    in_channel = ((z_c >= CHANNEL_Z0) & (z_c < CHANNEL_Z_EXIT)
                  & (r_c >= CHANNEL_R_IN) & (r_c <= CHANNEL_R_OUT))
    in_plume = (z_c >= CHANNEL_Z_EXIT) & (z_c <= CHANNEL_Z0 + GRID_MAX_Z) \
        & (r_c <= GRID_MAX_R)
    roi = (~iron) & (in_channel | in_plume)
    dV = 2 * np.pi * r_c * area
    W_roi = float(np.sum((Br_e[roi] ** 2 + Bz_e[roi] ** 2) / (2 * mu_0) * dV[roi]))

    # --- 6) CONTROL: pointwise max on the discharge grid (should diverge)
    zg = np.linspace(0.0, GRID_MAX_Z, GRID_N_Z) + CHANNEL_Z0
    rg = np.linspace(0.0, GRID_MAX_R, GRID_N_R)
    Zq, Rq = np.meshgrid(zg, rg, indexing="ij")
    Br_grid = _interp(points, tris, Br, Rq, Zq, mask=iron)
    Br_max_grid = float(np.nanmax(np.abs(Br_grid)))

    return dict(flux_exit=float(flux_exit), Br_int=float(Br_int),
                Br_peak=Br_peak, z_peak=float(z_peak), W_roi=W_roi,
                Br_max_grid=Br_max_grid)


def sweep_amr(passes=range(0, 15), amr_frac=0.05):
    rows = []
    for p in passes:
        pts, tris, psi, Br, Bz, dt = solve_any(p, amr_frac=amr_frac)
        m = metrics(pts, tris, psi, Br, Bz)
        m.update(passes=p, n_nodes=len(pts), n_tris=len(tris), t=dt)
        rows.append(m)
        print(f"  pass {p}: {len(pts):6d} nodes  {dt:5.1f} s  "
              f"flux={m['flux_exit']*1e6:8.4f} uWb  "
              f"Br_int={m['Br_int']*1e4*1e3:8.4f} G*mm  "
              f"Br_peak={m['Br_peak']*1e4:7.2f} G  "
              f"z_peak={m['z_peak']*1e3:6.3f} mm  "
              f"W={m['W_roi']*1e6:8.4f} uJ  "
              f"|Br|max={m['Br_max_grid']*1e4:8.1f} G", flush=True)
    return rows


def sweep_base(grids=((121, 221), (241, 441)), amr_passes=4, amr_frac=0.05):
    """Refine the BACKGROUND grid, not just the AMR region.

    Only conformal resolutions are admissible: check_conformal rejects grids
    on which a material edge falls between grid lines, because there the
    geometry itself moves with the mesh (see its docstring). The defaults are
    aligned; 61x111 and 181x331 are NOT (r = 35.5 mm falls between grid
    lines). 361x661 is aligned too but takes ~5 min — pass it explicitly.
    """
    rows = []
    for nr, nz in grids:
        r_n = np.linspace(0.0, DOMAIN_R, nr)
        z_n = np.linspace(0.0, 0.110, nz)
        bad = check_conformal(r_n, z_n)
        if bad:
            print(f"  base {nr}x{nz}: SKIPPED — non-conformal, material edges "
                  f"off-grid: {bad}")
            continue
        pts, tris, psi, Br, Bz, dt = solve_any(amr_passes, amr_frac=amr_frac,
                                               nr=nr, nz=nz)
        m = metrics(pts, tris, psi, Br, Bz)
        m.update(nr=nr, nz=nz, n_nodes=len(pts), n_tris=len(tris), t=dt)
        rows.append(m)
        print(f"  base {nr}x{nz}: {len(pts):6d} nodes  {dt:5.1f} s  "
              f"flux={m['flux_exit']*1e6:8.4f} uWb  "
              f"Br_int={m['Br_int']*1e4*1e3:8.4f} G*mm  "
              f"Br_peak={m['Br_peak']*1e4:7.2f} G  "
              f"z_peak={m['z_peak']*1e3:6.3f} mm  "
              f"W={m['W_roi']*1e6:8.4f} uJ  "
              f"|Br|max={m['Br_max_grid']*1e4:8.1f} G", flush=True)
    return rows


def _rel_drift(rows, key):
    """|x_i - x_last| / |x_last| — self-convergence against the finest mesh."""
    ref = rows[-1][key]
    return np.array([abs(r[key] - ref) / abs(ref) if ref else np.nan
                     for r in rows])


def report(rows):
    """Relative change from one pass to the next: the practical stopping test."""
    keys = ["flux_exit", "Br_int", "Br_peak", "z_peak", "W_roi", "Br_max_grid"]
    print("\n  step-to-step relative change (%)")
    print("  " + "pass".ljust(6) + "".join(k.rjust(13) for k in keys))
    for i in range(1, len(rows)):
        cells = []
        for k in keys:
            prev, cur = rows[i - 1][k], rows[i][k]
            cells.append(f"{abs(cur - prev) / abs(prev) * 100:12.3f}%")
        print("  " + f"{rows[i-1]['passes']}->{rows[i]['passes']}".ljust(6)
              + "".join(cells))


def plot(rows, out="mesh_convergence.png"):
    dof = np.array([r["n_nodes"] for r in rows], float)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    panels = [
        ("flux_exit", 1e6, "exit-plane flux, uWb", False),
        ("Br_int", 1e4 * 1e3, "mid-channel int|B_r|dz, G*mm", False),
        ("Br_peak", 1e4, "mid-channel peak |B_r|, G", False),
        ("z_peak", 1e3, "peak position z, mm", False),
        ("W_roi", 1e6, "field energy in ROI, uJ", False),
        ("Br_max_grid", 1e4, "grid-wide max |B_r|, G  (CONTROL)", True),
    ]
    for ax, (key, scale, label, control) in zip(axes.ravel(), panels):
        y = np.array([r[key] for r in rows]) * scale
        ax.plot(dof, y, "o-", color="crimson" if control else "C0")
        for r, yy in zip(rows, y):
            ax.annotate(str(r["passes"]), (r["n_nodes"], yy),
                        textcoords="offset points", xytext=(4, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("nodes")
        ax.set_title(label, fontsize=10,
                     color="crimson" if control else "black")
        ax.grid(alpha=0.3)
    fig.suptitle("FEM mesh convergence vs AMR passes (labels = amr_passes)")
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")

    # self-convergence, log-log
    fig2, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for key, style in [("flux_exit", "o-"), ("Br_int", "s-"),
                       ("Br_peak", "^-"), ("W_roi", "v-"),
                       ("Br_max_grid", "x--")]:
        d = _rel_drift(rows, key)[:-1]
        ax.loglog(dof[:-1], np.maximum(d, 1e-12), style, label=key)
    ax.set_xlabel("nodes"); ax.set_ylabel("|x - x_finest| / |x_finest|")
    ax.set_title("self-convergence against the finest mesh")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig2.savefig("mesh_convergence_rates.png", dpi=140)
    print("saved mesh_convergence_rates.png")


def main():
    print("AMR sweep (base grid 121x221, amr_frac=0.05):")
    rows = sweep_amr()
    report(rows)
    plot(rows)
    print("\nbase-grid sweep at amr_passes=4:")
    sweep_base()


if __name__ == "__main__":
    main()
