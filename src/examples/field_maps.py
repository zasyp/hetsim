# Dump every 2D field of a solved PlasmaState as its own colour map into a
# folder. The example scripts otherwise only show 1D midline slices; this
# lets you see the full (N_z, N_r) solution — inputs (B, lambda, n_e, n_n)
# and electron outputs (Te, phi, E_z, E_r, |E|) — one PNG per field.
#
# Run standalone (solves the SPT-70 first, then writes the maps):
#   python -m src.examples.field_maps
# Or from another script:  save_field_maps(state, "maps")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.solver.state import PlasmaState


# (attribute, filename, title, colorbar label, unit scale, cmap, signed)
# `signed` fields get a symmetric diverging scale centred on zero.
_FIELD_SPECS = [
    ("B",   "B_magnitude", "Magnetic field |B|",        "|B|, G",      1e4, "turbo",   False),
    ("Br",  "B_r",         "Radial magnetic field B_r", "B_r, G",      1e4, "turbo",   False),
    ("Bz",  "B_z",         "Axial magnetic field B_z",  "B_z, G",      1e4, "turbo",   False),
    ("lam", "lambda",      "Field-line label lambda",   "lambda",      1.0, "viridis", False),
    ("n_e", "n_e",         "Electron density n_e",      "n_e, m$^{-3}$", 1.0, "viridis", False),
    ("n_n", "n_n",         "Neutral density n_n",       "n_n, m$^{-3}$", 1.0, "viridis", False),
    ("Te",  "Te",          "Electron temperature T_e",  "T_e, eV",     1.0, "inferno", False),
    ("phi", "phi",         "Plasma potential phi",      "phi, V",      1.0, "viridis", False),
    ("E_z", "E_z",         "Axial electric field E_z",  "E_z, kV/m",   1e-3, "RdBu_r",  True),
    ("E_r", "E_r",         "Radial electric field E_r", "E_r, kV/m",   1e-3, "RdBu_r",  True),
]


def _plot_field(field, thruster, grid, title, label, scale, cmap, signed, out):
    """Write a single field as a masked pcolormesh map with channel walls."""
    z_mm = grid.z_nodes() * 1e3
    r_mm = grid.r_nodes() * 1e3
    L_mm = thruster.channel_length * 1e3

    # blank out the solid thruster body (z < L outside the channel annulus)
    Z, R = np.meshgrid(grid.z_nodes(), grid.r_nodes(), indexing="ij")
    body = (Z < thruster.channel_length) & (
        (R < thruster.r_min) | (R > thruster.r_max)
    )
    data = np.where(body, np.nan, field) * scale

    # Robust colour limits: the flat-face interface leak is fixed at the
    # source (field_from_flux one-sided gradients), but the pole-tip corner
    # singularity still leaves a few near-exit wall nodes with an unbounded,
    # mesh-dependent field that would otherwise saturate the scale. Clip to
    # the 1st/99th percentile of the finite data, as magnetic_field does for
    # its |B| map, and flag the clipping on the colourbar.
    finite = data[np.isfinite(data)]
    if signed:
        vmax = np.percentile(np.abs(finite), 99)
        vmin, extend = -vmax, "both"
    else:
        vmin, vmax = np.percentile(finite, [1, 99])
        extend = "max"

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    pcm = ax.pcolormesh(z_mm, r_mm, data.T, cmap=cmap, shading="gouraud",
                        vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=label, extend=extend)
    for r_wall in (thruster.r_min * 1e3, thruster.r_max * 1e3):
        ax.plot([0, L_mm], [r_wall, r_wall], "w-", lw=1.5)
    ax.axvline(L_mm, color="w", ls="--", lw=1)
    ax.set_xlabel("z, mm (anode at 0, exit plane dashed)")
    ax.set_ylabel("r, mm")
    ax.set_title(title)

    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_field_maps(state: PlasmaState, out_dir: str | Path = "maps") -> list[Path]:
    """Write one 2D colour map per available field of `state` into `out_dir`.

    Input fields (B, lambda, n_e, n_n) are always present; the electron
    outputs (Te, phi, E_z, E_r) are written only after a solve has filled
    them. Returns the list of PNG paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for attr, fname, title, label, scale, cmap, signed in _FIELD_SPECS:
        field = getattr(state, attr, None)
        if field is None:  # unsolved output — skip
            continue
        out = out_dir / f"{fname}.png"
        _plot_field(field, state.thruster, state.grid,
                    title, label, scale, cmap, signed, out)
        written.append(out)

    # |E| magnitude, if both components solved
    if state.E_z is not None and state.E_r is not None:
        out = out_dir / "E_magnitude.png"
        _plot_field(np.hypot(state.E_z, state.E_r), state.thruster, state.grid,
                    "Electric field |E|", "|E|, kV/m", 1e-3, "magma", False, out)
        written.append(out)

    print(f"saved {len(written)} field maps to {out_dir}/")
    return written


def main(out_dir: str = "maps"):
    from src.examples.common import spt70
    from src.structs.propellants import xenon
    from src.solver import FluidElectronSolver

    thruster, grid = spt70()
    state = PlasmaState.placeholder(grid, thruster, xenon())
    state = FluidElectronSolver(state).solve(verbose=True)
    save_field_maps(state, out_dir)


if __name__ == "__main__":
    main()
