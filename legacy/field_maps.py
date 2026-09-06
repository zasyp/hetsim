# Field maps of a PLACEHOLDER discharge — the pre-hybrid way of looking at
# the 2D solution.
#
# The drawing itself moved into the package (src/viz/fields.py) once the
# SPT-70 example started needing it; what is left here is the old standalone
# entry point, which solves the electron fluid on PlasmaState.placeholder —
# prescribed n_e and n_n — and dumps the maps. Superseded by
# `python -m src.examples.spt70`, whose densities come from the particles.
#
# Run:
#   python -m legacy.field_maps

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.config import spt70
from src.solver import FluidElectronSolver, PlasmaState
from src.structs.propellants import xenon
from src.viz import save_field_maps


def main(out_dir: str = "maps"):
    thruster, grid = spt70()
    state = PlasmaState.placeholder(grid, thruster, xenon())
    state = FluidElectronSolver(state).solve(verbose=True)
    save_field_maps(state, out_dir)


if __name__ == "__main__":
    main()
