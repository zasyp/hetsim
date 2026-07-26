# Fluid-electron solver package: assembles electron_liquid blocks 1-6 into
# a self-consistent Te / potential solve on a PlasmaState.
#
#   from src.solver import PlasmaState, FluidElectronSolver, SolverSettings
#
#   state = PlasmaState.placeholder(grid, thruster, gas)
#   FluidElectronSolver(state).solve()
#   state.Te, state.phi, state.E_z, state.diagnostics

from .state import PlasmaState
from .geometry import LayerGeometry
from .electron_fluid import FluidElectronSolver, SolverSettings

__all__ = [
    "PlasmaState",
    "LayerGeometry",
    "FluidElectronSolver",
    "SolverSettings",
]
