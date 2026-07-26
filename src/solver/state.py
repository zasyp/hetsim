# Plasma state — the field container the electron-fluid solver reads and
# writes. Holds the fixed inputs (magnetic field + heavy-species densities)
# and, after a solve, the electron results (Te, potential, E-field).
#
# Until the PIC ion / ion-continuity side is wired in, n_e and n_n are
# supplied externally; `placeholder` builds physically-scaled stand-in
# profiles so the electron solver can be exercised end-to-end today. Swap
# `placeholder` for real deposited densities once ions are pushed.

from dataclasses import dataclass, field

import numpy as np

from ..structs.classes import Grid2D, Thruster, WorkingSubstance
from ..magnetics.spt70_system import field_on_grid


@dataclass
class PlasmaState:
    grid: Grid2D
    thruster: Thruster
    gas: WorkingSubstance

    # inputs (on the (N_z, N_r) grid)
    Br: np.ndarray
    Bz: np.ndarray
    lam: np.ndarray
    n_e: np.ndarray
    n_n: np.ndarray

    # electron-solver outputs (filled by FluidElectronSolver.solve)
    Te: np.ndarray | None = None
    phi: np.ndarray | None = None
    E_z: np.ndarray | None = None
    E_r: np.ndarray | None = None
    Te_layers: np.ndarray | None = None
    phi_star: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def B(self) -> np.ndarray:
        return np.hypot(self.Br, self.Bz)

    @classmethod
    def placeholder(cls,
                    grid: Grid2D,
                    thruster: Thruster,
                    gas: WorkingSubstance,
                    n_e_peak: float = 2e17,
                    n_n_anode: float = 1e20,
                    ) -> "PlasmaState":
        """Build a state on the real SPT-70 field with prescribed 1-D axial
        density profiles broadcast over r — a stand-in for the eventual
        deposited PIC densities. n_e rises from the anode to a peak near the
        exit plane and decays into the plume; n_n falls exponentially as the
        propellant is ionized. Magnitudes and shapes are representative, not
        self-consistent.
        """
        Br, Bz, lam = field_on_grid(grid, thruster.B_r_max)
        z = grid.z_nodes()
        L = thruster.channel_length

        n_e = np.interp(z, [0.0, L, 2 * L], [0.05 * n_e_peak, n_e_peak, 0.25 * n_e_peak])
        n_n = n_n_anode * np.exp(-z / (0.4 * L))
        # floor the plume neutral density so nu_en / ionization stay finite
        n_n = np.maximum(n_n, 1e-4 * n_n_anode)

        ones_r = np.ones(grid.N_r)
        return cls(
            grid=grid, thruster=thruster, gas=gas,
            Br=Br, Bz=Bz, lam=lam,
            n_e=n_e[:, None] * ones_r,
            n_n=n_n[:, None] * ones_r,
        )
