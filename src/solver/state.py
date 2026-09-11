# Plasma state — the field container the electron-fluid solver reads and
# writes. Holds the fixed inputs (magnetic field + heavy-species densities)
# and, after a solve, the electron results (Te, potential, E-field).
#
# Two ways to fill the densities:
#
#   placeholder      prescribed 1-D profiles broadcast over r. Not a solution
#                    — a stand-in that lets the electron solver be exercised
#                    on its own. Read the SHAPE of anything it produces, never
#                    the magnitude.
#   from_particles   n_e and n_n deposited from the ion and neutral
#                    macroparticles, with n_e = n_i by quasineutrality. This
#                    is the self-consistent path; the hybrid loop calls
#                    update_densities on every electron update.

from dataclasses import dataclass, field

import numpy as np

from ..structs.classes import Grid2D, Thruster, WorkingSubstance, ParticleArray
from ..magnetics.spt70_system import field_on_grid
from ..ions.moments import number_density, smooth

# Floors on the deposited densities. Not cosmetic: the cross-field conductance
# that the potential solve inverts is proportional to n_e, so an empty cell
# would make the layer conductance singular; the collision frequencies divide
# by n_n. The floors are far below any density the model resolves (n_e runs
# 1e17-1e18 in the channel, n_n 1e19-1e20), so they only ever act in cells
# no particle has reached yet.
N_E_FLOOR = 1e15        # 1/m^3
N_N_FLOOR = 1e15        # 1/m^3


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

    @classmethod
    def from_particles(cls,
                       grid: Grid2D,
                       thruster: Thruster,
                       gas: WorkingSubstance,
                       ions: ParticleArray,
                       neutrals: ParticleArray,
                       smooth_passes: int = 2,
                       dV: np.ndarray | None = None,
                       ) -> "PlasmaState":
        """Build a state whose densities come from the macroparticles.

        n_e = n_i is quasineutrality, and it is an assumption, not a result:
        the model never solves Poisson, so it cannot represent charge
        separation. That is fine everywhere except inside the Debye sheath at
        the walls — which is exactly why the sheath is handled analytically
        (block 5) instead of being resolved.
        """
        Br, Bz, lam = field_on_grid(grid, thruster.B_r_max)
        state = cls(grid=grid, thruster=thruster, gas=gas, Br=Br, Bz=Bz, lam=lam,
                    n_e=np.full((grid.N_z, grid.N_r), N_E_FLOOR),
                    n_n=np.full((grid.N_z, grid.N_r), N_N_FLOOR))
        state.update_densities(ions, neutrals, smooth_passes=smooth_passes, dV=dV)
        return state

    def update_densities(self,
                         ions: ParticleArray,
                         neutrals: ParticleArray,
                         smooth_passes: int = 2,
                         dV: np.ndarray | None = None,
                         ) -> None:
        """Re-deposit n_e and n_n from the current particle populations.

        Mutates in place on purpose. The hybrid loop keeps ONE state and one
        FluidElectronSolver across the whole run: the solver holds the layer
        geometry (built once from the magnetic field) and the previous Te as a
        warm start, and it reads n_e / n_n off the state at every Gummel
        iteration. Rebuilding either object each electron update would throw
        away both and cost a hundred iterations instead of a handful.
        """
        n_i = smooth(number_density(ions, self.grid, dV), smooth_passes)
        n_n = smooth(number_density(neutrals, self.grid, dV), smooth_passes)
        self.n_e = np.maximum(n_i, N_E_FLOOR)
        self.n_n = np.maximum(n_n, N_N_FLOOR)
