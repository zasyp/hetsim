import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable

from ..utils.utils import load_xy_table

@dataclass
class Grid2D:
    max_z:float
    max_r:float
    N_r:int
    N_z:int
    min_z:float = 0.0
    min_r:float = 0.0

    def z_nodes(self):
        return np.linspace(self.min_z, self.max_z, self.N_z)
    
    def r_nodes(self):
        return np.linspace(self.min_r, self.max_r, self.N_r)

    def create_mesh(self):
        return np.meshgrid(
            np.linspace(self.min_r, self.max_r, self.N_r),
            np.linspace(self.min_z, self.max_z, self.N_z),
            indexing='ij'
            )


@dataclass
class ParticleArray:
    # structure-of-arrays: one object, six equal-length arrays,
    # element k of every array describes particle k
    _fields = ("z", "r", "v_z", "v_r", "v_theta", "weight")

    z:np.ndarray | None = None
    r:np.ndarray | None = None
    v_z:np.ndarray | None = None
    v_r:np.ndarray | None = None
    v_theta:np.ndarray | None = None
    weight:np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in self._fields:
            values = getattr(self, name)
            arr = np.empty(0) if values is None else np.asarray(values, dtype=np.float64)
            setattr(self, name, arr)

    def __len__(self):
        return self.z.size

    def extend(self, other:"ParticleArray"):
        for name in self._fields:
            setattr(self, name, np.concatenate([getattr(self, name), getattr(other, name)]))

    def keep(self, mask:np.ndarray):
        for name in self._fields:
            setattr(self, name, getattr(self, name)[mask])


@dataclass
class Thruster:
    r_min:float
    r_max:float
    channel_length:float
    mdot:float
    B_r_max:float
    voltage:int
    mass:float
    temperature_anode:float


@dataclass
class WorkingSubstance:
    """Propellant gas: particle mass plus electron-impact ionization
    data as a function of the reduced field E/N.

    Ionization data is meant to be loaded from a public database —
    typically LXCat (lxcat.net): either a pre-computed swarm table
    (E/N in Td, ionization rate coefficient k_iz in m^3/s, produced by
    the site's built-in BOLSIG+ solver from a chosen cross-section set
    such as Phelps/Biagi/SIGLO), or raw cross sections run through an
    external Boltzmann solver. Only the swarm-table path is implemented
    here; see load_lxcat_swarm_file.
    """
    name:str
    mass:float                                          # particle mass, kg
    ionization_table:np.ndarray | None = None           # (N, 2): [E/N (Td), k_iz (m^3/s)]
    maxwellian_table:np.ndarray | None = None           # (N, 3): [T_e (eV), k_iz (m^3/s), E_c (eV)]
    _k_iz_interp:Callable | None = field(default=None, repr=False, init=False)
    _k_iz_Te_interp:Callable | None = field(default=None, repr=False, init=False)
    _E_cost_interp:Callable | None = field(default=None, repr=False, init=False)

    def load_lxcat_swarm_file(self, filepath:str) -> None:
        """Load a two-column swarm table exported from LXCat:
        column 0 = reduced field E/N [Td], column 1 = ionization rate
        coefficient k_iz [m^3/s]. Builds a cubic interpolator over it,
        clamped to the table's edge values outside its range.
        """
        data, (k_iz,) = load_xy_table(filepath, ncols=2)
        self.ionization_table = data
        self._k_iz_interp = k_iz

    def ionization_rate_coefficient(self, E_N:float | np.ndarray) -> float | np.ndarray:
        """k_iz(E/N) in m^3/s, interpolated from the loaded LXCat table."""
        if self._k_iz_interp is None:
            raise RuntimeError(
                f"{self.name}: no ionization data loaded "
                "(call load_lxcat_swarm_file first)"
            )
        return self._k_iz_interp(E_N)

    def volumetric_ionization_rate(self, n_e:np.ndarray, N:np.ndarray, E_N:np.ndarray) -> np.ndarray:
        """S_iz = n_e * N * k_iz(E/N): ionization events per m^3 per s."""
        return n_e * N * self.ionization_rate_coefficient(E_N)

    def load_lxcat_maxwellian_table(self, filepath:str) -> None:
        """Load a three-column table of rate coefficients vs mean electron
        energy for a Maxwellian EEDF (LXCat/BOLSIG+ "rate coefficient vs
        mean energy" output, not the E/N swarm table): column 0 = T_e
        [eV], column 1 = ionization rate coefficient k_iz [m^3/s], column
        2 = effective ionization energy cost E_c [eV] (ionization +
        excitation losses per ionization event). Used by the electron
        fluid energy equation, which is parametrized by T_e directly.
        """
        data, (k_iz, e_cost) = load_xy_table(filepath, ncols=3)
        self.maxwellian_table = data
        self._k_iz_Te_interp = k_iz
        self._E_cost_interp = e_cost

    def ionization_rate_Te(self, Te:float | np.ndarray) -> float | np.ndarray:
        """k_iz(T_e) in m^3/s, assuming a Maxwellian EEDF at T_e."""
        if self._k_iz_Te_interp is None:
            raise RuntimeError(
                f"{self.name}: no Maxwellian ionization data loaded "
                "(call load_lxcat_maxwellian_table first)"
            )
        return self._k_iz_Te_interp(Te)

    def ionization_energy_cost(self, Te:float | np.ndarray) -> float | np.ndarray:
        """Effective energy cost E_c(T_e) [eV] per ionization event."""
        if self._E_cost_interp is None:
            raise RuntimeError(
                f"{self.name}: no Maxwellian ionization data loaded "
                "(call load_lxcat_maxwellian_table first)"
            )
        return self._E_cost_interp(Te) 