import numpy as np
import scipy.constants as cst
from dataclasses import dataclass, field
from collections.abc import Callable

from ..utils.utils import load_xy_table, load_hallthruster_table

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

    def thermal_speed(self, T:float) -> float:
        """sqrt(k_B T / m) — 1D thermal speed scale [m/s], T in K."""
        return np.sqrt(cst.Boltzmann * T / self.mass)

    def mean_speed(self, T:float) -> float:
        """sqrt(8 k_B T / (pi m)) — mean speed of a Maxwellian [m/s], T in K."""
        return np.sqrt(8 * cst.Boltzmann * T / (np.pi * self.mass))
    ionization_table:np.ndarray | None = None           # (N, 2): [E/N (Td), k_iz (m^3/s)]
    maxwellian_table:np.ndarray | None = None           # (N, 3): [T_e (eV), k_iz (m^3/s), E_c (eV)]
    elastic_table:np.ndarray | None = None              # (N, 2): [T_e (eV), k_en (m^3/s)]
    excitation_table:np.ndarray | None = None           # (N, 2): [T_e (eV), k_exc (m^3/s)]
    E_iz:float | None = None                            # ionization threshold, eV
    E_exc:float | None = None                           # excitation threshold, eV
    _k_iz_interp:Callable | None = field(default=None, repr=False, init=False)
    _k_iz_Te_interp:Callable | None = field(default=None, repr=False, init=False)
    _E_cost_interp:Callable | None = field(default=None, repr=False, init=False)
    _k_en_interp:Callable | None = field(default=None, repr=False, init=False)
    _k_exc_interp:Callable | None = field(default=None, repr=False, init=False)

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
        """Effective energy cost E_c(T_e) [eV] per ionization event:
        the ionization energy itself plus the excitation losses the
        electron population pays alongside each ionization.

        Two sources, in order of preference: a pre-computed E_c column
        from a 3-column Maxwellian table, if one was loaded; otherwise
        assembled from the HallThruster.jl-style reaction thresholds as

            E_c(Te) = E_iz + E_exc * k_exc(Te) / k_iz(Te),

        the excitation term being the mean number of excitation events
        per ionization event times the energy lost in each. k_iz is
        floored to keep the ratio finite at low Te, where both rates
        vanish; the resulting huge E_c is harmless because the energy
        equation only ever uses the product E_c * k_iz, which stays
        equal to E_iz*k_iz + E_exc*k_exc.
        """
        if self._E_cost_interp is not None:
            return self._E_cost_interp(Te)
        if self.E_iz is None:
            raise RuntimeError(
                f"{self.name}: no ionization-cost data loaded (call "
                "load_hallthruster_ionization or load_lxcat_maxwellian_table first)"
            )
        if self._k_exc_interp is None:
            return self.E_iz if np.isscalar(Te) else np.full_like(np.asarray(Te, dtype=float), self.E_iz)
        k_iz = np.maximum(self._k_iz_Te_interp(Te), 1e-30)
        return self.E_iz + self.E_exc * self._k_exc_interp(Te) / k_iz

    # --- HallThruster.jl reaction tables (Te-parametrized, Maxwellian EEDF) ---

    def load_hallthruster_ionization(self, filepath:str) -> None:
        """Load a HallThruster.jl single-ionization table (e.g.
        ionization_Xe_Xe+.dat): sets k_iz(T_e) and the ionization
        threshold E_iz [eV] parsed from the file header.
        """
        threshold, data, interp = load_hallthruster_table(filepath)
        if threshold is None:
            raise ValueError(f"{filepath}: no 'Ionization energy (eV):' header found")
        self.maxwellian_table = data
        self.E_iz = threshold
        self._k_iz_Te_interp = interp

    def load_hallthruster_elastic(self, filepath:str) -> None:
        """Load a HallThruster.jl elastic (momentum-transfer) table
        (e.g. elastic_Xe.dat): sets k_en(T_e) for the electron-neutral
        collision frequency nu_en = n_n * k_en.
        """
        _, data, interp = load_hallthruster_table(filepath)
        self.elastic_table = data
        self._k_en_interp = interp

    def load_hallthruster_excitation(self, filepath:str) -> None:
        """Load a HallThruster.jl excitation table (e.g.
        excitation_Xe.dat): sets k_exc(T_e) and the excitation threshold
        E_exc [eV] used in the effective ionization cost.
        """
        threshold, data, interp = load_hallthruster_table(filepath)
        if threshold is None:
            raise ValueError(f"{filepath}: no 'Excitation energy (eV):' header found")
        self.excitation_table = data
        self.E_exc = threshold
        self._k_exc_interp = interp

    def k_en(self, Te:float | np.ndarray) -> float | np.ndarray:
        """Elastic momentum-transfer rate coefficient k_en(T_e) [m^3/s],
        Maxwellian EEDF."""
        if self._k_en_interp is None:
            raise RuntimeError(
                f"{self.name}: no elastic-collision data loaded "
                "(call load_hallthruster_elastic first)"
            )
        return self._k_en_interp(Te)

    def k_exc(self, Te:float | np.ndarray) -> float | np.ndarray:
        """Excitation rate coefficient k_exc(T_e) [m^3/s], Maxwellian EEDF."""
        if self._k_exc_interp is None:
            raise RuntimeError(
                f"{self.name}: no excitation data loaded "
                "(call load_hallthruster_excitation first)"
            )
        return self._k_exc_interp(Te)


@dataclass
class Thruster:
    r_min:float
    r_max:float
    channel_length:float
    mdot:float
    B_r_max:float
    voltage:int
    propellant:WorkingSubstance
    temperature_anode:float
    # axial position [m] of the "cathode plane": the field line passing
    # through (z_cathode, r_mid) is the cathode-side boundary of the
    # electron-fluid lambda layers. A model knob, not hardware geometry;
    # keep it downstream of the anomalous-transport barrier (~1.14 L).
    z_cathode:float = 0.045

    @property
    def mass(self) -> float:
        """Propellant particle mass [kg] — shorthand for propellant.mass."""
        return self.propellant.mass

    def exit_area(self) -> float:
        """Annular channel cross-section pi*(r_max^2 - r_min^2) [m^2]."""
        return np.pi * (self.r_max ** 2 - self.r_min ** 2) 