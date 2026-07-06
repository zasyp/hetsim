from typing import Any
import numpy as np


class Grid2D():
    """Uniform 2-D (r, z) grid.

    Stores the axis limits, node counts and spacings, and exposes the node
    coordinates. `r_nodes()`/`z_nodes()` return the 1-D coordinate arrays
    (the analogue of Julia's `nodes(g)`), while `mesh()` returns the full
    2-D coordinate matrices for vectorised field evaluation.
    """

    def __init__(
            self,
            r_min: np.float64,
            r_max: np.float64,
            N_r: int,
            z_min: np.float64,
            z_max: np.float64,
            N_z: int,
            ):
        self.r_min = r_min
        self.r_max = r_max
        self.N_r = N_r
        self.z_min = z_min
        self.z_max = z_max
        self.N_z = N_z
        self.h_r = (r_max - r_min) / (N_r - 1)
        self.h_z = (z_max - z_min) / (N_z - 1)

    def r_nodes(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.linspace(self.r_min, self.r_max, self.N_r)

    def z_nodes(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.linspace(self.z_min, self.z_max, self.N_z)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, Z) coordinate matrices with shape (N_r, N_z)."""
        return np.meshgrid(self.r_nodes(), self.z_nodes(), indexing="ij")


class ParticleArray():
    """Structure-of-arrays macroparticle storage for one atomic species.

    All per-particle state lives in flat numpy arrays so that the push,
    boundary, deposition and MCC steps are pure vector operations -- with a
    Python object per particle those steps dominate the run time once the
    population reaches ~1e5 macroparticles.

    Only the first `n` slots are in use; of those, slots with
    `active == False` belong to particles that left the domain and are
    reclaimed by `compact()`. `mass` is the atom mass shared by every
    particle (charge states of one propellant species).

    Fields: z, r (m); v_r, v_z, v_theta (m/s); T (K); weight (atoms per
    macroparticle); charge (0 = neutral, 1 = Xe+, 2 = Xe++); active.
    """

    _FLOAT_FIELDS = ('z', 'r', 'v_r', 'v_z', 'v_theta', 'T', 'weight')

    def __init__(self, mass: float, capacity: int = 16384):
        self.mass = float(mass)
        self.n = 0
        capacity = max(int(capacity), 1)
        for name in self._FLOAT_FIELDS:
            setattr(self, name, np.zeros(capacity))
        self.charge = np.zeros(capacity, dtype=np.int8)
        self.active = np.zeros(capacity, dtype=bool)

    @property
    def capacity(self) -> int:
        return self.z.size

    def _ensure(self, extra: int) -> None:
        need = self.n + extra
        cap = self.capacity
        if need <= cap:
            return
        new_cap = max(2 * cap, need)
        for name in self._FLOAT_FIELDS + ('charge', 'active'):
            old = getattr(self, name)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[:self.n] = old[:self.n]
            setattr(self, name, new)

    def add(self, z, r, v_r, v_z, v_theta, charge, T, weight) -> None:
        """Append a batch of active particles (arrays or scalars of one
        common broadcast length)."""
        m = np.broadcast(z, r, v_r, v_z, v_theta, charge, T, weight).size
        if m == 0:
            return
        self._ensure(m)
        s = slice(self.n, self.n + m)
        self.z[s] = z
        self.r[s] = r
        self.v_r[s] = v_r
        self.v_z[s] = v_z
        self.v_theta[s] = v_theta
        self.charge[s] = charge
        self.T[s] = T
        self.weight[s] = weight
        self.active[s] = True
        self.n += m

    def compact(self) -> None:
        """Drop inactive slots, keeping the arrays contiguous."""
        keep = self.active[:self.n]
        m = int(np.count_nonzero(keep))
        if m == self.n:
            return
        for name in self._FLOAT_FIELDS + ('charge', 'active'):
            arr = getattr(self, name)
            arr[:m] = arr[:self.n][keep]
        self.n = m

    def count_active(self, charge: int | None = None) -> int:
        act = self.active[:self.n]
        if charge is None:
            return int(np.count_nonzero(act))
        return int(np.count_nonzero(act & (self.charge[:self.n] == charge)))
