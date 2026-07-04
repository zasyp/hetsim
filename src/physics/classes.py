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


class Particle():
    def __init__(
            self,
            z:np.float64,
            r:np.float64,
            v_r:np.float64,
            v_z:np.float64,
            mass:np.float64,
            charge:bool = False,
            T:np.float64 = np.float64(0.0),
            weight:np.float64 = np.float64(1.0),
            active:bool = True,
            ):
        self.z = z
        self.r = r
        self.v_z = v_z
        self.v_r = v_r
        self.mass = mass
        self.charge = charge
        self.T = T            # temperature carried by the particle
        self.weight = weight  # macroparticle statistical weight (real/macro)
        self.active = active  # False once the particle leaves the domain
