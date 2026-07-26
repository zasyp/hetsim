# Layer geometry — the lambda-layer bookkeeping shared by the potential
# and energy solves, built once per magnetic field and reused every
# iteration. Wraps electron_liquid.lambda_layers (block 3) so the solver
# and its diagnostics talk to layers through one object instead of passing
# idx / edges / centers / dV around by hand.

import numpy as np

from ..structs.classes import Grid2D, Thruster
from ..neutrals.neutrals import node_volume
from ..electron_liquid.lambda_layers import (
    lambda_range, build_layers, thruster_body_mask, layer_average, _bin,
)


class LayerGeometry:
    """Partition of the (z, r) grid into N_layers magnetic-field-line
    layers, plus the per-layer volumes, wall areas and a mid-channel z
    coordinate for plotting. Construct once from the field streamfunction
    lam; then use it to reduce any node field to layers and to place layer
    quantities back in space.
    """

    def __init__(self, grid: Grid2D, thruster: Thruster, lam: np.ndarray, N_layers: int):
        self.grid = grid
        self.thruster = thruster
        self.N_layers = N_layers
        self.lam = lam

        z, r = grid.z_nodes(), grid.r_nodes()
        self.dV = node_volume(grid)

        body = thruster_body_mask(grid, thruster)
        self.lam_a, self.lam_c = lambda_range(lam, z, r, thruster)
        self.edges, self.centers, self.idx = build_layers(
            lam, N_layers, self.lam_a, self.lam_c, body)
        self.d_lambda = self.edges[1] - self.edges[0]

        # lambda value OF EACH LAYER INDEX. build_layers bins in ascending
        # lambda (centers[0] = smallest) but then FLIPS the indices when
        # lambda decreases anode->cathode (lam_a > lam_c), so layer k then
        # sits at centers[N-1-k], not centers[k]. Pairing layer_values[k]
        # with the wrong lambda inverts every layer->grid map (potential,
        # Te), so keep the correctly-ordered per-layer lambda here.
        self.layer_lambda = (self.centers[::-1].copy()
                             if self.lam_a > self.lam_c else self.centers.copy())

        self.wall_area = self._layer_wall_area()
        self.z_layer = self._layer_axial_position()

    # --- reductions -------------------------------------------------------

    def average(self, field: np.ndarray) -> np.ndarray:
        """Volume-weighted layer average of a node field."""
        return layer_average(self.idx, field, self.dV, self.N_layers)

    def integrate_power(self, density: np.ndarray) -> np.ndarray:
        """Integrate a volumetric density [X/m^3] over each layer -> [X]."""
        return _bin(self.idx, density * self.dV, self.N_layers)

    def to_grid(self, layer_values: np.ndarray) -> np.ndarray:
        """Interpolate a per-layer quantity back onto every node by its
        local lambda (nodes outside the layered range clamp to the nearest
        boundary layer), matching solve_potential.potential_on_grid.
        """
        order = np.argsort(self.layer_lambda)
        return np.interp(self.lam.ravel(), self.layer_lambda[order],
                         layer_values[order]).reshape(self.lam.shape)

    # --- wall bookkeeping -------------------------------------------------

    def layer_wall_power(self, q_wall: np.ndarray) -> np.ndarray:
        """Total wall power [W] in each layer given a wall heat-flux field
        q_wall [W/m^2] (evaluated on the whole grid; only the wall rows are
        read). Each wall node owns a ring dA = 2 pi r dz (half rings at the
        ends) assigned to its field line's layer — the same binning as
        lambda_layers.layer_wall_area, but weighted by the local flux.
        """
        z, r = self.grid.z_nodes(), self.grid.r_nodes()
        dz = z[1] - z[0]
        wall_i = z <= self.thruster.channel_length + 1e-12
        P = np.zeros(self.N_layers)
        for r_wall in (self.thruster.r_min, self.thruster.r_max):
            j = np.argmin(np.abs(r - r_wall))
            dA = np.full(int(wall_i.sum()), 2 * np.pi * r_wall * dz)
            dA[0] *= 0.5
            dA[-1] *= 0.5
            P += _bin(self.idx[wall_i, j], q_wall[wall_i, j] * dA, self.N_layers)
        return P

    def _layer_wall_area(self) -> np.ndarray:
        from ..electron_liquid.lambda_layers import layer_wall_area
        return layer_wall_area(self.idx, self.grid, self.thruster, self.N_layers)

    def _layer_axial_position(self) -> np.ndarray:
        """Mean mid-channel z [m] of each layer, for plotting layer
        quantities against a physical axis. Empty layers fall back to their
        neighbours by interpolation.
        """
        z, r = self.grid.z_nodes(), self.grid.r_nodes()
        j_mid = np.argmin(np.abs(r - (self.thruster.r_min + self.thruster.r_max) / 2))
        col = self.idx[:, j_mid]
        z_layer = np.full(self.N_layers, np.nan)
        for k in range(self.N_layers):
            here = col == k
            if here.any():
                z_layer[k] = z[here].mean()
        # fill any empty layers by interpolation over layer index
        good = ~np.isnan(z_layer)
        if not good.all():
            z_layer[~good] = np.interp(np.flatnonzero(~good),
                                       np.flatnonzero(good), z_layer[good])
        return z_layer
