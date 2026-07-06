from math import floor
from typing import Any
import numpy as np
from scipy import ndimage

def interpolation_weights(
        node: float, grid: np.ndarray[Any, np.dtype[np.float64]]
        ) -> tuple[int, int, float, float]:
    """Linear interpolation weights for `node` on a uniform 1-D `grid`.

    Returns (idx0, idx1, w0, w1) such that a value sampled at `node` can be
    approximated as `w0*values[idx0] + w1*values[idx1]`. Nodes outside the
    grid are clamped to the nearest edge (weight 1.0 on the boundary index).
    """
    M = len(grid) - 1
    h = grid[1] - grid[0]
    min_node = grid[0]
    max_node = grid[-1]
    if node <= min_node:
        return 0, 1, 1.0, 0.0
    elif node >= max_node:
        return M-1, M, 0.0, 1.0
    
    idx = int(floor((node - min_node) / h))
    idx = max(0, min(idx, M - 1))
    w1 = (node - grid[idx]) / h
    w0 = 1.0 - w1

    return idx, idx+1, w0, w1


def smoothing(
        field:np.ndarray[Any, np.dtype[np.float64]],
        kernel1d:np.ndarray = np.array([1, 2, 1]) / 4.0,
              ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Smooth `field` by convolving `kernel1d` along every axis in turn."""
    for axis in range(field.ndim):
        field = ndimage.convolve1d(field, kernel1d, axis=axis)

    return field

def bilinear_gather(
        field: np.ndarray,
        r: np.ndarray[Any, np.dtype[np.float64]],
        z: np.ndarray[Any, np.dtype[np.float64]],
        pr: np.ndarray[Any, np.dtype[np.float64]],
        pz: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorised bilinear interpolation of a node field with shape
    (len(r), len(z)) at many points at once -- the grid-to-particle (gather)
    counterpart of `bilinear_interp`. Points outside the grid are clamped to
    its edges, matching the behaviour of `interpolation_weights`.
    """
    h_r = r[1] - r[0]
    h_z = z[1] - z[0]
    jr = np.clip(np.floor((pr - r[0]) / h_r).astype(int), 0, len(r) - 2)
    iz = np.clip(np.floor((pz - z[0]) / h_z).astype(int), 0, len(z) - 2)
    tr = np.clip((pr - r[jr]) / h_r, 0.0, 1.0)
    tz = np.clip((pz - z[iz]) / h_z, 0.0, 1.0)
    return (
        (1.0 - tr) * (1.0 - tz) * field[jr, iz]
        + (1.0 - tr) * tz * field[jr, iz + 1]
        + tr * (1.0 - tz) * field[jr + 1, iz]
        + tr * tz * field[jr + 1, iz + 1]
    )


def bilinear_interp(
        field: np.ndarray, r: np.ndarray, z: np.ndarray, r0: float, z0: float,
        ) -> float:
    """Bilinear interpolation of a 2-D (r, z) grid field at an arbitrary
    point (r0, z0), built from the project's shared 1-D
    `interpolation_weights` helper applied along each axis in turn.
    """
    ir0, ir1, wr0, wr1 = interpolation_weights(r0, r)
    iz0, iz1, wz0, wz1 = interpolation_weights(z0, z)
    return (
        wr0 * wz0 * field[ir0, iz0] + wr0 * wz1 * field[ir0, iz1]
        + wr1 * wz0 * field[ir1, iz0] + wr1 * wz1 * field[ir1, iz1]
    )
