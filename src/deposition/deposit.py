from ..structs.classes import ParticleArray, Grid2D

import numpy as np


def locate_particle(
        part: ParticleArray,
        grid: Grid2D
        ):

    dr = (grid.max_r - grid.min_r) / (grid.N_r - 1)
    dz = (grid.max_z - grid.min_z) / (grid.N_z - 1)

    zeta = (part.z - grid.min_z) / dz
    rho = (part.r - grid.min_r) / dr

    left_z_node = np.clip(zeta.astype(np.int64), 0, grid.N_z - 2)
    lower_r_node = np.clip(rho.astype(np.int64), 0, grid.N_r - 2)

    lt_right = zeta - left_z_node
    lt_upper = rho - lower_r_node

    return left_z_node, lower_r_node, lt_right, lt_upper


def scatter(
    part: ParticleArray,
    grid: Grid2D,
    left_z_node: np.ndarray,
    lower_r_node: np.ndarray,
    lt_right: np.ndarray,
    lt_upper: np.ndarray,
    q: np.ndarray | None = None
    ):

    values = part.weight if q is None else part.weight * q
    flat = left_z_node * grid.N_r + lower_r_node
    idx = np.concatenate([flat, flat + grid.N_r, flat + 1, flat + grid.N_r + 1])
    vals = np.concatenate([
        values * (1 - lt_right) * (1 - lt_upper),
        values * (lt_right) * (1 - lt_upper),
        values * (1 - lt_right) * (lt_upper),
        values * (lt_right) * (lt_upper),
    ])
    S = np.bincount(idx, weights=vals, minlength=grid.N_z * grid.N_r)
    return S.reshape(grid.N_z, grid.N_r)


def gather(
        field: np.ndarray,
        left_z_node: np.ndarray,
        lower_r_node: np.ndarray,
        lt_right: np.ndarray,
        lt_upper: np.ndarray
        ):

    return (
        field[left_z_node, lower_r_node] * (1 - lt_right) * (1 - lt_upper)
        + field[left_z_node + 1, lower_r_node] * (lt_right) * (1 - lt_upper)
        + field[left_z_node, lower_r_node + 1] * (1 - lt_right) * (lt_upper)
        + field[left_z_node + 1, lower_r_node + 1] * (lt_right) * (lt_upper)
    )
