# Lambda layers — block 3 of the electron-fluid model: reduction of the
# (z, r) grid to magnetic-field-line coordinates.
#
# After field_on_grid every node carries lam(z, r) — the magnetic
# streamfunction, constant along a field line. Electrons run freely
# along field lines and barely across them, so one line = one unknown of
# the quasi-1D electron model (Fife 1998; Hara 2019 review). The working
# range of lam (anode line -> cathode line) is split into N_layers
# uniform bins; every node gets its bin number in idx, or -1 if it
# belongs to no layer (lam outside the anode-cathode range, or inside
# the thruster body). Nodes with idx = -1 must stay OUT of every layer
# sum: snapping them to the edge layers is exactly the bug that painted
# the plume with the anode potential in the previous iteration.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from src.structs.classes import Thruster, Grid2D


def lambda_range(lam:np.ndarray,
                 z_nodes:np.ndarray,
                 r_nodes:np.ndarray,
                 thruster:Thruster,
                 ) -> tuple[float, float]:
    """Streamfunction values of the two boundary field lines, read along
    the mid-channel radius: (lam_anode, lam_cathode). The anode line
    passes through (z=0, r_mid), the cathode line through
    (z=thruster.z_cathode, r_mid) — z_cathode is a model knob, keep it
    downstream of the anomalous-transport barrier so the whole potential
    drop lands inside the layered region.
    """
    r_mid = (thruster.r_min + thruster.r_max) / 2
    j_mid = np.argmin(np.abs(r_nodes - r_mid))
    i_c = np.argmin(np.abs(z_nodes - thruster.z_cathode))

    line = lam[:i_c + 1, j_mid]
    d = np.diff(line)
    assert (d > 0).all() or (d < 0).all(), (
        "lambda is not monotonic along mid-channel (B_r changes sign "
        "before z_cathode) — layer ordering anode->cathode would break"
    )

    return lam[0, j_mid], lam[i_c, j_mid]


def thruster_body_mask(grid:Grid2D, thruster:Thruster) -> np.ndarray:
    """Boolean (N_z, N_r) mask of the thruster body: nodes upstream of
    the exit plane but outside the channel annulus. Field lines poke
    through the channel walls into this region, so without the mask
    "walled-in" nodes would join the layers of legitimate plasma nodes.
    The wall rows themselves (r = r_min, r_max) stay unmasked — they
    carry the wall-area binning of layer_wall_area.
    """
    Z, R = np.meshgrid(grid.z_nodes(), grid.r_nodes(), indexing="ij")
    return (Z < thruster.channel_length) & (
        (R < thruster.r_min) | (R > thruster.r_max)
    )


def build_layers(lam:np.ndarray,
                 N_layers:int,
                 lam_a:float,
                 lam_c:float,
                 body_mask:np.ndarray,
                 ):
    """Partition the working lambda range into N_layers uniform bins and
    assign every grid node to one: returns (edges, centers, idx), idx of
    the same shape as lam with values in [0, N_layers-1] or -1.

    Ordering convention: layer 0 touches the anode, layer N_layers-1 the
    cathode line — the potential solver's boundary conditions rely on it,
    so the map is flipped when lam decreases from anode to cathode.

    The clip below only repairs np.digitize's edge case (a node with
    lam exactly equal to the upper edge lands one bin past the end); it
    is applied BEFORE the out-of-range mask, so no outside node survives
    into an edge layer.
    """
    lo, hi = min(lam_a, lam_c), max(lam_a, lam_c)
    edges = np.linspace(lo, hi, N_layers + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    inside = (lam >= lo) & (lam <= hi)
    idx = np.digitize(lam, edges) - 1
    idx = np.clip(idx, 0, N_layers - 1)
    idx[~inside] = -1
    idx[body_mask] = -1

    if lam_a > lam_c:
        valid = idx >= 0
        idx[valid] = N_layers - 1 - idx[valid]

    return edges, centers, idx


def _bin(idx:np.ndarray, w:np.ndarray, N_layers:int) -> np.ndarray:
    """Sum the node weights w into their layers, skipping idx = -1."""
    valid = idx >= 0
    return np.bincount(idx[valid].ravel(), weights=w[valid].ravel(),
                       minlength=N_layers)


def layer_volume(idx:np.ndarray, dV:np.ndarray, N_layers:int) -> np.ndarray:
    """Plasma volume of every layer [m^3]: V_j = sum of node volumes."""
    return _bin(idx, dV, N_layers)


def layer_average(idx:np.ndarray, f:np.ndarray, dV:np.ndarray,
                  N_layers:int) -> np.ndarray:
    """Volume-weighted layer average <f>_j = sum(f dV) / V_j (used for
    n_e, n_n, later Te). Empty layers return 0."""
    V = _bin(idx, dV, N_layers)
    return _bin(idx, f * dV, N_layers) / np.maximum(V, 1e-30)


def layer_wall_area(idx:np.ndarray, grid:Grid2D, thruster:Thruster,
                    N_layers:int) -> np.ndarray:
    """Channel-wall area [m^2] belonging to each layer: every wall node
    (the two grid rows at r_min/r_max, z <= channel length) owns a ring
    dA = 2 pi r_wall dz (half rings at the ends), assigned to the layer
    of the field line that hits the wall there. Plume layers never touch
    the walls and get 0 — the wall-loss term of the energy equation
    switches off there by itself.
    """
    z = grid.z_nodes()
    r = grid.r_nodes()
    dz = z[1] - z[0]
    wall_i = z <= thruster.channel_length + 1e-12
    A = np.zeros(N_layers)
    for r_wall in (thruster.r_min, thruster.r_max):
        j = np.argmin(np.abs(r - r_wall))
        dA = np.full(wall_i.sum(), 2 * np.pi * r_wall * dz)
        dA[0] *= 0.5
        dA[-1] *= 0.5
        A += _bin(idx[wall_i, j], dA, N_layers)
    return A
