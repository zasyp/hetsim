import numpy as np

from ..structs.classes import Grid2D, Thruster, ParticleArray

def node_volume(grid:Grid2D, thruster:Thruster | None = None):
    """Control volume of every node [m^3]: the annulus between the two
    half-cell radii, times dz (half at the z ends).

    Pass the thruster to also halve the nodes that sit ON a channel wall —
    see below. Leaving it out keeps the old behaviour and misreads the
    near-wall density by a factor of two.

    The axis node is the exception. Its half-cell annulus is pi(dr/2)^2 dz,
    but that is not the volume CIC actually deposits into it: a particle at
    radius rho contributes with weight (1 - rho/dr), so a uniform density n
    puts n * 2pi dz * int_0^dr (1-rho/dr) rho drho = n * pi dr^2 dz / 3
    there. Dividing that by pi(dr/2)^2 dz reads back 4/3 n -- a 33% density
    spike on the axis, out of nowhere, which then feeds the plume layer
    averages and the Boltzmann term. Using the deposition's own volume,
    pi dr^2 dz / 3, makes the axis node read a uniform density correctly.
    (Every other node comes out exact already: the linear weights from the
    two neighbouring shells sum to its full annulus.)
    """
    dz = (grid.max_z - grid.min_z) / (grid.N_z - 1)
    dr = (grid.max_r - grid.min_r) / (grid.N_r - 1)
    r = grid.r_nodes()

    r_inner = np.maximum(r - dr/2, grid.min_r)
    r_outer = np.minimum(r + dr/2, grid.max_r)
    ring_sq = np.pi * (r_outer ** 2 - r_inner ** 2)
    if grid.min_r == 0.0:
        ring_sq[0] = np.pi * dr ** 2 / 3

    V = np.tile(ring_sq * dz, (grid.N_z, 1))
    V[0, :] *= 0.5
    V[-1, :] *= 0.5

    # Nodes lying exactly on a channel wall are the same bug as the axis, in
    # the other direction. CIC hands them deposits from ONE side only -- the
    # other side is thruster body, where no particle ever is -- but they are
    # being given a FULL annulus to divide by, so a uniform plasma reads half
    # its density there. (The integral is the same one as for the axis: a
    # uniform n deposits n * 2pi r_wall (dr/2) dz into the wall node, i.e.
    # exactly the plasma-side half-annulus.)
    #
    # That factor of two is not cosmetic. The Boltzmann term of the
    # thermalized potential is Te*ln(n_e/n_ref), so halving n at the wall
    # subtracts a spurious Te*ln2 -- 20-30 V at these temperatures -- from the
    # potential right at the wall, and that shows up as a radial field
    # pushing ions INTO the wall. The measured near-wall density ratio was
    # 0.21 of mid-channel where the Bohm pre-sheath says ~0.5.
    if thruster is not None:
        z = grid.z_nodes()
        in_channel = z <= thruster.channel_length
        for r_wall in (thruster.r_min, thruster.r_max):
            j = int(np.argmin(np.abs(grid.r_nodes() - r_wall)))
            if abs(grid.r_nodes()[j] - r_wall) < 1e-9:
                V[in_channel, j] *= 0.5
    return V


def density(S:np.ndarray, V:np.ndarray):
    return S / V


def free_flight(
        z:np.ndarray,
        r:np.ndarray,
        v_z:np.ndarray,
        v_r:np.ndarray,
        v_theta:np.ndarray,
        tau,
        tolerance:float = 1e-14
        ):
    # straight flight in 3D + rotation back into the r-z plane;
    # tau is a scalar dt or a per-particle array of flight times
    x = r + v_r * tau
    y = v_theta * tau
    new_z = z + v_z * tau
    new_r = np.hypot(x, y)

    on_axis = new_r <= tolerance
    safe_r = np.where(on_axis, 1.0, new_r)
    cos_a = x / safe_r
    sin_a = y / safe_r

    new_v_r = np.where(on_axis, np.hypot(v_r, v_theta), cos_a * v_r + sin_a * v_theta)
    new_v_theta = np.where(on_axis, 0.0, -sin_a * v_r + cos_a * v_theta)

    return new_z, new_r, new_v_r, new_v_theta


def push_neutrals(part:ParticleArray, dt:float):
    part.z, part.r, part.v_r, part.v_theta = free_flight(
        part.z, part.r, part.v_z, part.v_r, part.v_theta, dt
    )


def sample_wall_flux(n:int, v_th:float):
    # diffuse re-emission at wall temperature:
    # normal component from the flux Maxwellian, tangentials Gaussian
    v_normal = v_th * np.sqrt(-2 * np.log(1 - np.random.rand(n)))
    u1 = 1 - np.random.rand(n)
    u2 = np.random.rand(n)
    amp = v_th * np.sqrt(-2 * np.log(u1))
    v_tang1 = amp * np.cos(2 * np.pi * u2)
    v_tang2 = amp * np.sin(2 * np.pi * u2)
    return v_normal, v_tang1, v_tang2


def apply_boundaries(
        part:ParticleArray,
        thruster:Thruster,
        grid:Grid2D,
        v_th_wall:float,
        z_prev:np.ndarray,
        ):
    # domain = channel [0, L] x [r_min, r_max] plus the plume region
    # [L, grid.max_z] x [0, grid.max_r] behind the exit plane;
    # z_prev (position before the push) tells whether the particle
    # came from the channel or from the plume
    L = thruster.channel_length

    # open plume boundaries: the particle leaves the domain
    alive = (part.z <= grid.max_z) & (part.r <= grid.max_r)

    was_in_plume = z_prev > L

    # anode (z < 0): diffuse re-emission, normal is +z
    hit = alive & (part.z < 0.0)
    n = int(hit.sum())
    if n:
        part.z[hit] = 0.0
        v_n, t1, t2 = sample_wall_flux(n, v_th_wall)
        part.v_z[hit] = v_n
        part.v_r[hit] = t1
        part.v_theta[hit] = t2

    # channel walls: only particles that were inside the channel
    in_channel = alive & ~was_in_plume & (part.z <= L)

    # inner wall (r < r_min): normal is +r
    hit = in_channel & (part.r < thruster.r_min)
    n = int(hit.sum())
    if n:
        part.r[hit] = thruster.r_min
        v_n, t1, t2 = sample_wall_flux(n, v_th_wall)
        part.v_r[hit] = v_n
        part.v_z[hit] = t1
        part.v_theta[hit] = t2

    # outer wall (r > r_max): normal is -r
    hit = in_channel & (part.r > thruster.r_max)
    n = int(hit.sum())
    if n:
        part.r[hit] = thruster.r_max
        v_n, t1, t2 = sample_wall_flux(n, v_th_wall)
        part.v_r[hit] = -v_n
        part.v_z[hit] = t1
        part.v_theta[hit] = t2

    # front face of the thruster (z = L outside the channel annulus):
    # a plume particle flying back hits it, normal is +z
    hit = alive & was_in_plume & (part.z < L) & (
        (part.r < thruster.r_min) | (part.r > thruster.r_max)
    )
    n = int(hit.sum())
    if n:
        part.z[hit] = L
        v_n, t1, t2 = sample_wall_flux(n, v_th_wall)
        part.v_z[hit] = v_n
        part.v_r[hit] = t1
        part.v_theta[hit] = t2

    return alive
