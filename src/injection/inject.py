import numpy as np
from math import floor, sqrt, log, cos, sin, pi
import scipy.constants as cst
from ..structs.classes import Particle, Thruster

def rotate_plane(
        r:float,
        z:float,
        v_r:float,
        v_theta:float,
        v_z:float,
        tau:float,
        tolerance:float = 1e-14
        ):
    x = r + v_r * tau
    y = v_theta * tau
    z = z + v_z * tau

    new_r = sqrt(x ** 2 + y ** 2)
    if new_r > tolerance:
        cos_a = x / new_r
        sin_a = y / new_r

        new_v_r = cos_a * v_r + sin_a * v_theta
        new_v_theta = -sin_a * v_r + cos_a * v_theta
    else:
        new_v_r = np.hypot(v_r, v_theta)
        new_v_theta = 0.0

    return new_r, z, new_v_r, new_v_theta


def inject_on_grid(
        time_step:float,
        particles:list[Particle],
        thruster:Thruster,
        weight:float,
):
    r1 = thruster.r_min
    r2 = thruster.r_max

    N_frac = thruster.mdot * time_step / (thruster.mass * weight)
    N_inj = floor(N_frac)
    if np.random.rand() < N_frac - N_inj:
        N_inj += 1

    v_thermal = sqrt(thruster.temperature_anode * cst.Boltzmann / thruster.mass)

    for _ in range(N_inj):
        r = sqrt(r1 ** 2 + np.random.rand() * (r2**2 - r1**2))
        z = 0.0
        v_z = v_thermal * sqrt(-2 * log(1 - np.random.rand()))
        u1 = 1 - np.random.rand()
        u2 = np.random.rand()
        v_perp = v_thermal * sqrt(-2 * log(u1))
        v_r = v_perp * cos(2 * pi * u2)
        v_theta = v_perp * sin(2 * pi * u2)

        tau = time_step * np.random.rand()
        r, z, v_r, v_theta = rotate_plane(r, z, v_r, v_theta, v_z, tau)
        particles.append(Particle(weight, z, r, v_z, v_r, v_theta))
    return N_inj


