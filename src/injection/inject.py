import numpy as np
from math import floor
from ..structs.classes import ParticleArray, Thruster
from ..neutrals.neutrals import free_flight


def inject_on_grid(
        time_step:float,
        thruster:Thruster,
        weight:float,
) -> ParticleArray:
    r1 = thruster.r_min
    r2 = thruster.r_max

    N_frac = thruster.mdot * time_step / (thruster.mass * weight)
    N_inj = floor(N_frac)
    if np.random.rand() < N_frac - N_inj:
        N_inj += 1

    v_thermal = thruster.propellant.thermal_speed(thruster.temperature_anode)

    r = np.sqrt(r1 ** 2 + np.random.rand(N_inj) * (r2 ** 2 - r1 ** 2))
    z = np.zeros(N_inj)
    v_z = v_thermal * np.sqrt(-2 * np.log(1 - np.random.rand(N_inj)))
    u1 = 1 - np.random.rand(N_inj)
    u2 = np.random.rand(N_inj)
    v_perp = v_thermal * np.sqrt(-2 * np.log(u1))
    v_r = v_perp * np.cos(2 * np.pi * u2)
    v_theta = v_perp * np.sin(2 * np.pi * u2)

    # smear the birth moment over the step: each particle flies
    # a random fraction of dt before the step ends
    tau = time_step * np.random.rand(N_inj)
    z, r, v_r, v_theta = free_flight(z, r, v_z, v_r, v_theta, tau)

    return ParticleArray(z, r, v_z, v_r, v_theta, np.full(N_inj, weight))
