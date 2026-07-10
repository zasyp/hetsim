from typing import Any
import numpy as np
from math import floor, sqrt
import scipy.constants as cst
from ..structs.classes import *

def inject_on_grid(
        time_step:np.float64,
        particles:list[Particle],
        thruster:Thruster,
):
    r1 = thruster.r_min
    r2 = thruster.r_max

    N_frac = thruster.mdot * time_step / (thruster.mass * particles[0].weight)
    N_inj = floor(N_frac)

    v_thermal = sqrt(thruster.temperature_anode * cst.Boltzmann / thruster.mass)

    for _ in N_inj:
        r = sqrt(r1 ** 2 + np.random.rand() * (r2**2 - r1**2))
        z = 0.0
        v_z = v_thermal 
        
    
    return