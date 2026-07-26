# Shared setup for the example scripts: the reference SPT-70 discharge
# in one place, so the examples cannot drift out of sync with the class
# interfaces one by one.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from src.structs.classes import Grid2D, Thruster
from src.structs.propellants import xenon
from src.structs.wall_materials import bn_sio2


def spt70() -> tuple[Thruster, Grid2D]:
    """Reference SPT-70 setup used by every example: xenon at 2.5 mg/s,
    300 V, B_r,max = 150 G, channel annulus r 17.5-35 mm, L = 30 mm,
    anode at 750 K.

    The grid covers the channel (30 mm) plus a plume region behind the
    exit (30 mm more, radially from the axis out to 50 mm), 0.5 mm cells.
    """
    thruster = Thruster(
        r_min=0.0175, r_max=0.035, channel_length=0.03,
        mdot=2.5e-6, B_r_max=0.015, voltage=300,
        propellant=xenon(), temperature_anode=750.0,
        wall_material=bn_sio2(), z_cathode=0.045,
    )
    grid = Grid2D(max_z=0.06, max_r=0.05, N_r=101, N_z=121)
    return thruster, grid
