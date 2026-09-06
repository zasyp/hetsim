# The reference discharge, in one place.
#
# Every entry point — the SPT-70 example, the legacy checkpoint scripts —
# builds its thruster and grid from here, so none of them can drift out of
# sync with the class interfaces one at a time.

from .structs.classes import Grid2D, Thruster
from .structs.propellants import xenon
from .structs.wall_materials import bn_sio2


def spt70() -> tuple[Thruster, Grid2D]:
    """Reference SPT-70 discharge: xenon at 2.5 mg/s, 300 V, B_r,max = 100 G,
    channel annulus r 17.5-35 mm, L = 30 mm, anode at 750 K, BN-SiO2 walls.

    The grid covers the channel (30 mm) plus a plume region behind the exit
    (30 mm more, radially from the axis out to 50 mm) on 0.5 mm cells. Those
    0.5 mm are not free: every material edge of the magnetic circuit lands on
    a grid line at that spacing, which is what keeps the field interpolation
    off the iron (see magnetics.spt70_system.check_conformal).

    Geometry is an illustrative approximation of an SPT-70-class thruster —
    representative topology and scale, not manufacturer data.
    """
    thruster = Thruster(
        r_min=0.0175, r_max=0.035, channel_length=0.03,
        mdot=2.5e-6, B_r_max=0.01, voltage=300,
        propellant=xenon(), temperature_anode=750.0,
        wall_material=bn_sio2(), z_cathode=0.045,
    )
    grid = Grid2D(max_z=0.06, max_r=0.05, N_r=101, N_z=121)
    return thruster, grid
