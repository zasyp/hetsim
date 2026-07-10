import numpy as np

class Grid2D:
    def __init__(
        self,
        max_z:float,
        max_r:float,
        N_r:int,
        N_z:int,
        min_z:float = 0.0,
        min_r:float = 0.0,
    ):
        self.max_z = max_z
        self.max_r = max_r
        self.N_r = N_r
        self.N_z = N_z
        self.min_z = min_z
        self.min_r = min_r

    def z_nodes(self):
        return np.linspace(self.min_z, self.max_z, self.N_z)
    
    def r_nodes(self):
        return np.linspace(self.min_r, self.max_r, self.N_r)

    def create_mesh(self):
        return np.meshgrid(
            np.linspace(self.min_r, self.max_r, self.N_r),
            np.linspace(self.min_z, self.max_z, self.N_z),
            indexing='ij'
            )


class ParticleArray:
    # structure-of-arrays: one object, six equal-length arrays,
    # element k of every array describes particle k
    _fields = ("z", "r", "v_z", "v_r", "v_theta", "weight")

    def __init__(
            self,
            z=None,
            r=None,
            v_z=None,
            v_r=None,
            v_theta=None,
            weight=None,
    ) -> None:
        for name, values in zip(self._fields, (z, r, v_z, v_r, v_theta, weight)):
            arr = np.empty(0) if values is None else np.asarray(values, dtype=np.float64)
            setattr(self, name, arr)

    def __len__(self):
        return self.z.size

    def extend(self, other:"ParticleArray"):
        for name in self._fields:
            setattr(self, name, np.concatenate([getattr(self, name), getattr(other, name)]))

    def keep(self, mask:np.ndarray):
        for name in self._fields:
            setattr(self, name, getattr(self, name)[mask])


class Thruster:
    def __init__(
            self,
            r_min:float,
            r_max:float,
            channel_length:float,
            mdot:float,
            B_r_max:float,
            voltage:int,
            mass:float,
            temperature_anode:float
            ) -> None:
        self.r_min = r_min
        self.r_max = r_max
        self.channel_length = channel_length
        self.mdot = mdot
        self.B_r_max = B_r_max
        self.voltage = voltage
        self.mass = mass
        self.temperature_anode = temperature_anode
