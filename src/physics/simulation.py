"""Top-level 2D/3V hybrid PIC-MCC Hall thruster simulation.

Wires together every stage of the per-step cycle in the order the physics
requires (see the discussion of Chapter II of the thesis):

    1. neutral injection at the anode        (neutrals.particle_injection)
    2. leapfrog push of neutrals and ions    (pusher.push_particles)
    3. wall accommodation / exits            (pusher.apply_boundaries)
    4. particle-to-grid deposition per species (neutrals.deposit_moments)
    5. electron fluid update: mobility, 1-D Ohm's law -> I_T, phi, E,
       electron energy                        (electrons.ElectronFluid.solve)
    6. MCC ionization with collision multiplier (ionization.ionization_step)

Performance metrics (thrust, Isp, currents, utilization, 2.108-2.115) are
accumulated from the macroparticles that leave through the domain exits.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.classes import ParticleArray, Grid2D
from physics.neutrals import particle_injection, deposit_moments, node_volumes
from physics.pusher import (
    ChannelGeometry, ExitTally, push_particles, apply_boundaries, E_CHARGE, KB,
)
from physics.electrons import ElectronFluid
from physics.ionization import ionization_step, IonizationRates, default_rates

XE_MASS = np.float64(2.1801714e-25)  # xenon atom, kg
G0 = 9.80665                          # standard gravity, m/s^2


class FieldSet:
    """Deposited moments for one species."""

    def __init__(self, g: Grid2D):
        shape = (g.N_r, g.N_z)
        self.n = np.zeros(shape)
        self.v_r = np.zeros(shape)
        self.v_z = np.zeros(shape)
        self.v_theta = np.zeros(shape)
        self.T = np.zeros(shape)


class Simulation:
    """2D/3V hybrid PIC-MCC Hall thruster discharge.

    Parameters
    ----------
    g : full-domain grid (channel + near field).
    geom : channel geometry inside that grid.
    Br, Bz, lam : static magnetic field and streamfunction on the nodes
        (from `physics.magnetic`, e.g. `vacuum_magnetic_field` or the flux
        solver of the SPT-70 example).
    lambda_anode, lambda_cathode : streamfunction values of the virtual
        anode/cathode lines (`electrons.lambda_at` helps pick them).
    mdot : anode propellant mass flow rate, kg/s.
    dt : heavy-particle time step, s.
    n_inject : neutral macroparticles injected per step (thesis: 6-10).
    V_discharge : discharge voltage, V.
    T_anode : anode/wall temperature for injection and accommodation, K.
    w_min_frac : daughter weight floor of the MCC collision multiplier as a
        fraction of the injection macroparticle weight; bounds the
        macroparticle population growth (see `ionization`). 0 disables it.
    alpha_bohm_in, alpha_bohm_out : two-zone anomalous transport (LANDMARK
        convention): reduced anomalous collisionality inside the channel,
        Bohm-1/16 outside. A uniform 1/16 makes the channel far too
        conductive -- the electron backstreaming current then overheats the
        discharge (eps saturating at the ceiling, ionization spread over the
        whole channel instead of the exit layer).
    alpha_wall_energy : wall energy loss coefficient (2.107), applied inside
        the channel only -- the plume has no walls.
    """

    def __init__(
            self,
            g: Grid2D,
            geom: ChannelGeometry,
            Br: np.ndarray,
            Bz: np.ndarray,
            lam: np.ndarray,
            lambda_anode: float,
            lambda_cathode: float,
            *,
            mdot: float = 2.5e-6,
            m_atom: float = XE_MASS,
            dt: float = 5.0e-8,
            n_inject: int = 8,
            V_discharge: float = 300.0,
            T_anode: float = 1000.0,
            gamma_mcc: float = 8.0,
            w_min_frac: float = 0.05,
            alpha_bohm_in: float = 1.0 / 64.0,
            alpha_bohm_out: float = 1.0 / 16.0,
            alpha_wall_energy: float = 0.7,
            rates: IonizationRates | None = None,
            rng: np.random.Generator | None = None,
            electron_kwargs: dict | None = None,
            ):
        self.g = g
        self.geom = geom
        self.mdot = mdot
        self.m_atom = m_atom
        self.dt = dt
        self.n_inject = n_inject
        self.T_anode = T_anode
        self.gamma_mcc = gamma_mcc
        self.w_inject = mdot * dt / (m_atom * n_inject)
        self.w_min = w_min_frac * self.w_inject
        self.rates = rates if rates is not None else default_rates()
        self.rng = rng if rng is not None else np.random.default_rng()

        self.particles = ParticleArray(m_atom)
        self.volumes = node_volumes(g)
        self.mask = geom.plasma_mask(g)
        self.neutrals = FieldSet(g)
        self.ions1 = FieldSet(g)
        self.ions2 = FieldSet(g)
        self.n_e = np.zeros((g.N_r, g.N_z))

        # Two-zone anomalous transport / wall losses on the nodes: the
        # channel proper is everything upstream of the exit plane.
        _, Z = g.mesh()
        in_channel = self.mask & (Z <= geom.z_exit)
        ekw = dict(
            alpha_bohm=np.where(in_channel, alpha_bohm_in, alpha_bohm_out),
            alpha_wall_energy=np.where(in_channel, alpha_wall_energy, 0.0),
        )
        ekw.update(electron_kwargs or {})

        self.electrons = ElectronFluid(
            g, Br, Bz, lam, self.mask,
            lambda_anode, lambda_cathode,
            V_discharge=V_discharge,
            rates=self.rates,
            **ekw,
        )

        self.tally = ExitTally()
        self._tally_t0 = 0.0
        self.time = 0.0
        self.n_steps = 0

    def reset_tally(self) -> None:
        """Restart the exit-tally accumulation (e.g. once the discharge has
        ignited, so `performance()` reflects the steady state and not the
        ignition transient)."""
        self.tally = ExitTally()
        self._tally_t0 = self.time

    # ------------------------------------------------------------------

    def seed_ions(self, n0: float, n_macro: int = 2000, T_ion: float = 1000.0,
                  eps_seed: float | None = None) -> None:
        """Seed an initial quasineutral plasma of density `n0` (1/m^3) inside
        the channel so the discharge can ignite (with no ions there is no
        n_e, hence no ionization and no current). Optionally preheat the
        electron slices to `eps_seed` eV.
        """
        geom, g = self.geom, self.g
        vol = np.pi * (geom.r_out**2 - geom.r_in**2) * (geom.z_exit - g.z_min)
        w = n0 * vol / n_macro
        vth = np.sqrt(KB * T_ion / self.m_atom)
        u = self.rng.random(n_macro)
        r = np.sqrt(geom.r_in**2 + u * (geom.r_out**2 - geom.r_in**2))
        z = g.z_min + self.rng.random(n_macro) * (geom.z_exit - g.z_min)
        v = self.rng.normal(0.0, vth, (n_macro, 3))
        self.particles.add(z, r, v[:, 0], v[:, 1], v[:, 2],
                           1, T_ion, w)
        if eps_seed is not None:
            self.electrons.eps[:] = eps_seed

    def _deposit(self, charge: int, out: FieldSet) -> None:
        pa = self.particles
        select = pa.active[:pa.n] & (pa.charge[:pa.n] == charge)
        deposit_moments(self.g, pa, select, self.volumes,
                        out.n, out.v_r, out.v_z, out.v_theta, out.T)

    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Advance the discharge by one heavy-particle time step."""
        g, geom = self.g, self.geom
        pa = self.particles

        particle_injection(
            g, pa, self.rng,
            np.float64(self.mdot), self.m_atom, np.float64(self.T_anode),
            np.float64(self.dt), self.n_inject,
            r_in=np.float64(geom.r_in), r_out=np.float64(geom.r_out),
        )

        push_particles(g, pa, self.dt,
                       self.electrons.Er, self.electrons.Ez)
        n_exited = apply_boundaries(
            g, pa, geom, self.rng, self.dt, self.tally,
            T_wall=self.T_anode)

        self._deposit(0, self.neutrals)
        self._deposit(1, self.ions1)
        self._deposit(2, self.ions2)
        self.n_e = self.ions1.n + 2.0 * self.ions2.n

        ediag = self.electrons.solve(
            self.n_e, self.neutrals.n,
            self.ions1.n, self.ions1.v_r, self.ions1.v_z,
            self.ions2.n, self.ions2.v_r, self.ions2.v_z,
            dt=self.dt,
        )

        counts = ionization_step(
            g, pa,
            self.electrons.ne_used, self.electrons.eps_nodes,
            self.dt, self.rng, self.rates, gamma=self.gamma_mcc,
            w_min=self.w_min,
        )

        # Reclaim the slots of exited particles once they are a noticeable
        # fraction of the storage.
        n_active = pa.count_active()
        if pa.n - n_active > max(1024, pa.n // 10):
            pa.compact()

        self.time += self.dt
        self.n_steps += 1
        return {
            'time': self.time,
            'n_active': n_active,
            'n_exited': n_exited,
            'ionization': counts,
            **ediag,
        }

    # ------------------------------------------------------------------

    def performance(self) -> dict:
        """Time-averaged performance from the exit tally (2.108-2.115),
        accumulated since the last `reset_tally()` call."""
        t = max(self.time - self._tally_t0, 1e-300)
        w = self.tally.weight
        mdot_i = self.m_atom * (w[1] + w[2]) / t                      # (2.108)
        mdot_total = self.m_atom * w[0] / t + mdot_i                  # (2.109)
        I_i = E_CHARGE * (w[1] + 2.0 * w[2]) / t                      # (2.110)
        thrust = sum(self.tally.momentum_z.values()) / t              # (2.112)
        isp = thrust / (max(self.mdot, 1e-300) * G0)                  # (2.113)
        eta_u = mdot_i / max(self.mdot, 1e-300)                       # (2.115)
        return {
            'thrust_N': thrust,
            'Isp_s': isp,
            'ion_current_A': I_i,
            'mdot_ion': mdot_i,
            'mdot_exit_total': mdot_total,
            'propellant_utilization': eta_u,
        }
