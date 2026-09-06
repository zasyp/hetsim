# Beam diagnostics: thrust, specific impulse, exit velocity, divergence.
#
# Everything a Hall thruster is judged by is a moment of the particles crossing
# the plume boundary, so that is what this module accumulates. The ion pusher
# hands over each batch of escaping ions (apply_ion_boundaries returns them
# whole); BeamTally sums the moments and keeps two histograms.
#
# Why accumulate instead of reading the instantaneous flux: at any single
# macro-step only a few hundred macroparticles cross the boundary, so an
# instantaneous thrust is ~5% noise at best, and during a breathing cycle it
# swings by much more than that. Thrust is a cycle-averaged quantity by nature.
# Accumulating over a window and dividing by the elapsed time is both less
# noisy and the physically meaningful number.
#
# Definitions used here, stated because they differ between papers:
#
#   T      = axial momentum flux carried out through the plume boundary,
#            m * sum(w * v_z) / elapsed  [N]. Ions and escaping neutrals both
#            count; both really do carry momentum away.
#   Isp    = T / (mdot * g0)  [s], with mdot the ANODE flow, so an un-ionized
#            atom drifting out at thermal speed correctly drags Isp down.
#   theta  = atan2(v_r, v_z) per ion, the divergence half-angle. Reported as
#            the momentum-weighted mean and as cos(theta) averaged over the
#            beam — the latter is the divergence efficiency, the fraction of
#            the ion momentum that actually points along the axis.
#   V_beam = mean ion kinetic energy expressed in volts, (m v^2 / 2) / e.
#            Compare with the discharge voltage: the ratio is the voltage
#            utilization, and it is always below one because ions are born
#            spread through the acceleration layer, not all at the anode.
#
# What is NOT in the thrust here: pressure forces on the anode and the front
# face, and momentum carried by the injected propellant. Both are small for a
# Hall thruster (the beam dominates by an order of magnitude) but they are not
# zero, so treat T as the beam thrust, not the balance-of-forces thrust.

import numpy as np
import scipy.constants as cst

from ..structs.classes import ParticleArray, Thruster


class BeamTally:
    """Running moments of everything that leaves through the plume."""

    def __init__(self, mass: float, v_max: float = 3.0e4, n_bins: int = 60):
        self.mass = mass
        self.weight = 0.0          # real particles
        self.p_z = 0.0             # sum w*v_z      [particles * m/s]
        self.p_r = 0.0             # sum w*v_r
        self.energy = 0.0          # sum w*v^2/2    [particles * J/kg]
        self.cos_weighted = 0.0    # sum w*|v|*cos(theta)
        self.speed_weighted = 0.0  # sum w*|v|
        self.v_edges = np.linspace(0.0, v_max, n_bins + 1)
        self.v_hist = np.zeros(n_bins)
        self.theta_edges = np.linspace(0.0, np.pi / 2, n_bins + 1)
        self.theta_hist = np.zeros(n_bins)

        self.neutral_weight = 0.0
        self.neutral_p_z = 0.0

    def add_ions(self, part: ParticleArray) -> None:
        if len(part) == 0:
            return
        w = part.weight
        speed = np.sqrt(part.v_z ** 2 + part.v_r ** 2 + part.v_theta ** 2)
        theta = np.arctan2(np.abs(part.v_r), part.v_z)

        self.weight += float(w.sum())
        self.p_z += float((w * part.v_z).sum())
        self.p_r += float((w * np.abs(part.v_r)).sum())
        self.energy += float((w * 0.5 * speed ** 2).sum())
        self.speed_weighted += float((w * speed).sum())
        self.cos_weighted += float((w * speed * np.cos(theta)).sum())

        self.v_hist += np.histogram(speed, bins=self.v_edges, weights=w)[0]
        self.theta_hist += np.histogram(np.clip(theta, 0, np.pi / 2),
                                        bins=self.theta_edges, weights=w)[0]

    def add_neutrals(self, weight: np.ndarray, v_z: np.ndarray) -> None:
        """Atoms drifting out of the plume — un-ionized propellant. They carry
        little momentum each but all of the mass flow that failed to ionize,
        so they matter for Isp far more than for thrust."""
        if len(weight) == 0:
            return
        self.neutral_weight += float(weight.sum())
        self.neutral_p_z += float((weight * v_z).sum())

    # --- derived quantities ----------------------------------------------

    def report(self, thruster: Thruster, elapsed: float) -> dict:
        """Thrust, Isp and beam quality over the accumulated window."""
        if elapsed <= 0 or self.weight <= 0:
            return {}
        m, e = self.mass, cst.elementary_charge

        T_ion = m * self.p_z / elapsed
        T_neutral = m * self.neutral_p_z / elapsed
        T = T_ion + T_neutral

        v_mean = self.p_z / self.weight                 # mean axial velocity
        v_speed = self.speed_weighted / self.weight     # mean speed
        V_beam = m * (2 * self.energy / self.weight) / (2 * e)
        cos_mean = self.cos_weighted / max(self.speed_weighted, 1e-30)

        centres = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        theta_mean = float(np.sum(centres * self.theta_hist)
                           / max(self.theta_hist.sum(), 1e-30))
        # 95% momentum cone: the half-angle containing 95% of the beam
        cum = np.cumsum(self.theta_hist) / max(self.theta_hist.sum(), 1e-30)
        theta_95 = float(np.interp(0.95, cum, centres))

        I_beam = e * self.weight / elapsed
        mdot = thruster.mdot
        mdot_beam = m * self.weight / elapsed

        return dict(
            thrust_mN=T * 1e3,
            thrust_ion_mN=T_ion * 1e3,
            thrust_neutral_mN=T_neutral * 1e3,
            Isp_s=T / (mdot * cst.g) if mdot else 0.0,
            v_exit_mean_kms=v_mean * 1e-3,
            v_speed_mean_kms=v_speed * 1e-3,
            V_beam_V=V_beam,
            voltage_utilization=V_beam / thruster.voltage,
            mass_utilization=mdot_beam / mdot if mdot else 0.0,
            divergence_mean_deg=np.degrees(theta_mean),
            divergence_95_deg=np.degrees(theta_95),
            divergence_efficiency=cos_mean,
            I_beam_A=I_beam,
        )

    def histograms(self):
        """(speed centres [km/s], speed weights), (angle centres [deg], weights)."""
        vc = 0.5 * (self.v_edges[:-1] + self.v_edges[1:]) * 1e-3
        tc = np.degrees(0.5 * (self.theta_edges[:-1] + self.theta_edges[1:]))
        return (vc, self.v_hist), (tc, self.theta_hist)
