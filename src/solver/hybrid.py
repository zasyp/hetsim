# Hybrid solver: kinetic heavy species + fluid electrons, coupled.
#
# This is the loop the whole project was built toward. Blocks 1-6 gave a
# self-consistent electron temperature and potential FOR A GIVEN plasma
# density; that density was PlasmaState.placeholder, a prescribed profile.
# Here it stops being prescribed:
#
#     electrons (fluid)  --E, Te-->  ions & neutrals (particles)
#            ^                                    |
#            +--------- n_e = n_i, n_n -----------+
#
# Nothing in the loop is new physics — every piece already existed. What is
# new is that the arrow now closes.
#
# --- Two time steps -------------------------------------------------------
#
# The only thing in this model that needs a small step is the ion push: an ion
# that fell through 300 V crosses a 0.5 mm cell in 24 ns, so dt ~ 1e-8 s (see
# ions.cfl_dt). Everything else is slow:
#
#   neutrals    240 m/s — they cross a cell in 2 us, 200 ion steps
#   ionization  nu = n_e k_iz ~ 1e4 1/s, so nu*dt << 1 for any dt we use
#   electrons   have no time derivative at all; the fluid model is quasi-static
#
# So the loop runs on two cadences. The ion push repeats n_sub times at dt;
# injection, the neutral push, ionization and the electron solve happen once
# per macro-step at dt_slow = n_sub * dt. With n_sub = 50 the neutral moves
# 0.1 mm per macro-step, a fifth of a cell — still well resolved — and the
# expensive parts run 50 times less often.
#
# --- What to expect -------------------------------------------------------
#
# Do not expect a steady state. Once ionization is self-consistent the
# discharge breathes: the plasma eats the neutrals faster than the anode
# refills them, ionization collapses, the channel refills, and it repeats at
# 10-30 kHz. That is the breathing mode, a real and well-documented feature of
# Hall thrusters (Boeuf & Garrigues 1998), not a numerical instability. Any
# comparison with experiment has to be made on cycle-averaged quantities.

from dataclasses import dataclass, field

import numpy as np
import scipy.constants as cst

from ..structs.classes import Grid2D, Thruster, ParticleArray
from ..neutrals.neutrals import node_volume, push_neutrals, apply_boundaries
from ..injection.inject import inject_on_grid
from ..electron_liquid.ionization import apply_ionization_weighted
from ..ions.ions import push_ions, apply_ion_boundaries, cfl_dt
from ..ions.moments import seed_uniform, seed_plasma_profile
from ..ions.beam import BeamTally
from .state import PlasmaState
from .electron_fluid import FluidElectronSolver, SolverSettings


@dataclass
class HybridSettings:
    """Knobs of the coupled run. The defaults are tuned for SPT-70 on the
    reference 121x101 grid; `weight` is the one to touch first if the run is
    too slow or too noisy."""

    weight: float = 1.4e10
    # Real ATOMS per neutral macroparticle. Sets the cost/noise trade: the
    # steady-state neutral inventory is mdot/m * residence ~ 1.4e15 atoms, so
    # this weight means ~1e5 neutral macroparticles, ~50 per channel cell,
    # ~14% density noise per cell before smoothing. Halving it doubles the
    # cost and cuts noise by 1.4.

    weight_ratio: float = 30.0
    # Ion weight = weight / weight_ratio. Ions need their own, smaller weight
    # because n_i is ~100x below n_n: at the shared weight the channel would
    # hold under one ion macroparticle per cell. See
    # ionization.apply_ionization_weighted for how the transfer stays
    # unbiased across the ratio.

    dt: float | None = None          # None -> ions.cfl_dt(grid, thruster)
    n_sub: int = 50                  # ion pushes per macro-step
    smooth_passes: int = 2           # binomial filter passes on the densities
    T_wall: float = 750.0            # K, wall/anode re-emission temperature

    seed_n_n: float = 2.5e19         # 1/m^3, initial channel neutral density
    seed_n_i: float = 2e17           # 1/m^3, initial seed plasma
    seed_T_i: float = 1000.0         # K, seed ion temperature

    electron: SolverSettings = field(default_factory=SolverSettings)


class HybridSolver:
    """One coupled run. Construct, then call run().

        hybrid = HybridSolver(thruster, grid)
        hybrid.run(macro_steps=400, verbose=True)
        hybrid.history          # per-macro-step diagnostics

    The state and the electron solver are built ONCE and reused: the layer
    geometry depends only on the magnetic field (fixed), and keeping the
    electron solver alive means every electron update warm-starts from the
    previous Te instead of from a cold guess. That turns a ~40-iteration
    Gummel solve into a handful.
    """

    def __init__(self, thruster: Thruster, grid: Grid2D,
                 settings: HybridSettings | None = None,
                 rng: np.random.Generator | None = None):
        self.thruster = thruster
        self.grid = grid
        self.s = settings or HybridSettings()
        self.rng = rng or np.random.default_rng(0)

        self.dt = self.s.dt or cfl_dt(grid, thruster)
        self.ion_weight = self.s.weight / self.s.weight_ratio
        self.dt_slow = self.dt * self.s.n_sub
        self.time = 0.0

        gas = thruster.propellant
        self.q_over_m = cst.elementary_charge / thruster.mass
        self.v_th_anode = gas.thermal_speed(thruster.temperature_anode)
        self.v_th_wall = gas.thermal_speed(self.s.T_wall)
        self.dV = node_volume(grid, thruster)

        self.neutrals = seed_uniform(grid, thruster, self.s.seed_n_n,
                                     self.s.weight, rng=self.rng)
        self.ions = seed_plasma_profile(grid, thruster, self.s.seed_n_i,
                                        self.ion_weight,
                                        temperature=self.s.seed_T_i,
                                        rng=self.rng)

        # what the seeds put in, so mass_balance can subtract it later
        self.seeded = float(self.neutrals.weight.sum() + self.ions.weight.sum())

        self.state = PlasmaState.from_particles(
            grid, thruster, gas, self.ions, self.neutrals,
            smooth_passes=self.s.smooth_passes, dV=self.dV)
        self.electrons = FluidElectronSolver(self.state, self.s.electron)
        self.electrons.solve()          # first field, so step 1 has E and Te

        self.history: list[dict] = []
        self.totals: dict[str, float] = {}
        self.totals_at_reset: dict[str, float] = {}
        # beam moments accumulate from t=0; call reset_beam() after the
        # startup transient so thrust is measured on the settled discharge
        self.beam = BeamTally(thruster.mass)
        self.beam_t0 = 0.0

    # --- the two cadences -------------------------------------------------

    def _ion_substep(self) -> dict:
        """One ion push at dt: kick by E, drift, resolve surfaces.

        The recombined atoms go straight back into the neutral population —
        that flux is a real part of the neutral density near the anode, not a
        book-keeping detail.
        """
        if len(self.ions) == 0:
            return {}
        z_prev = self.ions.z.copy()      # copy, or it is a view of ions.z
        push_ions(self.ions, self.grid, self.state.E_z, self.state.E_r,
                  self.q_over_m, self.dt)
        alive, born, beam, tally = apply_ion_boundaries(
            self.ions, self.thruster, self.grid, z_prev,
            self.v_th_anode, self.v_th_wall)
        self.ions.keep(alive)
        if len(born):
            self.neutrals.extend(born)
        self.beam.add_ions(beam)
        return tally

    def _heavy_step(self) -> int:
        """Injection, the neutral push and ionization, all at dt_slow."""
        s = self.s
        self.neutrals.extend(
            inject_on_grid(self.dt_slow, self.thruster, s.weight))

        if len(self.neutrals):
            z_prev = self.neutrals.z.copy()
            push_neutrals(self.neutrals, self.dt_slow)
            alive = apply_boundaries(self.neutrals, self.thruster, self.grid,
                                     self.v_th_wall, z_prev)
            # atoms that drift out of the plume are a sink too — un-ionized
            # propellant leaving the thruster. Counting them is what makes the
            # mass balance close; without it the check reports a 25% "leak"
            # after a few hundred microseconds, which is simply this flux.
            gone = float(self.neutrals.weight[~alive].sum())
            self.beam.add_neutrals(self.neutrals.weight[~alive],
                                   self.neutrals.v_z[~alive])
            self.totals["neutral_escape"] = (
                self.totals.get("neutral_escape", 0.0) + gone)
            self.neutrals.keep(alive)

        return apply_ionization_weighted(
            self.neutrals, self.ions, self.grid, self.state.Te, self.state.n_e,
            self.thruster.propellant, self.dt_slow, self.ion_weight,
            rng=self.rng)

    def _electron_update(self) -> None:
        self.state.update_densities(self.ions, self.neutrals,
                                    smooth_passes=self.s.smooth_passes,
                                    dV=self.dV)
        self.electrons.solve()

    # --- driver -----------------------------------------------------------

    def macro_step(self) -> dict:
        """One dt_slow of the coupled system."""
        n_events = self._heavy_step()

        tally = {}
        for _ in range(self.s.n_sub):
            for k, v in self._ion_substep().items():
                tally[k] = tally.get(k, 0.0) + v
        for k, v in tally.items():
            self.totals[k] = self.totals.get(k, 0.0) + v

        self._electron_update()
        self.time += self.dt_slow

        e = cst.elementary_charge
        # currents [A]: weight is real particles, so weight*e/dt is amperes.
        # The channel walls and the front face are reported apart. They are
        # different failures: ions on the channel walls are the erosion and
        # efficiency loss the model is supposed to predict, while ions on
        # the front face are plume ions that turned around and came back,
        # i.e. a divergence problem. Adding them into one "I_wall" hid which
        # of the two was large.
        I_wall = e * (tally.get("inner_wall", 0.0)
                      + tally.get("outer_wall", 0.0)) / self.dt_slow
        I_front = e * tally.get("front_face", 0.0) / self.dt_slow
        I_anode_ion = e * tally.get("anode", 0.0) / self.dt_slow
        # Discharge current: what the supply pushes through the anode, ions
        # landing on it plus the electron back-current the fluid solve now
        # reports (see solve_potential on why that number used to be junk).
        I_anode_electron = float(self.state.diagnostics["I_anode_electron"])
        rec = dict(
            time=self.time,
            n_ion_macro=len(self.ions),
            n_neutral_macro=len(self.neutrals),
            I_beam=e * tally.get("beam", 0.0) / self.dt_slow,
            I_anode=I_anode_ion,
            I_wall=I_wall,
            I_front=I_front,
            I_iz=e * n_events * self.ion_weight / self.dt_slow,
            I_e_anode=I_anode_electron,
            I_d=I_anode_electron + I_anode_ion,
            Te_peak=float(self.state.diagnostics["Te_peak"]),
            n_e_peak=float(self.state.n_e.max()),
            n_n_anode=float(self.state.n_n[0].mean()),
            iterations=int(self.state.diagnostics["iterations"]),
        )
        self.history.append(rec)
        return rec

    def run(self, macro_steps: int, verbose: bool = False,
            every: int = 20, callback=None) -> list[dict]:
        """Advance the coupled system. callback(self, rec), if given, runs
        after every macro-step — that is the hook a time-average has to use,
        because in a breathing discharge every instantaneous field is a
        snapshot of one phase of the cycle and means nothing on its own."""
        for i in range(macro_steps):
            rec = self.macro_step()
            if callback is not None:
                callback(self, rec)
            if verbose and (i % every == 0 or i == macro_steps - 1):
                print(f"  t = {rec['time']*1e6:7.2f} us   "
                      f"ions {rec['n_ion_macro']:7d}  neut {rec['n_neutral_macro']:7d}   "
                      f"I_iz {rec['I_iz']:5.2f}  I_beam {rec['I_beam']:5.2f}  "
                      f"I_wall {rec['I_wall']:5.2f}  I_d {rec['I_d']:5.2f} A   "
                      f"Te {rec['Te_peak']:5.2f} eV   ({rec['iterations']} it)",
                      flush=True)
        return self.history

    def reset_beam(self) -> None:
        """Start the thrust window here. Call it once the startup transient is
        over: moments accumulated while the seed plasma is flushing describe
        the seed, not the thruster."""
        self.beam = BeamTally(self.thruster.mass)
        self.beam_t0 = self.time
        # surface tallies are cumulative from t=0; remember where the window
        # starts so the wall-loss profile can be differenced onto it too
        self.totals_at_reset = {k: (v.copy() if hasattr(v, "copy") else v)
                                for k, v in self.totals.items()}

    def performance(self) -> dict:
        """Thrust, Isp, exit velocity and divergence over the beam window."""
        return self.beam.report(self.thruster, self.time - self.beam_t0)

    # --- checks -----------------------------------------------------------

    def mass_balance(self) -> dict:
        """Where the propellant went, in real particles.

        Everything supplied — injected by the anode plus what the seeds put
        in — must sit in one of two places: still in the domain (as an atom or
        an ion), or gone through the plume. The plume sink has TWO channels,
        and both must be counted: ions as the beam, and un-ionized atoms
        drifting out. The walls and the anode are not sinks — they hand the
        atom back. If this does not close, a boundary is deleting particles it
        should recombine, or a sink is going uncounted.
        """
        injected = (self.thruster.mdot / self.thruster.mass) * self.time
        in_domain = float(self.neutrals.weight.sum() + self.ions.weight.sum())
        escaped = (self.totals.get("beam", 0.0)
                   + self.totals.get("neutral_escape", 0.0))
        residual = in_domain + escaped - injected - self.seeded
        supplied = injected + self.seeded
        return dict(injected=injected, seeded=self.seeded, in_domain=in_domain,
                    beam=self.totals.get("beam", 0.0),
                    neutral_escape=self.totals.get("neutral_escape", 0.0),
                    escaped=escaped, residual=residual,
                    rel_error=abs(residual) / supplied if supplied else 0.0)
