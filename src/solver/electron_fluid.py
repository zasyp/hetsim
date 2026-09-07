# Fluid-electron solver — the driver that assembles blocks 1-6 into a
# self-consistent electron state. Given the magnetic field and the
# heavy-species densities (PlasmaState), it finds the electron temperature
# Te and the plasma potential phi that satisfy, simultaneously:
#
#   * current continuity across the lambda layers (block 4, solve_potential)
#   * the electron energy balance on the layers (block 6, electron_energy)
#
# The two are coupled — the potential's cross-field conductance depends on
# Te through the collision frequency, and the energy balance is driven by
# the ohmic heating j.E built from the potential's field — so they are
# solved by alternating Picard iteration (a Gummel loop) under relaxation
# until Te stops moving.
#
# Unit convention: SI everywhere EXCEPT Te / potentials, in eV / volts.

from dataclasses import dataclass

import numpy as np

from ..electron_liquid.default_plasm_params import omega_ce
from ..electron_liquid.collisions import (
    neutral_collision, ionization_collision, coulomb_collision,
    anomaly_collision, anomalous_alpha, electron_collision, wall_collision,
    ANOMALY_PROFILE_PRESETS,
)
from ..electron_liquid.mobility import (
    zeroB_mobility, hall_parameter, perp_mobility, perp_thermal_conductivity,
)
from ..electron_liquid.layer_potential import (
    node_weight, layer_conductance, layer_ionization_current,
)
from ..electron_liquid.solve_potential import (
    solve_potential, potential_on_grid, electric_field,
)
from ..electron_liquid.electron_energy import (
    ohmic_heating, node_thermal_weight, solve_electron_energy,
)
from ..electron_liquid import sheath_interaction as sheath
from ..electron_liquid import power_balance as power
from .geometry import LayerGeometry
from .state import PlasmaState


@dataclass
class SolverSettings:
    n_layers: int = 40
    # Marks et al., arXiv:2507.08113, Sec. IV A: "the four-parameter model
    # with alpha_anom = 1/16, beta_anom ~ 0.99, z_anom ~ 1.05, and
    # L_anom ~ 0.38 may be a good starting choice when simulating OTHER
    # thrusters at 300 V" -- which is what this is. The SPT-100 and H9
    # tuples are that paper's posterior medians for those two specific
    # thrusters and carry their geometry with them; borrowing one of them
    # puts the transport barrier where THAT thruster's exit plane was.
    anomaly_preset: str = "GENERIC_300V"
    background_pressure_torr: float = 0.0
    Te_anode: float = 4.0          # eV, Dirichlet at the anode layer
    Te_cathode: float = 3.0        # eV, Dirichlet at the cathode layer
    Te_init: float = 12.0          # eV, interior starting guess
    Te_floor: float = 0.5          # eV, keep collisions/mobility finite
    sheath_edge_factor: float = 0.5  # n_e(sheath edge) / n_e(bulk)
    # Effective cross-field / inter-field-line thermal conduction number.
    # It sets how strongly heat spreads from the ohmically-heated
    # acceleration layers to the wall- and anode-touching layers that shed
    # it; too small and an isolated hot layer runs away. ~35 lands the
    # placeholder SPT-70 at a peak Te ~25 eV (~0.08 V_d, the usual Hall
    # rule of thumb) — this is the main calibration knob and should be
    # pinned against data / a kinetic run.
    kappa_coeff: float = 35.0
    # Whether the anomalous collision frequency enters the PERPENDICULAR
    # THERMAL conductivity as well as the mobility. Brick, Roberts & Jorns
    # (AIAA 2025-0298, Sec. IV): "we do not include the anomalous collision
    # frequency in the perpendicular conductivity. Our previous work showed
    # that doing so would artificially lower the electron temperature."
    # nu_anom dominates nu, so leaving it in inflates kappa_perp several
    # fold and lets the hot layers dump their heat into the cold ones.
    # True reproduces the pre-2025 behaviour.
    anom_in_heat_flux: bool = False
    relax: float = 0.2             # Te under-relaxation
    max_iter: int = 300
    tol: float = 1e-3              # eV, max |dTe| for convergence
    thermal_floor: float = 1e-18   # W/V, tiny G^T so empty faces stay coupled


class FluidElectronSolver:
    """Self-consistent electron temperature + potential on a PlasmaState.

    solver = FluidElectronSolver(state)
    solver.solve()          # fills state.Te, state.phi, state.E_z/E_r, ...
    """

    def __init__(self, state: PlasmaState, settings: SolverSettings | None = None):
        self.state = state
        self.s = settings or SolverSettings()
        s = self.s
        st = state

        self.geom = LayerGeometry(st.grid, st.thruster, st.lam, s.n_layers)
        self.r = st.grid.r_nodes()
        self.z = st.grid.z_nodes()
        self.B = st.B
        self.omega = omega_ce(self.B)

        # static anomalous-transport profile alpha(z) (does not depend on Te)
        preset = ANOMALY_PROFILE_PRESETS[s.anomaly_preset]
        self.alpha = anomalous_alpha(
            self.z, st.thruster.channel_length,
            background_pressure_torr=s.background_pressure_torr, **preset,
        )[:, None] * np.ones(st.grid.N_r)

        # Nodes that actually see a wall: inside the channel annulus and
        # upstream of the exit plane. Everywhere else (the plume, and the
        # wedge inside r_min) has no wall, so nu_w must be zero there —
        # notably at the transport barrier, which sits just PAST the exit.
        th = st.thruster
        rr = self.r[None, :]
        zz = self.z[:, None]
        self.in_channel = ((zz <= th.channel_length)
                           & (rr >= th.r_min) & (rr <= th.r_max))
        self.channel_width = th.r_max - th.r_min

        # initial Te on layers: interior at Te_init, ends at the BCs
        self.Te_layers = np.full(s.n_layers, s.Te_init)
        self.Te_layers[0] = s.Te_anode
        self.Te_layers[-1] = s.Te_cathode

    # --- per-iteration pieces --------------------------------------------

    def _transport(self, Te_grid):
        """Collision frequency, cross-field mobility and thermal
        conductivity on the grid at the current Te."""
        st = self.state
        # electron-wall term: needs the sheath drop, which depends on Te,
        # so it is rebuilt every Gummel pass alongside the other channels
        gamma_w = sheath.see_yield(Te_grid, st.thruster.wall_material)
        nu_w = np.where(
            self.in_channel,
            wall_collision(Te_grid, gamma_w, st.gas.mass, self.channel_width),
            0.0,
        )
        nu_en = neutral_collision(Te_grid, st.n_n, st.gas)
        nu_iz = ionization_collision(Te_grid, st.n_n, st.gas)
        nu_ei = coulomb_collision(st.n_e, Te_grid)
        nu_an = anomaly_collision(self.B, self.alpha)

        nu = electron_collision(nu_en, nu_iz, nu_ei, nu_an, nu_w)
        mu = perp_mobility(zeroB_mobility(nu), hall_parameter(self.omega, nu))

        if self.s.anom_in_heat_flux:
            mu_q = mu
        else:                       # see SolverSettings.anom_in_heat_flux
            nu_q = electron_collision(nu_en, nu_iz, nu_ei, 0.0, nu_w)
            mu_q = perp_mobility(zeroB_mobility(nu_q),
                                 hall_parameter(self.omega, nu_q))
        kappa = perp_thermal_conductivity(st.n_e, Te_grid, mu_q, self.s.kappa_coeff)
        return nu, mu, kappa

    def _potential(self, mu, Te_grid):
        """Solve current continuity for phi* and reconstruct the field.

        Returns the full potential phi (with the along-line Boltzmann term,
        the ion-accelerating field reported to the user) AND the field of
        the thermalized potential phi* alone. Only phi* is constant along a
        field line, so its gradient is the pure cross-field field that
        drives — and ohmically heats — the perpendicular electron current;
        using the full phi there would feed the Boltzmann term's ~Te growth
        straight back into the heating and drive Te unstable.
        """
        st, g = self.state, self.geom
        w = node_weight(mu, st.n_e, st.Br, st.Bz, self.r, g.dV)
        G_face = layer_conductance(g.idx, self.s.n_layers, g.d_lambda, w)
        k_iz = st.gas.ionization_rate_Te(Te_grid)
        dI_iz = layer_ionization_current(g.idx, st.n_e, st.n_n, k_iz, g.dV, self.s.n_layers)
        phi_star = solve_potential(G_face, dI_iz, st.thruster.voltage, 0.0)

        # Raw cross-field current across the anode-side layer face [A].
        # Summing the continuity rows telescopes to
        #     I_face[-1] - I_face[0] = sum(dI_iz),
        # so the two boundary faces differ by the ion current born between
        # them. UNRESOLVED: which boundary (if either) is the discharge
        # current. Identifying I_d = I_beam + this was tried and FAILS —
        # it goes negative during the transient and yields I_beam/I_d > 1,
        # which is impossible. Getting I_d out of this formulation needs
        # the sign convention of the layer solve pinned down properly;
        # until then this is a raw number, not a discharge current.
        self.I_anode_electron = float(G_face[0] * (phi_star[0] - phi_star[1]))

        phi = potential_on_grid(phi_star, g.layer_lambda, st.lam, self.Te_layers, st.n_e)
        E_z, E_r = electric_field(phi, self.z, self.r)

        phi_star_grid = g.to_grid(phi_star)
        E_z_star, E_r_star = electric_field(phi_star_grid, self.z, self.r)
        return phi_star, phi, E_z, E_r, E_z_star, E_r_star

    def _losses(self, Te_grid, E_z, E_r, mu):
        """Per-layer ohmic source [W] and linearized loss coefficient
        [W/V] = (wall + ionization + radiation power at Te) / Te.
        E_z, E_r here are the CROSS-FIELD (phi*) components."""
        st, g, s = self.state, self.geom, self.s

        p_ohmic = ohmic_heating(st.n_e, mu, E_z, E_r)
        source = g.integrate_power(p_ohmic)

        # wall (sheath) loss -> W per layer
        gamma = sheath.see_yield(Te_grid, st.thruster.wall_material)
        phi_s = sheath.sheath_potential(Te_grid, gamma, st.gas)
        q_wall = sheath.wall_energy_loss(Te_grid, s.sheath_edge_factor * st.n_e, phi_s)
        W_wall = g.layer_wall_power(q_wall)

        # inelastic volumetric losses -> W per layer
        W_ion = g.integrate_power(power.ionization_power_density(Te_grid, st.n_n, st.n_e, st.gas))
        rad_density = (st.n_n * st.n_e * st.gas.k_exc(Te_grid)
                       * st.gas.E_exc * 1.602176634e-19)
        W_rad = g.integrate_power(rad_density)

        loss = W_wall + W_ion + W_rad
        Te_layers_safe = np.maximum(self.Te_layers, s.Te_floor)
        loss_coeff = loss / Te_layers_safe
        return source, loss_coeff, dict(P_ohmic=source, W_wall=W_wall,
                                        W_ion=W_ion, W_rad=W_rad)

    def _thermal_conductance(self, kappa):
        st, g, s = self.state, self.geom, self.s
        wT = node_thermal_weight(kappa, st.Br, st.Bz, self.r, g.dV)
        G_T = layer_conductance(g.idx, s.n_layers, g.d_lambda, wT)
        return G_T + s.thermal_floor

    # --- driver -----------------------------------------------------------

    def solve(self, verbose: bool = False) -> PlasmaState:
        s, g, st = self.s, self.geom, self.state
        last = None
        for it in range(s.max_iter):
            Te_grid = np.maximum(g.to_grid(self.Te_layers), s.Te_floor)

            nu, mu, kappa = self._transport(Te_grid)
            phi_star, phi, E_z, E_r, E_z_star, E_r_star = self._potential(mu, Te_grid)
            source, loss_coeff, budget = self._losses(Te_grid, E_z_star, E_r_star, mu)
            G_T = self._thermal_conductance(kappa)

            Te_new = solve_electron_energy(G_T, source, loss_coeff,
                                           s.Te_anode, s.Te_cathode)
            Te_new = np.maximum(Te_new, s.Te_floor)

            delta = np.max(np.abs(Te_new - self.Te_layers))
            self.Te_layers = (1 - s.relax) * self.Te_layers + s.relax * Te_new
            last = (phi_star, phi, E_z, E_r, budget)
            if verbose and (it % 20 == 0 or delta < s.tol):
                print(f"  it {it:3d}   max|dTe| = {delta:8.4f} eV   "
                      f"Te_peak = {self.Te_layers.max():6.2f} eV")
            if delta < s.tol:
                break

        phi_star, phi, E_z, E_r, budget = last  # full-phi field for output
        st.Te_layers = self.Te_layers
        st.phi_star = phi_star
        st.Te = g.to_grid(self.Te_layers)
        st.phi = phi
        st.E_z, st.E_r = E_z, E_r
        st.diagnostics = self._final_diagnostics(budget, it, delta)
        return st

    def _final_diagnostics(self, budget, iters, delta) -> dict:
        d = {k: v.sum() for k, v in budget.items()}
        d["iterations"] = iters + 1
        d["converged"] = bool(delta < self.s.tol)
        d["residual_eV"] = float(delta)
        d["Te_peak"] = float(self.Te_layers.max())
        d["I_anode_electron"] = float(getattr(self, "I_anode_electron", 0.0))
        d["z_layer"] = self.geom.z_layer
        d["Te_layers"] = self.Te_layers.copy()
        return d
