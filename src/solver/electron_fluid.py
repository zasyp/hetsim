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
    node_weight, layer_conductance, layer_conductance_profile,
    layer_ionization_current, layer_currents, layer_log_density,
    face_thermal_force,
)
from ..electron_liquid.solve_potential import (
    solve_potential, potential_on_grid, electric_field,
)
from ..electron_liquid.electron_energy import (
    ohmic_heating, node_thermal_weight, solve_electron_energy,
)
from ..electron_liquid import sheath_interaction as sheath
from ..electron_liquid import power_balance as power
from ..ions.moments import smooth
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
    # Anchor the anomalous-transport barrier on the magnetic field maximum
    # instead of on the nominal channel length.
    #
    # The barrier is a MAGNETIC feature — it is where the Hall parameter
    # peaks — and Eq. (3) only places it with z_anom because the thrusters
    # it was fitted to (SPT-100, H9) have their |B| maximum essentially at
    # the exit plane, so "1.05 channel lengths from the anode" and "just
    # downstream of the field peak" are the same statement there. They are
    # not the same statement here: this geometry peaks at z = 0.90 L, 3 mm
    # INSIDE the channel. Taking z_anom literally then parks the barrier —
    # and with it the ohmic heating and the whole potential drop — 4.5 mm
    # downstream of the field peak and outside the channel, where there is
    # no wall to carry the heat away. That put the Te peak past the exit
    # plane, which the internal probe maps contradict: Linnell & Gallimore
    # (IEPC-2005-024, Figs. 10, 14) find the hot region starting just
    # UPSTREAM of the acceleration zone. It also ran Te to 45-60 eV at 300 V.
    # Those same maps do read 47-60 eV -- but at 500 and 600 V, and they say
    # the value is a saturation set by wall losses ("the electron temperature
    # is anticipated to saturate near 50-60 eV due to discharge channel wall
    # losses"), not a number that scales with V_d. At 300 V the rule of thumb
    # Te_max ~ 0.1 V_d puts the expected peak near 30 eV, so sitting at the
    # wall-loss saturation ceiling at 300 V means the electrons are being
    # given too much power, not that the model found the measured value.
    #
    # So keep the paper's OFFSET (z_anom - 1) L relative to the field peak
    # and move the anchor. False takes z_anom literally.
    barrier_on_field_peak: bool = True
    background_pressure_torr: float = 0.0
    Te_anode: float = 4.0          # eV, Dirichlet at the anode layer
    Te_cathode: float = 3.0        # eV, Dirichlet at the cathode layer
    Te_init: float = 12.0          # eV, interior starting guess
    Te_floor: float = 0.5          # eV, keep collisions/mobility finite
    sheath_edge_factor: float = 0.5  # n_e(sheath edge) / n_e(bulk)
    # Cross-field / inter-field-line thermal conduction number in
    # kappa_perp = coeff * e * n_e * Te * mu_perp. Classical values: 5/2
    # from kinetic theory (each diffusing electron carries its enthalpy),
    # 4.7 from Braginskii's magnetized limit. This is the Braginskii one.
    #
    # It used to be 35 -- seven times classical -- and that was not a
    # transport argument, it was a patch for the missing convective term in
    # the energy equation. With conduction as the only transport channel,
    # the ohmic heating deposited at the barrier had nowhere to go except
    # sideways into the neighbouring layers, so the only way to keep Te off
    # 70 eV was to crank the conduction until the hump spread over the whole
    # channel. That bought the right PEAK and paid for it with the wrong
    # SHAPE. Now that the electron flow convects its own enthalpy
    # (electron_energy.solve_electron_energy), the physical number works.
    kappa_coeff: float = 4.7
    # Whether the heating source is the full field work j_e.E (True) or only
    # the resistive dissipation eta_perp j_e^2 = sigma_perp |E*|^2 (False).
    #
    # j_e.E is the one that belongs with the (5/2)Te convective flux the
    # energy equation now carries -- the derivation, and the size of the
    # difference, are in electron_energy.ohmic_heating. False is kept to
    # reproduce runs made before the convective term existed.
    ohmic_on_full_field: bool = True
    # Whether the cross-field electron current carries the thermal-force term
    # (ln(n_e/n_ref) - 1) dTe/dlambda alongside dphi*/dlambda -- Koo 2005
    # Eq. (2.62), layer_potential.face_thermal_force. It is not small: on the
    # reference state it is 58% of the electrostatic term on the first face
    # off the anode (the one the discharge current is read from) and of the
    # opposite sign and comparable size through the plume.
    #
    # The term is physically right, and there is a clean check that it is:
    # with it, the face current is GAUGE INVARIANT. Rescaling n_ref by c
    # shifts phi* by Te*ln(c), which differs from layer to layer because Te
    # does, and f shifts by -ln(c); the two cancel exactly in
    # G*(dphi* + f*dTe). Without the term the current depends on the gauge.
    #
    # OFF by default because it destabilizes the coupled SEED STARTUP. Same
    # seed with and without it track each other to 1.9 us (I ~ 11-13 A);
    # then the run without it damps its overshoot to 0.5 A while the run with
    # it climbs to 65 A and 1.5M ions by 5.7 us. The growth is inside the
    # channel: the sparse anode layer is the gauge, so f saturates at the
    # clamp (+2) almost everywhere, dTe < 0 there makes the EMF oppose the
    # downstream current, the solve pulls potential drop into the channel
    # where the conductance is huge, the current rises, Te rises, dTe grows.
    # (Zeroing the two Dirichlet faces removes a real artifact but not this.)
    # Whether it is stable once the startup transient is over is open.
    thermal_force: bool = False
    # Whether the anomalous collision frequency enters the PERPENDICULAR
    # THERMAL conductivity as well as the mobility. Brick, Roberts & Jorns
    # (AIAA 2025-0298, Sec. IV): "we do not include the anomalous collision
    # frequency in the perpendicular conductivity. Our previous work showed
    # that doing so would artificially lower the electron temperature."
    # nu_anom dominates nu, so leaving it in inflates kappa_perp several
    # fold and lets the hot layers dump their heat into the cold ones.
    # True reproduces the pre-2025 behaviour.
    anom_in_heat_flux: bool = False
    # Anchor the thermalized-potential gauge on the anode layer density, so
    # that the physical phi equals V_d where solve_potential pins phi*.
    # False restores the old max(n_e) reference.
    anode_gauge: bool = True
    # Cap |ln(n_e/n_ref)| at this many e-foldings, i.e. the Boltzmann term at
    # this many Te. Past a few Te the Maxwellian tail that the relation
    # assumes is exhausted (a fraction exp(-k) of the population), so the
    # relation stops holding; None disables the cap.
    boltzmann_clamp: float | None = 3.0
    # Extra binomial smoothing passes applied to n_e BEFORE it enters the
    # Boltzmann term of potential_on_grid — and nowhere else.
    #
    # Everything else the solver does with n_e sums it over a layer, so the
    # PIC shot noise averages away. The Boltzmann term is the one place the
    # raw per-node density survives into a field the IONS are pushed by, and
    # taking its gradient turns noise into force: a 5% density wobble across
    # one cell at Te = 40 eV is 2 V over 0.25 mm, i.e. 8 kV/m — the same
    # order as the real acceleration field. The ions do not just rattle,
    # they random-walk in energy, because the noise is redrawn every
    # macro-step; a few microseconds of that sprays them into the walls.
    #
    # The term is a fluid closure and only means anything on scales the
    # fluid model resolves, so filtering it to a few cells is not a cosmetic
    # smoother — it is the term's own domain of validity. The along-line
    # density variation it exists to capture spans the channel width, tens
    # of cells, and passes straight through.
    boltzmann_smooth: int = 4
    # Te under-relaxation. The loop is Newton in the losses and Picard in
    # everything else (transport, ohmic source, face currents), so the
    # remaining stiffness is the Picard half and 0.5 is about where that
    # stops damping: 0.5 converges the placeholder state in 14 iterations,
    # 0.7 and 1.0 do not converge at all. It was 0.15 while the losses were
    # linearized on the secant, which needed the extra damping and paid for
    # it with four times the iterations.
    relax: float = 0.5
    max_iter: int = 300
    # eV, max |dTe| for convergence. 1e-2, not the old 1e-3: Te itself moves
    # by an eV or more between macro-steps of the hybrid loop, so resolving
    # the Gummel fixed point to a thousandth of an eV bought nothing and
    # doubled the iteration count.
    tol: float = 1e-2
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
        L = st.thruster.channel_length
        self.z_field_peak = self._field_peak()
        # shifting the query point by (L - z_Bmax) slides the whole Eq. (3)
        # profile so its barrier lands at z_Bmax + (z_anom - 1) L
        z_query = self.z + (L - self.z_field_peak if s.barrier_on_field_peak else 0.0)
        self.alpha = anomalous_alpha(
            z_query, L,
            background_pressure_torr=s.background_pressure_torr, **preset,
        )[:, None] * np.ones(st.grid.N_r)
        anchor = self.z_field_peak if s.barrier_on_field_peak else L
        self.z_barrier = float(anchor + (preset["z_anom"] - 1.0) * L)

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

        # Face currents, filled by _potential on every pass. None on the
        # first call means the energy equation runs conduction-only for one
        # iteration, which is the right cold start: there is no potential
        # solution yet to draw a flow direction from.
        self.I_face = None

        # initial Te on layers: interior at Te_init, ends at the BCs
        self.Te_layers = np.full(s.n_layers, s.Te_init)
        self.Te_layers[0] = s.Te_anode
        self.Te_layers[-1] = s.Te_cathode

    def _field_peak(self) -> float:
        """Axial position of the |B| maximum on the mid-channel radius [m].

        Read on one radius on purpose: the field also spikes at the pole
        tips, where the plasma is not, and a global argmax would anchor the
        barrier on the corner singularity instead of on the channel.
        """
        th = self.state.thruster
        j = int(np.argmin(np.abs(self.r - 0.5 * (th.r_min + th.r_max))))
        return float(self.z[int(np.argmax(self.B[:, j]))])

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

        # Gauge of phi*, shared by the thermal-force factor below and by the
        # Boltzmann term of potential_on_grid further down: both are written
        # against the SAME reference density or they are not talking about
        # the same phi*.
        n_ref = float(g.average(st.n_e)[0]) if self.s.anode_gauge else None

        # Thermal-force term of the cross-field Ohm's law (Koo 2005 Eq. 2.62):
        # the phi* derivation holds Te constant ALONG a field line, which is
        # what the lambda reduction is for, but nothing holds it constant
        # ACROSS lines, and Te moves an eV or more between layers.
        emf = None
        if self.s.thermal_force and n_ref:
            G_layer = layer_conductance_profile(g.idx, self.s.n_layers,
                                                g.d_lambda, w)
            ln_n = layer_log_density(g.idx, st.n_e, n_ref, w, self.s.n_layers,
                                     clamp=self.s.boltzmann_clamp)
            emf = face_thermal_force(ln_n, G_layer, G_face, self.Te_layers)

        phi_star = solve_potential(G_face, dI_iz, st.thruster.voltage, 0.0,
                                   emf_face=emf)
        self._phi_star = phi_star

        # Electron conduction current across every layer face [A], positive
        # downstream. The energy equation convects (5/2)Te with it, and its
        # anode-side value is the electron back-current the anode collects.
        # Summing the continuity rows telescopes to
        #     I_face[0] - I_face[-1] = sum(dI_iz),
        # anode collection = cathode supply + total ionization, so I_face[0]
        # IS a discharge current once the ion current to the anode is added
        # to it (hybrid.py does that). This was marked UNRESOLVED while
        # solve_potential carried the ion source with the wrong sign, which
        # made I_face[0] come out near zero — and negative in the transient,
        # which is what made I_beam/I_d > 1.
        # kept for the post-mortem dump: the layer resistance profile is the
        # only way to see WHERE the discharge current is set, and it cannot
        # be reconstructed from the saved fields alone.
        self.G_face = G_face
        self.dI_iz = dI_iz
        self.I_face = layer_currents(G_face, phi_star, emf_face=emf)
        self.I_anode_electron = float(self.I_face[0])
        self.I_ionization = float(dI_iz.sum())

        # n_ref (the gauge, anchored on the anode layer so that phi = V_d
        # there, which is the boundary condition solve_potential imposes on
        # phi*) was taken above, before the thermal-force term that shares it.
        n_boltz = (smooth(st.n_e, self.s.boltzmann_smooth)
                   if self.s.boltzmann_smooth else st.n_e)
        phi = potential_on_grid(phi_star, g.layer_lambda, st.lam,
                                self.Te_layers, n_boltz, n_ref=n_ref,
                                boltzmann_clamp=self.s.boltzmann_clamp)
        E_z, E_r = electric_field(phi, self.z, self.r)

        phi_star_grid = g.to_grid(phi_star)
        E_z_star, E_r_star = electric_field(phi_star_grid, self.z, self.r)
        return phi_star, phi, E_z, E_r, E_z_star, E_r_star

    def _loss_terms(self, Te_layers):
        """The three electron energy sinks, per layer [W], evaluated at a
        given layer temperature: wall sheath, ionization, radiation."""
        st, g, s = self.state, self.geom, self.s
        Te_grid = np.maximum(g.to_grid(Te_layers), s.Te_floor)

        gamma = sheath.see_yield(Te_grid, st.thruster.wall_material)
        phi_s = sheath.sheath_potential(Te_grid, gamma, st.gas)
        q_wall = sheath.wall_energy_loss(Te_grid, s.sheath_edge_factor * st.n_e, phi_s)
        W_wall = g.layer_wall_power(q_wall)

        W_ion = g.integrate_power(
            power.ionization_power_density(Te_grid, st.n_n, st.n_e, st.gas))
        rad_density = (st.n_n * st.n_e * st.gas.k_exc(Te_grid)
                       * st.gas.E_exc * 1.602176634e-19)
        W_rad = g.integrate_power(rad_density)
        return W_wall, W_ion, W_rad

    def _losses(self, E_z, E_r, mu, E_z_phi=None, E_r_phi=None):
        """Per-layer heating source [W], the loss at the current Te [W] and
        its derivative dLoss/dTe [W/V]. E_z, E_r are the CROSS-FIELD (phi*)
        components, which are what drives the current; E_z_phi, E_r_phi are
        the full electrostatic ones, and the source is j_e.E across the two
        (see electron_energy.ohmic_heating on why it is not j_e.E*).

        The derivative is the point. The losses are violently nonlinear in
        Te -- the wall term goes as Te^(3/2) times a sheath factor whose SEE
        yield is itself climbing a power law, so d(ln W_wall)/d(ln Te) runs
        to 4 and beyond near the space-charge limit. Linearizing them the
        cheap way, as loss = (loss(T0)/T0) * Te, uses the SECANT through the
        origin and so understates that slope several-fold. The energy solve
        then over-corrects, the Gummel loop answers by over-correcting back,
        and the pair settles into a two-cycle that under-relaxation only
        just fails to damp: on the placeholder state the hottest layer sat
        there swinging between 22 and 35 eV forever.

        Taking the real tangent instead -- one extra evaluation of the loss
        terms at Te*(1+eps) -- puts the true dLoss/dTe on the diagonal, which
        is a Newton step in the loss and kills the cycle.
        """
        st, g, s = self.state, self.geom, self.s

        if not s.ohmic_on_full_field:
            E_z_phi = E_r_phi = None
        p_ohmic = ohmic_heating(st.n_e, mu, E_z, E_r, E_z_phi, E_r_phi)
        source = g.integrate_power(p_ohmic)

        Te0 = np.maximum(self.Te_layers, s.Te_floor)
        W_wall, W_ion, W_rad = self._loss_terms(Te0)
        loss = W_wall + W_ion + W_rad

        eps = 1e-2
        loss_hi = sum(self._loss_terms(Te0 * (1.0 + eps)))
        # clip at zero: the losses only ever grow with Te, and a negative
        # slope from finite-difference noise would break the diagonal
        dloss = np.maximum((loss_hi - loss) / (eps * Te0), 0.0)

        return source, loss, dloss, Te0, dict(P_ohmic=source, W_wall=W_wall,
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
            source, loss, dloss, Te0, budget = self._losses(
                E_z_star, E_r_star, mu, E_z, E_r)
            G_T = self._thermal_conductance(kappa)

            Te_new = solve_electron_energy(G_T, source, loss, dloss, Te0,
                                           s.Te_anode, s.Te_cathode,
                                           I_face=self.I_face)
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
        d["I_ionization"] = float(getattr(self, "I_ionization", 0.0))
        d["I_face"] = np.asarray(getattr(self, "I_face", [])).copy()
        d["G_face"] = np.asarray(getattr(self, "G_face", [])).copy()
        d["dI_iz"] = np.asarray(getattr(self, "dI_iz", [])).copy()
        d["phi_star_layers"] = np.asarray(getattr(self, "_phi_star", [])).copy()
        d["z_layer"] = self.geom.z_layer
        d["z_barrier"] = float(self.z_barrier)
        d["z_field_peak"] = float(self.z_field_peak)
        d["Te_layers"] = self.Te_layers.copy()
        return d
