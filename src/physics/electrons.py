"""Fluid electron model of the hybrid Hall thruster code (thesis Secs.
2.2.2, 2.5, 2.6).

The static magnetic field reduces the 2-D problem to 1-D: electrons
equilibrate almost instantly along magnetic field lines (contours of the
streamfunction lambda), so every field line carries a single thermalized
potential (Morozov, eq. 2.13)

    phi*(lambda) = phi - (kB Te / e) ln(n_e / n_ref)                (2.33)

and a single electron energy eps(lambda). The `ElectronFluid` class
discretizes the channel + near-field plasma into `n_lines` lambda-slices
between the anode line lambda_a and the cathode line lambda_c and, once per
heavy-particle step, performs:

1. **Mobility** (2.46-2.50): classical cross-field mobility from
   electron-neutral collisions nu_en = n_a * 2.5e-13, augmented by Bohm
   (alpha_B * omega_Be) and/or wall (alpha_w * 1e7) anomalous collisionality.

2. **1-D Ohm's law / current conservation** (2.40-2.44): the total current
   I_T through every field line is equal, which yields a closed form for
   I_T given phi*(lambda_a) - phi*(lambda_c); back-substitution gives
   d(phi*)/d(lambda) per line (2.42) and hence phi*(lambda).
   Surface integrals along a field line are evaluated as volume sums over
   the nodes of the corresponding lambda-slice using dS = rB dV / d(lambda)
   (|grad lambda| = rB, eq. 2.7).

3. **Potential and field**: phi on every node from (2.33), smoothed, then
   E = -grad(phi) (2.45), zeroed inside the solid thruster body.

4. **Electron energy** (Sec. 2.6): explicit, subcycled update of eps per
   slice with ohmic heating j_e.E (2.53), inelastic losses to neutrals and
   ions and sheath wall losses (2.105-2.107), plus advection (5/3 eps u_e)
   and electron heat diffusion (10 mu eps / 9 * grad eps) along the lambda
   coordinate. This is a finite-volume simplification of the full
   volume-integrated discretization (2.97-2.104): same physics terms, same
   slice geometry, but assembled as a standard advection-diffusion-source
   update instead of the A1..A16 coefficient bookkeeping.

Electron temperature/energy conversion: eps [eV] = 3/2 kB Te / e (2.58).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.classes import Grid2D
from physics.neutrals import node_volumes
from physics.ionization import IonizationRates, default_rates
from numerical.numerical_funcs import smoothing, bilinear_interp

E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
KB = 1.380649e-23


def lambda_at(g: Grid2D, lam: np.ndarray, r0: float, z0: float) -> float:
    """Streamfunction value at an arbitrary point -- helper for picking the
    anode/cathode field lines (e.g. mid-channel at the anode plane and at the
    cathode plane)."""
    return bilinear_interp(lam, g.r_nodes(), g.z_nodes(), r0, z0)


class ElectronFluid:
    """Field-line-averaged electron fluid: potential, electric field and
    electron energy solver. Construct once, then call `solve()` every step.

    Parameters
    ----------
    g, Br, Bz, lam : grid, magnetic field components and streamfunction on
        the nodes (from `physics.magnetic`).
    plasma_mask : boolean (N_r, N_z) mask of plasma nodes
        (`ChannelGeometry.plasma_mask`).
    lambda_anode, lambda_cathode : streamfunction values of the virtual anode
        and cathode lines; the Ohm's law is solved between them.
    V_discharge : anode potential relative to the cathode line, V.
    n_ref : reference density of the thermalized potential (2.33), 1/m^3.
    n_floor : lower clamp on n_e in the Ohm solve; keeps the startup (no ions
        yet) well conditioned.
    eps_init, eps_cathode : initial slice energy and the fixed cathode-side
        boundary energy, eV.
    alpha_bohm, alpha_wall_mob : anomalous collisionality coefficients of
        (2.50) and (2.49); defaults give Bohm-1/16 and no wall term.
        `alpha_bohm` may also be a node field (N_r, N_z) for the standard
        two-zone anomalous transport (low inside the channel, Bohm outside).
    alpha_wall_energy, U_loss : wall energy loss model (2.107).
        `alpha_wall_energy` may also be a node field, e.g. nonzero only
        inside the channel where there are walls to lose energy to; it is
        volume-averaged per lambda-slice.
    rates : IonizationRates for the inelastic loss coefficients.
    """

    def __init__(
            self,
            g: Grid2D,
            Br: np.ndarray,
            Bz: np.ndarray,
            lam: np.ndarray,
            plasma_mask: np.ndarray,
            lambda_anode: float,
            lambda_cathode: float,
            n_lines: int = 25,
            V_discharge: float = 300.0,
            n_ref: float = 1.0e12,
            n_floor: float = 1.0e14,
            eps_init: float = 5.0,
            eps_cathode: float = 3.0,
            alpha_bohm: float | np.ndarray = 1.0 / 16.0,
            alpha_wall_mob: float = 0.0,
            alpha_wall_energy: float | np.ndarray = 0.4,
            U_loss: float = 20.0,
            rates: IonizationRates | None = None,
            subcycle_max: int = 200,
            eps_ceiling: float = 100.0,
            ):
        self.g = g
        self.Br = Br
        self.Bz = Bz
        self.mask = plasma_mask
        self.n_lines = n_lines
        self.V_d = V_discharge
        self.n_ref = n_ref
        self.n_floor = n_floor
        self.eps_cathode = eps_cathode
        self.alpha_bohm = alpha_bohm
        self.alpha_wall_mob = alpha_wall_mob
        self.alpha_wall_energy = alpha_wall_energy
        self.U_loss = U_loss
        self.rates = rates if rates is not None else default_rates()
        self.subcycle_max = subcycle_max
        self.eps_ceiling = eps_ceiling

        R, _ = g.mesh()
        self.B = np.hypot(Br, Bz)
        self.rB = np.maximum(R * self.B, 1e-12)  # |grad lambda|, eq. 2.7
        self.volumes = node_volumes(g)

        # lambda-slice binning between the anode and cathode lines. The
        # normalized coordinate x = (lam - lam_a)/(lam_c - lam_a) runs 0 -> 1
        # from anode to cathode regardless of the sign of the streamfunction.
        self.dlam = (lambda_cathode - lambda_anode) / n_lines  # signed
        x = (lam - lambda_anode) / (lambda_cathode - lambda_anode)
        self.bin_of_node = np.clip(
            np.floor(x * n_lines).astype(int), 0, n_lines - 1)
        self.lam_centers = lambda_anode + (np.arange(n_lines) + 0.5) * self.dlam

        # Nodes whose lambda falls outside the anode-cathode span (e.g. the
        # near-axis plume or the outer plume beyond the anode field line) do
        # not belong to any anode-cathode field line: they must not contribute
        # to the Ohm's-law / energy integrals, and the slice-to-node maps
        # (potential, energy) are not valid there either. Such nodes -- and
        # the solid body -- take the value of the nearest solved plasma node
        # instead; the index map for that fill is static, precompute it.
        # (Clipping them into the edge bins, as the bin map alone would do,
        # paints the whole outer plume with the *anode* potential and energy:
        # a large phi ~ V_d artifact on the potential map and a spurious
        # ionization avalanche where cold plume neutrals meet the anode-slice
        # electron energy.)
        self._solve_mask = plasma_mask & (x >= 0.0) & (x <= 1.0)
        self._fill_idx = tuple(ndimage.distance_transform_edt(
            ~self._solve_mask, sampling=(g.h_r, g.h_z),
            return_distances=False, return_indices=True))

        # Precomputed per-slice geometry over the solved plasma nodes.
        flat_bins = self.bin_of_node[self._solve_mask]
        self._flat_bins = flat_bins
        self.V_bin = np.maximum(
            np.bincount(flat_bins, self.volumes[self._solve_mask], n_lines),
            1e-300)
        rB_bin = np.bincount(
            flat_bins, (self.rB * self.volumes)[self._solve_mask],
            n_lines) / self.V_bin
        # Arclength of the slice along the normal coordinate: ds = dlam / rB.
        self.ds_bin = np.abs(self.dlam) / np.maximum(rB_bin, 1e-12)

        # Per-slice wall energy loss coefficient: volume-average of the
        # (possibly node-resolved) alpha_wall_energy over each slice.
        aw = np.broadcast_to(
            np.asarray(alpha_wall_energy, dtype=np.float64), self.rB.shape)
        self._aw_bin = self._bin_sum(aw * self.volumes) / self.V_bin

        # State and outputs.
        self.eps = np.full(n_lines, eps_init)
        self.phi = np.zeros((g.N_r, g.N_z))
        self.Er = np.zeros_like(self.phi)
        self.Ez = np.zeros_like(self.phi)
        self.I_T = 0.0
        self.ne_used = np.zeros_like(self.phi)

    # -- helpers ---------------------------------------------------------

    def _bin_sum(self, node_field: np.ndarray) -> np.ndarray:
        """Sum a node field over the solved plasma nodes of every slice."""
        return np.bincount(
            self._flat_bins, node_field[self._solve_mask], self.n_lines)

    def _surface_weight(self) -> np.ndarray:
        """dS weights: integral over a field line ~ sum f * rB * V / |dlam|."""
        return self.rB * self.volumes / np.abs(self.dlam)

    def mobility(self, n_a: np.ndarray, eps_nodes: np.ndarray) -> np.ndarray:
        """Cross-field electron mobility (2.46-2.50) on the nodes, m^2/(V*s)."""
        w_ce = E_CHARGE * self.B / M_E                      # (2.47)
        nu_m = (n_a * 2.5e-13                               # (2.48)
                + self.alpha_bohm * w_ce                    # (2.50)
                + self.alpha_wall_mob * 1.0e7)              # (2.49)
        nu_m = np.maximum(nu_m, 1.0)
        return (E_CHARGE / (M_E * nu_m)) / (1.0 + (w_ce / nu_m) ** 2)  # (2.46)

    # -- main entry point --------------------------------------------------

    def solve(
            self,
            n_e: np.ndarray,
            n_a: np.ndarray,
            n_i1: np.ndarray,
            ui_r1: np.ndarray,
            ui_z1: np.ndarray,
            n_i2: np.ndarray | None = None,
            ui_r2: np.ndarray | None = None,
            ui_z2: np.ndarray | None = None,
            dt: float = 1.0e-8,
            ) -> dict:
        """One electron update: mobility -> I_T -> phi* -> phi, E -> energy.

        Inputs are node fields deposited from the heavy particles: electron
        (= charge) density n_e, neutral density n_a, ion densities and mean
        velocities per charge state. Advances the slice energies self.eps by
        `dt` and refreshes self.phi, self.Er, self.Ez, self.I_T.
        """
        g = self.g
        nl = self.n_lines

        ne = np.where(self.mask, np.maximum(n_e, self.n_floor), 0.0)
        self.ne_used = ne

        # Slice-averaged electron temperature (K) and its lambda-gradient.
        Te_bin = (2.0 * self.eps / 3.0) * E_CHARGE / KB      # (2.58)
        dTe_dlam = np.gradient(Te_bin, self.lam_centers)

        eps_nodes = self.eps[self.bin_of_node]
        mu = self.mobility(n_a, eps_nodes)

        # --- per-slice Ohm's law integrals (2.42, 2.44) -------------------
        wS = self._surface_weight()
        ln_ne = np.where(self.mask, np.log(np.maximum(ne, 1.0) / self.n_ref), 0.0)

        G1 = self._bin_sum(E_CHARGE * ne * mu * self.rB * wS)
        # Relative conductivity floor: during startup, slices that hold only
        # the density floor are orders of magnitude more resistive than the
        # seeded ones, and the whole discharge voltage would drop (and
        # dissipate) across them, driving the energy solver into runaway.
        # Capping the conductivity contrast regularizes that transient and is
        # negligible once the plasma fills the anode-cathode span.
        G1 = np.maximum(G1, 1e-5 * np.max(G1) + 1e-30)
        G2 = self._bin_sum(E_CHARGE * ne * mu * self.rB * (ln_ne - 1.0) * wS)

        # Ion current through each line: u_perp = s * (Bz u_r - Br u_z)/B,
        # with s the sign that orients grad(lambda) from anode to cathode.
        s = np.sign(self.dlam)
        u_n1 = s * (self.Bz * ui_r1 - self.Br * ui_z1) / np.maximum(self.B, 1e-12)
        j_ion = E_CHARGE * n_i1 * u_n1
        if n_i2 is not None:
            u_n2 = s * (self.Bz * ui_r2 - self.Br * ui_z2) / np.maximum(self.B, 1e-12)
            j_ion = j_ion + 2.0 * E_CHARGE * n_i2 * u_n2
        Gi = self._bin_sum(j_ion * wS)

        term2 = (G2 / G1) * (KB / E_CHARGE) * dTe_dlam

        # Boundary thermalized potentials (2.33): anode at V_d, cathode at 0.
        ne_bin = self._bin_sum(ne * self.volumes) / self.V_bin
        ln_a = np.log(max(ne_bin[0], 1.0) / self.n_ref)
        ln_c = np.log(max(ne_bin[-1], 1.0) / self.n_ref)
        phi_star_a = self.V_d - (KB * Te_bin[0] / E_CHARGE) * ln_a
        phi_star_c = 0.0 - (KB * Te_bin[-1] / E_CHARGE) * ln_c

        # Total current from (2.44). I_lam is oriented along +lambda; the
        # physical discharge current (anode -> cathode positive) reported as
        # self.I_T differs by the sign s of dlam.
        S0 = np.sum(self.dlam / G1)
        S2 = np.sum(term2 * self.dlam)
        Si = np.sum((Gi / G1) * self.dlam)
        I_lam = -((phi_star_c - phi_star_a) + S2 - Si) / S0
        self.I_T = float(s * I_lam)

        # d(phi*)/d(lambda) per slice (2.42) and the phi* profile.
        dphi_dlam = -I_lam / G1 - term2 + Gi / G1
        phi_star = phi_star_a + (np.cumsum(dphi_dlam) - 0.5 * dphi_dlam) * self.dlam

        # --- 2-D potential and electric field (2.33, 2.45) ----------------
        Te_nodes = Te_bin[self.bin_of_node]
        phi = phi_star[self.bin_of_node] + (KB * Te_nodes / E_CHARGE) * ln_ne
        # The slice-to-node map is only valid on the solved anode-cathode
        # span; fill everything else (solid body, out-of-span plume) with the
        # nearest solved value so the gradient sees no artificial jump, then
        # smooth.
        phi = phi[self._fill_idx]
        phi = smoothing(phi)
        self.phi = phi

        r_nodes = g.r_nodes()
        z_nodes = g.z_nodes()
        self.Er = -np.gradient(phi, r_nodes, axis=0)
        self.Ez = -np.gradient(phi, z_nodes, axis=1)
        self.Er[~self.mask] = 0.0
        self.Ez[~self.mask] = 0.0

        # --- electron energy equation (Sec. 2.6, simplified FV form) ------
        # Node-level electron current density along the anode->cathode
        # normal (2.38) and the parallel-projected field for ohmic heating.
        dphi_dlam_nodes = dphi_dlam[self.bin_of_node]
        dTe_dlam_nodes = dTe_dlam[self.bin_of_node]
        j_e = (E_CHARGE * ne * mu * self.rB * s
               * (-dphi_dlam_nodes
                  - (ln_ne - 1.0) * (KB / E_CHARGE) * dTe_dlam_nodes))
        E_n = s * (self.Bz * self.Er - self.Br * self.Ez) / np.maximum(self.B, 1e-12)

        heat = self._bin_sum(j_e * E_n * self.volumes)          # W per slice
        neV = np.maximum(self._bin_sum(ne * self.volumes), 1e-300)
        na_bin = self._bin_sum(n_a * self.volumes) / self.V_bin
        ni_bin = self._bin_sum(n_i1 * self.volumes) / self.V_bin
        # Electron fluid velocity along the anode->cathode coordinate:
        # v_e = -j_e / (e n_e) -- electrons drift *against* the conventional
        # current, i.e. from the cathode side toward the anode, carrying the
        # ohmic heat of the exit region into the channel (not into the plume).
        u_bin = self._bin_sum((-j_e / (E_CHARGE * np.maximum(ne, 1.0)))
                              * self.volumes * self.mask) / self.V_bin
        # The drift used for energy advection cannot exceed the electron
        # thermal speed in this diffusive description.
        v_te = np.sqrt(2.0 * E_CHARGE * np.maximum(self.eps, 0.1) / M_E)
        u_bin = np.clip(u_bin, -v_te, v_te)
        mu_bin = self._bin_sum(mu * self.volumes) / self.V_bin

        eps = self.eps
        heating_rate = heat / (E_CHARGE * neV)                  # eV/s per slice

        # Explicit subcycled advection-diffusion-source update along the
        # slice arclength coordinate.
        ds = self.ds_bin
        D = 10.0 * mu_bin * np.maximum(eps, 0.1) / 9.0
        dt_stab = 0.4 * min(
            np.min(ds / np.maximum((5.0 / 3.0) * np.abs(u_bin), 1e-3)),
            np.min(ds**2 / np.maximum(2.0 * D, 1e-12)),
        )
        n_sub = int(np.clip(np.ceil(dt / dt_stab), 1, self.subcycle_max))
        dt_s = dt / n_sub

        for _ in range(n_sub):
            nu_loss = (na_bin * self.rates.loss_neutral(eps)
                       + ni_bin * self.rates.loss_ion(eps)
                       + self._aw_bin * 1.0e7
                       * np.exp(-self.U_loss / np.maximum(eps, 0.1)))  # 2.105-2.107

            # Upwind advection of eps by the electron drift (5/3 eps u term).
            deps = np.zeros_like(eps)
            adv = (5.0 / 3.0) * u_bin
            grad_up = np.where(
                adv > 0,
                (eps - np.roll(eps, 1)) / ds,
                (np.roll(eps, -1) - eps) / ds,
            )
            grad_up[0] = 0.0 if adv[0] > 0 else (eps[1] - eps[0]) / ds[0]
            grad_up[-1] = (eps[-1] - eps[-2]) / ds[-1] if adv[-1] > 0 else 0.0
            deps -= adv * grad_up

            # Heat diffusion (10 n mu eps / 9) grad eps, conservative form.
            flux = np.zeros(nl + 1)
            for f in range(1, nl):
                Df = 0.5 * (D[f - 1] + D[f])
                dsf = 0.5 * (ds[f - 1] + ds[f])
                flux[f] = -Df * (eps[f] - eps[f - 1]) / dsf
            deps -= (flux[1:] - flux[:-1]) / ds

            deps += heating_rate - nu_loss * eps
            eps = np.clip(eps + dt_s * deps, 0.1, self.eps_ceiling)
            eps[-1] = self.eps_cathode  # fixed cathode-side energy
            D = 10.0 * mu_bin * np.maximum(eps, 0.1) / 9.0

        self.eps = eps

        return {
            'I_T': self.I_T,
            'eps_max': float(np.max(eps)),
            'phi_min': float(np.min(self.phi)),
            'phi_max': float(np.max(self.phi)),
            'n_subcycles': n_sub,
        }

    # -- node-level views used by the MCC step ----------------------------

    @property
    def eps_nodes(self) -> np.ndarray:
        """Mean electron energy on the nodes (eV), for the ionization MCC.
        Out-of-span nodes take the nearest solved slice's energy (see
        __init__), not the clipped edge bin's."""
        return self.eps[self.bin_of_node][self._fill_idx] * self.mask

    @property
    def Te_nodes(self) -> np.ndarray:
        """Electron temperature on the nodes, K (2.58)."""
        return (2.0 * self.eps_nodes / 3.0) * E_CHARGE / KB
