# Electron collision frequencies [1/s] — block 1 of the electron-fluid
# model. The total momentum-transfer frequency nu_e sets how often an
# electron loses its direction, which is what lets it migrate across the
# magnetic field (see mobility, block 2).
#
# Unit convention: SI everywhere EXCEPT Te, which is in eV.
#
# Excitation collisions are omitted from nu_e: together with ionization
# they add <~20% to nu_en at the highest Te and are buried under the
# anomalous-transport uncertainty. Ionization matters elsewhere — as the
# ion source in the current-continuity equation and as the energy sink
# in the electron energy equation.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import scipy.constants as const

from src.electron_liquid.default_plasm_params import debye_radius, omega_ce
from src.structs.classes import WorkingSubstance

TE_FLOOR = 0.1        # eV; keeps the cold edges of the grid finite
LN_LAMBDA_FLOOR = 2.0  # standard floor for the Coulomb logarithm


def neutral_collision(Te:np.ndarray,
                      neutral_concentration:np.ndarray,
                      propellant:WorkingSubstance,
                      ) -> np.ndarray:
    """Electron-neutral momentum-transfer frequency nu_en = n_n * k_en(Te)."""
    return propellant.k_en(Te) * neutral_concentration


def ionization_collision(Te:np.ndarray,
                         neutral_concentration:np.ndarray,
                         propellant:WorkingSubstance,
                         ) -> np.ndarray:
    """Ionization collision frequency nu_iz = n_n * k_iz(Te)."""
    return propellant.ionization_rate_Te(Te) * neutral_concentration


def coulomb_logarithm(Te:np.ndarray,
                      electron_concentration:np.ndarray,
                      ) -> np.ndarray:
    """Coulomb logarithm ln(b_max / b_min) from first principles:
    b_max is the Debye length (fields are screened beyond it), b_min the
    classical distance of closest approach e / (4 pi eps0 Te), Te in eV.
    Floored at LN_LAMBDA_FLOOR; typical discharge values are ~10-15.
    """
    Te = np.maximum(Te, TE_FLOOR)
    b_max = debye_radius(Te, electron_concentration)
    b_min = const.elementary_charge / (4 * const.pi * const.epsilon_0 * Te)
    return np.maximum(np.log(b_max / b_min), LN_LAMBDA_FLOOR)


def coulomb_collision(electron_concentration:np.ndarray,
                      Te:np.ndarray,
                      ) -> np.ndarray:
    """Electron-ion momentum-transfer frequency [1/s] (NRL Plasma
    Formulary, converted to n_e in m^-3, Te in eV):

        nu_ei = 2.91e-12 * n_e * lnL * Te^(-3/2)
    """
    Te = np.maximum(Te, TE_FLOOR)
    ln_lambda = coulomb_logarithm(Te, electron_concentration)
    return 2.91e-12 * electron_concentration * ln_lambda * Te ** -1.5


# Calibrated 4+2-parameter anomalous-transport model (Marks & Jorns,
# "Uncertainty quantification of a multi-component Hall thruster model at
# varying facility pressures", arXiv:2507.08113, 2025 -- Eqs. 3-4, p.3;
# posterior medians in Tables IV-V, p.9). Each preset is the posterior
# median for (alpha_anom, beta_anom, z_anom, L_anom, delta_z_anom) fitted
# to thruster discharge-current/thrust data at 300 V; GENERIC_300V is the
# authors' suggested starting point for a thruster with no dedicated fit
# (they give no generic delta_z_anom, so it defaults to 0 -- no facility-
# pressure shift -- until fitted to a specific facility/thruster).
ANOMALY_PROFILE_PRESETS: dict[str, dict[str, float]] = {
    "SPT-100":      dict(alpha_anom=0.06, beta_anom=0.99, z_anom=1.14, L_anom=0.43, delta_z_anom=0.33),
    "H9":           dict(alpha_anom=0.13, beta_anom=0.98, z_anom=1.07, L_anom=0.43, delta_z_anom=0.18),
    "GENERIC_300V": dict(alpha_anom=1 / 16, beta_anom=0.99, z_anom=1.05, L_anom=0.38, delta_z_anom=0.0),
}

# Pressure-shift midpoint P0 [Torr], Eq. (4): found to transfer well
# across thrusters/propellants in the paper, so it is not one of the
# per-thruster calibrated parameters above.
P0_DEFAULT = 25e-6  # Torr (Torr-Xe / Torr-Kr as appropriate; propellant-corrected)


def anomalous_transport_profile(z_hat:np.ndarray,
                                alpha_anom:float = 1 / 16,
                                beta_anom:float = 0.99,
                                z_anom:float = 1.05,
                                L_anom:float = 0.38,
                                ) -> np.ndarray:
    """Axial profile of the inverse anomalous Hall parameter (i.e. the
    alpha that multiplies omega_ce in anomaly_collision), Eq. (3):

        alpha(z) = alpha_anom * (1 - beta_anom * exp(-((z_hat - z_anom)/L_anom)^2))

    z_hat is the axial coordinate normalized by the discharge channel
    length (anode at z_hat=0, nominal exit plane at z_hat=1) -- already
    including any pressure-driven shift (see anomaly_axial_shift /
    anomalous_alpha for the physical-coordinate version). alpha_anom sets
    the near-anode ceiling, beta_anom (~0.95-0.99) carves out a deep
    low-mobility trough around z_anom (~1.0-1.15, i.e. near/just past the
    exit plane) of width L_anom (~0.35-0.45), which is what produces the
    steep ion-acceleration region seen in experiments. See
    ANOMALY_PROFILE_PRESETS for calibrated tuples per thruster, or use a
    scalar alpha_anom=1/16 (Bohm) with beta_anom=0 to recover the old
    constant-alpha behavior.
    """
    return alpha_anom * (1 - beta_anom * np.exp(-((z_hat - z_anom) / L_anom) ** 2))


def anomaly_axial_shift(background_pressure_torr:float | np.ndarray,
                        channel_length:float,
                        delta_z_anom:float,
                        P0:float = P0_DEFAULT,
                        ) -> float | np.ndarray:
    """Upstream shift Delta_z(P_B) [m] of the anomalous-transport barrier
    with background (facility) pressure, Eq. (4): a ground-test artifact
    -- rising background pressure ingests extra neutrals through the exit
    plane, moving the ionization/acceleration region upstream relative to
    where it sits in space. The logistic form saturates at both low and
    high P_B/P0 so the shift stays bounded, magnitude <= delta_z_anom *
    channel_length; it is exactly 0 at P_B=0 by construction.

    background_pressure_torr : facility pressure P_B [Torr], propellant-
        corrected (Torr-Xe / Torr-Kr) -- 0 recovers the unshifted profile.
    delta_z_anom : shift magnitude, channel-length units (paper's fitted
        range ~0-0.5; see ANOMALY_PROFILE_PRESETS).
    P0 : shift midpoint pressure [Torr]; P0_DEFAULT = 25e-6 Torr is the
        paper's cross-thruster fit and rarely needs changing.
    """
    x = np.asarray(background_pressure_torr) / P0 - 1.0
    logistic = 1.0 / (1.0 + np.exp(-2.0 * x)) - 1.0 / (1.0 + np.e ** 2)
    return delta_z_anom * channel_length * logistic


def anomalous_alpha(z:np.ndarray,
                    channel_length:float,
                    alpha_anom:float = 1 / 16,
                    beta_anom:float = 0.99,
                    z_anom:float = 1.05,
                    L_anom:float = 0.38,
                    background_pressure_torr:float | np.ndarray = 0.0,
                    delta_z_anom:float = 0.0,
                    P0:float = P0_DEFAULT,
                    ) -> np.ndarray:
    """Full Eqs. (3)-(4) anomalous-transport coefficient alpha(z) =
    nu_anom/omega_ce at physical axial position(s) z [m] (anode at z=0),
    combining the Gaussian transport barrier with its pressure-dependent
    upstream shift:

        z_hat = (z + Delta_z(P_B)) / channel_length

    This is the one-stop entry point for "correct" alpha: pass a preset
    from ANOMALY_PROFILE_PRESETS (** to unpack) plus the operating
    background_pressure_torr, and feed the result into anomaly_collision.
    With background_pressure_torr=0 (or delta_z_anom=0) this reduces to
    anomalous_transport_profile(z/channel_length, ...), i.e. the in-space,
    unshifted profile.
    """
    shift = anomaly_axial_shift(background_pressure_torr, channel_length, delta_z_anom, P0)
    z_hat = (z + shift) / channel_length
    return anomalous_transport_profile(z_hat, alpha_anom, beta_anom, z_anom, L_anom)


def anomaly_collision(B:np.ndarray, alpha:float | np.ndarray = 1 / 16) -> np.ndarray:
    """Bohm-type anomalous frequency nu_anom = alpha * omega_ce. alpha is
    the empirical knob (classic 1/16); pass an array to use different
    values inside the channel and in the plume — e.g. the output of
    anomalous_transport_profile() for a literature-calibrated axial shape
    instead of a single constant.
    """
    return alpha * omega_ce(B)


def anomaly_collision_brick(electron_concentration:np.ndarray,
                            Te:np.ndarray,
                            c:float = 640.0,
                            ) -> np.ndarray:
    """Alternative anomalous frequency following the electron-ion scaling
    (Brick, Roberts & Jorns, "Numerical Investigation of Electron Energy
    Transport in Hall Thrusters", AIAA SciTech 2025-0298 -- Eq. 8, p.7):

        nu_anom = c * nu_ei = 2.9e-12 * c * n_e * lnL * Te^(-3/2)

    i.e. the classical electron-ion frequency (coulomb_collision) scaled by
    a single scalar c. This is a deliberate alternative to the Bohm-like
    anomaly_collision(): instead of imposing the transport barrier through a
    hand-tuned axial profile alpha(z), the minimum arises self-consistently
    from the Te^(-3/2) dependence -- nu_anom bottoms out where Te peaks (near
    peak B), which is what carves the steep electric field. Only the overall
    magnitude is tuned; the paper's fitted values are O(100-1000) per
    operating condition (Kr 300V/15A c=1000, Xe 300V/15A c=640, Kr 300V/30A
    c=250, Xe 600V/15A c=500; Brick 2025 Figs. 8-9). Because it depends on
    local n_e, Te rather than z, feed it plasma fields, not a grid position.

    Note: the paper's fully self-consistent update also applies a relaxation
    factor (Eq. 16, r=0.5) for numerical stability -- see relax_collision().
    """
    return c * coulomb_collision(electron_concentration, Te)


def relax_collision(nu_new:np.ndarray,
                    nu_old:np.ndarray,
                    r:float = 0.5,
                    ) -> np.ndarray:
    """Under-relaxed update of a self-consistent anomalous frequency (Brick
    2025, Eq. 16):

        nu = r * nu_new + (1 - r) * nu_old

    nu_new is anomaly_collision_brick() evaluated on the current-timestep
    fields, nu_old the previous accepted value. Damps the large transients
    that make an unrelaxed self-consistent closure go unstable; the paper
    uses r=0.5 as the largest value that stayed reliably stable.
    """
    return r * nu_new + (1 - r) * nu_old


def electron_collision(nu_en:np.ndarray,
                       nu_iz:np.ndarray,
                       nu_ei:np.ndarray,
                       nu_anom:np.ndarray,
                       ) -> np.ndarray:
    """Total effective electron momentum-transfer frequency [1/s]:
    the sum of the classical channels and the anomalous term."""
    return nu_en + nu_iz + nu_ei + nu_anom
