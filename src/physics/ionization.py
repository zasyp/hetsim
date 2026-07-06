"""Monte Carlo Collision (MCC) ionization for the hybrid PIC model
(thesis Sec. 2.4).

Every time step each neutral macroparticle runs the cascade of tests
  e- + Xe  -> 2e- + Xe+    P_C = n_e k_Xe+ (eps) dt      (2.21)
  e- + Xe  -> 3e- + Xe++   P_C = n_e k_Xe++(eps) dt      (2.22)
and each Xe+ macroparticle the stepwise test
  e- + Xe+ -> 2e- + Xe++   P_C = n_e k_step(eps) dt      (2.23)
with n_e and the mean electron energy eps gathered at the particle position.
(2.23 is written with n_Xe+ in the thesis; in the quasineutral model
n_e = n_Xe+ + 2 n_Xe++ is the same field to first order and is what the local
electron density physically multiplies, so n_e is used here.)

Because P_C ~ 1e-9..1e-6, raw MCC would produce too few ion macroparticles
for usable statistics. The collision multiplier technique (Sec. 2.4.3) is
therefore applied with the smooth adaptive multiplier of eqs. 2.25-2.26:
    P*_C = gamma* P_C,   gamma* = sqrt(gamma P_C) / P_C,
and a successful event splits the parent instead of converting it: the
daughter of the new species carries weight w/gamma*, the parent keeps
w (gamma* - 1)/gamma*. The mean ionized mass per step is unchanged while the
ion macroparticle count grows by the factor gamma*.

To keep that growth bounded, `w_min` caps the multiplier at gamma* <= w/w_min
so no daughter is ever lighter than `w_min`: without the cap, daughters that
recombine at the walls re-enter the cascade as ever-lighter neutrals and the
macroparticle population explodes with particles that carry negligible mass
but full per-particle cost. Once a particle's weight reaches w_min the test
falls back to plain MCC species conversion.

Rate coefficients: the thesis uses tabulated Maxwellian-averaged rates from
Garrigues et al. (2001). Here `default_rates()` loads equivalent tabulated
Maxwellian rate coefficients for xenon bundled in `physics/data/*.dat`
(taken from the open-source HallThruster.jl code of UM-PEPL, University of
Michigan; derived from LXCat cross-section data; the energy argument is the
mean electron energy eps = 3/2 Te, same convention as this module):

    ionization_Xe_Xe+.dat     e- + Xe  -> 2e- + Xe+    (12.13 eV)
    ionization_Xe_Xe2+.dat    e- + Xe  -> 3e- + Xe++   (33.10 eV)
    ionization_Xe+_Xe2+.dat   e- + Xe+ -> 2e- + Xe++   (20.98 eV)
    excitation_Xe.dat         e- + Xe  -> e- + Xe*     ( 8.32 eV)

The electron energy loss coefficients (2.105-2.106) are assembled from these
as k_loss = sum(E_threshold * k) / eps. If the data files are missing, the
analytic fallbacks below (Goebel & Katz fits plus threshold-scaled forms)
are used instead -- with a loud RuntimeWarning, since those fits are rough
stand-ins, not quantitative data.
"""
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.classes import ParticleArray, Grid2D
from numerical.numerical_funcs import bilinear_gather

E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31      # electron mass, kg

# Xenon thresholds, eV
E_ION_XE1 = 12.127    # Xe -> Xe+
E_EXC_XE = 11.6       # first excitation (dominant inelastic energy sink)
E_ION_XE2 = 33.3      # Xe -> Xe++ (double, direct)
E_STEP = 21.2         # Xe+ -> Xe++

_EPS_MIN = 0.05  # eV; below this all rates vanish for numerical safety


def _te_ev(eps):
    """Electron temperature (eV) from mean energy eps (eV): eps = 3/2 Te."""
    return np.maximum(2.0 * np.asarray(eps, dtype=np.float64) / 3.0, _EPS_MIN)


def _cbar(te_ev):
    """Mean thermal speed of Maxwellian electrons, m/s (Te in eV)."""
    return np.sqrt(8.0 * E_CHARGE * te_ev / (np.pi * M_E))


def k_ionization_single(eps):
    """<sigma*v> for e- + Xe -> 2e- + Xe+, m^3/s (Goebel & Katz fit)."""
    te = _te_ev(eps)
    poly = 3.97 + 0.643 * te - 0.0368 * te**2
    return 1.0e-20 * np.maximum(poly, 0.0) * np.exp(-E_ION_XE1 / te) * _cbar(te)


def k_excitation(eps):
    """<sigma*v> for the lumped excitation channel of Xe, m^3/s (G&K fit)."""
    te = _te_ev(eps)
    return 1.93e-19 * np.exp(-E_EXC_XE / te) / np.sqrt(te) * _cbar(te)


def k_ionization_double(eps):
    """<sigma*v> for direct e- + Xe -> 3e- + Xe++, m^3/s.

    Threshold-scaled stand-in (peak cross-section ~4% of single ionization,
    threshold 33.3 eV); only used if the LXCat tables are unavailable.
    """
    te = _te_ev(eps)
    poly = 3.97 + 0.643 * te - 0.0368 * te**2
    return 4.0e-22 * np.maximum(poly, 0.0) * np.exp(-E_ION_XE2 / te) * _cbar(te)


def k_ionization_stepwise(eps):
    """<sigma*v> for e- + Xe+ -> 2e- + Xe++, m^3/s (threshold 21.2 eV);
    stand-in of the same form as the single-ionization fit."""
    te = _te_ev(eps)
    poly = 1.0 + 0.2 * te
    return 1.0e-20 * poly * np.exp(-E_STEP / te) * _cbar(te)


def k_loss_neutral(eps):
    """Electron energy loss rate coefficient against neutrals, m^3/s, defined
    so that the loss frequency of eq. 2.105 is nu = n_a * k and the energy
    sink is n_e * eps * nu (eq. 2.54): k = (E1 k1 + Eexc kexc + E2 k2)/eps.
    """
    eps = np.maximum(np.asarray(eps, dtype=np.float64), _EPS_MIN)
    return (E_ION_XE1 * k_ionization_single(eps)
            + E_EXC_XE * k_excitation(eps)
            + E_ION_XE2 * k_ionization_double(eps)) / eps


def k_loss_ion(eps):
    """Energy loss rate coefficient against Xe+ (eq. 2.106 analogue)."""
    eps = np.maximum(np.asarray(eps, dtype=np.float64), _EPS_MIN)
    return E_STEP * k_ionization_stepwise(eps) / eps


class RateTable:
    """Tabulated Maxwellian rate coefficient k(eps), log-log interpolated.

    Below the first tabulated point the coefficient is extrapolated with the
    Arrhenius form k = A exp(-E/eps) fitted to the first two points -- a
    plain endpoint clamp would overestimate threshold-limited rates by orders
    of magnitude at low energy. Above the last point the value is clamped.
    """

    def __init__(self, eps_ev: np.ndarray, k_m3s: np.ndarray):
        self._eps = np.asarray(eps_ev, dtype=np.float64)
        self._log_eps = np.log(self._eps)
        self._log_k = np.log(np.maximum(np.asarray(k_m3s, np.float64), 1e-300))
        # Arrhenius tail: ln k = ln k0 - E_fit (1/eps - 1/eps0).
        inv = 1.0 / self._eps
        self._E_fit = max(
            (self._log_k[1] - self._log_k[0]) / max(inv[0] - inv[1], 1e-300),
            0.0)

    def __call__(self, eps):
        eps = np.maximum(np.asarray(eps, dtype=np.float64), _EPS_MIN)
        k = np.exp(np.interp(np.log(eps), self._log_eps, self._log_k))
        low = eps < self._eps[0]
        if np.any(low):
            tail = np.exp(
                self._log_k[0]
                - self._E_fit * (1.0 / eps - 1.0 / self._eps[0]))
            k = np.where(low, tail, k)
        return k


@dataclass
class IonizationRates:
    single: Callable = k_ionization_single
    double: Callable = k_ionization_double
    stepwise: Callable = k_ionization_stepwise
    loss_neutral: Callable = k_loss_neutral
    loss_ion: Callable = k_loss_ion


DATA_DIR = Path(__file__).resolve().parent / 'data'


def load_rate_table(path: Path | str) -> tuple[float, RateTable]:
    """Read one bundled rate table.

    Format (HallThruster.jl reaction files): the first line carries the
    threshold energy, the second is a column header, then two columns of
    mean electron energy (eV) and Maxwellian rate coefficient (m^3/s).
    Returns (threshold_eV, RateTable).
    """
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        first = f.readline()
    threshold = float(first.split(':')[1])
    data = np.loadtxt(path, skiprows=2)
    # Log-log interpolation cannot use the eps = 0-ish leading points with
    # k = 0; drop them (RateTable clamps below the first kept point anyway).
    keep = data[:, 1] > 0.0
    return threshold, RateTable(data[keep, 0], data[keep, 1])


_DEFAULT_RATES: IonizationRates | None = None


def default_rates() -> IonizationRates:
    """Tabulated xenon rates from `physics/data` (see module docstring),
    loaded once and cached. Falls back to the analytic fits -- with a
    RuntimeWarning -- only if the tables cannot be read."""
    global _DEFAULT_RATES
    if _DEFAULT_RATES is not None:
        return _DEFAULT_RATES

    try:
        E1, k1 = load_rate_table(DATA_DIR / 'ionization_Xe_Xe+.dat')
        E2, k2 = load_rate_table(DATA_DIR / 'ionization_Xe_Xe2+.dat')
        Es, ks = load_rate_table(DATA_DIR / 'ionization_Xe+_Xe2+.dat')
        Ex, kx = load_rate_table(DATA_DIR / 'excitation_Xe.dat')
    except (OSError, IndexError, ValueError) as err:
        warnings.warn(
            f"could not load the LXCat rate tables from {DATA_DIR} ({err}); "
            "falling back to the rough analytic fits",
            RuntimeWarning, stacklevel=2)
        _DEFAULT_RATES = IonizationRates()
        return _DEFAULT_RATES

    def loss_neutral(eps):
        e = np.maximum(np.asarray(eps, dtype=np.float64), _EPS_MIN)
        return (E1 * k1(e) + Ex * kx(e) + E2 * k2(e)) / e

    def loss_ion(eps):
        e = np.maximum(np.asarray(eps, dtype=np.float64), _EPS_MIN)
        return Es * ks(e) / e

    _DEFAULT_RATES = IonizationRates(
        single=k1, double=k2, stepwise=ks,
        loss_neutral=loss_neutral, loss_ion=loss_ion,
    )
    return _DEFAULT_RATES


def _split(
        pa: ParticleArray,
        idx: np.ndarray,
        p_c: np.ndarray,
        gamma: float,
        w_min: float,
        u: np.ndarray,
        new_charge: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one collision test with the adaptive multiplier (2.25-2.26) for
    the particles at slots `idx`, vectorised.

    Successful boosted tests split the parent: the parent keeps
    w (gamma* - 1)/gamma* and the daughter (returned as slot indices plus
    daughter weights) carries w/gamma* with the new charge state. Where the
    boosted probability would not help (gamma* <= 1, i.e. P_C >= 1/gamma, or
    the parent is already at the `w_min` weight floor), the test falls back
    to plain MCC species conversion (Sec. 2.4.1): the parent itself switches
    charge state in place.

    Returns (n_converted, daughter_slots, daughter_weights).
    """
    pos = p_c > 0.0
    p_safe = np.where(pos, p_c, 1.0)
    g_star = np.sqrt(gamma * p_safe) / p_safe
    if w_min > 0.0:
        g_star = np.minimum(g_star, pa.weight[idx] / w_min)

    plain = pos & (g_star <= 1.0) & (u < p_c)
    pa.charge[idx[plain]] = new_charge

    split = pos & (g_star > 1.0) & (u < g_star * p_c)
    sidx = idx[split]
    w_d = pa.weight[sidx] / g_star[split]
    pa.weight[sidx] -= w_d
    return int(np.count_nonzero(plain)), sidx, w_d


def ionization_step(
        g: Grid2D,
        particles: ParticleArray,
        n_e: np.ndarray,
        eps: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        rates: IonizationRates | None = None,
        gamma: float = 8.0,
        w_min: float = 0.0,
        ) -> dict:
    """One MCC ionization pass over all active macroparticles.

    `n_e` (1/m^3) and `eps` (mean electron energy, eV) are node fields of
    shape (N_r, N_z); both are gathered bilinearly at each particle position.
    Daughter ions produced by the collision-multiplier splits are appended to
    `particles`. `w_min` is the daughter weight floor of the multiplier cap
    (0 disables it; see the module docstring).

    Returns event counts: {'single': .., 'double': .., 'stepwise': ..}.
    """
    if rates is None:
        rates = default_rates()

    counts = {'single': 0, 'double': 0, 'stepwise': 0}
    n = particles.n
    if n == 0:
        return counts
    act = particles.active[:n]
    charge = particles.charge[:n]

    r_nodes = g.r_nodes()
    z_nodes = g.z_nodes()

    def gather(idx):
        pr = particles.r[idx]
        pz = particles.z[idx]
        ne_p = bilinear_gather(n_e, r_nodes, z_nodes, pr, pz)
        eps_p = bilinear_gather(eps, r_nodes, z_nodes, pr, pz)
        return ne_p, eps_p

    daughters: list[tuple[np.ndarray, np.ndarray, int]] = []

    # Neutrals: single (2.21), then double (2.22) for those still neutral.
    idx0 = np.flatnonzero(act & (charge == 0))
    if idx0.size:
        ne_p, eps_p = gather(idx0)
        u = rng.random((idx0.size, 2))
        p1 = rates.single(eps_p) * ne_p * dt
        conv, sidx, w_d = _split(particles, idx0, p1, gamma, w_min,
                                 u[:, 0], 1)
        counts['single'] += conv + sidx.size
        daughters.append((sidx, w_d, 1))

        still = charge[idx0] == 0  # survived the first test (2.22)
        idx0b = idx0[still]
        p2 = rates.double(eps_p[still]) * ne_p[still] * dt
        conv, sidx, w_d = _split(particles, idx0b, p2, gamma, w_min,
                                 u[still, 1], 2)
        counts['double'] += conv + sidx.size
        daughters.append((sidx, w_d, 2))

    # Xe+ stepwise ionization (2.23).
    idx1 = np.flatnonzero(act & (charge == 1))
    if idx1.size:
        ne_p, eps_p = gather(idx1)
        ps = rates.stepwise(eps_p) * ne_p * dt
        conv, sidx, w_d = _split(particles, idx1, ps, gamma, w_min,
                                 rng.random(idx1.size), 2)
        counts['stepwise'] += conv + sidx.size
        daughters.append((sidx, w_d, 2))

    for sidx, w_d, q in daughters:
        if sidx.size:
            particles.add(
                particles.z[sidx], particles.r[sidx],
                particles.v_r[sidx], particles.v_z[sidx],
                particles.v_theta[sidx],
                q, particles.T[sidx], w_d,
            )

    return counts
