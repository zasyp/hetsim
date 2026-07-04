from typing import Any, Callable
import numpy as np

# --- physical constants (SI, energies handled separately in eV) -----------
M_E = 9.1093837e-31   # electron mass, kg
Q_E = 1.6021766e-19   # elementary charge, C (also J per eV)

# --- xenon (the usual Hall-thruster propellant) ---------------------------
XENON_IONIZATION_ENERGY = 12.1298     # first ionization potential, eV
XENON_OUTER_ELECTRONS = 6             # equivalent electrons in the 5p^6 shell
LOTZ_CONSTANT = 4.5e-18               # Lotz prefactor a, m^2 * eV^2
#   (the customary 4.5e-14 cm^2*eV^2 converted to m^2*eV^2)


def lotz_cross_section(
        E: np.ndarray[Any, np.dtype[np.float64]] | float,
        E_iz: float = XENON_IONIZATION_ENERGY,
        q: int = XENON_OUTER_ELECTRONS,
        a: float = LOTZ_CONSTANT,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Single-shell Lotz electron-impact ionization cross-section (m^2).

    sigma(E) = a * q * ln(E / E_iz) / (E * E_iz)   for E >= E_iz, else 0,

    with the incident electron energy `E` and the ionization threshold `E_iz`
    in eV, `q` the number of equivalent electrons in the outer shell and `a`
    the Lotz prefactor (m^2*eV^2). This is a compact, physically-motivated
    default; for quantitative work pass a measured cross-section instead
    (see `tabulated_cross_section`), the rest of the pipeline is unchanged.
    """
    E = np.asarray(E, dtype=np.float64)
    above = E > E_iz
    # np.where evaluates both branches, so guard the log against E <= 0/E_iz.
    safe_E = np.where(above, E, E_iz * 2.0)
    sigma = a * q * np.log(safe_E / E_iz) / (safe_E * E_iz)
    return np.where(above, sigma, 0.0)


def tabulated_cross_section(
        energy_table: np.ndarray[Any, np.dtype[np.float64]],
        sigma_table: np.ndarray[Any, np.dtype[np.float64]],
        ) -> Callable[[np.ndarray], np.ndarray]:
    """Build a cross-section callable sigma(E) from measured `(energy, sigma)`
    data by linear interpolation.

    `energy_table` is in eV, `sigma_table` in m^2. Energies below the first
    tabulated point (i.e. below threshold) return 0; above the last point the
    cross-section is held flat. The returned callable is a drop-in replacement
    for `lotz_cross_section` in `maxwellian_rate_coefficient`.
    """
    energy_table = np.asarray(energy_table, dtype=np.float64)
    sigma_table = np.asarray(sigma_table, dtype=np.float64)

    def sigma(E: np.ndarray[Any, np.dtype[np.float64]] | float) -> np.ndarray:
        E = np.asarray(E, dtype=np.float64)
        return np.interp(E, energy_table, sigma_table, left=0.0, right=sigma_table[-1])

    return sigma


def maxwellian_rate_coefficient(
        T_e: np.ndarray[Any, np.dtype[np.float64]] | float,
        cross_section: Callable[[np.ndarray], np.ndarray] = lotz_cross_section,
        E_iz: float = XENON_IONIZATION_ENERGY,
        n_points: int = 512,
        energy_span: float = 30.0,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Ionization rate coefficient k_iz(T_e) = <sigma(v) v> (m^3/s), obtained
    by averaging the cross-section over an isotropic Maxwellian electron
    energy distribution at temperature `T_e` (in eV):

        k_iz = (2/sqrt(pi)) * sqrt(2 e / m_e) * T_e^{-3/2}
               * integral_{E_iz}^{inf} sigma(E) E exp(-E/T_e) dE ,

    with `E` in eV. This is the quantity that couples the neutral continuity
    sink to the electron temperature: feed a T_e field in and get a k_iz
    field of the same shape out (scalars in -> scalar out).

    The integral is truncated at E_iz + `energy_span`*max(T_e) (the Maxwellian
    tail is negligible far beyond a few T_e) and evaluated on `n_points` nodes
    by the trapezoidal rule, fully vectorised over the `T_e` array. Cells with
    T_e <= 0 return k_iz = 0.
    """
    T_e = np.asarray(T_e, dtype=np.float64)
    orig_shape = T_e.shape
    Te = T_e.ravel()

    k = np.zeros_like(Te)
    hot = Te > 0.0
    if not np.any(hot):
        return k.reshape(orig_shape) if orig_shape else np.float64(0.0)

    Te_hot = Te[hot]
    E_max = E_iz + energy_span * float(np.max(Te_hot))
    E = np.linspace(E_iz, E_max, n_points)               # (n_points,)
    sigma = np.asarray(cross_section(E), dtype=np.float64)

    # integrand[c, e] for cell c (temperature Te_hot[c]) and energy E[e]
    integrand = sigma[None, :] * E[None, :] * np.exp(-E[None, :] / Te_hot[:, None])
    integral = np.trapezoid(integrand, E, axis=1)

    prefactor = (2.0 / np.sqrt(np.pi)) * np.sqrt(2.0 * Q_E / M_E) * Te_hot ** -1.5
    k[hot] = prefactor * integral

    return k.reshape(orig_shape) if orig_shape else np.float64(k[0])


def ionization_frequency(
        n_e: np.ndarray[Any, np.dtype[np.float64]],
        T_e: np.ndarray[Any, np.dtype[np.float64]],
        cross_section: Callable[[np.ndarray], np.ndarray] = lotz_cross_section,
        E_iz: float = XENON_IONIZATION_ENERGY,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Local ionization frequency nu = n_e * k_iz(T_e) (1/s).

    This is the loss rate *per neutral atom*: it multiplies the neutral
    density linearly in the continuity sink, which is exactly what
    `solve_neutral_continuity` consumes as its `ionization_freq` argument.
    """
    k_iz = maxwellian_rate_coefficient(T_e, cross_section, E_iz)
    return n_e * k_iz


def ionization_rate(
        n_n: np.ndarray[Any, np.dtype[np.float64]],
        n_e: np.ndarray[Any, np.dtype[np.float64]],
        T_e: np.ndarray[Any, np.dtype[np.float64]],
        cross_section: Callable[[np.ndarray], np.ndarray] = lotz_cross_section,
        E_iz: float = XENON_IONIZATION_ENERGY,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Volumetric ionization rate S_iz = n_n * n_e * k_iz(T_e)
    (ionization events per m^3 per s).

    This is simultaneously the neutral sink and the ion/electron source, so it
    is the natural handoff between the neutral solver and a plasma/electron
    model. Multiply by the macroparticle weight and time step to convert into
    the number of neutral macroparticles removed (and ions created) per cell.
    """
    return n_n * ionization_frequency(n_e, T_e, cross_section, E_iz)
