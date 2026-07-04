"""Neutral xenon flow and ionization in an SPT-70-class Hall thruster channel.

This example ties together the three pieces of the neutral model:

  1. `deposit`                  -- scatter injected neutral macroparticles onto
                                   the axisymmetric (r, z) grid to get the raw,
                                   ionization-free (n, v_r, v_z, T) fields
                                   (the free-molecular / "cold gas" picture).
  2. `ionization.*`             -- from a prescribed electron temperature and
                                   plasma density, build the ionization rate
                                   coefficient k_iz(T_e) from collision
                                   cross-sections and the loss frequency
                                   nu = n_e * k_iz.
  3. `solve_neutral_continuity` -- solve the steady continuity equation with
                                   that loss term, using the deposited velocity
                                   field, to get the *actual* neutral density
                                   depleted by ionization.

The contrast between the density in step 1 (no ionization) and step 3 (with
ionization) is the physics of interest: neutrals stream in from the anode, and
are eaten away in the near-exit region where the electrons are hot -- exactly
the ionization/acceleration layer of a Hall thruster.

Geometry and operating point (mass flow, temperatures, densities) are an
illustrative SPT-70-class approximation -- representative topology and scale,
not certified data. Swap in your own numbers for real analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from physics.classes import Grid2D, Particle
from physics.neutrals import deposit, solve_neutral_continuity
from physics.ionization import (
    maxwellian_rate_coefficient, ionization_frequency, ionization_rate,
)
from numerical.numerical_funcs import smoothing

# --- physical constants -----------------------------------------------------
K_B = 1.380649e-23          # Boltzmann constant, J/K
AMU = 1.66053907e-27        # atomic mass unit, kg
M_XE = 131.293 * AMU        # xenon atom mass, kg

# --- SPT-70-class channel geometry (meters) ---------------------------------
CHANNEL_R_IN = 0.021        # inner wall radius
CHANNEL_R_OUT = 0.035       # outer wall radius
ANODE_Z = 0.020            # anode / injection plane
EXIT_Z = 0.045             # channel exit plane

DOMAIN_R = 0.050           # radial extent of the simulation domain
DOMAIN_Z = 0.060           # axial extent (channel + a bit of near plume)

# --- operating point --------------------------------------------------------
MASS_FLOW = 2.5e-6          # anode xenon mass flow, kg/s (2.5 mg/s)
T_GAS = 700.0              # neutral gas temperature at the anode, K
V_DRIFT = 150.0            # mean axial injection speed of the neutrals, m/s

# Representative electron-temperature and plasma-density profiles. Both peak in
# the near-exit ionization layer; these stand in for a plasma/electron solver.
TE_ANODE = 3.0             # electron temperature at the anode, eV
TE_PEAK = 22.0            # peak electron temperature near the exit, eV
TE_PEAK_Z = 0.043         # axial location of the T_e peak, m
TE_WIDTH = 0.006          # Gaussian half-width of the T_e hump, m

NE_PEAK = 3.0e17          # peak plasma (electron) density, m^-3
NE_PEAK_Z = 0.042         # axial location of the n_e peak, m
NE_WIDTH = 0.007          # Gaussian half-width of the n_e hump, m

N_PARTICLES = 40000        # neutral macroparticles injected for the deposition
RNG_SEED = 12345


def channel_area() -> float:
    """Cross-sectional area of the annular discharge channel, m^2."""
    return np.pi * (CHANNEL_R_OUT**2 - CHANNEL_R_IN**2)


def inlet_number_density() -> float:
    """Anode neutral number density n_in (m^-3) implied by the mass flow,
    from mdot = m_Xe * n_in * v_drift * A_channel.
    """
    return MASS_FLOW / (M_XE * V_DRIFT * channel_area())


def sample_neutrals(grid: Grid2D, n_in: float) -> np.ndarray:
    """Sample `N_PARTICLES` neutral macroparticles filling the channel annulus
    between the anode and the exit, with a drifting-Maxwellian velocity
    (axial drift `V_DRIFT` plus a thermal spread at `T_GAS`).

    The macroparticle weight is chosen so that a uniform fill reproduces the
    target inlet density `n_in`: total real atoms = n_in * (fill volume), split
    evenly over the macroparticles.
    """
    rng = np.random.default_rng(RNG_SEED)

    # area-uniform radial sampling across the annulus: r = sqrt(U over r^2)
    u = rng.random(N_PARTICLES)
    r = np.sqrt(CHANNEL_R_IN**2 + u * (CHANNEL_R_OUT**2 - CHANNEL_R_IN**2))
    z = rng.uniform(ANODE_Z, EXIT_Z, N_PARTICLES)

    sigma_th = np.sqrt(K_B * T_GAS / M_XE)          # 1-D thermal speed, m/s
    v_z = V_DRIFT + sigma_th * rng.standard_normal(N_PARTICLES)
    v_r = sigma_th * rng.standard_normal(N_PARTICLES)

    fill_volume = channel_area() * (EXIT_Z - ANODE_Z)
    weight = n_in * fill_volume / N_PARTICLES

    particles = np.empty(N_PARTICLES, dtype=object)
    for k in range(N_PARTICLES):
        particles[k] = Particle(
            z=z[k], r=r[k], v_r=v_r[k], v_z=v_z[k], mass=M_XE,
            T=T_GAS, weight=weight,
        )
    return particles


def electron_fields(grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """Prescribed electron temperature T_e (eV) and plasma density n_e (m^-3)
    fields: axial Gaussian humps in the near-exit ionization layer, confined
    radially to the channel annulus (they stand in for a real electron model).
    """
    R, Z = grid.mesh()

    te = TE_ANODE + (TE_PEAK - TE_ANODE) * np.exp(-((Z - TE_PEAK_Z) / TE_WIDTH) ** 2)
    ne = NE_PEAK * np.exp(-((Z - NE_PEAK_Z) / NE_WIDTH) ** 2)

    in_channel = (R >= CHANNEL_R_IN) & (R <= CHANNEL_R_OUT)
    te = np.where(in_channel, te, TE_ANODE)
    ne = np.where(in_channel, ne, 0.0)
    return te, ne


def build_velocity_for_continuity(
        grid: Grid2D, n_dep: np.ndarray, v_r: np.ndarray, v_z: np.ndarray, n_in: float,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Use the deposited velocity field for the continuity solve, but fall back
    to the mean axial drift in cells the macroparticles barely reached (where
    the deposited velocity is just statistical noise). This keeps the transport
    well-posed downstream of the ionization layer, where the gas is depleted.

    Cells the deposition left with (near-)zero axial speed -- either barely
    reached by particles, or lifted above the density threshold only by the
    smoothing of `n_dep` while their velocity stayed at the empty-cell default
    of 0 -- would give a matrix row with no outflow (singular), so they are
    forced onto the drift as well.
    """
    thin = (n_dep < 0.02 * n_in) | (np.abs(v_z) < 1.0)
    v_z_c = np.where(thin, V_DRIFT, v_z)
    v_r_c = np.where(thin, 0.0, v_r)
    # The *mean* neutral velocity is smooth; the per-cell deposited value still
    # carries thermal shot noise. Feeding that noise into the upwind solve
    # makes mass pile up wherever the velocity spuriously converges (n > n_in),
    # so smooth the velocity field before using it for transport.
    for _ in range(4):
        v_z_c = smoothing(v_z_c)
        v_r_c = smoothing(v_r_c)
    return v_r_c, v_z_c


def _draw_channel(ax: plt.Axes) -> None:
    """Overlay the discharge-channel walls (black) on a (z, r) axes."""
    ax.plot([ANODE_Z, EXIT_Z], [CHANNEL_R_IN, CHANNEL_R_IN], 'k', lw=2, zorder=4)
    ax.plot([ANODE_Z, EXIT_Z], [CHANNEL_R_OUT, CHANNEL_R_OUT], 'k', lw=2, zorder=4)
    ax.plot([ANODE_Z, ANODE_Z], [CHANNEL_R_IN, CHANNEL_R_OUT], 'k', lw=2, zorder=4)
    ax.axvline(EXIT_Z, color='0.4', ls='--', lw=1, zorder=2)


def plot_results(
        grid: Grid2D, n_dep: np.ndarray, te: np.ndarray, ne: np.ndarray,
        s_iz: np.ndarray, n_cont: np.ndarray, n_cont_no_iz: np.ndarray,
        ) -> plt.Figure:
    """Six-panel overview of the neutral flow and its ionization."""
    R, Z = grid.mesh()
    r_mid = 0.5 * (CHANNEL_R_IN + CHANNEL_R_OUT)
    i_mid = int(np.argmin(np.abs(grid.r_nodes() - r_mid)))
    z = grid.z_nodes()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)

    def field_panel(ax, field, title, label, cmap, scale=1.0):
        cf = ax.contourf(Z, R, field * scale, levels=30, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=label)
        _draw_channel(ax)
        ax.set_title(title)
        ax.set_ylabel('r, m')

    field_panel(axes[0, 0], n_dep, 'Deposited neutral density (no ionization)',
                'n, 1e19 m^-3', 'viridis', 1e-19)
    field_panel(axes[0, 1], te, 'Electron temperature (prescribed)',
                'T_e, eV', 'inferno')
    field_panel(axes[0, 2], ne, 'Plasma density (prescribed)',
                'n_e, 1e17 m^-3', 'plasma', 1e-17)
    field_panel(axes[1, 0], n_cont, 'Neutral density from continuity (with ionization)',
                'n, 1e19 m^-3', 'viridis', 1e-19)
    field_panel(axes[1, 1], s_iz, 'Ionization rate S_iz',
                'S_iz, 1e23 m^-3 s^-1', 'magma', 1e-23)

    ax = axes[1, 2]
    ax.plot(z, n_dep[i_mid] * 1e-19, label='deposited (no ioniz.)', lw=2)
    ax.plot(z, n_cont_no_iz[i_mid] * 1e-19, '--', label='continuity, no ioniz.', lw=1.5)
    ax.plot(z, n_cont[i_mid] * 1e-19, label='continuity, with ioniz.', lw=2)
    ax.set_xlabel('z, m')
    ax.set_ylabel('n at mid-radius, 1e19 m^-3')
    ax.set_title('Axial neutral density profiles')
    ax.axvline(EXIT_Z, color='0.4', ls='--', lw=1)
    ax.legend(fontsize=8, loc='upper right')
    axr = ax.twinx()
    axr.plot(z, te[i_mid], color='tab:red', alpha=0.5, lw=1.2)
    axr.set_ylabel('T_e, eV', color='tab:red')
    axr.tick_params(axis='y', labelcolor='tab:red')

    for ax in axes[1, :3]:
        ax.set_xlabel('z, m')

    fig.suptitle('SPT-70-like neutral xenon flow and ionization '
                 '(illustrative geometry & operating point)')
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the full neutral-flow + ionization pipeline and report diagnostics."""
    nr, nz = 61, 121
    grid = Grid2D(0.0, DOMAIN_R, nr, ANODE_Z, DOMAIN_Z, nz)

    n_in = inlet_number_density()
    print(f"channel area           : {channel_area()*1e4:.2f} cm^2")
    print(f"inlet neutral density  : {n_in:.3e} m^-3")
    print(f"neutral thermal speed  : {np.sqrt(K_B*T_GAS/M_XE):.1f} m/s "
          f"(drift {V_DRIFT:.0f} m/s)")

    # --- step 1: deposit injected macroparticles -> raw neutral fields -------
    particles = sample_neutrals(grid, n_in)
    shape = (nr, nz)
    n_dep = np.zeros(shape); v_r = np.zeros(shape); v_z = np.zeros(shape); T = np.zeros(shape)
    nvr = np.zeros(shape); nvz = np.zeros(shape); nT = np.zeros(shape)
    deposit(grid, particles, n_dep, v_r, v_z, T, nvr, nvz, nT)
    for _ in range(3):
        n_dep = smoothing(n_dep)   # damp the macroparticle shot noise
    print(f"deposited density (mid): {n_dep[nr//2, nz//4]:.3e} m^-3 "
          f"(target {n_in:.3e})")

    # --- step 2: ionization from prescribed electron fields ------------------
    te, ne = electron_fields(grid)
    k_iz_peak = maxwellian_rate_coefficient(TE_PEAK)
    nu = ionization_frequency(ne, te)     # loss frequency, 1/s
    print(f"k_iz at T_e={TE_PEAK:.0f} eV  : {k_iz_peak:.3e} m^3/s")
    print(f"max ionization freq    : {nu.max():.3e} 1/s "
          f"(mean free time {1.0/nu.max()*1e6:.2f} us)")

    # --- step 3: steady continuity with the ionization sink ------------------
    v_r_c, v_z_c = build_velocity_for_continuity(grid, n_dep, v_r, v_z, n_in)
    inlet_mask = np.zeros(shape, dtype=bool)
    inlet_mask[:, 0] = True
    r_nodes = grid.r_nodes()
    in_annulus = (r_nodes >= CHANNEL_R_IN) & (r_nodes <= CHANNEL_R_OUT)
    inlet_density = np.zeros(shape)
    inlet_density[in_annulus, 0] = n_in

    n_cont = solve_neutral_continuity(
        grid, v_r_c, v_z_c, inlet_mask, inlet_density, ionization_freq=nu)
    n_cont_no_iz = solve_neutral_continuity(
        grid, v_r_c, v_z_c, inlet_mask, inlet_density)

    # ionization rate uses the *actual* (depleted) neutral density
    s_iz = ionization_rate(n_cont, ne, te)

    # --- diagnostics: mass balance and propellant utilization ----------------
    _, _, _, _, _, _, vol = _cell_volumes(grid)
    injected_flux = n_in * V_DRIFT * channel_area()       # atoms/s
    ionized_total = float(np.sum(s_iz * vol))             # atoms/s
    utilization = ionized_total / injected_flux
    print(f"injected neutral flux  : {injected_flux:.3e} atoms/s")
    print(f"total ionization rate  : {ionized_total:.3e} atoms/s")
    print(f"propellant utilization : {utilization*100:.1f} %")

    i_mid = int(np.argmin(np.abs(r_nodes - 0.5*(CHANNEL_R_IN+CHANNEL_R_OUT))))
    n_exit = n_cont[i_mid, int(np.argmin(np.abs(grid.z_nodes()-EXIT_Z)))]
    print(f"neutral depletion at exit: n_exit/n_in = {n_exit/n_in:.2f} at mid-radius")

    fig = plot_results(grid, n_dep, te, ne, s_iz, n_cont, n_cont_no_iz)
    fig.savefig(Path(__file__).with_name('spt70_neutral_flow.png'), dpi=150)
    plt.show()


def _cell_volumes(grid: Grid2D):
    """Expose the continuity control-volume geometry for the mass-balance
    diagnostic (thin re-export of the neutrals module's private helper)."""
    from physics.neutrals import _cell_geometry
    return _cell_geometry(grid)


if __name__ == '__main__':
    main()
