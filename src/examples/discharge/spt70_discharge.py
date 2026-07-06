"""SPT-70-class Hall thruster discharge: full 2D/3V hybrid PIC-MCC example.

Chains together every part of the code base:

1. The magnetic system of the SPT-70 example (iron circuit + coils) is
   solved with the axisymmetric flux solver and calibrated to ~200 G at the
   channel exit; the flux function psi is the magnetic streamfunction lambda.
2. The field and streamfunction are interpolated onto the discharge grid
   (anode plane -> near field, Fig. 2.3 of the reference thesis).
3. A `Simulation` is assembled: neutral injection at the anode, leapfrog
   heavy-particle transport with wall accommodation, per-species deposition
   (Ruyten radial shape factors), the field-line electron fluid (1-D Ohm's
   law -> discharge current I_T, thermalized potential -> phi, E; electron
   energy equation) and MCC ionization with the collision-multiplier, using
   the tabulated xenon rate coefficients bundled in physics/data.
4. The discharge is seeded with a dilute quasineutral plasma and evolved;
   time traces and 2-D maps are saved next to this script.

Run:  python spt70_discharge.py [--steps N] [--no-show]
The default 4000 steps (~100 us of discharge time) take a few minutes;
the discharge current typically ignites and settles within the first
30-50 us. Use fewer steps for a quick look.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physics.classes import Grid2D
from physics.magnetic import solve_axisymmetric_flux, field_from_flux
from physics.pusher import ChannelGeometry
from physics.electrons import lambda_at
from physics.simulation import Simulation
from numerical.numerical_funcs import bilinear_gather
from examples.magnetic_field.spt70_magnetic_field import (
    build_spt70_system, calibrate_field_scale,
    CHANNEL_R_IN, CHANNEL_R_OUT, CHANNEL_Z0, CHANNEL_Z_EXIT,
)

# --- discharge domain (Fig. 2.3): anode plane to the near-field ------------
R_MAX = 0.055          # radial extent of the simulation domain, m
Z_MAX = 0.085          # axial extent (anode at CHANNEL_Z0 = 0.020 m)
N_R, N_Z = 50, 50      # ~1.25 mm cells in both directions
Z_CATHODE = 0.065      # virtual cathode line plane (2 cm past the exit)

# --- operating point (SPT-70-like) ------------------------------------------
MDOT = 2.5e-6          # anode mass flow, kg/s (~2.5 mg/s Xe)
V_DISCHARGE = 300.0    # discharge voltage, V
DT = 2.5e-8            # heavy-particle time step, s
N_INJECT = 15          # neutral macroparticles injected per step
SEED_DENSITY = 1e17    # ignition seed plasma density, 1/m^3
SEED_EPS = 8.0         # ignition electron energy, eV


def build_discharge_field() -> tuple[Grid2D, np.ndarray, np.ndarray, np.ndarray]:
    """Solve + calibrate the SPT-70 magnetic system and interpolate
    (Br, Bz, lambda) onto the discharge grid."""
    r_m, z_m, mu_r, jphi = build_spt70_system(121, 201)
    psi = solve_axisymmetric_flux(r_m, z_m, mu_r, jphi)
    Br_m, Bz_m = field_from_flux(psi, r_m, z_m)
    scale = calibrate_field_scale(Br_m, Bz_m, r_m, z_m)
    psi, Br_m, Bz_m = psi * scale, Br_m * scale, Bz_m * scale

    g = Grid2D(0.0, R_MAX, N_R, CHANNEL_Z0, Z_MAX, N_Z)
    R, Z = g.mesh()
    pr, pz = R.ravel(), Z.ravel()
    Br = bilinear_gather(Br_m, r_m, z_m, pr, pz).reshape(R.shape)
    Bz = bilinear_gather(Bz_m, r_m, z_m, pr, pz).reshape(R.shape)
    lam = bilinear_gather(psi, r_m, z_m, pr, pz).reshape(R.shape)
    return g, Br, Bz, lam


def run(n_steps: int, show: bool = True) -> None:
    t0 = time.perf_counter()
    g, Br, Bz, lam = build_discharge_field()
    geom = ChannelGeometry(r_in=CHANNEL_R_IN, r_out=CHANNEL_R_OUT,
                           z_exit=CHANNEL_Z_EXIT)

    r_mid = 0.5 * (CHANNEL_R_IN + CHANNEL_R_OUT)
    lam_a = lambda_at(g, lam, r_mid, CHANNEL_Z0 + 1e-4)
    lam_c = lambda_at(g, lam, r_mid, Z_CATHODE)
    print(f"magnetic field ready ({time.perf_counter()-t0:.1f} s); "
          f"lambda anode/cathode = {lam_a:.3e} / {lam_c:.3e}")

    sim = Simulation(
        g, geom, Br, Bz, lam, lam_a, lam_c,
        mdot=MDOT, dt=DT, n_inject=N_INJECT, V_discharge=V_DISCHARGE,
        rng=np.random.default_rng(1),
    )
    sim.seed_ions(n0=SEED_DENSITY, n_macro=4000, eps_seed=SEED_EPS)

    hist = {'t': [], 'I_T': [], 'n_active': [], 'n_ions': [], 'eps_max': []}
    t0 = time.perf_counter()
    # Ignition transient is over well before 60% of the default run; restart
    # the performance tally there so the numbers describe the steady state.
    i_reset = int(0.6 * n_steps)
    for i in range(n_steps):
        if i == i_reset:
            sim.reset_tally()
        d = sim.step()
        hist['t'].append(d['time'] * 1e6)
        hist['I_T'].append(d['I_T'])
        hist['n_active'].append(d['n_active'])
        pa = sim.particles
        hist['n_ions'].append(int(np.count_nonzero(
            pa.active[:pa.n] & (pa.charge[:pa.n] > 0))))
        hist['eps_max'].append(d['eps_max'])
        if i % max(1, n_steps // 20) == 0:
            print(f"step {i:5d}  t={d['time']*1e6:7.2f} us  "
                  f"N={d['n_active']:6d} (ions {hist['n_ions'][-1]:6d})  "
                  f"I_T={d['I_T']:9.4f} A  eps_max={d['eps_max']:6.1f} eV  "
                  f"ioniz/step={sum(d['ionization'].values()):4d}")
    print(f"{n_steps} steps in {time.perf_counter()-t0:.1f} s")

    perf = sim.performance()
    print("\n--- performance over the last 40% of the run (2.108-2.115) ---")
    print(f"thrust                 : {perf['thrust_N']*1e3:8.3f} mN")
    print(f"specific impulse       : {perf['Isp_s']:8.1f} s")
    print(f"ion beam current       : {perf['ion_current_A']:8.3f} A")
    print(f"discharge current I_T  : {np.mean(hist['I_T'][-n_steps//5:]):8.3f} A"
          f"  (mean over last 20% of the run)")
    print(f"propellant utilization : {perf['propellant_utilization']:8.3f}")

    _plot(sim, g, geom, hist, show)


def _plot(sim, g, geom, hist, show):
    R, Z = g.mesh()
    ex = [g.z_min * 1e3, g.z_max * 1e3, g.r_min * 1e3, g.r_max * 1e3]

    def walls(ax):
        ax.plot([geom.z_exit * 1e3] * 2, [geom.r_in * 1e3, 0], 'k-', lw=1)
        for rr in (geom.r_in, geom.r_out):
            ax.plot([g.z_min * 1e3, geom.z_exit * 1e3], [rr * 1e3] * 2,
                    'k-', lw=1.5)

    # Show only the plasma domain: inside the solid thruster body the fields
    # are fill/extension values with no physical meaning.
    body = ~sim.mask

    def masked(f):
        return np.where(body, np.nan, f)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    panels = [
        (masked(sim.electrons.phi), 'plasma potential phi, V', dict()),
        (masked(np.log10(np.maximum(sim.n_e, 1e12))),
         'log10 n_e [1/m^3]', dict()),
        (masked(np.log10(np.maximum(sim.neutrals.n, 1e14))),
         'log10 n_a [1/m^3]', dict()),
        (masked(sim.electrons.eps_nodes), 'mean electron energy, eV', dict()),
    ]
    for ax, (f, title, kw) in zip(axs.ravel(), panels):
        im = ax.imshow(f, origin='lower', aspect='auto',
                       extent=[ex[0], ex[1], ex[2], ex[3]], **kw)
        fig.colorbar(im, ax=ax)
        walls(ax)
        ax.set_title(title)
    for ax in axs[-1]:
        ax.set_xlabel('z, mm')
    for ax in axs[:, 0]:
        ax.set_ylabel('r, mm')
    fig.suptitle('SPT-70-class discharge: 2-D fields at '
                 f't = {sim.time*1e6:.1f} us')
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name('spt70_discharge_fields.png'), dpi=150)

    fig2, axs2 = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs2[0].plot(hist['t'], hist['I_T'])
    axs2[0].set_ylabel('I_T, A')
    axs2[1].plot(hist['t'], hist['n_active'], label='all')
    axs2[1].plot(hist['t'], hist['n_ions'], label='ions')
    axs2[1].set_ylabel('macroparticles')
    axs2[1].legend()
    axs2[2].plot(hist['t'], hist['eps_max'])
    axs2[2].set_ylabel('max eps, eV')
    axs2[2].set_xlabel('t, us')
    fig2.suptitle('Discharge development')
    fig2.tight_layout()
    fig2.savefig(Path(__file__).with_name('spt70_discharge_history.png'), dpi=150)

    if show:
        plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    run(args.steps, show=not args.no_show)
