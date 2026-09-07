# The SPT-70 calculation: the whole model, end to end, into one folder.
#
# This is the project's single example. It runs everything the code currently
# knows how to do and writes the result out rather than printing a number:
#
#   magnetostatics   nodal-P1 FEM with AMR at the pole tips, B differentiated
#                    from a continuous psi so div B is machine-zero
#                    (magnetics.fem, reached through field_on_grid)
#   heavy species    kinetic neutrals and ions as macroparticles — flux
#                    Maxwellian injection at the anode, cylindrical push,
#                    MCC ionization, wall and anode re-emission
#   electrons        fluid, on lambda-layers of the real field: collisions,
#                    anomalous cross-field mobility, thermalized potential,
#                    the energy equation with wall losses (blocks 1-6)
#   coupling         the two halves drive each other — see solver.hybrid
#
# The scripts under legacy/ each answer one question about one block and print
# to the terminal; several of them still run on prescribed densities. This one
# is the "what does the model actually say" report.
#
#   python -m src.examples.spt70                  # 300 steps -> out/
#   python -m src.examples.spt70 1500 run_a       # longer, named folder
#
# What lands in the folder:
#
#   setup.txt       the configuration that produced everything else.
#   maps/           one PNG per 2D field: B, lambda, n_e, n_n, Te, phi, E_z,
#                   E_r, |E| plus the ion velocity moments v_z, v_r, |v| and
#                   the ionization rate S_iz.
#   traces.png      currents, populations and peak Te against time — where the
#                   breathing mode shows up if it shows up.
#   profiles.png    mid-channel axial cuts of everything at the end of the run.
#   beam.png        exit speed and divergence angle distributions.
#   performance.txt thrust, Isp, exit velocity, divergence, utilizations,
#                   mass balance.
#   history.csv     the per-macro-step record, for plotting elsewhere.
#
# THE BEAM WINDOW. Thrust is meaningless during startup: while the seed plasma
# flushes out, the ions crossing the boundary are the seed, not the discharge.
# So the beam moments are reset after `settle_fraction` of the run and the
# performance numbers describe only the remainder. If the run is short, that
# remainder is still transient — the file says so.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from src.config import spt70
from src.viz import save_field_maps, plot_field
from src.solver import HybridSolver, HybridSettings
from src.ions.moments import mean_velocity
from src.electron_liquid.ionization import ionization_source
from src.magnetics.spt70_system import DOMAIN_R, DOMAIN_Z, check_conformal


def main(macro_steps: int = 3000, out_dir: str = "out",
         settle_fraction: float = 0.5, grid=None, electron=None):
    """Run the reference discharge and write the report folder.

    grid / electron override the Grid2D and the SolverSettings from
    config.spt70() — they exist so a resolution or calibration sweep can
    launch several runs side by side without editing the source between
    them. Both default to the reference configuration.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thruster, ref_grid = spt70()
    grid = ref_grid if grid is None else grid
    settings = HybridSettings(seed_n_i=1e17)
    if electron is not None:
        settings.electron = electron
    hybrid = HybridSolver(thruster, grid, settings)

    settle = int(macro_steps * settle_fraction)
    print(f"dt {hybrid.dt:.2e} s (ions) / {hybrid.dt_slow:.2e} s (everything else)")
    print(f"{macro_steps} macro-steps = {macro_steps*hybrid.dt_slow*1e6:.0f} us; "
          f"beam window opens at step {settle}\n")

    _write_setup(hybrid, out / "setup.txt", macro_steps, settle)

    every = max(1, macro_steps // 15)
    hybrid.run(settle, verbose=True, every=every)
    hybrid.reset_beam()
    print(f"  --- beam window opens at t = {hybrid.time*1e6:.1f} us ---")
    hybrid.run(macro_steps - settle, verbose=True, every=every)

    _write_maps(hybrid, out / "maps")
    _write_traces(hybrid, out / "traces.png")
    _write_profiles(hybrid, out / "profiles.png")
    _write_beam(hybrid, out / "beam.png")
    _write_history(hybrid, out / "history.csv")
    _write_performance(hybrid, out / "performance.txt", macro_steps, settle)

    print(f"\neverything written to {out}/")
    return hybrid


# --- setup ----------------------------------------------------------------

def _write_setup(hybrid, path: Path, macro_steps, settle):
    """Record what produced this folder.

    A report that does not say which discharge, grid and closures it came from
    is a folder of pretty pictures. Written BEFORE the run, so it survives a
    crash or a Ctrl-C halfway through — that is exactly when you want to know
    what was being run.
    """
    thr, g, s = hybrid.thruster, hybrid.grid, hybrid.s
    e = s.electron
    dr = (g.max_r - g.min_r) / (g.N_r - 1)
    dz = (g.max_z - g.min_z) / (g.N_z - 1)
    bad = check_conformal(np.linspace(0.0, DOMAIN_R, 121),
                          np.linspace(0.0, DOMAIN_Z, 221))

    lines = [
        "SPT-70 run — setup",
        "=" * 62,
        "",
        "thruster",
        "-" * 62,
        f"  channel            r {thr.r_min*1e3:.1f}-{thr.r_max*1e3:.1f} mm, "
        f"L = {thr.channel_length*1e3:.0f} mm",
        f"  propellant         {thr.propellant.name}, mdot (anode) = {thr.mdot*1e6:.2f} mg/s",
        f"  V_d discharge voltage  {thr.voltage} V",
        f"  B_r max            {thr.B_r_max*1e4:.0f} G (calibration target at "
        f"the mid-channel exit)",
        f"  anode / walls      {thr.temperature_anode:.0f} K, "
        f"{thr.wall_material.name}",
        "",
        "grid",
        "-" * 62,
        f"  {g.N_z} x {g.N_r} nodes, dz = {dz*1e3:.2f} mm, dr = {dr*1e3:.2f} mm",
        f"  z {g.min_z*1e3:.0f}-{g.max_z*1e3:.0f} mm (anode at 0), "
        f"r {g.min_r*1e3:.0f}-{g.max_r*1e3:.0f} mm",
        "",
        "magnetostatics",
        "-" * 62,
        "  nodal-P1 FEM, conformal mesh, AMR at the pole tips (magnetics.fem)",
        "  B is differentiated from the interpolated psi on this grid, so",
        "  div B = 0 to machine precision and lambda is continuous through the",
        "  iron. FEM solve is cached across the run — the field never changes.",
        f"  base FEM mesh conformal: {'yes' if not bad else f'NO — {bad}'}",
        "",
        "heavy species (kinetic)",
        "-" * 62,
        f"  W_n neutral weight     {s.weight:.2e} real atoms per macroparticle",
        f"  ion weight         {hybrid.ion_weight:.2e} "
        f"(= weight / {s.weight_ratio:g})",
        f"  dt      ion push step  {hybrid.dt:.2e} s (CFL-limited)",
        f"  dt_slow            {hybrid.dt_slow:.2e} s "
        f"(= {s.n_sub} x dt: injection, neutrals, ionization, electrons)",
        f"  density smoothing  {s.smooth_passes} binomial passes",
        f"  seeds              n_n = {s.seed_n_n:.2e}, n_i = {s.seed_n_i:.2e} 1/m^3"
        f" at {s.seed_T_i:.0f} K",
        f"  wall re-emission   {s.T_wall:.0f} K",
        "",
        "electron fluid",
        "-" * 62,
        f"  N_lambda field-line layers  {e.n_layers}",
        f"  anomalous transport {e.anomaly_preset} preset, "
        f"p_B = {e.background_pressure_torr:g} Torr",
        f"  Te boundaries      anode {e.Te_anode} eV, cathode {e.Te_cathode} eV",
        f"  kappa_coeff        {e.kappa_coeff:g}   dimensionless number in"
        f" kappa_perp = coeff*e*n_e*Te*mu_perp;",
        f"                     classical values are 5/2 (kinetic theory) and"
        f" 4.7 (Braginskii, magnetized limit)",
        f"  Gummel loop        relax {e.relax:g}, tol {e.tol:g} eV, "
        f"max {e.max_iter} iter",
        "",
        "run",
        "-" * 62,
        f"  macro-steps        {macro_steps} "
        f"({macro_steps*hybrid.dt_slow*1e6:.0f} us)",
        f"  beam window        from step {settle} "
        f"({(macro_steps-settle)*hybrid.dt_slow*1e6:.0f} us)",
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)


# --- maps -----------------------------------------------------------------

def _write_maps(hybrid, map_dir: Path):
    """Electron/field maps from field_maps, plus the ion-specific ones."""
    st = hybrid.state
    save_field_maps(st, map_dir)

    v_z, v_r = mean_velocity(hybrid.ions, hybrid.grid)
    S_iz = ionization_source(st.Te, st.n_e, st.n_n, st.gas)

    extra = [
        (v_z, "ion_v_z", "Ion mean axial velocity", "v_z, km/s", 1e-3,
         "RdBu_r", True),
        (v_r, "ion_v_r", "Ion mean radial velocity", "v_r, km/s", 1e-3,
         "RdBu_r", True),
        (np.hypot(v_z, v_r), "ion_speed", "Ion mean speed", "|v|, km/s", 1e-3,
         "plasma", False),
        (S_iz, "S_iz", "Ionization rate S_iz", "S_iz, m$^{-3}$s$^{-1}$", 1.0,
         "inferno", False),
    ]
    for field, name, title, label, scale, cmap, signed in extra:
        plot_field(field, hybrid.thruster, hybrid.grid, title, label,
                    scale, cmap, signed, map_dir / f"{name}.png")
    print(f"saved 4 ion maps to {map_dir}/")


# --- plots ----------------------------------------------------------------

def _write_traces(hybrid, path: Path):
    h = hybrid.history
    t = np.array([x["time"] for x in h]) * 1e6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    for key, label in [("I_iz", "ionization"), ("I_beam", "beam"),
                       ("I_wall", "walls"), ("I_anode", "anode")]:
        ax[0].plot(t, [x[key] for x in h], label=label, lw=1.2)
    ax[0].set_ylabel("current, A"); ax[0].set_title("currents")
    ax[0].legend(fontsize=8)

    ax[1].plot(t, [x["n_ion_macro"] for x in h], label="ions")
    ax[1].plot(t, [x["n_neutral_macro"] for x in h], label="neutrals")
    ax[1].set_ylabel("macroparticles"); ax[1].set_title("populations")
    ax[1].legend(fontsize=8)

    ax[2].plot(t, [x["Te_peak"] for x in h], color="crimson")
    ax[2].set_ylabel("peak Te, eV"); ax[2].set_title("electron temperature")

    for a in ax:
        a.set_xlabel("t, us"); a.grid(alpha=0.3)
        a.axvline(hybrid.beam_t0 * 1e6, color="0.4", ls=":", lw=1)
    fig.suptitle("time traces (dotted line: beam window opens)")
    fig.savefig(path, dpi=140); plt.close(fig)


def _write_profiles(hybrid, path: Path):
    st, g, thr = hybrid.state, hybrid.grid, hybrid.thruster
    z = g.z_nodes() * 1e3
    jm = int(np.argmin(np.abs(g.r_nodes() - 0.5 * (thr.r_min + thr.r_max))))
    L = thr.channel_length * 1e3
    v_z, _ = mean_velocity(hybrid.ions, g)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    ax[0, 0].semilogy(z, st.n_e[:, jm], label="n_e = n_i")
    ax[0, 0].semilogy(z, st.n_n[:, jm], label="n_n")
    ax[0, 0].set_ylabel("density, m$^{-3}$"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(z, st.Te[:, jm], color="crimson")
    ax[0, 1].set_ylabel("Te, eV")
    ax[1, 0].plot(z, st.phi[:, jm])
    ax[1, 0].set_ylabel("phi, V")
    ax[1, 1].plot(z, v_z[:, jm] * 1e-3, color="darkgreen")
    ax[1, 1].set_ylabel("ion mean v_z, km/s")
    for a in ax.ravel():
        a.set_xlabel("z, mm"); a.grid(alpha=0.3)
        a.axvline(L, color="0.5", ls="--", lw=1)
    fig.suptitle("mid-channel axial profiles at the end of the run")
    fig.savefig(path, dpi=140); plt.close(fig)


def _write_beam(hybrid, path: Path):
    (vc, vh), (tc, th) = hybrid.beam.histograms()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax[0].step(vc, vh, where="mid")
    ax[0].set_xlabel("exit speed, km/s"); ax[0].set_ylabel("ions (weighted)")
    ax[0].set_title("beam speed distribution")
    ax[1].step(tc, th, where="mid", color="darkorange")
    ax[1].set_xlabel("divergence half-angle, deg"); ax[1].set_ylabel("ions (weighted)")
    ax[1].set_title("beam divergence")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"beam over {(hybrid.time - hybrid.beam_t0)*1e6:.0f} us")
    fig.savefig(path, dpi=140); plt.close(fig)


def _write_history(hybrid, path: Path):
    h = hybrid.history
    keys = list(h[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for rec in h:
            f.write(",".join(f"{rec[k]:.6g}" for k in keys) + "\n")


# --- numbers --------------------------------------------------------------

def _I_ceiling(hybrid) -> float:
    """Largest beam current the mass flow can support [A]: every atom singly
    ionized and every ion reaching the beam, I_max = e * mdot / m_i. eta_m is
    measured against it, and no I_b can exceed it (1.837 A for xenon at
    2.5 mg/s) -- which is why I_b is not the ~2.2 A datasheet DISCHARGE
    current I_d, a quantity this model does not yet compute.
    """
    import scipy.constants as cst
    th = hybrid.thruster
    return cst.elementary_charge * th.mdot / th.mass


def _write_performance(hybrid, path: Path, macro_steps, settle):
    perf = hybrid.performance()
    mb = hybrid.mass_balance()
    window = (hybrid.time - hybrid.beam_t0) * 1e6
    residence = (hybrid.thruster.channel_length
                 / hybrid.thruster.propellant.thermal_speed(
                     hybrid.thruster.temperature_anode)) * 1e6

    lines = [
        "SPT-70 run — performance",
        "=" * 60,
        f"macro-steps          {macro_steps} ({hybrid.time*1e6:.1f} us total)",
        f"beam window          {window:.1f} us (from step {settle})",
        f"neutral residence    {residence:.0f} us  <- nothing equilibrates faster",
        "",
        "thrust and efficiency",
        "-" * 60,
    ]
    if perf:
        lines += [
            f"  T       thrust                  {perf['thrust_mN']:.3f} mN"
            f"   (ions {perf['thrust_ion_mN']:.3f}, "
            f"neutrals {perf['thrust_neutral_mN']:.3f})",
            f"  I_sp    specific impulse        {perf['Isp_s']:.0f} s"
            f"   (on total mdot)",
            f"  I_b     beam ion current        {perf['I_beam_A']:.3f} A"
            f"   (ceiling e*mdot/m_i = {_I_ceiling(hybrid):.3f} A)",
            "",
            "ion velocities",
            "-" * 60,
            f"  <v_z>   mean axial velocity     {perf['v_exit_mean_kms']:.2f} km/s",
            f"  <|v|>   mean speed              {perf['v_speed_mean_kms']:.2f} km/s",
            f"  V_b     beam voltage            {perf['V_beam_V']:.1f} V "
            f"(V_d = {hybrid.thruster.voltage} V)",
            f"  eta_v   voltage utilization     {perf['voltage_utilization']:.3f}"
            f"   = V_b / V_d",
            f"  eta_m   mass utilization        {perf['mass_utilization']:.3f}"
            f"   = I_b / (e*mdot/m_i)",
            "",
            "divergence",
            "-" * 60,
            f"  <theta> mean half-angle         {perf['divergence_mean_deg']:.1f} deg",
            f"  theta95 95% cone half-angle     {perf['divergence_95_deg']:.1f} deg",
            f"  eta_d   divergence efficiency   {perf['divergence_efficiency']:.3f}"
            f"   = <cos theta>",
        ]
    else:
        lines.append("  no ions crossed the plume boundary in the beam window.")

    lines += [
        "",
        "mass balance [particles]",
        "-" * 60,
        f"  N_inj   injected by the anode   {mb['injected']:.4e}",
        f"  N_seed  present at t = 0        {mb['seeded']:.4e}",
        f"  N_dom   still in the domain     {mb['in_domain']:.4e}",
        f"  N_beam  left as beam ions       {mb['beam']:.4e}",
        f"  N_esc   left as un-ionized gas  {mb['neutral_escape']:.4e}",
        f"  ---     residual                {mb['residual']:.4e}"
        f"  (rel {mb['rel_error']:.2e})",
        "",
        "  The residual is the stochastic injector, not a leak: inject_on_grid",
        "  emits a whole number of macroparticles per step and rounds the",
        "  remainder randomly, so the realised flow random-walks around mdot.",
        "",
        "read this before quoting any number above",
        "-" * 60,
    ]
    if hybrid.time * 1e6 < 3 * residence:
        lines.append(
            f"  WARNING: the run is {hybrid.time*1e6:.0f} us, under three neutral\n"
            f"  residence times. These numbers describe a startup transient, not\n"
            f"  a thruster. Use them to check that the machinery works, nothing\n"
            f"  more.")
    else:
        lines.append(
            "  The run is long enough to have refilled the channel several\n"
            "  times, but a self-sustained breathing cycle still has to be\n"
            "  confirmed in traces.png before these numbers mean anything.")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    folder = sys.argv[2] if len(sys.argv) > 2 else "out"
    main(steps, folder)
