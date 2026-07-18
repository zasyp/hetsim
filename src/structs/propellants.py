# Ready-made propellant gases: WorkingSubstance factories wired to the
# HallThruster.jl reaction tables copied into <repo>/reactions
# (see reactions/explanation.md for provenance).
#
# Each factory loads single ionization (ground -> +1) only; the
# multi-charge tables (e.g. ionization_Xe_Xe2+.dat) sit unused until the
# model grows multiply-charged ions. Data completeness differs per gas:
# Xe and Kr have the full set, N2 has no excitation table (its E_c
# reduces to the bare ionization threshold), and Ar has ionization only
# — no elastic table means no nu_en, so argon() cannot feed the electron
# fluid model until an elastic_Ar table is added.

from pathlib import Path

import scipy.constants as cst

from .classes import WorkingSubstance

REACTIONS_DIR = Path(__file__).resolve().parents[2] / "reactions"


def _build(
        name:str,
        mass_amu:float,
        elastic:bool = True,
        excitation:bool = True,
        reactions_dir:str | Path = REACTIONS_DIR,
        ) -> WorkingSubstance:
    d = Path(reactions_dir)
    gas = WorkingSubstance(name=name, mass=mass_amu * cst.atomic_mass)
    gas.load_hallthruster_ionization(d / f"ionization_{name}_{name}+.dat")
    if elastic:
        gas.load_hallthruster_elastic(d / f"elastic_{name}.dat")
    if excitation:
        gas.load_hallthruster_excitation(d / f"excitation_{name}.dat")
    return gas


def xenon() -> WorkingSubstance:
    return _build("Xe", 131.293)


def krypton() -> WorkingSubstance:
    return _build("Kr", 83.798)


def argon() -> WorkingSubstance:
    # ionization data only: no elastic/excitation tables in reactions/
    return _build("Ar", 39.948, elastic=False, excitation=False)


def nitrogen() -> WorkingSubstance:
    # no excitation table in reactions/
    return _build("N2", 2 * 14.0067, excitation=False)
