# Ready-made channel wall materials: WallMaterial factories wired to the
# SEE-yield coefficient tables in <repo>/materials/walls
# (see materials/walls/explanation.md for provenance and references).
#
# Each file is a `key value` list (a, b, gamma_2plusb) with `#` comments;
# the coefficients are the Maxwellian-averaged SEE fit of Goebel & Katz,
# Fundamentals of Electric Propulsion, Ch. 7, Table 7-1 / Eq. 7.3-30.

from pathlib import Path

from .classes import WallMaterial

WALLS_DIR = Path(__file__).resolve().parents[2] / "materials" / "walls"


def _load(name:str, walls_dir:str | Path = WALLS_DIR) -> WallMaterial:
    """Read materials/walls/<name>.txt (`key value` lines, `#` comments)
    and build the WallMaterial with its a, b, gamma_2plusb coefficients.
    """
    params:dict[str, float] = {}
    with open(Path(walls_dir) / f"{name}.txt") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, value = line.split()
            params[key] = float(value)
    return WallMaterial(name=name, **params)


def boron_nitride() -> WallMaterial:
    return _load("BN")


def bn_sio2() -> WallMaterial:
    return _load("BNSiO2")


def alumina() -> WallMaterial:
    return _load("Al2O3")


def stainless_steel() -> WallMaterial:
    return _load("stainless_steel")
