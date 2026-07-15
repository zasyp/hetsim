# Shared helpers for reading tabulated data files exported from public
# plasma-physics databases (LXCat and similar) and turning them into
# interpolators. Used by WorkingSubstance to load ionization-rate data,
# but written generically so any (x, y1, y2, ...) table can reuse it.

from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d


def read_table(filepath:str, ncols:int = None, comments:str = "#") -> np.ndarray:
    """Read a whitespace-separated numeric table, skipping comment lines.

    Raises with a clear message instead of letting a cryptic numpy/scipy
    error surface, since these files are typically hand-exported from a
    website (e.g. LXCat) and easy to get slightly wrong (wrong column
    count, stray header line, empty file).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")

    data = np.loadtxt(path, comments=comments, ndmin=2)

    if data.size == 0:
        raise ValueError(f"{path}: file contains no data rows")

    if ncols is not None and data.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {data.shape[1]}")

    if not np.isfinite(data).all():
        raise ValueError(f"{path}: contains NaN/inf values")

    return data


def sort_by_first_column(data:np.ndarray) -> np.ndarray:
    """Sort table rows by column 0 (ascending).

    interp1d requires a strictly increasing x-array, but tables exported
    by hand from a website aren't always already sorted.
    """
    order = np.argsort(data[:, 0])
    sorted_data = data[order]

    if np.any(np.diff(sorted_data[:, 0]) <= 0):
        raise ValueError("first column has duplicate/repeated values after sorting")

    return sorted_data


_MIN_POINTS = {"linear": 2, "quadratic": 3, "cubic": 4}


def clamped_interpolator(x:np.ndarray, y:np.ndarray, kind:str = "cubic"):
    """Interpolator over (x, y) that holds the edge values constant
    outside the table range, instead of extrapolating wildly - swarm/rate
    tables from LXCat should not be trusted far outside their span.
    """
    required = _MIN_POINTS.get(kind, 2)
    if len(x) < required:
        raise ValueError(
            f"need at least {required} points for kind={kind!r} interpolation, "
            f"got {len(x)} (pass a lower-order kind, e.g. 'linear', or a bigger table)"
        )

    return interp1d(
        x, y,
        kind=kind,
        bounds_error=False,
        fill_value=(y[0], y[-1]),
    )


def load_xy_table(filepath:str, ncols:int, comments:str = "#", kind:str = "cubic"):
    """Read an (N, ncols) table and build a clamped interpolator for every
    column after the first (column 0 is the independent variable, e.g.
    E/N or T_e).

    Returns (raw_data, interpolators), where interpolators is a tuple of
    length ncols - 1, one per dependent column, in column order.
    """
    data = read_table(filepath, ncols=ncols, comments=comments)
    data = sort_by_first_column(data)

    x = data[:, 0]
    interpolators = tuple(
        clamped_interpolator(x, data[:, col], kind=kind)
        for col in range(1, ncols)
    )
    return data, interpolators
