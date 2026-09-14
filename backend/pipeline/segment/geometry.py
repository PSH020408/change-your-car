"""P2-1 — track geometry from position telemetry.

Three things have to be right before a curvature number means anything.

SCALE. FastF1's X/Y are integers in some fixed unit, and every downstream
number (curvature in 1/m, corner radius in m) depends on knowing it. Rather
than hardcode a convention that could be wrong, the scale is DERIVED: we
already know the lap's true length in metres from `add_distance()`, so
    metres per unit = lap_distance_m / path_length_in_xy_units
This is self-calibrating and stays correct if the convention ever changes.

RESAMPLING. Features must never be resampled (DECISIONS.md D1) because
interpolating a speed trace invents driving that did not happen. Geometry is
the opposite case: the racing line is a smooth spatial curve, sampled at
uneven intervals only because the car's speed varied. Interpolating it onto a
uniform arc-length grid recovers the curve, it does not invent it. The
distinction is between a signal that is genuinely discontinuous (throttle,
brake) and one that is continuous by construction (position).

SMOOTHING. Curvature is a second derivative, so it amplifies noise
quadratically. Raw GPS jitter of a metre becomes a phantom corner. The
Savitzky-Golay window is expressed in METRES and converted to samples, so the
same setting behaves identically on a 10 m and a 20 m grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:                                   # SciPy is in requirements, but this
    from scipy.signal import savgol_filter          # module is the one piece
    _HAVE_SCIPY = True                 # of the pipeline worth keeping
except ImportError:                    # runnable on a bare numpy install —
    _HAVE_SCIPY = False                # see _savgol_numpy below.


@dataclass
class TrackGeometry:
    distance_m: np.ndarray        # uniform grid
    x_m: np.ndarray               # metres, smoothed
    y_m: np.ndarray
    curvature_1pm: np.ndarray     # signed: + left, - right
    radius_m: np.ndarray          # 1/|k|, clipped
    lap_length_m: float
    metres_per_unit: float
    closure_error_m: float        # excess beyond the expected one-step gap
    n_samples_raw: int
    grid_step_m: float = 10.0


def calibrate_scale(x: np.ndarray, y: np.ndarray, lap_distance_m: float) -> float:
    """Metres per X/Y unit, derived from the lap's known length."""
    seg = np.hypot(np.diff(x), np.diff(y))
    path_units = float(np.nansum(seg))
    if path_units <= 0 or not np.isfinite(lap_distance_m) or lap_distance_m <= 0:
        return 1.0
    return float(lap_distance_m / path_units)


def resample_to_grid(distance: np.ndarray, *channels: np.ndarray,
                     step_m: float = 10.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Linear interpolation onto a uniform arc-length grid.

    Only ever applied to position channels — see the module docstring.
    """
    d = np.asarray(distance, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    if len(d) < 4:
        raise ValueError("not enough position samples to build a grid")

    # enforce strict monotonicity so np.interp is well defined
    keep = np.concatenate([[True], np.diff(d) > 0])
    d = d[keep]
    # linspace, not arange: arange drops the final partial step, which on a
    # closed circuit shows up as a phantom gap at start-finish.
    n = max(int(round((d[-1] - d[0]) / step_m)), 3)
    grid = np.linspace(d[0], d[-1], n, endpoint=False)

    out = []
    for c in channels:
        c = np.asarray(c, dtype=float)[ok][keep]
        out.append(np.interp(grid, d, c))
    return grid, out


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _savgol_numpy(v: np.ndarray, window: int, polyorder: int,
                  wrap: bool = True) -> np.ndarray:
    """Savitzky-Golay via least squares, so SciPy stays optional.

    Convolution with coefficients from the pseudo-inverse of the Vandermonde
    matrix over the window's offsets — the same construction SciPy uses.

    Edges are padded CIRCULARLY, because a racing circuit is a closed loop:
    the samples before the start line are the ones at the end of the lap. A
    reflecting pad instead mirrors the curve back on itself at start-finish,
    which flattens the geometry exactly where most circuits are straight but
    can invent or erase curvature on the ones that are not. Measured on a
    synthetic circle, reflecting padding left an 18 m closure error; wrapping
    brings it under a metre.
    """
    half = window // 2
    offs = np.arange(-half, half + 1, dtype=float)
    vander = np.vander(offs, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(vander)[0]            # row 0 = smoothed value
    pad = "wrap" if wrap else "reflect"
    padded = np.pad(v, half, mode=pad)
    return np.convolve(padded, coeffs[::-1], mode="valid")


def _smooth1d(v: np.ndarray, window: int, polyorder: int,
              wrap: bool = True) -> np.ndarray:
    if _HAVE_SCIPY:
        # SciPy grew a "wrap" mode for exactly this case.
        return savgol_filter(v, window, polyorder,
                             mode="wrap" if wrap else "interp")
    return _savgol_numpy(v, window, polyorder, wrap=wrap)


def smooth_xy(x: np.ndarray, y: np.ndarray, step_m: float,
              window_m: float = 60.0, polyorder: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay with the window specified in metres, not samples."""
    win = _odd(max(polyorder + 2, int(round(window_m / max(step_m, 1e-6)))))
    win = min(win, _odd(len(x) - 1) if len(x) > polyorder + 2 else polyorder + 3)
    if win <= polyorder or len(x) <= win:
        return x, y
    return _smooth1d(x, win, polyorder), _smooth1d(y, win, polyorder)


def curvature(x: np.ndarray, y: np.ndarray, step_m: float,
              closed: bool = True) -> np.ndarray:
    """Signed curvature of a uniformly-sampled planar curve.

        k = (x' y" - y' x") / (x'^2 + y'^2)^(3/2)

    Sign is kept (positive = left-hand turn) because corner direction is a
    real feature: a driver's braking and traction biases are not symmetric
    between left and right handers.
    """
    if closed:
        # np.gradient uses one-sided differences at the ends; on a loop the
        # true neighbour is the other end, so wrap before differentiating.
        xw, yw = np.r_[x[-2:], x, x[:2]], np.r_[y[-2:], y, y[:2]]
        dx, dy = np.gradient(xw, step_m), np.gradient(yw, step_m)
        ddx, ddy = np.gradient(dx, step_m), np.gradient(dy, step_m)
        dx, dy, ddx, ddy = dx[2:-2], dy[2:-2], ddx[2:-2], ddy[2:-2]
    else:
        dx, dy = np.gradient(x, step_m), np.gradient(y, step_m)
        ddx, ddy = np.gradient(dx, step_m), np.gradient(dy, step_m)
    denom = (dx * dx + dy * dy) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (dx * ddy - dy * ddx) / denom
    return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)


def build(frame: pd.DataFrame, lap_distance_m: float, step_m: float = 10.0,
          smooth_window_m: float = 60.0) -> TrackGeometry:
    """Position telemetry for one lap -> calibrated, smoothed geometry."""
    need = {"distance_m", "pos_x", "pos_y"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"reference lap is missing {sorted(missing)}")

    d = frame["distance_m"].to_numpy(dtype=float)
    xu = frame["pos_x"].to_numpy(dtype=float)
    yu = frame["pos_y"].to_numpy(dtype=float)

    finite = np.isfinite(d) & np.isfinite(xu) & np.isfinite(yu)
    d, xu, yu = d[finite], xu[finite], yu[finite]
    if len(d) < 20:
        raise ValueError(f"only {len(d)} usable position samples")

    scale = calibrate_scale(xu, yu, lap_distance_m)
    grid, (xg, yg) = resample_to_grid(d, xu * scale, yu * scale, step_m=step_m)
    xs, ys = smooth_xy(xg, yg, step_m, window_m=smooth_window_m)
    k = curvature(xs, ys, step_m)

    with np.errstate(divide="ignore"):
        radius = np.where(np.abs(k) > 1e-9, 1.0 / np.abs(k), np.inf)

    return TrackGeometry(
        distance_m=grid,
        x_m=xs, y_m=ys,
        curvature_1pm=k,
        radius_m=np.clip(radius, 0.0, 1e5),
        lap_length_m=float(grid[-1] - grid[0]),
        metres_per_unit=scale,
        # A lap is a closed loop, so the final sample should sit exactly one
        # grid step short of the first. Anything beyond that means the
        # reference lap was cut badly or the position stream drifted, and
        # every segment boundary downstream is suspect.
        closure_error_m=max(0.0, float(
            np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) - step_m),
        grid_step_m=float(step_m),
        n_samples_raw=int(len(d)),
    )
