"""P2-1 — track geometry from position telemetry.

Three things have to be right before a curvature number means anything.

SCALE. FastF1's X/Y are integers in some fixed unit, and every downstream
number (curvature in 1/m, corner radius in m) depends on knowing it. Rather
than hardcode a convention that could be wrong, the scale is DERIVED: we
already know the lap's true length in metres from `add_distance()`, so
    metres per unit = lap_distance_m / path_length_in_xy_units
This is self-calibrating and stays correct if the convention ever changes.

DERIVATIVES COME FROM THE RAW SAMPLES, NOT FROM A GRID. The first version
interpolated position linearly onto a uniform grid and then differentiated.
That is wrong in a way that looks fine until you check the physics: linear
interpolation turns the racing line into a POLYLINE, whose curvature is zero
inside each segment and a spike at every knot. On the 2022 Australian GP that
produced corners implying 9-10 g of lateral load - roughly twice what an F1
car can generate - and five "straights" containing 21-122 m radii.

So curvature is computed by fitting a local weighted polynomial in arc length
to the RAW, unevenly-spaced samples and differentiating that analytically.
Uneven spacing is handled natively, there is no interpolation to create
artefacts, and the window is specified in metres so it means the same thing
at every speed. Only the already-smooth results are put on a grid afterwards.

PHYSICS IS THE FINAL CHECK. Curvature and speed together give lateral load,
v^2/r. An F1 car peaks around 5-6 g, so a curvature implying more than that
at the speed actually driven is not a corner - it is noise, and gets rejected
rather than classified.
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


# An F1 car's peak lateral load. Anything above this is a numerical artefact,
# not a corner: the highest figures ever recorded in F1 sit around 6 g.
MAX_PHYSICAL_LATERAL_G = 6.5

# Calibrated, not chosen. Swept window x order against a jittered circle of
# known radius (accuracy) and a 70 m hairpin between straights (resolution),
# at 0.2 / 0.5 / 1.0 m of position jitter:
#
#   window  order |  curvature error @ jitter 0.2 / 0.5 / 1.0 m
#   ------------- + ------------------------------------------
#     40 m    3   |    4.3%  /  16.4%  /  79.5%     <- the original setting
#     60 m    3   |    1.7%  /   0.2%  /  42.3%
#     90 m    2   |    0.6%  /   0.1%  /   1.0%     <- only one stable across all
#    130 m    2   |    0.5%  /   0.5%  /   0.2%     (but smears short corners)
#
# A quadratic resists chasing noise where a cubic follows it, and 90 m is the
# widest window that still recovers a 70 m corner at full amplitude.
DEFAULT_SMOOTH_WINDOW_M = 90.0
DEFAULT_POLY_ORDER = 2


@dataclass
class TrackGeometry:
    # --- required ---
    distance_m: np.ndarray        # uniform grid
    x_m: np.ndarray               # metres, smoothed
    y_m: np.ndarray
    curvature_1pm: np.ndarray     # signed: + left, - right
    radius_m: np.ndarray          # 1/|k|, clipped
    lap_length_m: float
    metres_per_unit: float
    closure_error_m: float        # excess beyond the expected one-step gap
    n_samples_raw: int
    # --- optional ---
    speed_kph: np.ndarray | None = None
    lateral_g: np.ndarray | None = None
    implausible_fraction: float = 0.0   # share of raw samples above MAX_PHYSICAL_LATERAL_G
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


def local_poly_derivatives(s: np.ndarray, v: np.ndarray, window_m: float,
                           order: int = 3, closed: bool = True,
                           period: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value and first two derivatives of v(s), from a local weighted fit.

    For each sample, fit a polynomial in (s - s_i) over a +/- window/2 metre
    neighbourhood with tricube weights, then read the derivatives straight off
    the coefficients. Because the fit uses the real arc-length positions,
    uneven spacing needs no interpolation — which is the whole point: the
    polyline artefacts that come from interpolating position first are what
    produced 10 g corners.

    On a closed circuit the neighbourhood wraps through start-finish.
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(s)
    if n < order + 3:
        raise ValueError(f"need at least {order + 3} samples, got {n}")

    L = period if period is not None else (s[-1] - s[0]) * n / max(n - 1, 1)
    half = window_m / 2.0
    val = np.empty(n)
    d1 = np.empty(n)
    d2 = np.empty(n)

    for i in range(n):
        ds = s - s[i]
        if closed and L > 0:
            ds = (ds + L / 2.0) % L - L / 2.0
        m = np.abs(ds) <= half
        if m.sum() < order + 2:                       # widen rather than fail
            m = np.zeros(n, dtype=bool)
            m[np.argsort(np.abs(ds))[: order + 3]] = True
        dsm = ds[m]
        w = np.clip(1.0 - (np.abs(dsm) / (half + 1e-9)) ** 3, 0.0, None) ** 3
        w = np.maximum(w, 1e-6)
        sw = np.sqrt(w)
        A = np.vander(dsm, order + 1, increasing=True) * sw[:, None]
        coef, *_ = np.linalg.lstsq(A, v[m] * sw, rcond=None)
        val[i] = coef[0]
        d1[i] = coef[1] if order >= 1 else 0.0
        d2[i] = 2.0 * coef[2] if order >= 2 else 0.0
    return val, d1, d2


def curvature_from_raw(s: np.ndarray, x: np.ndarray, y: np.ndarray,
                       window_m: float = 90.0, order: int = 2
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed curvature at each raw sample, plus the smoothed x/y."""
    xs, dx, ddx = local_poly_derivatives(s, x, window_m, order)
    ys, dy, ddy = local_poly_derivatives(s, y, window_m, order)
    denom = (dx * dx + dy * dy) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (dx * ddy - dy * ddx) / denom
    return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0), xs, ys


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
          smooth_window_m: float = 90.0, poly_order: int = 2) -> TrackGeometry:
    """Position telemetry for one lap -> calibrated, physically-checked geometry."""
    need = {"distance_m", "pos_x", "pos_y"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"reference lap is missing {sorted(missing)}")

    d = frame["distance_m"].to_numpy(dtype=float)
    xu = frame["pos_x"].to_numpy(dtype=float)
    yu = frame["pos_y"].to_numpy(dtype=float)
    vraw = (pd.to_numeric(frame["speed_kph"], errors="coerce").to_numpy(dtype=float)
            if "speed_kph" in frame.columns else None)

    finite = np.isfinite(d) & np.isfinite(xu) & np.isfinite(yu)
    d, xu, yu = d[finite], xu[finite], yu[finite]
    if vraw is not None:
        vraw = vraw[finite]
    if len(d) < 20:
        raise ValueError(f"only {len(d)} usable position samples")

    keep = np.concatenate([[True], np.diff(d) > 0])
    d, xu, yu = d[keep], xu[keep], yu[keep]
    if vraw is not None:
        vraw = vraw[keep]

    scale = calibrate_scale(xu, yu, lap_distance_m)
    xm, ym = xu * scale, yu * scale

    # Curvature on the RAW samples — no interpolation, no polyline artefacts.
    k_raw, xs_raw, ys_raw = curvature_from_raw(
        d, xm, ym, window_m=smooth_window_m, order=poly_order)

    # Physics gate. A curvature implying more lateral load than the car can
    # generate is noise; zero it rather than let it become a corner.
    implausible = 0.0
    if vraw is not None and np.isfinite(vraw).any():
        v_ms = np.nan_to_num(vraw, nan=0.0) / 3.6
        lat_g_raw = (v_ms ** 2) * np.abs(k_raw) / 9.81
        bad = lat_g_raw > MAX_PHYSICAL_LATERAL_G
        implausible = float(bad.mean())
        k_raw = np.where(bad, 0.0, k_raw)

    # Only smooth results go on the grid; curvature is now a smooth scalar, so
    # interpolating IT is safe in a way interpolating position never was.
    n = max(int(round((d[-1] - d[0]) / step_m)), 3)
    grid = np.linspace(d[0], d[-1], n, endpoint=False)
    xg = np.interp(grid, d, xs_raw)
    yg = np.interp(grid, d, ys_raw)
    kg = np.interp(grid, d, k_raw)

    vg = np.interp(grid, d, np.nan_to_num(vraw, nan=0.0)) if vraw is not None else None
    lat_g = ((vg / 3.6) ** 2 * np.abs(kg) / 9.81) if vg is not None else None

    with np.errstate(divide="ignore"):
        radius = np.where(np.abs(kg) > 1e-9, 1.0 / np.abs(kg), np.inf)

    return TrackGeometry(
        distance_m=grid,
        x_m=xg, y_m=yg,
        curvature_1pm=kg,
        radius_m=np.clip(radius, 0.0, 1e5),
        speed_kph=vg,
        lateral_g=lat_g,
        implausible_fraction=round(implausible, 4),
        lap_length_m=float(grid[-1] - grid[0]),
        metres_per_unit=scale,
        closure_error_m=max(0.0, float(
            np.hypot(xg[-1] - xg[0], yg[-1] - yg[0])) - step_m),
        n_samples_raw=int(len(d)),
        grid_step_m=float(step_m),
    )
