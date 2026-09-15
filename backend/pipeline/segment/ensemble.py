"""Ensemble reference: the circuit's shape from many laps, not one.

Every geometry result so far came from a single lap's position trace. Position
jitter on an F1 car is roughly half a metre to a metre, and curvature is a
second derivative, so that jitter is amplified into the corner count — which
is why the same circuit produced 10, 14 and 14 turns in three different years
from three different template laps. The circuit did not change; the template
did.

Averaging N laps reduces independent jitter by roughly sqrt(N). Twelve laps
cut it by a factor of three or so, which is more than any smoothing window can
deliver without smearing real corners. This is the standard motorsport
approach (an ensemble centreline) and it removes the single-lap lottery.

Two things have to be right for the average to be sharper rather than
blurrier than any one lap:

  PHASE. Distance zero has to be the same physical point on every lap. FastF1
  cuts laps at the timing line, but the first telemetry sample can land
  anywhere in the next 20-50 m, so laps are out of phase by that much. Averaging
  out-of-phase corners smears them. Each lap is shifted to maximise the circular
  cross-correlation of its speed profile with the anchor lap's — braking points
  are the same on every lap, so speed profiles lock together to within a grid
  step.

  LINE. Different drivers take different lines. That is exactly what we want
  averaged out: we are after the circuit's shape, not any one driver's path,
  and the median across laps is the centreline the pipeline should have been
  using from the start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EnsembleResult:
    frame: pd.DataFrame          # distance_m, pos_x, pos_y, speed_kph on a grid
    n_laps: int
    lap_length_m: float
    phase_shifts_m: list[float]
    rejected: int


def _resample(frame: pd.DataFrame, grid: np.ndarray, scale: float) -> dict[str, np.ndarray] | None:
    d = pd.to_numeric(frame["distance_m"], errors="coerce").to_numpy(dtype=float) * scale
    cols = {}
    for name in ("pos_x", "pos_y", "speed_kph"):
        v = pd.to_numeric(frame.get(name), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(d) & np.isfinite(v)
        if ok.sum() < 20:
            return None
        dd, vv = d[ok], v[ok]
        order = np.argsort(dd)
        dd, vv = dd[order], vv[order]
        keep = np.concatenate([[True], np.diff(dd) > 0])
        cols[name] = np.interp(grid, dd[keep], vv[keep], period=float(grid[-1] + (grid[1] - grid[0])))
    return cols


def _phase_shift(anchor_speed: np.ndarray, speed: np.ndarray, max_shift_pts: int) -> int:
    """Circular shift (in grid points) that best aligns `speed` to the anchor."""
    a = anchor_speed - np.nanmean(anchor_speed)
    b = speed - np.nanmean(speed)
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    # circular cross-correlation via FFT
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    lags = np.arange(len(a))
    lags = np.where(lags > len(a) // 2, lags - len(a), lags)
    mask = np.abs(lags) <= max_shift_pts
    best = int(np.argmax(np.where(mask, corr, -np.inf)))
    return int(lags[best])


def build_ensemble(frames: list[pd.DataFrame], lap_length_m: float,
                   step_m: float = 5.0, max_phase_m: float = 80.0,
                   min_laps: int = 3) -> EnsembleResult:
    """Median circuit shape from several laps, phase-aligned on speed."""
    n = max(int(round(lap_length_m / step_m)), 10)
    grid = np.arange(n, dtype=float) * step_m

    sampled, rejected = [], 0
    for f in frames:
        raw_len = float(pd.to_numeric(f["distance_m"], errors="coerce").max())
        if not raw_len or not np.isfinite(raw_len):
            rejected += 1
            continue
        cols = _resample(f, grid, lap_length_m / raw_len)
        if cols is None:
            rejected += 1
            continue
        sampled.append(cols)

    if len(sampled) < min_laps:
        raise ValueError(f"only {len(sampled)} usable laps for an ensemble (need {min_laps})")

    anchor = sampled[0]["speed_kph"]
    max_pts = int(round(max_phase_m / step_m))
    shifts = []
    aligned = {"pos_x": [], "pos_y": [], "speed_kph": []}
    for cols in sampled:
        k = _phase_shift(anchor, cols["speed_kph"], max_pts)
        shifts.append(float(k * step_m))
        for name in aligned:
            aligned[name].append(np.roll(cols[name], k))

    med = {name: np.nanmedian(np.vstack(v), axis=0) for name, v in aligned.items()}
    frame = pd.DataFrame({"distance_m": grid, "pos_x": med["pos_x"],
                          "pos_y": med["pos_y"], "speed_kph": med["speed_kph"]})
    return EnsembleResult(frame=frame, n_laps=len(sampled), lap_length_m=lap_length_m,
                          phase_shifts_m=shifts, rejected=rejected)
