"""P9 — putting a real lap on the racing line and fitting the three car parameters.

Shared by the calibration report (pipeline) and the live engine (API): the
engine fits the same three numbers to the baseline lap it is about to
modify, so what the HUD shows is anchored to exactly the calibration the
report grades.

Fixed constants (set once by an 8-circuit sweep, applied to every lap —
see docs/PHYSICS_ENGINE.md):
    power 480 kW    lap-average wheel power: ERS deployment limits,
                    lift-and-coast, shifts and driveline losses folded in
    driven 0.60     rear-axle share of the grip available for traction
    brake  1.00     braking may use the full friction limit
    load sensitivity 0 (tested 0.15 / 0.30: worse on every circuit)
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.physics import qss as Q

POWER_KW = 480.0
DRIVEN_SHARE = 0.60
BRAKE_SHARE = 1.00
BOUNDS = {"mu": (1.2, 2.8), "cl_a": (2.5, 6.5), "cd_a": (0.8, 1.8)}
X0 = np.array([1.6, 4.0, 1.2])
MAX_PHASE_M = 80.0


def default_car(mass_kg: float, rho: float) -> Q.Car:
    return Q.Car(mass_kg=mass_kg, rho=rho, power_kw=POWER_KW, driven_axle_share=DRIVEN_SHARE,
                 brake_share=BRAKE_SHARE)


# ------------------------------------------------------------------ line I/O
def line_to_store(line: pd.DataFrame, meta: dict) -> dict:
    """Compact form of line.parquet for the baseline store (JSON, ~40 kB/circuit)."""
    return {"grid_step_m": float(meta["grid_step_m"]), "lap_length_m": float(meta["lap_length_m"]),
            "smooth_window_m": float(meta["smooth_window_m"]),
            "elevation_used": bool(meta["elevation"].get("used")),
            "distance_m": [round(float(x), 1) for x in line["distance_m"]],
            "curvature_1pm": [round(float(x), 6) for x in line["curvature_1pm"]],
            "grade": [round(float(x), 5) for x in line["grade"]],
            "drs_share": [round(float(x), 2) for x in line["drs_share"]],
            "speed_kph": [round(float(x), 1) for x in line["speed_kph"]]}


def load_store_line(path: Path) -> pd.DataFrame:
    doc = json.loads(Path(path).read_text())
    return pd.DataFrame({k: np.asarray(doc[k], dtype=float)
                         for k in ("distance_m", "curvature_1pm", "grade", "drs_share", "speed_kph")})


# ------------------------------------------------------------- lap on line
def phase_align(real_v: np.ndarray, ref_v: np.ndarray, step_m: float) -> int:
    """Grid shift that best aligns the lap's speed trace to the line's median speed."""
    a = np.nan_to_num(ref_v - np.nanmean(ref_v))
    b = np.nan_to_num(real_v - np.nanmean(real_v))
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    lags = np.arange(len(a))
    lags = np.where(lags > len(a) // 2, lags - len(a), lags)
    ok = np.abs(lags) <= int(round(MAX_PHASE_M / step_m))
    return int(lags[int(np.argmax(np.where(ok, corr, -np.inf)))])


def lap_on_line(trace: dict | pd.DataFrame, line: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Real speed (km/h) and DRS state on the line's grid, phase-aligned.

    Returns (v_kph, drs_open, shift_pts): the trace was rolled by `shift_pts`
    grid points to sit on the line; undo it with -shift when mapping a
    line-grid quantity back onto the trace's own distance axis.
    """
    grid = line["distance_m"].to_numpy(float)
    step = float(grid[1] - grid[0])
    lap_len = float(grid[-1] + step)
    d = np.asarray(trace["distance_m"], float)
    v = np.asarray(trace["speed_kph"], float)
    drs = np.asarray(trace["drs_open"], bool).astype(float)
    ok = np.isfinite(d) & np.isfinite(v)
    d, v, drs = d[ok], v[ok], drs[ok]
    order = np.argsort(d)
    d, v, drs = d[order], v[order], drs[order]
    keep = np.concatenate([[True], np.diff(d) > 0])
    d, v, drs = d[keep], v[keep], drs[keep]
    vg = np.interp(grid, d, v, period=lap_len)
    dg = np.interp(grid, d, drs, period=lap_len) >= 0.5
    k = phase_align(vg, line["speed_kph"].to_numpy(float), step)
    return np.roll(vg, k), np.roll(dg, k), k


# ------------------------------------------------------------------- fitting
def fit_lap(line: pd.DataFrame, v_real_kph: np.ndarray, drs: np.ndarray, base: Q.Car) -> dict:
    """Fit mu, ClA, CdA to the speed trace (never to the lap time)."""
    from scipy.optimize import least_squares          # SciPy only where the fit runs

    d = line["distance_m"].to_numpy(float)
    k = line["curvature_1pm"].to_numpy(float)
    g = line["grade"].to_numpy(float)
    n = len(d)

    def resid(x):
        car = replace(base, mu=x[0], cl_a=x[1], cd_a=x[2])
        return Q.solve(d, k, g, drs, car).speed_kph[:n] - v_real_kph

    lo = np.array([BOUNDS[p][0] for p in ("mu", "cl_a", "cd_a")])
    hi = np.array([BOUNDS[p][1] for p in ("mu", "cl_a", "cd_a")])
    t0 = time.perf_counter()
    res = least_squares(resid, X0, bounds=(lo, hi), method="trf", diff_step=1e-3,
                        max_nfev=80, xtol=1e-4, ftol=1e-5)
    mu, cl, cd = (float(v) for v in res.x)
    car = replace(base, mu=mu, cl_a=cl, cd_a=cd)
    prof = Q.solve(d, k, g, drs, car)
    r = prof.speed_kph[:n] - v_real_kph
    at_bound = bool(np.any(np.isclose(res.x, lo, rtol=0, atol=1e-3)) or np.any(np.isclose(res.x, hi, rtol=0, atol=1e-3)))
    return {"mu": mu, "cl_a": cl, "cd_a": cd, "rms_kph": float(np.sqrt(np.mean(r * r))),
            "bias_kph": float(np.mean(r)), "max_abs_kph": float(np.max(np.abs(r))),
            "sim_lap_time_s": prof.lap_time_s, "n_evals": int(res.nfev), "at_bound": at_bound,
            "fit_ms": round((time.perf_counter() - t0) * 1000, 1), "car": car, "profile": prof}
