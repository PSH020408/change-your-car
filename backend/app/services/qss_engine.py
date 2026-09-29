"""P9 — the physics engine behind the simulator, in differential form.

Every request runs the QSS lap solver three times on the circuit's line:

    prof0  the car fitted to the baseline lap (mu, ClA, CdA; cached per lap)
    prof1  the same car with the setup sliders applied (P3 modifiers ->
           downforce %, drag %, mechanical grip %, fuel kg)
    prof2  prof1 with one grip multiplier chosen so that the lap-time change
           equals the ML's conditions estimate (tyre, temperature)

and reports DIFFERENCES: segment deltas are prof1 - prof0 (setup) and
prof2 - prof1 (conditions); the simulated speed trace is the REAL trace
plus (prof2 - prof0). A point-mass solver brakes 10-20 m later than a
driver and its own speed trace is 12 km/h RMS off the real one; in the
difference that structural error cancels, and the trace keeps the shape
the driver actually produced. This is the standard way lap simulations are
used for setup work, and it retires the warp-and-clamp reconstruction and
its residual term (defect #28).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pipeline.physics import qss as Q, qss_fit as F
from pipeline.physics.modifiers import PhysicsState


@dataclass
class QssLap:
    line: pd.DataFrame
    d: np.ndarray                 # line grid
    k: np.ndarray
    g: np.ndarray
    drs: np.ndarray               # DRS open where the baseline lap opened it
    v_real: np.ndarray            # baseline speed on the grid, phase-aligned
    shift_pts: int
    car0: Q.Car
    prof0: Q.Profile
    fit: dict                     # rms_kph, lap error, parameters


@dataclass
class QssResult:
    physics_s: np.ndarray         # per segment, prof1 - prof0
    physics_lo_s: np.ndarray
    physics_hi_s: np.ndarray
    ml_s: np.ndarray              # per segment, prof2 - prof1 (ML lap delta redistributed by the engine)
    total_s: np.ndarray           # prof2 - prof0
    dv1_kph_on_trace: np.ndarray  # setup-only speed change at each baseline trace sample
    dv2_kph_on_trace: np.ndarray  # setup + conditions speed change at each baseline trace sample
    lap_delta_s: float
    grip_multiplier_ml: float
    car1: Q.Car
    car2: Q.Car
    solves: int


class QssService:
    def __init__(self, baselines_dir: Path, aero_u: float = 0.30, grip_u: float = 0.50):
        self.baselines_dir = Path(baselines_dir)
        self.aero_u, self.grip_u = aero_u, grip_u
        self._lines: dict[str, pd.DataFrame] = {}
        self._laps: dict[str, QssLap] = {}

    # ------------------------------------------------------------ lines
    def line_path(self, season: int, event: str) -> Path:
        return self.baselines_dir / str(season) / event / "line.json"

    def available(self, season: int, event: str) -> bool:
        return self.line_path(season, event).exists()

    def line(self, season: int, event: str) -> pd.DataFrame:
        key = f"{season}/{event}"
        if key not in self._lines:
            self._lines[key] = F.load_store_line(self.line_path(season, event))
        return self._lines[key]

    # ------------------------------------------------------------ laps
    def lap(self, season: int, event: str, lap: dict, trace: pd.DataFrame, fuel_kg: float,
            air_temp_c: float | None) -> QssLap:
        uid = str(lap["lap_uid"])
        if uid in self._laps:
            return self._laps[uid]
        line = self.line(season, event)
        v_real, drs, shift = F.lap_on_line(trace, line)
        base = F.default_car(mass_kg=798.0 + fuel_kg, rho=Q.air_density(air_temp_c))
        fit = F.fit_lap(line, v_real, drs, base)
        q = QssLap(line=line, d=line["distance_m"].to_numpy(float), k=line["curvature_1pm"].to_numpy(float),
                   g=line["grade"].to_numpy(float), drs=drs, v_real=v_real, shift_pts=shift,
                   car0=fit["car"], prof0=fit["profile"],
                   fit={k: v for k, v in fit.items() if k not in ("car", "profile")})
        self._laps[uid] = q
        return q

    # ------------------------------------------------------------ simulate
    def _solve(self, q: QssLap, car: Q.Car) -> Q.Profile:
        return Q.solve(q.d, q.k, q.g, q.drs, car)

    @staticmethod
    def _bounds(segments: list[dict]) -> list[tuple[float, float]]:
        return [(float(s["start_m"]), float(s["end_m"])) for s in segments]

    def simulate(self, q: QssLap, state: PhysicsState, ml_lap_delta_s: float, segments: list[dict],
                 trace_distance_m: np.ndarray) -> QssResult:
        bounds = self._bounds(segments)
        t0 = q.prof0.segment_times(bounds)
        grip_pct = (1.0 + state.mech_grip_pct / 100.0) * state.grip_multiplier * 100.0 - 100.0
        car1 = q.car0.with_setup(downforce_pct=state.downforce_pct, drag_pct=state.drag_pct,
                                 mech_grip_pct=grip_pct, fuel_delta_kg=state.fuel_delta_kg)
        prof1 = self._solve(q, car1)
        t1 = prof1.segment_times(bounds)
        solves = 1

        # coefficient bands: every relative effect at (1-u) and (1+u)
        def scaled(f: float) -> Q.Car:
            return q.car0.with_setup(downforce_pct=state.downforce_pct * (1 + f * self.aero_u),
                                     drag_pct=state.drag_pct * (1 + f * self.aero_u),
                                     mech_grip_pct=grip_pct * (1 + f * self.grip_u) if abs(state.mech_grip_pct) > 1e-9 else grip_pct,
                                     fuel_delta_kg=state.fuel_delta_kg)
        if any(abs(x) > 1e-9 for x in (state.downforce_pct, state.drag_pct, grip_pct)):
            t_lo = self._solve(q, scaled(-1.0)).segment_times(bounds) - t0
            t_hi = self._solve(q, scaled(+1.0)).segment_times(bounds) - t0
            solves += 2
        else:
            t_lo = t_hi = t1 - t0

        # ML conditions as one grip multiplier on the setup car
        m = 1.0
        prof2, car2 = prof1, car1
        if abs(ml_lap_delta_s) > 1e-4:
            def f(mult: float) -> float:
                nonlocal solves
                solves += 1
                return self._solve(q, replace(car1, mu=car1.mu * mult)).lap_time_s - prof1.lap_time_s - ml_lap_delta_s
            lo, hi = 0.85, 1.15
            flo, fhi = f(lo), f(hi)
            if flo * fhi < 0:
                m = float(brentq(f, lo, hi, xtol=1e-4, maxiter=30))
            else:                                   # asked for more than +-15% grip can give: clamp
                m = lo if abs(flo) < abs(fhi) else hi
            car2 = replace(car1, mu=car1.mu * m)
            prof2 = self._solve(q, car2)
            solves += 1
        t2 = prof2.segment_times(bounds)

        # speed changes on the line grid -> back onto the baseline trace's own axis
        n = len(q.d)
        step = float(q.d[1] - q.d[0])
        lap_len = float(q.d[-1] + step)
        td = np.asarray(trace_distance_m, float) % lap_len

        def on_trace(prof: Q.Profile) -> np.ndarray:
            dv = np.roll(prof.speed_kph[:n] - q.prof0.speed_kph[:n], -q.shift_pts)
            return np.interp(td, q.d, dv, period=lap_len)

        return QssResult(physics_s=t1 - t0, physics_lo_s=np.minimum(t_lo, t_hi), physics_hi_s=np.maximum(t_lo, t_hi),
                         ml_s=t2 - t1, total_s=t2 - t0,
                         dv1_kph_on_trace=on_trace(prof1), dv2_kph_on_trace=on_trace(prof2),
                         lap_delta_s=prof2.lap_time_s - q.prof0.lap_time_s,
                         grip_multiplier_ml=m, car1=car1, car2=car2, solves=solves)
