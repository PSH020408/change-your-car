"""P9-2 — quasi-steady-state point-mass lap simulation.

The car is a point of mass m on the racing line. At every metre the line
prescribes a curvature k(s) and a gradient dz/ds; the car brings power P,
drag area CdA, downforce area ClA and a friction coefficient mu. From these
alone the speed profile v(s) follows in three sweeps:

  1. the grip limit in every corner:   mu (m g cos(th) + L(v)) >= m v^2 |k|
  2. a backward sweep from each limit, braking as hard as the remaining grip
     (friction ellipse) plus drag plus gravity allow
  3. a forward sweep, accelerating on the lesser of traction and power

v(s) is the point-wise minimum; lap time is the integral of ds / v. Nothing
here is fitted to the lap being simulated except the three parameters the
calibration step exposes (mu, ClA, CdA) - and those are fitted to the SPEED
trace, never to the lap time, so the lap-time error stays an honest measure.

Sign conventions: `grade` is dz/ds (positive uphill). `curvature_1pm` keeps
its sign from the geometry step; only |k| matters here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import math

import numpy as np

G = 9.81
R_AIR = 287.05
V_CAP_MS = 400.0 / 3.6          # numerical ceiling, never reached by the physics
V_FLOOR_MS = 5.0                # keeps 1/v finite at start-up


def air_density(air_temp_c: float | None, pressure_pa: float = 101325.0) -> float:
    t = 20.0 if air_temp_c is None or not np.isfinite(air_temp_c) else float(air_temp_c)
    return pressure_pa / (R_AIR * (t + 273.15))


@dataclass(frozen=True)
class Car:
    mass_kg: float = 806.0          # 798 dry + a qualifying fuel load
    power_kw: float = 780.0         # at the wheels, ICE + MGU-K deployed
    cd_a: float = 1.20              # drag area, m^2
    cl_a: float = 4.00              # downforce area, m^2
    mu: float = 1.60                # tyre-road friction, effective
    brake_share: float = 1.00       # share of the grip limit reachable in pure braking
    # Only the rear axle drives. Its share of the normal load (static weight
    # distribution plus the rear-biased floor) caps traction out of slow
    # corners; the fronts contribute nothing to acceleration.
    driven_axle_share: float = 1.00
    # Tyre load sensitivity: the friction coefficient falls as the vertical
    # load rises, mu_eff = mu * (N / g)^(-s) with N the normal acceleration
    # (gravity + aero). s = 0 is the classic Coulomb point mass; racing slicks
    # sit around 0.1-0.3. This is what lets a car pull 2.5 g at 90 km/h and
    # only 5 g, not 7, at 250 km/h.
    load_sensitivity: float = 0.0
    drs_drag_cut: float = 0.12      # rear-wing drag removed with DRS open
    drs_lift_cut: float = 0.08      # downforce removed with DRS open
    rho: float = 1.20               # air density, kg/m^3

    def with_setup(self, *, downforce_pct: float = 0.0, drag_pct: float = 0.0,
                   mech_grip_pct: float = 0.0, fuel_delta_kg: float = 0.0) -> "Car":
        """The P3 slider effects, expressed on the engine's own parameters."""
        return replace(self,
                       cl_a=self.cl_a * (1.0 + downforce_pct / 100.0),
                       cd_a=self.cd_a * (1.0 + drag_pct / 100.0),
                       mu=self.mu * (1.0 + mech_grip_pct / 100.0),
                       mass_kg=self.mass_kg + fuel_delta_kg)


@dataclass
class Profile:
    distance_m: np.ndarray
    speed_ms: np.ndarray
    time_s: np.ndarray              # cumulative, time_s[0] = 0
    v_limit_ms: np.ndarray          # corner grip limit alone
    accel_ms2: np.ndarray           # longitudinal, along the lap
    mode: np.ndarray                # 0 grip-limited corner, 1 traction, 2 power, 3 braking

    @property
    def lap_time_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def speed_kph(self) -> np.ndarray:
        return self.speed_ms * 3.6

    def segment_times(self, bounds_m: list[tuple[float, float]]) -> np.ndarray:
        """Time spent between start/end distances (a segment may wrap the line)."""
        d, t = self.distance_m, self.time_s
        lap_len = float(d[-1])                      # the closing point is on the grid
        out = []
        for a, b in bounds_m:
            ta = float(np.interp(a % lap_len, d, t))
            # an end exactly on the line closes the lap, it does not restart it
            tb = self.lap_time_s if abs(b - lap_len) < 1e-6 else float(np.interp(b % lap_len, d, t))
            out.append(tb - ta if b >= a else (self.lap_time_s - ta) + tb)
        return np.asarray(out)


def _lift(car: Car, v: np.ndarray, drs: np.ndarray) -> np.ndarray:
    cl = car.cl_a * np.where(drs, 1.0 - car.drs_lift_cut, 1.0)
    return 0.5 * car.rho * cl * v * v


def _drag(car: Car, v: np.ndarray, drs: np.ndarray) -> np.ndarray:
    cd = car.cd_a * np.where(drs, 1.0 - car.drs_drag_cut, 1.0)
    return 0.5 * car.rho * cd * v * v


def grip_accel(car: Car, normal_ms2: np.ndarray | float) -> np.ndarray | float:
    """Maximum friction acceleration for a given normal acceleration (load)."""
    if car.load_sensitivity <= 0.0:
        return car.mu * normal_ms2
    return car.mu * G * (normal_ms2 / G) ** (1.0 - car.load_sensitivity)


def corner_limit(car: Car, curvature_1pm: np.ndarray, grade: np.ndarray,
                 drs: np.ndarray) -> np.ndarray:
    """Highest steady speed the grip allows at each point.

    mu_eff(N) * N = v^2 |k|  with  N = g cos(th) + rho ClA v^2 / (2 m).
    Without load sensitivity this is closed-form:
        v^2 = mu g cos / (|k| - mu rho ClA / (2 m)),
    unbounded when aero alone would hold the corner at any speed (the
    denominator is <= 0). With load sensitivity it is solved by a few
    fixed-point passes from the closed-form start.
    """
    k = np.abs(curvature_1pm)
    cos_th = 1.0 / np.sqrt(1.0 + grade * grade)
    cl = car.cl_a * np.where(drs, 1.0 - car.drs_lift_cut, 1.0)
    aero = car.rho * cl / (2.0 * car.mass_kg)
    denom = k - car.mu * aero
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where(denom > 1e-9, car.mu * G * cos_th / denom, np.inf)
    v = np.minimum(np.sqrt(v2), V_CAP_MS)
    if car.load_sensitivity > 0.0:
        for _ in range(30):
            n_acc = G * cos_th + aero * v * v
            a_lat = grip_accel(car, n_acc)
            with np.errstate(divide="ignore", invalid="ignore"):
                v_new = np.where(k > 1e-9, np.sqrt(a_lat / k), V_CAP_MS)
            v = np.minimum(0.5 * v + 0.5 * v_new, V_CAP_MS)
    return v


def solve(distance_m: np.ndarray, curvature_1pm: np.ndarray, grade: np.ndarray,
          drs_open: np.ndarray | None, car: Car, passes: int = 2) -> Profile:
    """Speed profile on a closed line. `distance_m` must be a uniform grid.

    The two sweeps are sequential by nature, so they run as plain Python
    loops over floats (numpy scalars are several times slower here); a
    1,100-point line solves in a few milliseconds.
    """
    d = np.asarray(distance_m, dtype=float)
    k_arr = np.abs(np.asarray(curvature_1pm, dtype=float))
    g_arr = np.asarray(grade, dtype=float) if grade is not None else np.zeros_like(d)
    n = len(d)
    if n < 10:
        raise ValueError("line too short")
    ds = float(d[1] - d[0])
    drs_arr = np.zeros(n, dtype=bool) if drs_open is None else np.asarray(drs_open, dtype=bool)

    v_lim = corner_limit(car, k_arr, g_arr, drs_arr)
    v = np.minimum(v_lim, V_CAP_MS).tolist()
    mode = [0] * n

    # per-point constants
    cos_th = (1.0 / np.sqrt(1.0 + g_arr * g_arr)).tolist()
    g_sin = (G * g_arr / np.sqrt(1.0 + g_arr * g_arr)).tolist()
    k = k_arr.tolist()
    half_rho = 0.5 * car.rho
    cl_pt = (half_rho * car.cl_a * np.where(drs_arr, 1.0 - car.drs_lift_cut, 1.0) / car.mass_kg).tolist()
    cd_pt = (half_rho * car.cd_a * np.where(drs_arr, 1.0 - car.drs_drag_cut, 1.0) / car.mass_kg).tolist()
    mu, share, driven = car.mu, car.brake_share, car.driven_axle_share
    p_over_m = car.power_kw * 1000.0 / car.mass_kg
    two_ds = 2.0 * ds
    floor2 = V_FLOOR_MS * V_FLOOR_MS

    ls = car.load_sensitivity

    def grip_long(j: int, vj: float) -> float:
        v2 = vj * vj
        n_acc = G * cos_th[j] + cl_pt[j] * v2
        a_max = mu * n_acc if ls <= 0.0 else mu * G * (n_acc / G) ** (1.0 - ls)
        if a_max <= 0.0:
            return 0.0
        r = v2 * k[j] / a_max
        rem = 1.0 - r * r
        return a_max * math.sqrt(rem) if rem > 0.0 else 0.0

    for _ in range(passes):
        # backward: brake from v[j] into v[j-1]; wraps the loop once more than needed
        for i in range(2 * n, 0, -1):
            j = i % n
            jm = j - 1 if j > 0 else n - 1
            vj = v[j]
            a_b = grip_long(j, vj) * share + cd_pt[j] * vj * vj + g_sin[j]
            if a_b < 0.0:
                a_b = 0.0
            v_reach = math.sqrt(vj * vj + two_ds * a_b)
            if v_reach < v[jm]:
                v[jm] = v_reach
                mode[jm] = 3
        # forward: accelerate from v[j] into v[j+1]
        for i in range(0, 2 * n):
            j = i % n
            jp = j + 1 if j < n - 1 else 0
            vj = v[j] if v[j] > V_FLOOR_MS else V_FLOOR_MS
            a_trac = grip_long(j, vj) * driven
            a_pow = p_over_m / vj - cd_pt[j] * vj * vj
            a_f = (a_trac if a_trac < a_pow else a_pow) - g_sin[j]
            v2 = vj * vj + two_ds * a_f
            v_reach = math.sqrt(v2 if v2 > floor2 else floor2)
            if v_reach < v[jp]:
                v[jp] = v_reach
                mode[jp] = 1 if a_trac < a_pow else 2

    v_arr = np.clip(np.asarray(v), V_FLOOR_MS, V_CAP_MS)
    inv = 1.0 / v_arr
    step_t = 0.5 * (inv + np.roll(inv, -1)) * ds        # trapezoid on 1/v, last step closes the loop
    t = np.concatenate([[0.0], np.cumsum(step_t[:-1])])
    lap_time = float(step_t.sum())
    accel = (np.roll(v_arr, -1) ** 2 - v_arr ** 2) / (2.0 * ds)
    # The profile carries the closing point (start-finish again) so that
    # interpolation across the line is seamless: n + 1 samples, time_s[-1] = lap time.
    return Profile(distance_m=np.append(d, d[-1] + ds),
                   speed_ms=np.append(v_arr, v_arr[0]),
                   time_s=np.append(t, lap_time),
                   v_limit_ms=np.append(v_lim, v_lim[0]),
                   accel_ms2=np.append(accel, accel[0]),
                   mode=np.append(np.asarray(mode, dtype=np.int8), mode[0]))


def speed_on(profile: Profile, at_m: np.ndarray) -> np.ndarray:
    """Simulated speed (km/h) interpolated at arbitrary distances."""
    lap_len = float(profile.distance_m[-1])
    return np.interp(np.asarray(at_m, dtype=float) % lap_len, profile.distance_m, profile.speed_kph)
