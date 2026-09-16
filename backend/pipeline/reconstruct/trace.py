"""P5 — segment deltas -> a continuous, physically legal simulated trace.

The model (P4) and the physics layer (P3) each answer per SEGMENT: "this
corner is 0.12 s slower". The HUD overlays a full speed trace on the real
one, so the scalar has to become a curve. This module warps the BASELINE
lap's own trace — never invents one — so that:

  1. every segment's integrated time moves by exactly its delta (bisection
     on a shape-weighted speed scale: a corner slows most at its apex, a
     straight most where it is fastest, because that is where the physics
     of a downforce or drag change acts);
  2. the result stays inside what a car can do: lateral grip
     v^2 k <= mu g (1 + v^2/v0^2), traction P/(m v) and braking limits from
     configs/physics.yaml. A requested delta the physics refuses is reported
     as a residual, not silently absorbed;
  3. throttle / brake / gear / DRS are re-derived from the new speed curve
     with thresholds fitted on the baseline lap itself, so a zero delta
     reproduces the driver's own pedals.

Everything here is kinematics on a distance grid: t = sum(ds / v).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.physics.modifiers import PhysicsConfig, default_config

G = 9.81
KPH = 3.6
MIN_SPEED_SCALE = 0.25          # a warp may not slow a sample below a quarter of its speed
DRS_OPEN_CODES = {10, 12, 14}   # FastF1 raw DRS codes meaning "open"


# ------------------------------------------------------------------ helpers
def integrate_lap_time(speed_kph: np.ndarray, distance_m: np.ndarray) -> float:
    """Trapezoidal sum of ds / v over the grid, seconds."""
    v = np.asarray(speed_kph, float) / KPH
    d = np.asarray(distance_m, float)
    ds = np.diff(d)
    vm = 0.5 * (v[1:] + v[:-1])
    return float(np.sum(ds / np.maximum(vm, 1.0)))


def _segment_slices(distance_m: np.ndarray, segments: list[dict]) -> list[tuple[int, int]]:
    """Index range [i0, i1) of each segment's SAMPLES on the grid.

    For TIME, use `_tiled(i0, i1, n)`: the interval from a segment's last
    sample to the next segment's first sample belongs to nobody otherwise,
    and 23 such straddles at 0.24 s each made the lap total disagree with
    the sum of segments by ~0.05 s on real telemetry.
    """
    out = []
    lap_len = max(float(s["end_m"]) for s in segments)
    for s in segments:
        a, b = float(s["start_m"]), float(s["end_m"])
        if s.get("wraps_start_finish") or b < a:
            mask = (distance_m >= a) | (distance_m < b)
        else:
            mask = (distance_m >= a) & (distance_m < b)
        idx = np.flatnonzero(mask)
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
    return out


def _tiled(i0: int, i1: int, n: int) -> tuple[int, int]:
    """Sample range whose intervals tile the lap: include the next segment's first sample."""
    return i0, min(i1 + 1, n)


def shape_weights(v: np.ndarray, kind: str) -> np.ndarray:
    """Where inside a segment a time change is spent (0.25 .. 1).

    Corner / kink: at the apex — that is where grip sets the speed.
    Straight: at the top speed — that is where drag sets it. The 0.25 floor
    keeps the whole segment moving a little so the joins stay smooth.
    """
    if len(v) == 0:
        return v
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-6:
        return np.ones_like(v)
    x = (v - lo) / (hi - lo)
    return 0.25 + 0.75 * (x if kind == "straight" else 1.0 - x)


def warp_segment(v_kph: np.ndarray, d_m: np.ndarray, target_time_s: float, w: np.ndarray,
                 tol_s: float = 1e-4) -> tuple[np.ndarray, float]:
    """Scale the speed as v / (1 + k w) so the segment's time hits the target.

    Time is monotonic in k, so bisection is exact to `tol_s`. k > 0 slows
    the segment, k < 0 speeds it up; the scale never drops below
    MIN_SPEED_SCALE. `w` may be one sample shorter than `v_kph`: the
    trailing sample then belongs to the next segment and is held fixed
    (weight 0) while its interval still counts toward this segment's time.
    """
    if len(w) < len(v_kph):
        w = np.concatenate([w, np.zeros(len(v_kph) - len(w))])

    def time_at(k: float) -> float:
        return integrate_lap_time(v_kph / np.maximum(1.0 + k * w, MIN_SPEED_SCALE), d_m)
    lo, hi = -0.75, 3.0                             # 1+k*w in [0.25, 4]
    if time_at(hi) < target_time_s:                 # slower than a quarter speed: cap
        return v_kph / np.maximum(1.0 + hi * w, MIN_SPEED_SCALE), time_at(hi)
    if time_at(lo) > target_time_s:                 # faster than 4x: cap
        return v_kph / np.maximum(1.0 + lo * w, MIN_SPEED_SCALE), time_at(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        t = time_at(mid)
        if abs(t - target_time_s) < tol_s:
            break
        if t < target_time_s:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return v_kph / np.maximum(1.0 + k * w, MIN_SPEED_SCALE), time_at(k)


# ------------------------------------------------------------------ warping
def warp_speed_trace(distance_m: np.ndarray, speed_kph: np.ndarray, segments: list[dict],
                     deltas_s: dict[int, float] | list[float]) -> tuple[np.ndarray, pd.DataFrame]:
    """Warp the whole lap; return (new speed, per-segment report)."""
    d = np.asarray(distance_m, float)
    v = np.asarray(speed_kph, float).copy()
    out = v.copy()
    rows = []
    n = len(d)
    for (i0, i1), s in zip(_segment_slices(d, segments), segments):
        idx = int(s["index"])
        delta = float(deltas_s[idx] if isinstance(deltas_s, dict) else deltas_s[idx])
        if i1 - i0 < 3:
            rows.append({"segment_index": idx, "kind": s["kind"], "baseline_s": 0.0, "requested_s": delta,
                         "warped_s": 0.0}); continue
        t0, t1 = _tiled(i0, i1, n)
        seg_v, seg_d = out[t0:t1].copy(), d[t0:t1]          # `out`: earlier segments already warped
        base_t = integrate_lap_time(v[t0:t1], seg_d)
        w = shape_weights(v[i0:i1], str(s["kind"]))
        new_v, got = warp_segment(seg_v, seg_d, integrate_lap_time(seg_v, seg_d) + delta, w)
        out[i0:i1] = new_v[: i1 - i0]
        rows.append({"segment_index": idx, "kind": s["kind"], "baseline_s": base_t, "requested_s": delta,
                     "warped_s": got - base_t})
    return out, pd.DataFrame(rows)


# ------------------------------------------------------------------ physics
@dataclass
class Envelope:
    mu: float
    v0_ms: float
    power_w: float
    mass_kg: float
    brake_share: float

    @classmethod
    def from_config(cls, cfg: PhysicsConfig | None = None, fuel_kg: float = 40.0,
                    grip_scale: float = 1.0) -> "Envelope":
        cfg = cfg or default_config()
        car = cfg.raw["car"]
        return cls(mu=cfg.coeff("car", "mu_mechanical").value * grip_scale,
                   v0_ms=cfg.coeff("car", "aero_crossover_kph").value / KPH,
                   power_w=cfg.coeff("car", "power_kw").value * 1000.0,
                   mass_kg=float(car["mass_dry_kg"]) + fuel_kg,
                   brake_share=cfg.coeff("car", "brake_share_of_grip").value)

    def grip_accel(self, v_ms: np.ndarray) -> np.ndarray:
        """Total grip available at this speed, m/s^2 (mechanical + aero)."""
        return self.mu * G * (1.0 + (v_ms / self.v0_ms) ** 2)

    def max_corner_speed(self, curvature_1pm: np.ndarray) -> np.ndarray:
        """v^2 k <= mu g (1 + v^2/v0^2)  ->  v_max, inf where aero always wins."""
        k = np.abs(np.asarray(curvature_1pm, float))
        a = k - self.mu * G / self.v0_ms ** 2
        out = np.full_like(k, np.inf)
        ok = a > 1e-9
        out[ok] = np.sqrt(self.mu * G / a[ok])
        return out

    def power_accel(self, v_ms: np.ndarray) -> np.ndarray:
        """What full throttle can add at this speed before drag, m/s^2."""
        return self.power_w / (self.mass_kg * np.maximum(v_ms, 5.0))

    def traction_accel(self, v_ms: np.ndarray) -> np.ndarray:
        return np.minimum(self.power_accel(v_ms), self.grip_accel(v_ms))

    def drag_decel(self, v_ms: np.ndarray, v_max_ms: float) -> np.ndarray:
        """Drag as a deceleration, scaled so that at the lap's top speed it
        eats the whole engine: a_drag(v_max) = P / (m v_max) -> a_drag ~ v^2."""
        return self.power_accel(np.array([v_max_ms]))[0] * (v_ms / max(v_max_ms, 1.0)) ** 2

    def braking_accel(self, v_ms: np.ndarray) -> np.ndarray:
        return self.brake_share * self.grip_accel(v_ms)


def segment_curvature(distance_m: np.ndarray, segments: list[dict]) -> np.ndarray:
    """Curvature on the grid from the segment table: a raised-cosine bump to
    the segment's peak curvature inside corners and kinks, ~0 on straights."""
    d = np.asarray(distance_m, float)
    k = np.zeros_like(d)
    for (i0, i1), s in zip(_segment_slices(d, segments), segments):
        if i1 - i0 < 2 or s["kind"] == "straight":
            continue
        peak = abs(float(s.get("peak_curvature_1pm") or 0.0))
        n = i1 - i0
        bump = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(n) + 0.5) / n))   # 0 at edges, 1 mid
        k[i0:i1] = peak * bump
    return k


HEADROOM = 0.15   # the baseline proves a load was possible; simulation may exceed it by this much


def effective_curvature(curvature_1pm: np.ndarray, baseline_kph: np.ndarray | None, env: Envelope,
                        headroom: float = HEADROOM) -> np.ndarray:
    """Segment-shape curvature, capped at the tightest curvature the baseline
    lap could physically have taken at its own speed (plus headroom)."""
    k = np.abs(np.asarray(curvature_1pm, float))
    if baseline_kph is None:
        return k
    v0 = np.asarray(baseline_kph, float) / KPH
    k_proof = env.grip_accel(v0) / np.maximum(v0, 1.0) ** 2
    return np.minimum(k, k_proof / (1.0 + headroom))


def clamp_to_gg_envelope(distance_m: np.ndarray, speed_kph: np.ndarray, curvature_1pm: np.ndarray,
                         env: Envelope, baseline_kph: np.ndarray | None = None,
                         headroom: float = HEADROOM) -> tuple[np.ndarray, dict]:
    """Cap the trace at what the tyres and the engine allow.

    1. lateral: v <= v_max(k)
    2. traction (forward pass): v_{i+1}^2 <= v_i^2 + 2 a_trac ds
    3. braking (backward pass): v_i^2 <= v_{i+1}^2 + 2 a_brake ds

    The envelope is ANCHORED ON THE BASELINE LAP: a real lap is proof that
    its loads were possible, so wherever the model's limit is tighter than
    what the baseline actually did, the baseline wins (plus `headroom`).
    Curvature from the segment table is a shape, not a survey, and the
    coefficients are grade B; without the anchor the baseline itself would
    be "clamped" and a zero delta would not return the real lap.
    """
    d = np.asarray(distance_m, float)
    v = np.asarray(speed_kph, float) / KPH
    ds = np.diff(d)
    k = effective_curvature(curvature_1pm, baseline_kph, env, headroom)
    if baseline_kph is not None:
        # the baseline's own per-interval acceleration, computed EXACTLY the way
        # the passes below test it (two-point, v^2 difference over 2 ds) - a
        # central-difference estimate under-reads sharp braking onsets and
        # let the baseline itself be clamped on real telemetry
        v0 = np.asarray(baseline_kph, float) / KPH
        a0_mid = np.abs(v0[1:] ** 2 - v0[:-1] ** 2) / (2.0 * np.maximum(ds, 1e-6)) * (1.0 + headroom)
    else:
        a0_mid = np.zeros(len(ds))
    v_lat = env.max_corner_speed(k)
    n0 = int(np.sum(v > v_lat + 1e-9))
    v = np.minimum(v, v_lat)
    n_trac = n_brake = 0
    for i in range(len(v) - 1):                                   # traction
        a = max(env.traction_accel(np.array([v[i]]))[0], a0_mid[i])
        vmax = np.sqrt(v[i] ** 2 + 2 * a * ds[i])
        if v[i + 1] > vmax + 1e-9:
            v[i + 1] = vmax; n_trac += 1
    for i in range(len(v) - 2, -1, -1):                           # braking
        a = max(env.braking_accel(np.array([v[i + 1]]))[0], a0_mid[i])
        vmax = np.sqrt(v[i + 1] ** 2 + 2 * a * ds[i])
        if v[i] > vmax + 1e-9:
            v[i] = vmax; n_brake += 1
    return v * KPH, {"lateral_clamped": n0, "traction_clamped": n_trac, "braking_clamped": n_brake}


# ------------------------------------------------------------- channels
def longitudinal_accel(distance_m: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """a = v dv/ds, m/s^2, central differences."""
    v = np.asarray(speed_kph, float) / KPH
    d = np.asarray(distance_m, float)
    dv = np.gradient(v, d, edge_order=1)
    return v * dv


def fit_brake_threshold(a_long: np.ndarray, brake_on: np.ndarray,
                        grid=(-1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -10.0)) -> float:
    """Deceleration below which this driver's brake light was on, fitted on
    the baseline lap by agreement — so a zero delta gives the driver's pedals back."""
    b = np.asarray(brake_on).astype(bool)
    best, best_t = -1.0, -3.0
    for t in grid:
        agree = float(np.mean((a_long < t) == b))
        if agree > best:
            best, best_t = agree, t
    return best_t


def synthesize_channels(distance_m: np.ndarray, speed_kph: np.ndarray, baseline: pd.DataFrame,
                        env: Envelope) -> pd.DataFrame:
    """Throttle / brake / gear / DRS for a warped speed trace.

    brake    a_long below the driver's fitted threshold
    throttle share of the traction limit being used when accelerating; full
             when holding top speed; zero when braking or lifting
    gear     nearest baseline gear for that speed (the ratio set is the car's)
    DRS      open exactly where the baseline lap had it open (a track zone)
    """
    d = np.asarray(distance_m, float)
    a = longitudinal_accel(d, speed_kph)
    a_base = longitudinal_accel(d, baseline["speed_kph"].to_numpy(float))
    thr = fit_brake_threshold(a_base, baseline["brake_on"].to_numpy()) if "brake_on" in baseline else -3.0
    brake = a < thr
    v_ms = np.asarray(speed_kph, float) / KPH
    # Throttle is the share of the engine needed to produce this acceleration
    # AND beat drag: at top speed a is ~0 but drag takes everything, so the
    # pedal is flat. Coasting shows as a ~ -drag -> 0%.
    demand = a + env.drag_decel(v_ms, float(np.max(v_ms)))
    throttle = np.clip(100.0 * demand / env.power_accel(v_ms), 0.0, 100.0)
    throttle[brake] = 0.0
    gear = np.zeros(len(d), dtype=int)
    if "gear" in baseline and baseline["gear"].notna().any():
        g = pd.to_numeric(baseline["gear"], errors="coerce")
        table = pd.DataFrame({"g": g, "v": baseline["speed_kph"]}).dropna().groupby("g")["v"].median()
        table = table[table.index > 0].sort_values()
        if len(table):
            centres, gears = table.to_numpy(float), table.index.to_numpy(int)
            gear = gears[np.abs(np.asarray(speed_kph, float)[:, None] - centres[None, :]).argmin(axis=1)]
    drs = np.zeros(len(d), dtype=bool)
    if "drs_raw" in baseline:
        drs = pd.to_numeric(baseline["drs_raw"], errors="coerce").isin(DRS_OPEN_CODES).to_numpy()
    return pd.DataFrame({"distance_m": d, "speed_kph": np.asarray(speed_kph, float),
                         "throttle_pct": np.round(throttle, 1), "brake_on": brake, "gear": gear,
                         "drs_open": drs, "accel_long_ms2": np.round(a, 3)})


# --------------------------------------------------------------- pipeline
@dataclass
class Reconstruction:
    trace: pd.DataFrame            # distance_m, time_s, speed_kph, throttle_pct, brake_on, gear, drs_open
    segments: pd.DataFrame         # per segment: baseline_s, requested_s, warped_s, achieved_s, residual_s
    lap_time_baseline_s: float
    lap_time_s: float
    requested_delta_s: float
    achieved_delta_s: float
    clamps: dict = field(default_factory=dict)
    official_lap_time_s: float | None = None

    @property
    def time_scale(self) -> float:
        """Official lap time / integrated baseline time. The 240 ms samples
        miss the partial intervals just after and just before the line
        (~0.5 s on a 90 s lap, -0.5%); the HUD's time axis is scaled by
        this so the baseline reads its official time. Deltas are
        unaffected: both traces carry the same gap."""
        if not self.official_lap_time_s or self.lap_time_baseline_s <= 0:
            return 1.0
        return float(self.official_lap_time_s / self.lap_time_baseline_s)

    @property
    def integration_error_s(self) -> float:
        """|achieved - requested| where the physics allowed the request."""
        ok = self.segments["clamped"] == False  # noqa: E712
        return float(np.abs(self.segments.loc[ok, "achieved_s"] - self.segments.loc[ok, "requested_s"]).sum())


def reconstruct(baseline: pd.DataFrame, segments: list[dict], deltas_s: dict[int, float] | list[float],
                cfg: PhysicsConfig | None = None, fuel_kg: float = 40.0, grip_scale: float = 1.0,
                clamp: bool = True, official_lap_time_s: float | None = None) -> Reconstruction:
    """Baseline telemetry (distance_m, speed_kph, brake_on, gear, drs_raw ...)
    + per-segment deltas -> simulated trace with a per-segment audit."""
    base = baseline.sort_values("distance_m").drop_duplicates("distance_m").reset_index(drop=True)
    d = base["distance_m"].to_numpy(float)
    v0 = base["speed_kph"].to_numpy(float)
    env = Envelope.from_config(cfg, fuel_kg=fuel_kg, grip_scale=grip_scale)

    v_warp, rep = warp_speed_trace(d, v0, segments, deltas_s)
    clamps: dict = {}
    v_fin = v_warp
    if clamp:
        v_fin, clamps = clamp_to_gg_envelope(d, v_warp, segment_curvature(d, segments), env, baseline_kph=v0)

    # per-segment achieved time after clamping
    ach = []
    n = len(d)
    for (i0, i1), s in zip(_segment_slices(d, segments), segments):
        t0, t1 = _tiled(i0, i1, n)
        ach.append(integrate_lap_time(v_fin[t0:t1], d[t0:t1]) - integrate_lap_time(v0[t0:t1], d[t0:t1])
                   if i1 - i0 >= 3 else 0.0)
    rep["achieved_s"] = ach
    rep["residual_s"] = rep["achieved_s"] - rep["requested_s"]
    rep["clamped"] = rep["residual_s"].abs() > 0.02

    ch = synthesize_channels(d, v_fin, base, env)
    v_ms = v_fin / KPH
    ds = np.diff(d, prepend=d[0])
    vm = np.concatenate([[v_ms[0]], 0.5 * (v_ms[1:] + v_ms[:-1])])
    ch.insert(1, "time_s", np.cumsum(ds / np.maximum(vm, 1.0)))
    t_base, t_new = integrate_lap_time(v0, d), integrate_lap_time(v_fin, d)
    out = Reconstruction(trace=ch, segments=rep, lap_time_baseline_s=t_base, lap_time_s=t_new,
                         requested_delta_s=float(rep["requested_s"].sum()), achieved_delta_s=t_new - t_base,
                         clamps=clamps, official_lap_time_s=official_lap_time_s)
    if official_lap_time_s:
        out.trace["time_s"] = out.trace["time_s"] * out.time_scale
    return out
