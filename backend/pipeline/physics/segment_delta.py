"""P3 — physical state -> time delta per segment.

The ML (P4) answers "how does THIS car on THIS tyre in THIS weather compare
to the reference lap". This module answers the question the ML cannot: what
a setup change does. The two are added (decision: separate, additive — see
claude/P3-SCOPE.md), each with its own interval.

THE MODEL, IN THREE LINES
  straight  v_max ~ (P / Cd)^(1/3)          drag sets the top speed
  corner    v^2  ~ grip * R                 grip sets the speed, aero is part of grip
  mass      a = (F - D) / m                 fuel slows the acceleration phases

HOW MUCH OF A CORNER'S GRIP IS AERO
Downforce grows with v^2, the car's weight does not, so the share of grip
that comes from the wings and floor is  alpha(v) = v^2 / (v^2 + v0^2)  where
v0 is the speed at which downforce equals the car's weight (~150 km/h; fitted
from our own corner data by physics-check #3). A 60 km/h hairpin barely
notices a wing change (alpha ~ 0.14); a 250 km/h sweep is almost all aero
(alpha ~ 0.74). This is why the same wing click is worth different time in
different segments, and why the pipeline segmented the lap at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.physics.modifiers import SCALES, PhysicsConfig, PhysicsState, SetupInput, default_config, physics_state

CORNER_KINDS = ("low_speed_corner", "medium_speed_corner", "high_speed_corner")
KINK_WEIGHT = 0.3   # a kink is nearly flat-out; grip changes move its time far less than a corner's


def aero_fraction(speed_kph: float, v0_kph: float) -> float:
    """Share of cornering grip that is aerodynamic at this speed, in [0, 1)."""
    v2 = max(float(speed_kph), 0.0) ** 2
    return v2 / (v2 + float(v0_kph) ** 2) if v2 > 0 else 0.0


def terminal_fraction(speed_mean_kph: float, speed_max_kph: float) -> float:
    """How much of a straight is spent near its top speed (0..1).

    mean/max close to 1 = a long flat-out run where drag rules; well below 1
    = a short blast out of a corner where acceleration (mass) rules.
    """
    if not speed_max_kph or speed_max_kph <= 0:
        return 0.5
    r = float(speed_mean_kph) / float(speed_max_kph)
    # mean/max of 0.80 -> 0.0 ; 1.00 -> 1.0
    return float(min(max((r - 0.80) / 0.20, 0.0), 1.0))


@dataclass(frozen=True)
class SegmentDelta:
    delta_s: float
    drag_s: float
    grip_s: float
    mass_s: float


def _grip_pct_total(state: PhysicsState, alpha: float) -> float:
    """Relative change of total cornering grip at aero share `alpha` (%)."""
    aero_term = alpha * state.downforce_pct
    mech_term = (1.0 - alpha) * state.mech_grip_pct
    thermal = state.thermal_grip_pct
    weather = 100.0 * (state.grip_multiplier - 1.0)
    return aero_term + mech_term + thermal + weather


def segment_time_delta(kind: str, time_s: float, speed_min_kph: float, speed_mean_kph: float,
                       speed_max_kph: float, state: PhysicsState,
                       cfg: PhysicsConfig | None = None, lap_time_s: float | None = None,
                       fuel_lap_penalty_s: float = 0.0) -> SegmentDelta:
    """Time change of one segment under `state`. Positive = slower."""
    cfg = cfg or default_config()
    if not np.isfinite(time_s) or time_s <= 0:
        return SegmentDelta(0.0, 0.0, 0.0, 0.0)
    v0 = cfg.coeff("car", "aero_crossover_kph").at(state.scale)
    kw = cfg.raw["fuel"]["kind_weight"]

    drag_s = grip_s = 0.0
    if kind == "straight":
        f_term = terminal_fraction(speed_mean_kph, speed_max_kph)
        expo = cfg.coeff("aero", "drag_to_topspeed_exponent").at("nominal")
        dv_over_v = -expo * state.drag_pct / 100.0           # more drag -> lower v_max
        drag_s = -time_s * f_term * dv_over_v                 # slower -> +time
    else:
        v_ref = speed_min_kph if np.isfinite(speed_min_kph) and speed_min_kph > 0 else speed_mean_kph
        alpha = aero_fraction(v_ref, v0)
        g_pct = _grip_pct_total(state, alpha)
        dv_over_v = 0.5 * g_pct / 100.0                        # v ~ sqrt(grip)
        w = KINK_WEIGHT if kind == "kink" else 1.0
        grip_s = -time_s * w * dv_over_v
        # corner drag: a high-speed corner is partly power-limited too
        if kind == "high_speed_corner":
            expo = cfg.coeff("aero", "drag_to_topspeed_exponent").at("nominal")
            drag_s = time_s * 0.3 * expo * state.drag_pct / 100.0

    # Fuel: ONE calibrated number (s per kg per lap, grade B) shared out by
    # segment time and kind weight — corners and their exits carry more of
    # it than a flat-out straight. Not also modelled as mass inside the grip
    # or acceleration terms: that would count the same kilogram twice.
    mass_s = 0.0
    if fuel_lap_penalty_s and lap_time_s:
        mass_s = fuel_lap_penalty_s * (time_s * float(kw.get(kind, 1.0))) / lap_time_s

    return SegmentDelta(delta_s=drag_s + grip_s + mass_s, drag_s=drag_s, grip_s=grip_s, mass_s=mass_s)


def lap_physics_delta(segments: pd.DataFrame, setup: SetupInput, *, session: str = "Q",
                      baseline_fuel_kg: float | None = None, baseline_track_temp_c: float | None = None,
                      roughness: float | None = None, cfg: PhysicsConfig | None = None) -> pd.DataFrame:
    """Per-segment physics deltas with an interval, for one baseline lap.

    `segments` needs: segment_kind, segment_time_s, speed_min_kph,
    speed_mean_kph, speed_max_kph (the gold feature-store columns). Returns
    the same rows plus physics_delta_s / _lo / _hi and the split
    (drag / grip / mass) at nominal coefficients.

    The interval is the spread of the effect over the coefficient sets
    "low" and "high" (every coefficient at value*(1-u) / value*(1+u)). It
    is an effect-SIZE band, not a statistical one — the honest statement
    for grade-C coefficients is "somewhere between half and one-and-a-half
    of this", and that is what the band says.
    """
    from pipeline.physics.modifiers import fuel_lap_penalty_s
    cfg = cfg or default_config()
    lap_time = float(pd.to_numeric(segments["segment_time_s"], errors="coerce").sum())
    out = segments.copy()
    cols: dict[str, list[float]] = {s: [] for s in SCALES}
    split = {"drag": [], "grip": [], "mass": []}
    # normalise kind-weights so a baseline lap's fuel penalty sums exactly to the lap penalty
    kw = cfg.raw["fuel"]["kind_weight"]
    wsum = float(sum(float(kw.get(k, 1.0)) * t for k, t in
                     zip(segments["segment_kind"], pd.to_numeric(segments["segment_time_s"], errors="coerce").fillna(0))))
    norm = lap_time / wsum if wsum > 0 else 1.0

    for scale in SCALES:
        st = physics_state(setup, session=session, baseline_fuel=baseline_fuel_kg,
                           baseline_track_temp_c=baseline_track_temp_c, roughness=roughness,
                           cfg=cfg, scale=scale)
        pen = fuel_lap_penalty_s(st.fuel_delta_kg, cfg, scale) * norm
        for _, r in segments.iterrows():
            d = segment_time_delta(str(r["segment_kind"]), float(r["segment_time_s"]),
                                   float(r.get("speed_min_kph", np.nan)), float(r.get("speed_mean_kph", np.nan)),
                                   float(r.get("speed_max_kph", np.nan)), st, cfg,
                                   lap_time_s=lap_time, fuel_lap_penalty_s=pen)
            cols[scale].append(d.delta_s)
            if scale == "nominal":
                split["drag"].append(d.drag_s); split["grip"].append(d.grip_s); split["mass"].append(d.mass_s)

    nom, lo, hi = (np.asarray(cols[s], dtype=float) for s in SCALES)
    out["physics_delta_s"] = nom
    out["physics_delta_lo_s"] = np.minimum.reduce([nom, lo, hi])
    out["physics_delta_hi_s"] = np.maximum.reduce([nom, lo, hi])
    out["physics_drag_s"] = split["drag"]
    out["physics_grip_s"] = split["grip"]
    out["physics_mass_s"] = split["mass"]
    return out


def lap_summary(deltas: pd.DataFrame) -> dict:
    """Whole-lap numbers for the HUD header."""
    return {
        "physics_delta_s": float(deltas["physics_delta_s"].sum()),
        "physics_delta_lo_s": float(deltas["physics_delta_lo_s"].sum()),
        "physics_delta_hi_s": float(deltas["physics_delta_hi_s"].sum()),
        "drag_s": float(deltas["physics_drag_s"].sum()),
        "grip_s": float(deltas["physics_grip_s"].sum()),
        "mass_s": float(deltas["physics_mass_s"].sum()),
    }
