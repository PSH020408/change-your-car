"""P2-5..P2-7 gate tests — the feature store's semantics.

These pin the decisions, not the plumbing: what a target means, what effort
measures, why driver style is compared segment-wise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import driver_style, effort, segment_features, setup_proxy


def _lap(n: int = 60, speed=250.0, throttle=100, brake=False, t0: float = 0.0,
         d0: float = 0.0, step: float = 5.0, dt: float = 0.24) -> pd.DataFrame:
    return pd.DataFrame({
        "distance_m": d0 + np.arange(n) * step,
        "speed_kph": np.full(n, speed, dtype=float),
        "throttle_pct": np.full(n, throttle, dtype=float),
        "brake_on": np.full(n, brake, dtype=bool),
        "session_time_s": t0 + np.arange(n) * dt,
        "drs_raw": np.zeros(n, dtype=float),
    })


# --------------------------------------------------------------- effort
def test_a_push_lap_and_a_cruise_lap_are_distinguishable():
    """2024 Monaco kept 89% of a procession because the pace gate could not
    see the difference. Effort is built from the pedals, not from lap time."""
    push = pd.concat([_lap(40, throttle=100, brake=False),
                      _lap(20, throttle=0, brake=True)], ignore_index=True)
    cruise = _lap(60, throttle=40, brake=False)

    e_push = effort.lap_effort(push)
    e_cruise = effort.lap_effort(cruise)
    assert e_push["effort_index"] > e_cruise["effort_index"]
    assert effort.classify_effort(e_push["effort_index"]) == "push"
    assert effort.classify_effort(e_cruise["effort_index"]) == "cruise"


def test_effort_uses_pedals_not_lap_time():
    """Deriving effort from pace would be circular: pace is what the setup
    effect shows up in, so it cannot also decide which laps show it."""
    src = __import__("inspect").getsource(effort.lap_effort)
    assert "lap_time" not in src and "pace" not in src


def test_coasting_counts_against_effort():
    coasting = _lap(60, throttle=5, brake=False)
    assert effort.lap_effort(coasting)["coast_frac"] > 0.9
    assert effort.lap_effort(coasting)["effort_index"] < 0.2


# ------------------------------------------------------ segment features
def _seg(idx=0, kind="straight", start=0.0, end=300.0, radius=None):
    return {"index": idx, "kind": kind, "start_m": start, "end_m": end,
            "length_m": end - start, "min_radius_m": radius,
            "direction": "straight", "sector": 1}


def test_segment_time_integrates_intervals_not_endpoints():
    """A dropout inside a segment must not be charged to the driver as time."""
    tel = _lap(40, dt=0.24)
    tel.loc[20:, "session_time_s"] += 5.0            # a 5 s hole mid-segment
    f = segment_features.features_for_segment(tel, _seg(end=200.0), 1000.0)
    assert f["segment_time_s"] < 12.0, "the hole must not be counted as driving"
    assert f["segment_time_gap_s"] > 4.0, "but it must be reported"


def test_brake_point_is_reported_as_a_fraction_of_the_segment():
    tel = _lap(60, brake=False)
    tel.loc[30:, "brake_on"] = True
    f = segment_features.features_for_segment(tel, _seg(end=300.0), 1000.0)
    assert 0.45 < f["brake_point_frac"] < 0.55
    assert 0.45 < f["brake_frac"] < 0.55


def test_a_segment_wrapping_start_finish_collects_both_ends():
    lap_len = 1000.0
    tel = pd.concat([_lap(20, d0=0.0, step=5.0), _lap(20, d0=900.0, step=5.0)],
                    ignore_index=True)
    got = segment_features.segment_slice(tel, 900.0, lap_len + 100.0, lap_len)
    assert len(got) == 40, "a wrapped segment must not silently lose half itself"


def test_lateral_load_is_computed_from_the_segments_own_radius():
    tel = _lap(40, speed=180.0)
    f = segment_features.features_for_segment(
        tel, _seg(kind="medium_speed_corner", end=200.0, radius=100.0), 1000.0)
    expected = (180 / 3.6) ** 2 / 100.0 / 9.81
    assert abs(f["lateral_g_peak"] - expected) < 0.01


# ---------------------------------------------------------- setup proxies
def test_missing_traps_are_imputed_from_their_own_location():
    tel = _lap(200, step=10.0)
    tel["speed_kph"] = np.linspace(100, 320, 200)
    segs = [_seg(0, "straight", 0.0, 1200.0), _seg(1, "straight", 1200.0, 1500.0)]
    lap = pd.Series({"speed_i1_kph": np.nan, "speed_i2_kph": 210.0,
                     "speed_fl_kph": np.nan, "speed_st_kph": np.nan})
    out = setup_proxy.impute_traps(lap, tel, segs, [500.0, 1200.0], 2000.0)

    assert out["speed_i1_imputed"] and np.isfinite(out["speed_i1_kph"])
    assert not out["speed_i2_imputed"], "a real reading must never be overwritten"
    assert out["speed_i2_kph"] == 210.0
    assert out["speed_st_imputed"]
    assert out["traps_imputed_count"] == 3


def test_the_speed_trap_is_taken_from_the_longest_straight():
    segs = [_seg(0, "straight", 0.0, 200.0), _seg(1, "straight", 400.0, 1500.0),
            _seg(2, "low_speed_corner", 200.0, 400.0)]
    assert setup_proxy.longest_straight(segs)["index"] == 1


def test_aero_balance_needs_both_ends_of_the_trade_off():
    tel = _lap(100, step=10.0)
    lap = pd.Series({"speed_i1_kph": 200.0, "speed_i2_kph": 200.0,
                     "speed_fl_kph": 300.0, "speed_st_kph": 320.0})
    out = setup_proxy.impute_traps(lap, tel, [_seg(0)], [300.0, 600.0], 1000.0)
    assert abs(out["aero_balance_proxy"] - 1.6) < 1e-6
    assert out["drag_proxy_kph"] == 320.0 and out["downforce_proxy_kph"] == 200.0


# ---------------------------------------------------------- driver style
def _style_frame():
    rows = []
    for seg in range(4):
        for drv, brake, apex in (("AGG", 0.80, 150.0), ("MID", 0.60, 140.0),
                                 ("CON", 0.40, 130.0)):
            for lap in range(10):
                rows.append({"season": 2024, "event": "X", "session": "R",
                             "segment_index": seg, "segment_kind": "low_speed_corner",
                             "driver": drv, "brake_point_frac": brake,
                             "speed_min_kph": apex, "throttle_on_frac": 1 - brake})
    return pd.DataFrame(rows)


def test_style_is_measured_against_the_field_on_the_same_segment():
    """Averaging a driver's braking point across a season compares Monaco's
    hairpin with Monza's Parabolica and measures the calendar, not the driver."""
    assert driver_style.GROUP == ["season", "event", "session", "segment_index"]
    z = driver_style.segment_wise_z(_style_frame())
    per_seg = z.groupby(["segment_index"])["z_brake_point_frac"].mean()
    assert np.allclose(per_seg.values, 0.0, atol=1e-9), "z-scores centre per segment"


def test_the_aggressive_driver_scores_above_the_conservative_one():
    bias = driver_style.driver_bias(_style_frame(), min_samples=5)
    got = bias.set_index("driver")["aggression_index"]
    assert got["AGG"] > got["MID"] > got["CON"]


def test_a_bias_from_too_few_segments_is_blanked_not_trusted():
    thin = _style_frame().head(6)
    bias = driver_style.driver_bias(thin, min_samples=20)
    zcols = [c for c in bias.columns if c.startswith("bias_")]
    assert bias[zcols].isna().all().all()
