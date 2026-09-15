"""P2-5..P2-7 gate tests — the feature store's semantics.

These pin the decisions, not the plumbing: what a target means, what effort
measures, why driver style is compared segment-wise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import driver_style, effort, run as frun, segment_features, setup_proxy


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


def test_a_dropout_is_reported_even_though_it_counts_toward_the_total():
    """Excluding the gap made segment times stop summing to the lap, so it is
    reported rather than removed — a corrupted lap must not look clean."""
    tel = _lap(40, dt=0.24)
    tel.loc[20:, "session_time_s"] += 5.0            # a 5 s hole mid-segment
    f = segment_features.features_for_segment(tel, _seg(end=200.0), 1000.0,
                                              time_span_s=14.4)
    assert f["segment_time_gap_s"] > 4.0


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


# ------------------------------------------------- distance-axis alignment
def test_each_lap_is_rescaled_onto_the_reference_axis():
    """add_distance integrates speed x dt, so every lap gets its own length.

    Segments are defined on the reference lap's axis; a 2% disagreement puts a
    boundary ~90 m away from where it belongs, which is a quarter of a 380 m
    corner. The difference then lands in the target as if a driver caused it.
    """
    short = _lap(100, step=10.0)                       # 990 m
    long_ = _lap(100, step=10.2)                       # 1009.8 m
    a = segment_features.align_distance(short, 1000.0)
    b = segment_features.align_distance(long_, 1000.0)
    assert abs(a["distance_m"].max() - 1000.0) < 1e-6
    assert abs(b["distance_m"].max() - 1000.0) < 1e-6
    assert a["distance_scale"].iloc[0] != b["distance_scale"].iloc[0]


def test_alignment_keeps_segment_boundaries_comparable_between_laps():
    seg = _seg(kind="low_speed_corner", start=400.0, end=600.0)
    short = segment_features.align_distance(_lap(100, step=10.0), 1000.0)
    long_ = segment_features.align_distance(_lap(100, step=10.2), 1000.0)
    n_short = len(segment_features.segment_slice(short, seg["start_m"], seg["end_m"], 1000.0))
    n_long = len(segment_features.segment_slice(long_, seg["start_m"], seg["end_m"], 1000.0))
    assert abs(n_short - n_long) <= 1, "the same segment must collect the same samples"


# ------------------------------------------------------------- the target
def _target_frame(times, gaps=None, effort_class="push"):
    n = len(times)
    return pd.DataFrame({
        "season": 2024, "event": "X", "session": "Q", "segment_index": 0,
        "lap_uid": [f"l{i}" for i in range(n)],
        "segment_time_s": times,
        "segment_time_gap_s": gaps if gaps is not None else [0.0] * n,
        "lap_effort_class": effort_class,
    })


def test_a_lap_with_a_gap_cannot_set_the_reference():
    """The gap interval is excluded from the integration on purpose, so such a
    lap's time is UNDERSTATED. Letting it win the minimum measures every
    honest lap against a time nobody drove."""
    df = _target_frame([2.70, 2.72, 2.75, 1.10], gaps=[0.0, 0.0, 0.0, 1.6])
    out = frun.add_targets(df)
    assert out["segment_reference_s"].iloc[0] > 2.5
    assert (out["segment_delta_s"] < 1.0).all()


def test_the_reference_is_a_low_quantile_not_a_bare_minimum():
    """min over 88 laps is the most outlier-sensitive statistic available."""
    df = _target_frame([2.70] * 40 + [2.10])
    out = frun.add_targets(df)
    assert out["segment_reference_s"].iloc[0] > 2.10


def test_cruise_laps_do_not_define_the_best():
    push = _target_frame([2.70, 2.72], effort_class="push")
    cruise = _target_frame([4.50, 4.60], effort_class="cruise")
    cruise["lap_uid"] = ["c0", "c1"]
    out = frun.add_targets(pd.concat([push, cruise], ignore_index=True))
    assert out["segment_reference_s"].iloc[0] < 3.0
    assert out.loc[out["lap_uid"] == "c0", "segment_delta_s"].iloc[0] > 1.0


# ---------------------------------------------------------- the invariant
def test_segment_times_summing_to_the_lap_is_checked():
    """A broken target is invisible until something asserts the total."""
    good = pd.DataFrame({
        "lap_uid": ["a"] * 4 + ["b"] * 4,
        "segment_time_s": [20.0, 20.0, 20.0, 18.0] + [20.0, 20.0, 20.0, 19.0],
        "lap_time_s": [78.0] * 4 + [79.0] * 4,
    })
    assert frun.check_segment_times_sum_to_the_lap(good)["passes"]

    broken = good.copy()
    broken.loc[broken["lap_uid"] == "a", "segment_time_s"] = 12.0   # sums to 48
    rep = frun.check_segment_times_sum_to_the_lap(broken)
    assert not rep["passes"]
    assert rep["laps_over_1pct"] >= 1


# ------------------------------------------- the boundary-interval defect
def test_segment_times_tile_the_lap_without_shedding_a_sampling_period():
    """N samples give N-1 intervals, so summing interior intervals sheds one
    sampling period per segment. Across 35 segments that was ~8 s of an 80 s
    lap — a 7% error the invariant caught on the first real build."""
    n, dt, step = 300, 0.24, 5.0
    tel = _lap(n, dt=dt, step=step)
    lap_len = n * step
    total_time = (n - 1) * dt

    edges = np.linspace(0, lap_len, 11)        # 10 tiling segments
    segs = [_seg(i, "straight", float(edges[i]), float(edges[i + 1]))
            for i in range(10)]
    built = segment_features.features_for_lap(tel, segs, lap_len)
    summed = float(built["segment_time_s"].sum())
    assert abs(summed - total_time) / total_time < 0.01, (
        f"segments summed to {summed:.2f}s against a {total_time:.2f}s lap")


def test_a_gap_is_reported_without_being_removed_from_the_total():
    """The total has to tile the lap, so the gap can no longer be excluded —
    but it must still be visible, or a corrupted lap looks clean."""
    tel = _lap(40, dt=0.24, step=5.0)
    tel.loc[20:, "session_time_s"] += 3.0
    seg = _seg(end=200.0)
    built = segment_features.features_for_lap(tel, [seg], 1000.0)
    assert built["segment_time_gap_s"].iloc[0] > 2.5
    assert built["segment_time_s"].iloc[0] > 3.0, "the gap stays in the total"


# ------------------------------------------------ per-circuit calibration
def test_the_smoothing_window_scales_with_the_circuit():
    """Monaco packs 19 turns into 3.3 km; a road course runs nearer 370 m per
    turn. The global 90 m window found 8 of Monaco's 19."""
    from pipeline.segment import geometry as G
    assert G.auto_window_m(3337.0) < G.auto_window_m(7004.0)
    assert 30.0 <= G.auto_window_m(1000.0) <= 90.0
    assert 30.0 <= G.auto_window_m(20000.0) <= 90.0
    # calibrated on the 19-session grids: 40-50 m matched published counts
    # where 90 m found two-thirds of them
    assert 45.0 <= G.auto_window_m(5300.0) <= 60.0


def test_a_circuit_override_beats_the_auto_scaled_window():
    from pipeline.segment import run as srun
    base = {"geometry_step_m": 10.0, "poly_order": 2}
    auto = srun.circuit_cfg(base, None, 3337.0)
    override = srun.circuit_cfg(base, {"segmentation": {"smooth_window_m": 50.0}}, 3337.0)
    assert override["smooth_window_m"] == 50.0
    assert auto["smooth_window_m"] != 50.0


def test_a_null_global_window_lets_the_auto_scale_apply():
    """The global config carried a hard 90 m that beat the auto-scale for every
    circuit: the calibration grids said 40-50 m and 90 m was applied to
    thirteen of fourteen circuits regardless. null must mean auto."""
    from pipeline.segment import run as srun
    from pipeline.segment import geometry as G
    for base in ({"smooth_window_m": None}, {}):
        cfg = srun.circuit_cfg(base, None, 5336.0)
        assert abs(cfg["smooth_window_m"] - G.auto_window_m(5336.0)) < 1e-9
    hard = srun.circuit_cfg({"smooth_window_m": 90.0}, None, 5336.0)
    assert hard["smooth_window_m"] == 90.0, "an explicit global value is still honoured"


def test_gap_closure_scales_with_the_window_unless_pinned():
    """A fixed 30 m gap merged Bahrain's corner complexes at every window."""
    from pipeline.segment import run as srun
    auto = srun.circuit_cfg({"smooth_window_m": None, "min_gap_m": None}, None, 5336.0)
    assert abs(auto["min_gap_m"] - auto["smooth_window_m"] / 3.0) < 0.15
    pinned = srun.circuit_cfg({"smooth_window_m": None, "min_gap_m": None},
                              {"segmentation": {"min_gap_m": 15.0}}, 3337.0)
    assert pinned["min_gap_m"] == 15.0


# ------------------------------------------------------ coverage gate
def test_partial_coverage_laps_are_excluded_not_stretched():
    """Rescaling a half-lap to full length stretches half a circuit across the
    whole. The 19-session build carried a 1.82x rescale and a 92 s
    per-segment delta because of it."""
    assert frun.COVERAGE_MIN <= 0.96 and frun.COVERAGE_MAX >= 1.04
    half = 0.55
    assert not (frun.COVERAGE_MIN <= half <= frun.COVERAGE_MAX)
    fine = 0.982
    assert frun.COVERAGE_MIN <= fine <= frun.COVERAGE_MAX
