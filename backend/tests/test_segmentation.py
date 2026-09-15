"""P2 gate tests — geometry and segmentation against known-answer shapes.

Curvature is a second derivative, so an implementation can look plausible and
be wrong by a factor. These use synthetic tracks whose curvature is known
analytically, which is the only way to catch that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.segment import geometry as G, segmentation as S, svg as V

XY_UNIT = 10.0          # pretend the position feed is in 1/10 m


def _circle(radius: float, n: int = 900, jitter: float = 0.0, seed: int = 5):
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = radius * np.cos(th), radius * np.sin(th)
    if jitter:
        x = x + rng.normal(0, jitter, n)
        y = y + rng.normal(0, jitter, n)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return pd.DataFrame({"distance_m": d, "pos_x": x * XY_UNIT, "pos_y": y * XY_UNIT,
                         "speed_kph": np.full(n, 150.0)}), float(d[-1])


def _oval(radius: float = 120.0, straight: float = 600.0, n: int = 400):
    a1 = np.linspace(-np.pi / 2, np.pi / 2, n, endpoint=False)
    a2 = np.linspace(np.pi / 2, 3 * np.pi / 2, n, endpoint=False)
    pts = np.vstack([
        np.c_[np.linspace(0, straight, n, endpoint=False), np.zeros(n)],
        np.c_[straight + radius * np.cos(a1), radius + radius * np.sin(a1)],
        np.c_[np.linspace(straight, 0, n, endpoint=False), np.full(n, 2 * radius)],
        np.c_[radius * np.cos(a2), radius + radius * np.sin(a2)],
    ])
    d = np.r_[0, np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
    speed = np.where(np.abs(G.curvature(pts[:, 0], pts[:, 1], 1.0)) > 0.004, 110.0, 300.0)
    frame = pd.DataFrame({"distance_m": d, "pos_x": pts[:, 0] * XY_UNIT,
                          "pos_y": pts[:, 1] * XY_UNIT})
    return frame, float(d[-1]), d, speed


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("radius", [80.0, 200.0, 450.0])
def test_curvature_matches_the_analytic_value_for_a_circle(radius):
    frame, length = _circle(radius)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    measured = float(np.mean(np.abs(geo.curvature_1pm)))
    assert abs(measured - 1 / radius) / (1 / radius) < 0.02


def test_scale_is_derived_not_assumed():
    """Every curvature in 1/m depends on the X/Y unit; deriving it from the
    lap's known length keeps the pipeline correct if the convention changes."""
    frame, length = _circle(200.0)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    assert abs(geo.metres_per_unit - 1 / XY_UNIT) / (1 / XY_UNIT) < 0.01


def test_closed_loop_padding_keeps_start_finish_continuous():
    """Reflecting pads mirror the curve back on itself at start-finish.

    On a circle that showed up as an 18 m closure error; a circuit is a loop,
    so the samples before the start line are the ones at the end of the lap.
    """
    frame, length = _circle(200.0)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    assert geo.closure_error_m < 2.0


def test_curvature_sign_distinguishes_left_from_right():
    frame, length = _circle(150.0)
    left = G.build(frame, lap_distance_m=length, step_m=10.0)
    frame_r = frame.assign(pos_y=-frame["pos_y"])
    right = G.build(frame_r, lap_distance_m=length, step_m=10.0)
    assert np.mean(left.curvature_1pm) > 0 > np.mean(right.curvature_1pm)


def test_geometry_rejects_a_lap_with_too_few_samples():
    frame = pd.DataFrame({"distance_m": [0, 1, 2], "pos_x": [0, 1, 2], "pos_y": [0, 0, 0]})
    with pytest.raises(ValueError):
        G.build(frame, lap_distance_m=2.0)


# -------------------------------------------------------------- segmentation
def test_oval_resolves_to_two_corners_and_two_straights():
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    cfg = {"curvature_threshold_1pm": 0.0035, "min_gap_m": 30.0,
           "min_segment_len_m": 40.0,
           "corner_speed_bins": {"low": [0, 120], "medium": [120, 200],
                                 "high": [200, 400]}}
    segs = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)
    corners = [s for s in segs if S.is_corner_kind(s.kind)]
    straights = [s for s in segs if s.kind == "straight"]
    assert len(corners) == 2 and len(straights) == 2


def test_a_corner_straddling_start_finish_is_counted_once():
    """Rebuilding segments from the un-rolled mask split it in two, which on
    the oval produced a 10 m phantom corner at distance zero."""
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    cfg = {"curvature_threshold_1pm": 0.0035, "min_gap_m": 30.0,
           "min_segment_len_m": 40.0, "corner_speed_bins": {}}
    segs = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)
    wrapped = [s for s in segs if s.wraps_start_finish]
    assert len(wrapped) == 1
    assert wrapped[0].end_m > geo.lap_length_m       # runs past the line
    assert not any(s.length_m < 40.0 for s in segs)  # no phantom fragments


def test_segments_tile_the_whole_lap_without_gaps_or_overlap():
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    cfg = {"curvature_threshold_1pm": 0.0035, "min_gap_m": 30.0,
           "min_segment_len_m": 40.0, "corner_speed_bins": {}}
    segs = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)
    total = sum(s.length_m for s in segs)
    assert abs(total - geo.lap_length_m) <= geo.grid_step_m + 1.0


def test_sweep_agrees_with_what_segmentation_actually_emits():
    """A sweep table reporting a different count than the pipeline produces is
    worse than none — the threshold gets chosen against the wrong column."""
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    kw = {"min_gap_m": 30.0, "min_segment_len_m": 40.0}
    sweep = S.sweep_thresholds(geo.curvature_1pm, geo.grid_step_m, [0.0035, 0.005], **kw)
    for t, expected in sweep.items():
        cfg = {"curvature_threshold_1pm": float(t), **kw, "corner_speed_bins": {}}
        segs = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)
        assert S.count_turns(segs)["turns"] == expected


def test_threshold_above_the_actual_curvature_finds_no_corners():
    """R=120 m is k=0.0083; a 0.010 threshold must find nothing."""
    frame, length, _, _ = _oval(radius=120.0)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    assert S.sweep_thresholds(geo.curvature_1pm, geo.grid_step_m, [0.010])["0.0100"] == 0


def test_corner_speed_classification_uses_apex_not_entry():
    bins = {"low": (0, 120), "medium": (120, 200), "high": (200, 400)}
    assert S.classify_corner(95.0, bins) == "low_speed_corner"
    assert S.classify_corner(160.0, bins) == "medium_speed_corner"
    assert S.classify_corner(250.0, bins) == "high_speed_corner"


def test_straights_get_no_apex_speed():
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    cfg = {"curvature_threshold_1pm": 0.0035, "min_gap_m": 30.0,
           "min_segment_len_m": 40.0, "corner_speed_bins": {}}
    segs = S.build_segments(geo, cfg, raw_distance_m=raw_d, raw_speed_kph=raw_v)
    for s in segs:
        if s.kind == "straight":
            assert s.apex_speed_kph is None
            assert s.min_speed_kph is not None


def test_double_apex_corner_is_not_split_in_two():
    """Curvature dips between the two apexes; min_gap_m closes it.

    The corner is padded with straight on both sides — without that, an array
    that is 'corner' at index 0 and at index -1 looks like a start-finish
    straddle to the wrap logic, which is correct for a lap and wrong for a
    bare arc.
    """
    n = 300
    a = np.linspace(0, np.pi, n)
    arc = 0.008 * (1 - 0.7 * np.exp(-((a - np.pi / 2) ** 2) / 0.02))  # dip at centre
    k = np.concatenate([np.zeros(60), arc, np.zeros(60)])
    mask = S.corner_mask(k, 10.0, 0.0035, min_gap_m=200.0, min_segment_len_m=40.0)
    assert S.count_corners(mask) == 1, "a wide gap setting must fuse the two apexes"
    tight = S.corner_mask(k, 10.0, 0.0035, min_gap_m=5.0, min_segment_len_m=40.0)
    assert S.count_corners(tight) == 2, "a tight setting must leave them separate"


def test_microsectors_tile_the_lap():
    ms = S.microsectors(5000.0, 28)
    assert len(ms) == 28
    assert ms[0]["start_m"] == 0.0
    assert abs(ms[-1]["end_m"] - 5000.0) < 0.1
    for a, b in zip(ms, ms[1:]):
        assert abs(a["end_m"] - b["start_m"]) < 0.1


def test_sector_boundaries_come_from_the_laps_own_time_axis():
    d = np.linspace(0, 5000, 500)
    t = np.linspace(100.0, 190.0, 500)          # lap starts at t=100, 90 s long
    b = S.sector_boundaries_from_times(d, t, lap_start_s=100.0,
                                       sector1_s=30.0, sector2_s=30.0)
    assert len(b) == 2
    assert abs(b[0] - 5000 / 3) < 30 and abs(b[1] - 2 * 5000 / 3) < 30


# ---------------------------------------------------------------------- svg
def test_track_map_preserves_aspect_ratio():
    """Stretching a circuit to fill the box would misplace every overlay."""
    frame, length, _, _ = _oval(radius=120.0, straight=900.0)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    m = V.build(geo)
    xs = [p["x"] for p in m["distance_index"]]
    ys = [p["y"] for p in m["distance_index"]]
    aspect_track = (max(geo.x_m) - min(geo.x_m)) / (max(geo.y_m) - min(geo.y_m))
    aspect_svg = (max(xs) - min(xs)) / (max(ys) - min(ys))
    assert abs(aspect_track - aspect_svg) / aspect_track < 0.05


def test_track_map_is_small_enough_for_a_lightweight_hud():
    frame, length, _, _ = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    m = V.build(geo)
    assert m["path"].startswith("M ") and m["path"].endswith(" Z")
    assert len(m["path"]) < 20000        # the whole point is a light 2D HUD
    assert len(m["distance_index"]) > 5


# ------------------------------------------- noise robustness (calibration)
@pytest.mark.parametrize("jitter", [0.2, 0.5, 1.0])
def test_curvature_survives_realistic_position_jitter(jitter):
    """The setting that mattered, pinned.

    The original 40 m / cubic window gave 16% curvature error at 0.5 m of
    jitter and 80% at 1.0 m — which is what produced 10 g corners on the 2022
    Australian GP. 90 m / quadratic holds under a few percent throughout.
    """
    frame, length = _circle(200.0, n=300, jitter=jitter)
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    measured = float(np.mean(np.abs(geo.curvature_1pm)))
    assert abs(measured - 1 / 200.0) / (1 / 200.0) < 0.08


def test_the_old_narrow_window_is_measurably_worse():
    """Guards the calibration itself: if someone narrows the window back, the
    accuracy loss is a test failure rather than a silent regression."""
    frame, length = _circle(200.0, n=300, jitter=0.5)
    good = G.build(frame, lap_distance_m=length, step_m=10.0,
                   smooth_window_m=90.0, poly_order=2)
    bad = G.build(frame, lap_distance_m=length, step_m=10.0,
                  smooth_window_m=40.0, poly_order=3)
    e_good = abs(np.mean(np.abs(good.curvature_1pm)) - 0.005) / 0.005
    e_bad = abs(np.mean(np.abs(bad.curvature_1pm)) - 0.005) / 0.005
    assert e_good < e_bad


def test_curvature_implying_impossible_g_is_rejected():
    """A radius that would need more grip than exists is noise, not a corner."""
    n = 300
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = 200.0
    x, y = R * np.cos(th), R * np.sin(th)
    d = np.linspace(0, 2 * np.pi * R, n, endpoint=False)
    # a single sample displaced hard sideways — the classic GPS glitch
    x[150] += 12.0
    frame = pd.DataFrame({"distance_m": d, "pos_x": x * XY_UNIT, "pos_y": y * XY_UNIT,
                          "speed_kph": np.full(n, 300.0)})
    geo = G.build(frame, lap_distance_m=2 * np.pi * R, step_m=10.0)
    assert geo.lateral_g is not None
    assert float(np.nanmax(geo.lateral_g)) <= G.MAX_PHYSICAL_LATERAL_G + 1e-6


# ------------------------------------------------------ physics invariants
def test_physics_check_flags_a_corner_over_the_grip_limit():
    seg = S.Segment(index=0, kind="high_speed_corner", start_m=0, end_m=100,
                    length_m=100, mean_curvature_1pm=0.02, peak_curvature_1pm=0.02,
                    min_radius_m=44.5, direction="right", apex_speed_kph=233.0)
    rep = S.check_physics([seg], 0.0035)
    assert not rep["passes"]
    assert rep["corners_over_g_limit"][0]["lateral_g"] > 6.5


def test_physics_check_flags_a_straight_that_hides_a_corner():
    seg = S.Segment(index=0, kind="straight", start_m=0, end_m=660, length_m=660,
                    mean_curvature_1pm=0.0, peak_curvature_1pm=0.0,
                    min_radius_m=21.4, direction="straight")
    rep = S.check_physics([seg], 0.0035)
    assert not rep["passes"]
    assert rep["straights_hiding_a_corner"][0]["radius_m"] == 21.4


def test_physics_check_passes_on_a_plausible_lap():
    segs = [
        S.Segment(index=0, kind="low_speed_corner", start_m=0, end_m=90, length_m=90,
                  mean_curvature_1pm=0.02, peak_curvature_1pm=0.02,
                  min_radius_m=50.0, direction="left", apex_speed_kph=110.0),
        S.Segment(index=1, kind="straight", start_m=90, end_m=800, length_m=710,
                  mean_curvature_1pm=0.0, peak_curvature_1pm=0.0,
                  min_radius_m=900.0, direction="straight"),
    ]
    assert S.check_physics(segs, 0.0035)["passes"]


def test_a_straight_at_the_classification_boundary_is_not_a_failure():
    """R=281 m against a 286 m threshold is a 2% judgement call, not a defect.

    The first version flagged it, which turned the invariant into noise and
    would have buried a real 21 m violation in the same list.
    """
    seg = S.Segment(index=8, kind="straight", start_m=0, end_m=100, length_m=100,
                    mean_curvature_1pm=0.003, peak_curvature_1pm=0.0036,
                    min_radius_m=280.9, direction="straight")
    rep = S.check_physics([seg], 0.0035)
    assert rep["passes"]
    assert rep["straights_hiding_a_corner"] == []
    assert rep["straights_at_the_boundary"][0]["index"] == 8


def test_sweep_full_reports_composition_and_physics_per_threshold():
    frame, length, raw_d, raw_v = _oval()
    geo = G.build(frame, lap_distance_m=length, step_m=10.0)
    cfg = {"min_gap_m": 30.0, "min_segment_len_m": 40.0,
           "corner_speed_bins": {"low": [0, 120], "medium": [120, 200],
                                 "high": [200, 400]}}
    rows = S.sweep_full(geo, cfg, [0.0025, 0.0035],
                        raw_distance_m=raw_d, raw_speed_kph=raw_v)
    for r in rows:
        assert r["turns"] == 2
        assert r["low"] + r["medium"] + r["high"] == r["corners"]
        assert r["corners"] + r["kinks"] == r["turns"]
        assert r["physics_passes"] is True
        # the threshold's physical meaning travels with it
        assert 0.5 < r["lateral_g_at_300kph"] < 10


def test_slice_gap_is_reported_separately_from_closure():
    """A telemetry lap that does not cover the full circuit cannot close to
    zero; charging that to the smoothing hid how incomplete the slice was."""
    frame, length = _circle(200.0, n=400)
    cut = frame.iloc[: int(len(frame) * 0.94)].copy()      # drop the last 6%
    geo = G.build(cut, lap_distance_m=float(cut["distance_m"].iloc[-1]), step_m=10.0)
    assert geo.slice_gap_m > 20.0
    assert geo.closure_error_m < geo.slice_gap_m


def test_kinks_are_judged_against_the_approach_speed_not_their_own_entry():
    """Inside a corner the car has already finished braking, so its first
    sample is the slowed speed — comparing against that makes every corner
    look flat. The approach is the previous segment's exit."""
    segs = [
        S.Segment(index=0, kind="straight", start_m=0, end_m=500, length_m=500,
                  mean_curvature_1pm=0.0, peak_curvature_1pm=0.0, min_radius_m=900.0,
                  direction="straight", exit_speed_kph=300.0, max_speed_kph=300.0),
        # braked hard for this one: 300 -> 120
        S.Segment(index=1, kind="low_speed_corner", start_m=500, end_m=600,
                  length_m=100, mean_curvature_1pm=0.02, peak_curvature_1pm=0.02,
                  min_radius_m=50.0, direction="left",
                  apex_speed_kph=120.0, entry_speed_kph=125.0, exit_speed_kph=150.0,
                  max_speed_kph=150.0),
        # took this one flat: approached at 150, apex 150
        S.Segment(index=2, kind="high_speed_corner", start_m=600, end_m=640,
                  length_m=40, mean_curvature_1pm=0.005, peak_curvature_1pm=0.005,
                  min_radius_m=200.0, direction="left",
                  apex_speed_kph=150.0, entry_speed_kph=150.0, exit_speed_kph=160.0,
                  max_speed_kph=160.0),
    ]
    S._mark_kinks(segs)
    assert segs[1].kind == "low_speed_corner", "a corner braked for stays a corner"
    assert segs[2].kind == "kink"


def test_a_curve_taken_flat_is_a_kink_not_a_corner():
    """Albert Park returned 40 m of curvature at 305 km/h with entry == apex
    == exit as a 'high speed corner'. Real geometry, but nobody lifted — and
    a setup change is not felt there the way it is in a corner."""
    worked = S.Segment(index=0, kind="high_speed_corner", start_m=0, end_m=150,
                       length_m=150, mean_curvature_1pm=0.006,
                       peak_curvature_1pm=0.006, min_radius_m=160.0,
                       direction="left", apex_speed_kph=230.0, entry_speed_kph=300.0)
    flat = S.Segment(index=1, kind="high_speed_corner", start_m=200, end_m=240,
                     length_m=40, mean_curvature_1pm=0.005,
                     peak_curvature_1pm=0.005, min_radius_m=180.0,
                     direction="left", apex_speed_kph=305.0, entry_speed_kph=305.0)
    assert worked.apex_speed_kph < S.KINK_SPEED_RETENTION * worked.entry_speed_kph
    assert flat.apex_speed_kph >= S.KINK_SPEED_RETENTION * flat.entry_speed_kph


def test_published_turn_counts_include_kinks():
    segs = [
        S.Segment(index=0, kind="low_speed_corner", start_m=0, end_m=90, length_m=90,
                  mean_curvature_1pm=0.02, peak_curvature_1pm=0.02,
                  min_radius_m=50.0, direction="left", apex_speed_kph=110.0),
        S.Segment(index=1, kind="kink", start_m=90, end_m=130, length_m=40,
                  mean_curvature_1pm=0.005, peak_curvature_1pm=0.005,
                  min_radius_m=200.0, direction="left", apex_speed_kph=305.0),
        S.Segment(index=2, kind="straight", start_m=130, end_m=900, length_m=770,
                  mean_curvature_1pm=0.0, peak_curvature_1pm=0.0,
                  min_radius_m=900.0, direction="straight"),
    ]
    t = S.count_turns(segs)
    assert t == {"corners": 1, "kinks": 1, "turns": 2}


def test_one_marginal_straight_does_not_fail_a_whole_circuit():
    """The gate exists to catch a 21 m hairpin hiding in a straight. Failing a
    circuit over a single 263 m radius makes the alarm worthless."""
    segs = [S.Segment(index=i, kind="straight", start_m=i*100, end_m=i*100+80,
                      length_m=80, mean_curvature_1pm=0.0, peak_curvature_1pm=0.0,
                      min_radius_m=900.0, direction="straight") for i in range(15)]
    segs[8].min_radius_m = 263.0
    rep = S.check_physics(segs, 0.0025)
    assert rep["passes"], "one borderline straight in fifteen is not contamination"

    segs[3].min_radius_m = 21.0
    segs[5].min_radius_m = 30.0
    assert not S.check_physics(segs, 0.0025)["passes"]


def test_over_g_always_fails_regardless_of_budget():
    seg = S.Segment(index=0, kind="high_speed_corner", start_m=0, end_m=100,
                    length_m=100, mean_curvature_1pm=0.02, peak_curvature_1pm=0.02,
                    min_radius_m=44.5, direction="right", apex_speed_kph=233.0)
    assert not S.check_physics([seg], 0.0025)["passes"]
