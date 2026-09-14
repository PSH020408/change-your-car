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


def _circle(radius: float, n: int = 900):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = radius * np.cos(th) * XY_UNIT, radius * np.sin(th) * XY_UNIT
    d = np.linspace(0, 2 * np.pi * radius, n, endpoint=False)
    return pd.DataFrame({"distance_m": d, "pos_x": x, "pos_y": y}), 2 * np.pi * radius


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
    corners = [s for s in segs if s.kind.endswith("_corner")]
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
        assert len([s for s in segs if s.kind.endswith("_corner")]) == expected


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
