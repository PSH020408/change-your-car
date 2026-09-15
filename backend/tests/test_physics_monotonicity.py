"""P3 gate — the physics layer must be directionally right before the ML
layer is allowed to add to it.

These tests know nothing about coefficient sizes (those are graded and
banded in physics.yaml). They pin the SIGNS, the zero at baseline, the
one deliberate non-monotonicity (ride height), and that the segment layer
spends a wing change where the physics says it should.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.physics import modifiers as M, segment_delta as D


# ------------------------------------------------------------- baseline = 0
def test_baseline_sliders_change_nothing():
    st = M.physics_state(M.SetupInput(), session="Q")
    assert st.downforce_pct == 0 and st.drag_pct == 0 and st.mech_grip_pct == 0
    assert st.fuel_delta_kg == 0 and st.grip_multiplier == 1.0 and st.balance_index == 0
    assert st.warning is None


def test_every_coefficient_carries_a_grade():
    grades = M.default_config().grades()
    assert grades, "physics.yaml must declare graded coefficients"
    assert set(grades.values()) <= {"A", "B", "C"}
    assert grades["fuel.s_per_kg_per_lap"] == "B"
    assert grades["ground_effect.downforce_pct_full_range"] == "C"


# ------------------------------------------------------------------ wings
def test_more_rear_wing_increases_downforce_and_drag():
    low, high = M.wing_to_aero(0.5, 0.2), M.wing_to_aero(0.5, 0.8)
    assert high.downforce_pct > low.downforce_pct
    assert high.drag_pct > low.drag_pct


def test_front_wing_moves_balance_forward_and_rear_wing_backward():
    assert M.wing_to_aero(0.9, 0.5).balance_pts > 0
    assert M.wing_to_aero(0.5, 0.9).balance_pts < 0


def test_rear_wing_costs_more_drag_per_downforce_than_the_front():
    f, r = M.wing_to_aero(1.0, 0.5), M.wing_to_aero(0.5, 1.0)
    assert r.drag_pct / r.downforce_pct > f.drag_pct / f.downforce_pct


def test_wing_effect_is_monotonic_over_the_whole_slider():
    df = [M.wing_to_aero(0.5, s).downforce_pct for s in np.linspace(0, 1, 21)]
    assert all(b > a for a, b in zip(df, df[1:]))


# ------------------------------------------------------------ ride height
def test_lowering_the_car_adds_downforce_until_it_bottoms():
    """The 2022 story: lower is better until the floor stalls."""
    g = [M.ride_height_to_ground_effect(h).downforce_pct for h in (1.0, 0.75, 0.5, 0.3)]
    assert all(b > a for a, b in zip(g, g[1:])), "monotonic above the bottoming threshold"
    floor = M.ride_height_to_ground_effect(0.0).downforce_pct
    assert floor < M.ride_height_to_ground_effect(0.3).downforce_pct, "and it collapses at the bottom"


def test_floor_downforce_is_cheaper_in_drag_than_wing_downforce():
    fl = M.ride_height_to_ground_effect(0.25)
    rw = M.wing_to_aero(0.5, 0.75)
    assert fl.drag_pct / fl.downforce_pct < rw.drag_pct / rw.downforce_pct


# ------------------------------------------------------------- suspension
def test_soft_suspension_helps_on_rough_tracks_and_stiff_on_smooth():
    assert M.suspension_to_mechanical_grip(0.2, roughness=1.0).grip_pct > 0
    assert M.suspension_to_mechanical_grip(0.8, roughness=1.0).grip_pct < 0
    assert M.suspension_to_mechanical_grip(0.8, roughness=0.0).grip_pct > 0


def test_stiffer_front_means_understeer():
    assert M.suspension_to_mechanical_grip(0.5, front_rear_split=0.9).balance_pts < 0


def test_suspension_is_a_small_effect():
    """Grade C: nothing here may rival a wing click."""
    worst = max(abs(M.suspension_to_mechanical_grip(s, roughness=r).grip_pct)
                for s in (0.0, 1.0) for r in (0.0, 1.0))
    assert worst < abs(M.wing_to_aero(0.5, 1.0).downforce_pct) / 3


# ------------------------------------------------------------------- fuel
def test_more_fuel_costs_time_and_scales_with_the_configured_rate():
    cfg = M.default_config()
    assert M.fuel_lap_penalty_s(10.0, cfg) == pytest.approx(10.0 * cfg.coeff("fuel", "s_per_kg_per_lap").value)
    assert M.fuel_lap_penalty_s(-10.0, cfg) < 0


def test_baseline_fuel_guess_is_light_in_qualifying_and_burns_down_in_the_race():
    q = M.baseline_fuel_kg("Q")
    r1, r50 = M.baseline_fuel_kg("R", 1, 57), M.baseline_fuel_kg("R", 50, 57)
    assert q < 15
    assert r1 > r50 > q


# ------------------------------------------------------------------- tyre
def test_tyre_grip_peaks_inside_temperature_window():
    cold, ideal, hot = (M.tyre_thermal_grip(t, "soft", 1) for t in (15.0, 35.0, 58.0))
    assert ideal > cold and ideal > hot
    assert ideal == 0.0


def test_tyre_loses_grip_with_age_and_softs_age_faster():
    assert M.tyre_thermal_grip(35.0, "soft", 10) < M.tyre_thermal_grip(35.0, "soft", 1)
    assert M.tyre_thermal_grip(45.0, "hard", 20) > M.tyre_thermal_grip(35.0, "soft", 20)


def test_thermal_loss_saturates_outside_the_window():
    """A 40-degree miss is not worse than a 30-degree miss; the bell is clipped."""
    assert M.tyre_thermal_grip(-5.0, "medium") == M.tyre_thermal_grip(5.0, "medium")


# ---------------------------------------------------------------- weather
def test_wet_is_slower_than_inter_is_slower_than_dry():
    d, i, w = (M.weather_grip_multiplier(x) for x in ("dry", "inter", "wet"))
    assert d == 1.0 and d > i > w > 0.5


def test_weather_band_never_exceeds_dry():
    assert M.weather_grip_multiplier("inter", scale="high") <= 1.0
    with pytest.raises(ValueError):
        M.weather_grip_multiplier("snow")


# ---------------------------------------------------------------- balance
def test_balance_warnings_fire_in_the_right_direction():
    under = M.physics_state(M.SetupInput(front_wing=0.0, rear_wing=1.0, suspension_split=1.0))
    over = M.physics_state(M.SetupInput(front_wing=1.0, rear_wing=0.0, suspension_split=0.0))
    assert under.balance_index < 0 and under.warning == "understeer"
    assert over.balance_index > 0 and over.warning == "oversteer"
    assert -1.0 <= under.balance_index <= 1.0


# ---------------------------------------------------------- segment layer
def _lap():
    """A toy lap: hairpin, sweeper, kink, two straights (a long one)."""
    return pd.DataFrame({
        "segment_index": [0, 1, 2, 3, 4],
        "segment_kind": ["straight", "low_speed_corner", "straight", "high_speed_corner", "kink"],
        "segment_time_s": [14.0, 6.0, 5.0, 4.0, 2.0],
        "speed_min_kph": [180.0, 70.0, 120.0, 240.0, 270.0],
        "speed_mean_kph": [300.0, 95.0, 200.0, 250.0, 280.0],
        "speed_max_kph": [320.0, 130.0, 260.0, 265.0, 290.0],
    })


def test_aero_fraction_is_small_in_hairpins_and_large_in_sweeps():
    assert D.aero_fraction(70.0, 150.0) < 0.2
    assert D.aero_fraction(240.0, 150.0) > 0.7
    assert D.aero_fraction(0.0, 150.0) == 0.0


def test_more_wing_costs_time_on_the_straight_and_wins_it_in_fast_corners():
    out = D.lap_physics_delta(_lap(), M.SetupInput(rear_wing=0.9), session="Q")
    by = out.set_index("segment_index")["physics_delta_s"]
    assert by[0] > 0, "long straight: slower"
    assert by[3] < 0, "fast sweeper: faster"
    assert abs(by[1]) < abs(by[3]), "the hairpin barely notices the wing"


def test_less_wing_is_the_mirror_image():
    more = D.lap_physics_delta(_lap(), M.SetupInput(rear_wing=0.9), session="Q")["physics_delta_s"]
    less = D.lap_physics_delta(_lap(), M.SetupInput(rear_wing=0.1), session="Q")["physics_delta_s"]
    assert np.allclose(more.to_numpy(), -less.to_numpy(), atol=1e-9)


def test_fuel_penalty_sums_to_the_configured_lap_penalty():
    cfg = M.default_config()
    out = D.lap_physics_delta(_lap(), M.SetupInput(fuel_kg=58.0), session="Q", baseline_fuel_kg=8.0, cfg=cfg)
    expect = 50.0 * cfg.coeff("fuel", "s_per_kg_per_lap").value
    assert out["physics_mass_s"].sum() == pytest.approx(expect)
    assert (out["physics_mass_s"] > 0).all()


def test_wet_track_slows_every_corner_and_no_straight():
    out = D.lap_physics_delta(_lap(), M.SetupInput(weather="wet"), session="Q")
    corners = out[out["segment_kind"] != "straight"]
    straights = out[out["segment_kind"] == "straight"]
    assert (corners["physics_delta_s"] > 0).all()
    assert (straights["physics_delta_s"] == 0).all()


def test_interval_contains_the_nominal_and_widens_for_grade_c():
    lap = _lap()
    wing = D.lap_physics_delta(lap, M.SetupInput(rear_wing=1.0), session="Q")
    floor = D.lap_physics_delta(lap, M.SetupInput(ride_height=0.3), session="Q")
    for out in (wing, floor):
        assert (out["physics_delta_lo_s"] <= out["physics_delta_s"] + 1e-12).all()
        assert (out["physics_delta_hi_s"] >= out["physics_delta_s"] - 1e-12).all()
    rel = lambda o: (o["physics_delta_hi_s"] - o["physics_delta_lo_s"]).abs().sum() / o["physics_delta_s"].abs().sum()
    assert rel(floor) > rel(wing), "grade C must be declared less certain than grade B"


def test_baseline_setup_is_exactly_zero_everywhere():
    out = D.lap_physics_delta(_lap(), M.SetupInput(), session="Q")
    assert np.allclose(out[["physics_delta_s", "physics_delta_lo_s", "physics_delta_hi_s"]].to_numpy(), 0.0)
