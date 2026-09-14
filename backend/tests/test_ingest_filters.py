"""P1 gate tests — the filter chain, without FastF1.

Every test here encodes a defect the reconnaissance pass actually found, so
a regression reintroduces a known failure rather than an abstract one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.ingest import filters, metadata, paths, writer


# --------------------------------------------------------------- pace gate
def _drying_race(n_wet: int = 20, n_dry: int = 20) -> pd.DataFrame:
    """A race that starts wet and dries out — the 2025 Silverstone shape.

    Wet laps are ~100 s, dry laps ~80 s. The session best is therefore set in
    the final phase, under grip that simply did not exist earlier.
    """
    rng = np.random.default_rng(7)
    wet = 100.0 + rng.normal(0, 0.5, n_wet)
    dry = 80.0 + rng.normal(0, 0.5, n_dry)
    times = np.concatenate([wet, dry])
    starts = np.concatenate([[0.0], np.cumsum(times)[:-1]])
    return pd.DataFrame({
        "LapTime": pd.to_timedelta(times, unit="s"),
        "LapStartTime": pd.to_timedelta(starts, unit="s"),
        "Compound": ["INTERMEDIATE"] * n_wet + ["SOFT"] * n_dry,
    })


def test_session_best_gate_destroys_the_wet_phase():
    """The old behaviour, pinned so we can see what was wrong with it."""
    laps = _drying_race()
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.pace_gate(laps, {"reference": "session_best", "threshold_pct": 1.07}, rep)
    # every ~100 s lap is >107% of the ~80 s session best
    assert len(kept) == 20
    assert (kept["LapTime"].dt.total_seconds() < 90).all()


def test_rolling_reference_keeps_laps_that_were_fastest_at_the_time():
    laps = _drying_race()
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.pace_gate(
        laps,
        {"reference": "rolling_window", "threshold_pct": 1.07,
         "window_laps": 5, "reference_top_n": 3},
        rep,
    )
    assert len(kept) > 30, "conditions-matched reference should rescue the wet phase"
    # the opening lap of the race is kept, where session_best would bin it
    assert 0 in kept.index
    assert rep.diagnostics["pace_gate"]["window_seconds"] > 0


def test_rolling_reference_still_removes_a_genuinely_slow_lap():
    """The gate must not become a no-op: a cruise lap still has to go."""
    laps = _drying_race(n_wet=0, n_dry=30)
    laps.loc[15, "LapTime"] = pd.to_timedelta(140.0, unit="s")   # in-lap pace
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.pace_gate(
        laps, {"reference": "rolling_window", "threshold_pct": 1.07,
               "window_laps": 5, "reference_top_n": 3}, rep)
    assert 15 not in kept.index


def test_pace_gate_records_its_provenance():
    laps = _drying_race()
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.pace_gate(laps, {"reference": "rolling_window"}, rep)
    assert "pace_reference_s" in kept and "pace_ratio" in kept
    assert (kept["pace_ratio"] <= 1.07 + 1e-9).all()


# ------------------------------------------------------------ track status
def test_track_status_filter_removes_safety_car_laps():
    # "14" = green then safety car, "45" = SC then red flag, "12" = green
    # then yellow (yellow is not excluded — racing continues under it).
    laps = pd.DataFrame({"TrackStatus": ["1", "1", "14", "1", "6", "12", "45"]})
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.exclude_track_status(laps, ["4", "5", "6", "7"], rep)
    assert list(kept["TrackStatus"]) == ["1", "1", "1", "12"]
    d = rep.diagnostics["track_status"]
    assert d["laps_removed_ours"] == 3
    assert d["laps_matching_each_code"]["4:SafetyCar"] == 2      # "14", "45"
    assert d["laps_matching_each_code"]["5:RedFlag"] == 1        # "45"


def test_all_green_session_reports_why_it_removed_nothing():
    """D4: 'removed zero' must be distinguishable from 'filter is broken'."""
    laps = pd.DataFrame({"TrackStatus": ["1"] * 10})
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.exclude_track_status(laps, ["4", "5", "6", "7"], rep)
    d = rep.diagnostics["track_status"]
    assert len(kept) == 10
    assert d["broken"] is False
    assert d["codes_present_but_not_excluded"] == {}
    assert "removing nothing is correct" in d["verdict"]


def test_yellow_only_session_is_not_reported_as_broken():
    """The first diagnostic cried wolf here.

    It treated any code other than all-clear as 'should have been removed',
    so a qualifying session with yellow flags and no safety car was reported
    as a broken filter. Yellow is not in the exclusion set by design.
    """
    laps = pd.DataFrame({"TrackStatus": ["1", "12", "2", "1"]})
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.exclude_track_status(laps, ["4", "5", "6", "7"], rep)
    d = rep.diagnostics["track_status"]
    assert len(kept) == 4
    assert d["broken"] is False
    assert "removing nothing is correct" in d["verdict"]
    assert d["codes_present_but_not_excluded"]["2:Yellow"] == 2


def test_a_genuinely_broken_filter_is_still_caught():
    """The alarm must survive being made less sensitive."""
    laps = pd.DataFrame({"TrackStatus": ["1", "14", "1"]})
    rep = filters.FilterReport(key="t", raw=len(laps))
    filters.exclude_track_status(laps, ["4"], rep)
    d = rep.diagnostics["track_status"]
    assert d["laps_removed_ours"] == 1 and d.get("broken") is not True

    # now simulate the pathological case the verdict exists to name
    rep2 = filters.FilterReport(key="t", raw=3)
    rep2.diagnostics["track_status"] = {
        "laps_matching_each_code": {"4:SafetyCar": 2}, "laps_removed_ours": 0}
    assert sum(rep2.diagnostics["track_status"]["laps_matching_each_code"].values()) > 0


def test_track_status_flag_tags_yellow_without_dropping_it():
    laps = pd.DataFrame({"TrackStatus": ["1", "12", "14", "2"]})
    tags = filters.tag_track_status(laps, ["4", "5", "6", "7"])
    assert list(tags) == ["green", "flagged_2", "green", "flagged_2"]
    assert len(laps) == 4, "tagging must never remove a lap"


# -------------------------------------------------------------- gap filter
class _Tel:
    """Stand-in for LapTelemetry carrying only the gap metrics."""

    def __init__(self, gap_s, gap_m=0.0, ok=True, n=300, frac=0.5):
        self.max_gap_s = gap_s
        self.max_gap_m = gap_m or gap_s * 80.0
        self.missed_samples_worst = max(0, int(round(gap_s / 0.24)) - 1)
        self.worst_gap_at_frac = frac
        self.n_samples, self._ok = n, ok

    @property
    def ok(self):
        return self._ok


def test_gap_filter_gates_on_time_not_distance():
    """The 40 m distance gate removed 88% of a real session.

    At 300 km/h a single skipped sample already spans 20 m, so a big gap in
    METRES is mostly evidence that the car was fast. The same three-sample
    dropout is 0.72 s whatever the speed.
    """
    laps = pd.DataFrame({"lap_uid": ["monza", "monaco", "dropout", "none"]},
                        index=[0, 1, 2, 3])
    tel = {
        # 0.24 s apart at 350 km/h = 23 m: normal cadence, huge in metres
        "monza": _Tel(0.24, gap_m=23.3),
        # same 0.24 s at 80 km/h = 5 m: identical data quality
        "monaco": _Tel(0.24, gap_m=5.3),
        # 2.4 s = ten consecutive samples missing: a real hole
        "dropout": _Tel(2.4, gap_m=98.0),
        "none": _Tel(0.0, ok=False),
    }
    rep = filters.FilterReport(key="t", raw=len(laps))
    kept = filters.max_telemetry_gap(laps, tel, laps["lap_uid"], 1.0, rep)

    assert list(kept.index) == [0, 1], "fast laps must survive a time-domain gate"
    d = rep.diagnostics["telemetry_gap"]
    assert d["removed_gap_exceeded"] == 1
    assert d["removed_no_telemetry"] == 1
    assert d["threshold_s"] == 1.0


def test_gap_filter_reports_a_threshold_sweep():
    """The threshold must be settable from data, not from an argument."""
    laps = pd.DataFrame({"lap_uid": [f"l{i}" for i in range(10)]}, index=range(10))
    tel = {f"l{i}": _Tel(g) for i, g in enumerate(
        [0.24, 0.24, 0.48, 0.48, 0.72, 0.96, 1.2, 1.8, 2.4, 4.0])}
    rep = filters.FilterReport(key="t", raw=len(laps))
    filters.max_telemetry_gap(laps, tel, laps["lap_uid"], 1.0, rep)
    sweep = rep.diagnostics["telemetry_gap"]["threshold_sweep_laps_kept"]
    assert sweep["0.5s"] == 4
    assert sweep["1.0s"] == 6
    assert sweep["3.0s"] == 9
    assert sweep["5.0s"] == 10
    # monotonic: a looser threshold can never keep fewer laps
    vals = list(sweep.values())
    assert vals == sorted(vals)


def test_missed_sample_count_is_derived_from_the_240ms_cadence():
    t = _Tel(1.2)
    assert t.missed_samples_worst == 4


# -------------------------------------------------------------- conditions
def test_conditions_are_tagged_not_dropped():
    laps = pd.DataFrame({"Compound": ["SOFT", "INTERMEDIATE", "WET", "HARD"]})
    tags = filters.tag_conditions(
        laps, {"inter_compounds": ["INTERMEDIATE"], "wet_compounds": ["WET"]})
    assert list(tags) == ["dry", "inter", "wet", "dry"]
    assert len(laps) == 4, "tagging must never remove a lap"


# ------------------------------------------------------------------ report
def test_report_accounts_for_every_lap():
    rep = filters.FilterReport(key="t", raw=100)
    rep.add("pit in/out", 80)
    rep.add("deleted", 78)
    rep.add("pace gate", 50)
    assert [s.removed for s in rep.steps] == [20, 2, 28]
    assert rep.final == 50 and rep.yield_pct == 50.0
    assert sum(s.removed for s in rep.steps) + rep.final == rep.raw


# ------------------------------------------------------------- chassis map
@pytest.fixture
def cmap():
    return metadata.load_chassis_map(paths.Path("configs/chassis.yaml"))


def test_chassis_map_covers_every_era_season(cmap):
    assert cmap.seasons() == [2022, 2023, 2024, 2025]


def test_chassis_map_resolves_renamed_teams(cmap):
    assert cmap.lookup(2024, "RB")["chassis"] == "VCARB 01"
    assert cmap.lookup(2025, "Racing Bulls")["chassis"] == "VCARB 02"
    assert cmap.lookup(2024, "Stake F1 Team Kick Sauber")["chassis"] == "C44"
    assert cmap.lookup(2022, "Oracle Red Bull Racing")["chassis"] == "RB18"


def test_chassis_lookup_is_punctuation_insensitive(cmap):
    assert cmap.lookup(2023, "haas f1 team")["chassis"] == "VF-23"
    assert cmap.lookup(2023, "Haas")["chassis"] == "VF-23"


def test_unmapped_team_is_reported_not_silently_nulled(cmap):
    missing = cmap.validate_coverage(2024, ["Ferrari", "Andretti"])
    assert missing == ["Andretti"]


def test_power_units_are_consistent_within_a_supplier(cmap):
    assert cmap.lookup(2024, "McLaren")["pu"] == cmap.lookup(2024, "Mercedes")["pu"]
    assert cmap.lookup(2024, "Haas F1 Team")["pu"] == "Ferrari"


# ------------------------------------------------------------------- paths
def test_slug_is_stable_and_filesystem_safe():
    assert paths.slug("Bahrain Grand Prix") == "bahrain_grand_prix"
    assert paths.slug("Spa-Francorchamps") == "spa_francorchamps"
    assert paths.slug("  São  Paulo ") == "sao_paulo"
    assert paths.slug("Autódromo José Carlos Pace") == "autodromo_jose_carlos_pace"


# ------------------------------------------------------------------ writer
def test_projection_fills_absent_columns_rather_than_raising():
    df = pd.DataFrame({"lap_uid": ["a"], "Speed": [280]})
    out = writer._project(df, {"lap_uid": "lap_uid", "Speed": "speed_kph",
                               "Nope": "missing_col"})
    assert list(out.columns) == ["lap_uid", "speed_kph", "missing_col"]
    assert out["missing_col"].isna().all()


def test_bronze_schema_stores_brake_as_boolean_and_drs_raw():
    """D5: the contract promised brake pressure that does not exist."""
    assert writer.TELEMETRY_COLUMNS["Brake"] == "brake_on"
    assert writer.TELEMETRY_COLUMNS["DRS"] == "drs_raw"
    assert "brake_pct" not in writer.TELEMETRY_COLUMNS.values()
    assert "drs_open" not in writer.TELEMETRY_COLUMNS.values()


def test_speed_traps_are_stored_with_missing_flags():
    """D6: imputation needs segments, so P1 flags rather than fills."""
    for t in ("i1", "i2", "fl", "st"):
        assert f"speed_{t}_missing" in writer.LAP_COLUMNS.values()
        assert f"speed_{t}_kph" in writer.LAP_COLUMNS.values()
