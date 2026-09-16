"""P6 gate — the engine answers the contract from a baseline store alone."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydantic")

from app.schemas import domain as S                       # noqa: E402
from app.services.engine import Engine, NotFound          # noqa: E402
from tests.test_reconstruct import _lap                    # noqa: E402


def _store(tmp_path):
    """A one-session baseline store built from the toy lap."""
    base, segs = _lap()
    a = np.gradient(base["speed_kph"].to_numpy() / 3.6, base["distance_m"].to_numpy()) * base["speed_kph"].to_numpy() / 3.6
    trace = {"distance_m": base["distance_m"].tolist(), "speed_kph": base["speed_kph"].tolist(),
             "throttle_pct": base["throttle_pct"].tolist(), "brake_on": [bool(x) for x in base["brake_on"]],
             "gear": [int(x) for x in base["gear"]], "drs_open": [bool(x == 12) for x in base["drs_raw"]]}
    lap = {"lap_uid": "2024_toy_Q_VER_003", "driver": "VER", "team": "Red Bull Racing", "chassis": "RB20",
           "power_unit": "Honda RBPT", "lap_number": 3, "lap_time_s": 60.0, "compound": "SOFT", "tyre_life": 2,
           "fresh_tyre": False, "track_temp_c": 35.0, "air_temp_c": 25.0, "telemetry_quality": "clean",
           "effort_class": "push", "gap_ahead_s": 8.0, "sector_times_s": [20.0, 20.0, 20.0], "condition": "dry",
           "trace": trace}
    segments = [{"index": s["index"], "kind": s["kind"], "start_m": s["start_m"], "end_m": s["end_m"],
                 "length_m": s["end_m"] - s["start_m"], "sector": 1 + s["index"] * 3 // 5,
                 "min_radius_m": 1 / s["peak_curvature_1pm"] if s["peak_curvature_1pm"] else None,
                 "direction": "left", "peak_curvature_1pm": s["peak_curvature_1pm"], "wraps_start_finish": False} for s in segs]
    doc = {"season": 2024, "event": "toy_grand_prix", "event_name": "Toy Grand Prix", "session": "Q", "circuit": "Toy",
           "track": {"view_box": "0 0 1000 1000", "path": "M 0 0 L 1 1", "sector_boundaries_m": [1000.0, 2000.0],
                     "lap_length_m": 3000.0, "published_turns": 2, "measured_turns": 2},
           "segments": segments, "session_temps": {"track_temp_c": 35.0, "air_temp_c": 25.0},
           "drivers": {"VER": {"team": "Red Bull Racing", "chassis": "RB20", "power_unit": "Honda RBPT",
                               "laps": {"representative": lap, "fastest": {"alias_of": "representative", "lap_uid": lap["lap_uid"]}},
                               "available": [{"lap_uid": lap["lap_uid"], "lap_number": 3, "lap_time_s": 60.0, "compound": "SOFT",
                                              "tyre_life": 2, "effort_class": "push", "telemetry_quality": "clean", "gap_ahead_s": 8.0}]}}}
    root = tmp_path / "baselines"
    (root / "2024" / "toy_grand_prix").mkdir(parents=True)
    (root / "2024" / "toy_grand_prix" / "Q.json").write_text(json.dumps(doc))
    (root / "index.json").write_text(json.dumps({"seasons": {"2024": {"toy_grand_prix": {
        "event_name": "Toy Grand Prix", "circuit": "Toy",
        "sessions": {"Q": {"drivers": ["VER"], "chassis": ["RB20"], "temps": doc["session_temps"]}}}}}}))
    return root


@pytest.fixture
def engine(tmp_path):
    return Engine(baselines_dir=_store(tmp_path), model_dir=tmp_path / "no-models")


def test_catalog_comes_from_the_store_index(engine):
    assert engine.seasons() == [2024]
    assert engine.events(2024)[0]["event"] == "toy_grand_prix"
    d = engine.drivers(2024, "toy_grand_prix", "Q")
    assert d[0]["driver"] == "VER" and d[0]["chassis"] == "RB20"
    with pytest.raises(NotFound):
        engine.drivers(2024, "toy_grand_prix", "R")


def test_baseline_response_is_the_real_lap_on_the_display_grid(engine):
    r = engine.baseline_response(S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER"))
    assert r.lap.lap_time_s == 60.0 and r.lap.chassis == "RB20"
    assert len(r.trace.distance_m) == len(r.trace.speed_kph) == len(r.trace.time_s)
    assert r.trace.distance_m[1] - r.trace.distance_m[0] == 20.0
    assert not any(r.trace.interpolated), "a 10 m raw grid never leaves a 20 m display sample uncovered"
    assert abs(r.trace.time_s[-1] - 60.0) < 1.0, "time axis scaled to the official lap time"
    assert len(r.segments) == 5 and r.track.lap_length_m == 3000.0


def test_baseline_setup_returns_the_real_lap_and_one_note(engine):
    req = S.SimulationRequest(baseline=S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER"))
    r = engine.simulate(req)
    assert abs(r.lap.delta_s) < 0.005
    assert r.lap.refused_s == 0.0
    assert np.allclose(r.simulated.speed_kph, r.baseline.speed_kph, atol=0.2)
    assert len(r.engineer_log) == 1 and r.engineer_log[0].channel == "model"
    assert r.model_version == "none"


def test_more_rear_wing_moves_time_from_corners_to_straights(engine):
    req = S.SimulationRequest(baseline=S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER"),
                              setup=S.CarSetup(rear_wing=0.9))
    r = engine.simulate(req)
    by = {s.kind.value: s for s in r.segments}
    assert by["straight"].physics_s > 0 and by["low_speed_corner"].physics_s < 0
    assert all(s.total_lo_s <= s.total_s <= s.total_hi_s for s in r.segments)
    assert r.lap.delta_lo_s <= r.lap.delta_s <= r.lap.delta_hi_s or r.lap.refused_s > 0
    assert abs(sum(s.achieved_s for s in r.segments) - r.lap.delta_s) < 0.03
    assert r.physics.downforce_pct > 0 and r.physics.drag_pct > 0
    assert any(n.channel == "aero" for n in r.engineer_log)
    assert len(r.grades) == 8 and {g.grade for g in r.grades} <= {"A", "B", "C"}
    assert len(r.simulated.distance_m) == len(r.baseline.distance_m)


def test_understeer_setup_raises_the_balance_warning(engine):
    req = S.SimulationRequest(baseline=S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER"),
                              setup=S.CarSetup(front_wing=0.0, rear_wing=1.0, suspension_split=1.0))
    r = engine.simulate(req)
    assert r.physics.warning == "understeer"
    assert any(n.channel == "balance" and n.severity == "warning" for n in r.engineer_log)


def test_wet_weather_without_a_model_is_physics_only_and_says_so(engine):
    req = S.SimulationRequest(baseline=S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER"),
                              environment=S.Environment(weather=S.Weather.WET, track_temp_c=20.0))
    r = engine.simulate(req)
    assert r.lap.delta_s > 1.0, "wet grip x0.65 costs seconds"
    assert r.physics.grip_multiplier == pytest.approx(0.65)
    channels = {n.channel for n in r.engineer_log}
    assert "weather" in channels and "model" in channels     # no registered model -> conditions ignored, said aloud


def test_unknown_driver_and_lap_are_404_material(engine):
    with pytest.raises(NotFound):
        engine.load_baseline(S.BaselineRef(season=2024, event="toy_grand_prix", driver="ZZZ"))
    with pytest.raises(NotFound):
        engine.load_baseline(S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER", lap="2024_toy_Q_VER_099"))
    b = engine.load_baseline(S.BaselineRef(season=2024, event="toy_grand_prix", driver="VER", lap="fastest"))
    assert b.lap["lap_uid"] == "2024_toy_Q_VER_003", "fastest aliases the representative lap here"


def test_http_contract(engine, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx", reason="fastapi.testclient needs httpx: .venv/bin/pip install httpx")
    from fastapi.testclient import TestClient
    from app import main
    from app.core import state
    monkeypatch.setattr(state, "get_engine", lambda: engine)
    main.app.dependency_overrides[state.get_engine] = lambda: engine
    c = TestClient(main.app)
    assert c.get("/api/meta/seasons").json() == [2024]
    r = c.get("/api/baseline", params={"season": 2024, "event": "toy_grand_prix", "driver": "VER"})
    assert r.status_code == 200 and r.json()["lap"]["driver"] == "VER"
    r = c.post("/api/simulate", json={"baseline": {"season": 2024, "event": "toy_grand_prix", "driver": "VER"},
                                      "setup": {"rear_wing": 0.8}})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"lap", "segments", "baseline", "simulated", "physics", "grades", "engineer_log", "model_version"}
    assert c.post("/api/simulate", json={"baseline": {"season": 2024, "event": "toy_grand_prix", "driver": "ZZZ"}}).status_code == 404
    assert c.post("/api/simulate", json={"baseline": {"season": 2024, "event": "toy_grand_prix", "driver": "VER"},
                                         "setup": {"rear_wing": 1.7}}).status_code == 422
