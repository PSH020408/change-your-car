"""P9 engine integration: differential QSS on a real stored baseline (skips without data/SciPy)."""
from pathlib import Path

import numpy as np
import pytest

from app.schemas import domain as S

BASELINES = Path(__file__).resolve().parents[1] / ".." / "data" / "artifacts" / "baselines"
MODELS = Path(__file__).resolve().parents[1] / ".." / "data" / "artifacts" / "models"


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("scipy")
    if not (BASELINES / "2024" / "bahrain_grand_prix" / "line.json").exists():
        pytest.skip("no line.json in the baseline store (run make line)")
    from app.services.engine import Engine
    return Engine(BASELINES, MODELS)


def _ref(session="Q", driver="VER"):
    return S.BaselineRef(season=2024, event="bahrain_grand_prix", session=session, driver=driver, lap="representative")


def test_zero_setup_is_zero_delta(engine):
    r = engine.simulate(S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(), environment=S.Environment()))
    assert r.engine_mode == "qss"
    assert abs(r.lap.delta_s) < 1e-3
    assert all(abs(s.physics_s) < 1e-4 for s in r.segments)
    assert r.qss_fit is not None and r.qss_fit.speed_rms_kph < 25.0
    assert r.computed_ms < 600.0                     # first call includes the 3-parameter fit


def test_split_sums_to_headline_without_rebuild_term(engine):
    req = S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(rear_wing=0.8, front_wing=0.6, fuel_kg=30.0),
                              environment=S.Environment())
    r = engine.simulate(req)
    parts = sum(s.physics_s + s.ml_s - s.refused_s for s in r.segments)
    assert abs(r.lap.delta_s - parts) < 0.02          # sample-boundary rounding only
    assert r.lap.refused_s == 0.0
    assert r.computed_ms < 150.0                      # cached fit: solves only


def test_more_fuel_is_slower_and_more_wing_cuts_top_speed(engine):
    base = engine.simulate(S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(), environment=S.Environment()))
    fuel = engine.simulate(S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(fuel_kg=30.0), environment=S.Environment()))
    wing = engine.simulate(S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(rear_wing=1.0), environment=S.Environment()))
    assert fuel.lap.delta_s > 0.3
    assert max(wing.simulated.speed_kph) < max(base.simulated.speed_kph)


def test_conditions_enter_as_grip_multiplier(engine):
    r = engine.simulate(S.SimulationRequest(baseline=_ref(), setup=S.CarSetup(),
                                            environment=S.Environment(compound=S.Compound.HARD, tyre_life=15)))
    if abs(r.lap.ml_s) > 1e-3:
        assert r.qss_fit.grip_multiplier_ml != 1.0
        assert abs(sum(s.ml_s for s in r.segments) - r.lap.ml_s) < 2e-3   # per-segment rounding
