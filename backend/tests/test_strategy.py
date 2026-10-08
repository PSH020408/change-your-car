"""P11 strategy mode: sequence parsing, fitting to race distance, and a real-store score (skips without data)."""
from pathlib import Path

import pytest

from app.schemas import domain as S
from app.services.strategy import StrategyService

BASELINES = Path(__file__).resolve().parents[1] / ".." / "data" / "artifacts" / "baselines"
MODELS = Path(__file__).resolve().parents[1] / ".." / "data" / "artifacts" / "models"


def test_parse_and_fit():
    seq = "SOFT 13 → HARD 20 → HARD 24"
    st = StrategyService.parse_sequence(seq)
    assert st == [("SOFT", 13), ("HARD", 20), ("HARD", 24)]
    assert StrategyService.fit_to_race(st, 57) == [("SOFT", 13), ("HARD", 20), ("HARD", 24)]
    assert StrategyService.fit_to_race(st, 60)[-1] == ("HARD", 27)
    assert StrategyService.fit_to_race(st, 50)[-1] == ("HARD", 17)


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("scipy")
    if not (BASELINES / "2024" / "bahrain_grand_prix" / "race.json").exists():
        pytest.skip("no race.json in the store (run make race-timing)")
    from app.services.engine import Engine
    return Engine(BASELINES, MODELS)


def _req(stints):
    return S.StrategyRequest(baseline=S.BaselineRef(season=2024, event="bahrain_grand_prix", session="R",
                                                    driver="VER", lap="representative"),
                             stints=[S.Stint(compound=S.Compound(c), laps=n) for c, n in stints])


def test_real_strategy_scores_and_sums(engine):
    r = engine.strategy.score(_req([("SOFT", 15), ("HARD", 21), ("SOFT", 21)]))
    assert r.race_laps == 57 and r.pit_loss_s > 15 and r.pit_loss_n > 5
    y = r.yours
    assert not y.refused
    assert y.stops == 2 and len(y.laps) == 57
    assert abs(y.driving_s - sum(l.predicted_s for l in y.laps)) < 1e-6
    assert abs(y.race_s - (y.driving_s + y.pit_s)) < 1e-6
    assert 57 * 90 < y.race_s < 57 * 110                     # a plausible Bahrain race, not nonsense
    assert any(c.winner for c in r.cards)


def test_out_of_data_stint_is_refused_not_invented(engine):
    r = engine.strategy.score(_req([("MEDIUM", 40), ("HARD", 17)]))
    assert r.yours.refused and "MEDIUM" in r.yours.refused[0]
    assert r.delta_to_best_s is None


def test_more_fuel_early_laps_are_slower(engine):
    r = engine.strategy.score(_req([("SOFT", 15), ("HARD", 21), ("SOFT", 21)]))
    laps = r.yours.laps
    assert laps[1].fuel_delta_s > laps[-1].fuel_delta_s
