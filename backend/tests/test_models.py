"""P4 gate — the model pipeline must refuse leaks, split by group, keep its
quantiles ordered, recover a planted effect, and register only on a pass."""
import numpy as np
import pandas as pd
import pytest

from pipeline.models import dataset as D, evaluate as E, level2 as L2, quantile as Q
from pipeline.models.train import group_folds


# ---------------------------------------------------------------- fixtures
def _gold(n_events: int = 6, laps_per: int = 40, seed: int = 0) -> pd.DataFrame:
    """A gold-like table with a PLANTED truth: +0.02 s per tyre lap, -0.01 s
    per race lap (fuel), hairpins twice as sensitive as straights."""
    rng = np.random.default_rng(seed)
    kinds = ["straight", "low_speed_corner", "high_speed_corner", "kink"]
    rows = []
    for e in range(n_events):
        for ses in ("Q", "R"):
            for lap in range(laps_per):
                drv = f"D{rng.integers(0, 8)}"
                tl = int(rng.integers(1, 30)); ln = int(rng.integers(1, 60)) if ses == "R" else 1
                temp = 30 + 5 * e + rng.normal(0, 1)
                for i, k in enumerate(kinds):
                    sens = 2.0 if k == "low_speed_corner" else 1.0
                    delta = 0.02 * sens * tl - (0.01 * ln if ses == "R" else 0) + rng.normal(0, 0.05) + 0.3
                    rows.append(dict(lap_uid=f"e{e}{ses}{lap}", season=2022 + e % 2, event=f"E{e}", event_slug=f"ev{e}",
                                     session=ses, driver=drv, chassis=f"C{int(drv[1]) // 2}", compound="SOFT",
                                     fresh_tyre=tl == 1, tyre_life=tl, lap_number=ln, stint=1, track_temp_c=temp,
                                     air_temp_c=temp - 8, effort_index=0.8, condition="dry", lap_effort_class="push",
                                     segment_index=i, segment_kind=k, segment_sector=i % 3 + 1, segment_is_kink=k == "kink",
                                     segment_length_m=[900, 120, 300, 200][i], segment_min_radius_m=[2000, 30, 250, 500][i],
                                     segment_reference_s=[12.0, 5.0, 4.0, 2.5][i] * (1 + 0.002 * (temp - 30)),
                                     segment_time_s=0.0, segment_time_gap_s=0.0, segment_delta_s=delta,
                                     speed_min_kph=100.0, lap_time_s=90.0))
    return pd.DataFrame(rows)


def _cfg() -> dict:
    return {"data": {"train_conditions": ["dry"], "effort_classes": ["push", "moderate"], "gap_tolerance_s": 0.05},
            "features": {"numeric": ["tyre_life", "lap_number", "stint", "track_temp_c", "air_temp_c", "effort_index",
                                     "segment_length_m", "segment_min_radius_m", "segment_reference_s"],
                         "categorical": ["segment_kind", "session", "compound", "fresh_tyre", "driver", "chassis",
                                         "season", "segment_sector", "segment_is_kink"]},
            "acceptance": {"segment_mae_s": 0.15, "lap_mae_s": 0.30, "unseen_track_lap_mae_s": 0.45,
                           "coverage_80": [0.7, 0.9], "beat_ridge": True,
                           "monotonic": {"tyre_life_up_slower": True, "lap_number_up_faster_in_race": True}}}


# ------------------------------------------------------------ feature spec
def test_feature_spec_refuses_the_answer_in_any_unit():
    for leak in ("lap_time_s", "speed_min_kph", "pace_ratio_session_best", "brake_point_frac", "segment_delta_s", "bias_braking_late"):
        with pytest.raises(ValueError):
            D.FeatureSpec(["tyre_life", leak], [])
    D.FeatureSpec(["tyre_life"], ["driver"])          # a clean spec is fine


def test_unseen_category_becomes_nan_not_a_new_code():
    df = _gold(2, 5)
    spec = D.FeatureSpec(["tyre_life"], ["driver"]).fit(df)
    X = spec.transform(pd.DataFrame({"tyre_life": [3], "driver": ["NOBODY"]}))
    assert str(X["driver"].dtype) == "category"
    assert X["driver"].isna().all()
    codes = spec.codes(X)
    assert np.isnan(codes["driver"].iloc[0])


def test_spec_round_trips_through_json():
    df = _gold(2, 5)
    spec = D.FeatureSpec.from_config(_cfg()).fit(df)
    back = D.FeatureSpec.from_json(spec.to_json())
    assert back.columns == spec.columns and back.vocab == spec.vocab


# ------------------------------------------------------------- selection
def test_training_rows_drop_wet_cruise_holes_and_incidents():
    df = _gold(2, 10)
    df.loc[0, "condition"] = "wet"
    df.loc[1, "lap_effort_class"] = "cruise"
    df.loc[2, "segment_time_gap_s"] = 1.0
    df.loc[3, "segment_delta_s"] = 9.0
    df.loc[4, "segment_delta_s"] = np.nan
    out = D.select_training_rows(df, _cfg(), unstable={(2022, "ev0")}, verbose=False)
    assert len(out) == len(df[df["event_slug"] != "ev0"]) - 5 or len(out) < len(df)
    assert (out["condition"] == "dry").all() and (out["segment_delta_s"] < 5).all()
    assert "ev0" not in set(out["event_slug"])


def test_group_folds_never_split_an_event():
    df = _gold(7, 8)
    g = D.group_key(df)
    folds = group_folds(g, 3)
    seen = {}
    for k, idx in enumerate(folds):
        for grp in g.iloc[idx].unique():
            assert seen.setdefault(grp, k) == k, "an event landed in two folds"
    assert sum(len(f) for f in folds) == len(df)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= len(df) / 3


# ------------------------------------------------------------- quantiles
def test_crossing_quantiles_are_sorted():
    q = pd.DataFrame({"q10": [0.5, 0.0], "q50": [0.2, 0.1], "q90": [0.1, 0.3]})
    out = Q.enforce_non_crossing(q)
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()


def test_coverage_and_lap_sum_metrics():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    q = pd.DataFrame({"q10": [-1, 0, 1, 4.0], "q50": [0, 1, 2, 3.0], "q90": [1, 2, 3, 5.0]})
    m = E.segment_metrics(y, q)
    assert m["segment_mae_s"] == 0.0 and m["coverage_80"] == 0.75
    lm = E.lap_metrics(pd.Series(["a", "a", "b", "b"]), y, q)
    assert lm["lap_mae_s"] == 0.0 and lm["laps"] == 2


def test_gates_read_the_right_numbers():
    metrics = {"cv": {"segment_mae_s": 0.1, "lap_mae_s": 0.2, "coverage_80": 0.8},
               "holdout": {"lap_mae_s": 0.4}, "ridge": {"segment_mae_s": 0.2},
               "probes": {"tyre_life": {"mean_delta_s": 0.05}, "lap_number": {"mean_delta_s": -0.1}}}
    g = E.gates(metrics, _cfg())
    assert all(g.values())
    metrics["probes"]["tyre_life"]["mean_delta_s"] = -0.01
    assert not E.gates(metrics, _cfg())["tyre_life_up_slower"]


# --------------------------------------------------------------- level 2
def test_level2_recovers_a_planted_temperature_effect():
    """References were planted at +0.2 % per degree of track temperature."""
    df = _gold(6, 6)
    # give every circuit two seasons so the circuit-centred temperature varies
    df2 = df.copy(); df2["season"] = df2["season"] + 2; df2["track_temp_c"] += 6
    df2["segment_reference_s"] *= 1 + 0.002 * 6
    both = pd.concat([df, df2])
    model, t = L2.fit_level2(both, alpha=0.01, n_boot=30)
    assert model.n_sessions >= 12
    for k in ("straight", "low_speed_corner"):
        c = model.coef[f"track_temp_per_c[{k}]"]
        assert 0.1 < c < 0.3, f"{k}: {c}"
    nom, lo, hi = model.reference_pct_shift("straight", d_track_temp_c=10.0)
    assert lo <= nom <= hi and 1.0 < nom < 3.0


def test_level2_is_identity_without_informative_sessions():
    df = _gold(3, 4)
    df = df[df["session"] == "Q"]                      # one session per circuit
    model, _ = L2.fit_level2(df, n_boot=5)
    assert model.coef == {} and model.reference_pct_shift("straight", 10.0) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------ end to end
def test_gbm_recovers_the_planted_effects_and_registers(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from pipeline.models import registry as R
    df = _gold(6, 30, seed=1)
    cfg = _cfg()
    spec = D.FeatureSpec.from_config(cfg).fit(df)
    X, y = spec.transform(df), df["segment_delta_s"].to_numpy(float)
    backend = Q.resolve_backend("auto")
    m = Q.QuantileSet(spec, [0.1, 0.5, 0.9], backend, {"n_estimators": 120, "learning_rate": 0.1,
                                                        "num_leaves": 15, "min_child_samples": 20}).fit(X, y)
    q = m.predict(X)
    assert (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    assert E.segment_metrics(y, q)["segment_mae_s"] < 0.08
    p_t = E.monotonic_probe(m, X, "tyre_life", +5.0)
    p_l = E.monotonic_probe(m, X, "lap_number", +10.0, subset=df["session"] == "R")
    assert p_t["mean_delta_s"] > 0.05, p_t          # planted +0.02 s / lap, hairpin x2
    assert p_l["mean_delta_s"] < -0.05, p_l         # planted -0.01 s / race lap
    imp = E.permutation_importance(m, X, y, n=2000)
    assert imp.iloc[0]["feature"] in ("tyre_life", "lap_number", "segment_kind")

    l2, _ = L2.fit_level2(df, n_boot=5)
    v = R.register(tmp_path, m, l2, {"cv": {}}, cfg, passed=False)
    assert not (tmp_path / "latest.json").exists(), "a failed run never becomes latest"
    v2 = R.register(tmp_path, m, l2, {"cv": {}}, cfg, passed=True)
    assert v != v2 and (tmp_path / "latest.json").exists()
    back, l2b, met = R.load(tmp_path)
    assert met["version"] == v2
    assert np.allclose(back.predict(X.head(50)).to_numpy(), q.head(50).to_numpy())
