"""P4 — what a model has to prove before it is registered.

Error is measured out-of-group (GroupKFold on season|event, plus one event
never seen at all), because the question the HUD asks is "what happens on a
lap nobody has driven", not "what happened on a lap in the training set".

Two non-error checks matter as much as the MAE:
  coverage    80% of true deltas must fall inside [q10, q90] — an interval
              that is too narrow is a lie, one that is too wide is useless
  monotonic   the model must slow down with tyre age and speed up as race
              fuel burns off; a model that gets the MAE by memorising
              drivers and violates that is rejected
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def segment_metrics(y: np.ndarray, q: pd.DataFrame) -> dict:
    y = np.asarray(y, dtype=float)
    lo, mid, hi = q["q10"].to_numpy(), q["q50"].to_numpy(), q["q90"].to_numpy()
    return {
        "segment_mae_s": float(np.mean(np.abs(y - mid))),
        "segment_median_ae_s": float(np.median(np.abs(y - mid))),
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
        "interval_width_median_s": float(np.median(hi - lo)),
        "pinball_q10": pinball(y, lo, 0.1), "pinball_q50": pinball(y, mid, 0.5), "pinball_q90": pinball(y, hi, 0.9),
        "n": int(len(y)),
    }


def lap_metrics(lap_uid: pd.Series, y: np.ndarray, q: pd.DataFrame) -> dict:
    """A lap's delta is the sum of its segments'; the HUD shows that number."""
    d = pd.DataFrame({"lap": lap_uid.to_numpy(), "y": np.asarray(y, float),
                      "p": q["q50"].to_numpy(), "lo": q["q10"].to_numpy(), "hi": q["q90"].to_numpy()})
    g = d.groupby("lap").sum(numeric_only=True)
    return {
        "lap_mae_s": float(np.mean(np.abs(g["y"] - g["p"]))),
        "lap_median_ae_s": float(np.median(np.abs(g["y"] - g["p"]))),
        "lap_coverage_80_naive_sum": float(np.mean((g["y"] >= g["lo"]) & (g["y"] <= g["hi"]))),
        "laps": int(len(g)),
    }


def clean_push_mask(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """The HUD's population: push laps with clean air at the line."""
    d = cfg.get("data", {})
    ok = np.ones(len(df), dtype=bool)
    if "lap_effort_class" in df:
        ok &= df["lap_effort_class"].isin(d.get("gate_effort", ["push"])).to_numpy()
    if "gap_ahead_s" in df and d.get("clean_air_gap_s") is not None:
        gap = pd.to_numeric(df["gap_ahead_s"], errors="coerce")
        ok &= (gap.isna() | (gap >= float(d["clean_air_gap_s"]))).to_numpy()
    return ok


def counterfactual_lap_metrics(df: pd.DataFrame, y: np.ndarray, q: pd.DataFrame) -> dict:
    """Error of the quantity the HUD actually shows.

    The simulator takes a REAL lap as baseline and shows how it would change
    under other conditions: shown = actual_baseline + (p(new) - p(baseline)).
    Whatever a lap carries that no pre-lap feature can know - traffic, a
    management phase, damage - is in both p terms and cancels. So the error
    that matters is the error of the DIFFERENCE, measured here between every
    lap and the same driver's fastest lap of that session, per segment, then
    summed to the lap.
    """
    d = pd.DataFrame({
        "key": df["season"].astype(str) + "|" + df["event_slug"].astype(str) + "|" + df["session"].astype(str)
               + "|" + df["driver"].astype(str),
        "lap": df["lap_uid"].to_numpy(), "seg": df["segment_index"].to_numpy(),
        "lt": pd.to_numeric(df["lap_time_s"], errors="coerce").to_numpy(),
        "y": np.asarray(y, float), "p": q["q50"].to_numpy()})
    best = d.groupby("key")["lt"].transform("min")
    base = d[d["lt"] == best].drop_duplicates(["key", "seg"])[["key", "seg", "lap", "y", "p"]] \
            .rename(columns={"lap": "base_lap", "y": "y_b", "p": "p_b"})
    m = d.merge(base, on=["key", "seg"], how="inner")
    m = m[m["lap"] != m["base_lap"]]
    if not len(m):
        return {"cf_segment_mae_s": float("nan"), "cf_lap_mae_s": float("nan"), "cf_lap_median_ae_s": float("nan"), "cf_laps": 0}
    m["dy"], m["dp"] = m["y"] - m["y_b"], m["p"] - m["p_b"]
    g = m.groupby("lap")[["dy", "dp"]].sum()
    return {"cf_segment_mae_s": float(np.mean(np.abs(m["dy"] - m["dp"]))),
            "cf_lap_mae_s": float(np.mean(np.abs(g["dy"] - g["dp"]))),
            "cf_lap_median_ae_s": float(np.median(np.abs(g["dy"] - g["dp"]))),
            "cf_lap_mae_naive_s": float(np.mean(np.abs(g["dy"]))),   # predict "no change"
            "cf_laps": int(len(g))}


def lap_noise_floor(df: pd.DataFrame, y: np.ndarray) -> dict:
    """How different are two CONSECUTIVE laps of the same driver on the same
    tyres, with nothing changed but one lap of fuel and one lap of wear?

    Whatever separates them is driver execution and track noise that no
    pre-lap feature can know. Half of that difference's MAE (two noisy laps
    were subtracted) is the floor a counterfactual lap error cannot beat.
    The gate is judged against it: a model within 1.25x of the floor is
    doing what the data allows; one far above it is leaving signal unused.
    """
    d = pd.DataFrame({
        "key": df["season"].astype(str) + "|" + df["event_slug"].astype(str) + "|" + df["session"].astype(str)
               + "|" + df["driver"].astype(str) + "|" + df["stint"].astype(str),
        "lap_n": pd.to_numeric(df["lap_number"], errors="coerce").to_numpy(),
        "lap": df["lap_uid"].to_numpy(), "seg": df["segment_index"].to_numpy(), "y": np.asarray(y, float)})
    per_lap = d.groupby(["key", "lap_n", "lap"])["y"].sum().reset_index().sort_values(["key", "lap_n"])
    prev = per_lap.groupby("key").shift(1)
    consecutive = per_lap[(per_lap["lap_n"] - prev["lap_n"]) == 1]
    if len(consecutive) < 5:
        return {"noise_floor_lap_s": float("nan"), "consecutive_pair_mae_s": float("nan"), "pairs": int(len(consecutive))}
    diff = (consecutive["y"] - prev.loc[consecutive.index, "y"]).abs()
    return {"noise_floor_lap_s": float(diff.mean() / np.sqrt(2)), "consecutive_pair_mae_s": float(diff.mean()),
            "pairs": int(len(consecutive))}


def skill(cf: dict) -> float:
    """1 - error / 'no change' error: the share of lap-to-lap change the model explains."""
    if not cf or not np.isfinite(cf.get("cf_lap_mae_naive_s", np.nan)) or cf["cf_lap_mae_naive_s"] <= 0:
        return float("nan")
    return 1.0 - cf["cf_lap_mae_s"] / cf["cf_lap_mae_naive_s"]


def by_kind(kind: pd.Series, y: np.ndarray, q: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"kind": kind.to_numpy(), "ae": np.abs(np.asarray(y, float) - q["q50"].to_numpy()),
                      "inside": (np.asarray(y) >= q["q10"].to_numpy()) & (np.asarray(y) <= q["q90"].to_numpy())})
    return d.groupby("kind").agg(mae_s=("ae", "mean"), coverage_80=("inside", "mean"), n=("ae", "size"))


def monotonic_probe(model, X: pd.DataFrame, col: str, shift: float, subset: pd.Series | None = None,
                    n: int = 20000, seed: int = 0) -> dict:
    """Mean change of q50 when `col` is shifted by `shift` on the same rows.

    Not a partial-dependence plot — one number, one direction, so the gate
    can be a boolean. Positive = the shift makes the lap SLOWER.
    """
    rows = X if subset is None else X[subset.to_numpy()]
    if len(rows) > n:
        rows = rows.sample(n, random_state=seed)
    base = model.predict(rows)["q50"].to_numpy()
    shifted = rows.copy()
    shifted[col] = shifted[col].astype(float) + shift
    alt = model.predict(shifted)["q50"].to_numpy()
    return {"col": col, "shift": shift, "mean_delta_s": float(np.mean(alt - base)),
            "share_slower": float(np.mean(alt > base)), "n": int(len(rows))}


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """MAE increase when one column is shuffled — model-agnostic, no extra deps."""
    rng = np.random.default_rng(seed)
    if len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    base = float(np.mean(np.abs(y - model.predict(X)["q50"].to_numpy())))
    rows = []
    for c in X.columns:
        Xp = X.copy()
        # keep the column's dtype (a shuffled category column must stay a
        # category column, or LightGBM sees a different feature set)
        perm = rng.permutation(len(Xp))
        Xp[c] = Xp[c].iloc[perm].set_axis(Xp.index)
        mae = float(np.mean(np.abs(y - model.predict(Xp)["q50"].to_numpy())))
        rows.append({"feature": c, "mae_increase_s": mae - base})
    return pd.DataFrame(rows).sort_values("mae_increase_s", ascending=False).reset_index(drop=True)


def gates(metrics: dict, cfg: dict) -> dict[str, bool]:
    a = cfg["acceptance"]
    cv, ho = metrics["cv"], metrics.get("holdout", {})
    lo, hi = a.get("coverage_80", [0.7, 0.9])
    cov = ho.get("coverage_80_calibrated", cv.get("coverage_80_calibrated", cv["coverage_80"])) if ho \
        else cv.get("coverage_80_calibrated", cv["coverage_80"])
    def cf(m: dict) -> dict:
        return m.get("clean_push") or {"cf_lap_mae_s": m.get("cf_lap_mae_s", np.inf),
                                       "cf_lap_mae_naive_s": m.get("cf_lap_mae_naive_s", np.nan)}
    floor = cv.get("noise_floor_lap_s", np.nan)
    ratio = float(a.get("lap_mae_vs_noise_floor", 1.25))
    out = {
        "segment_mae": cv["segment_mae_s"] <= float(a["segment_mae_s"]),
        # the lap gate: within `ratio` of what consecutive laps of the same
        # driver on the same tyres differ by anyway, AND explaining at least
        # `lap_skill_min` of the lap-to-lap change
        "clean_push_lap_within_noise_floor": (cf(cv)["cf_lap_mae_s"] <= ratio * floor) if np.isfinite(floor) else
                                             (cf(cv)["cf_lap_mae_s"] <= float(a.get("lap_mae_s", 0.30))),
        "clean_push_lap_skill": skill(cf(cv)) >= float(a.get("lap_skill_min", 0.35)),
        "unseen_track_clean_push_lap_skill": (skill(cf(ho)) >= float(a.get("unseen_track_lap_skill_min", 0.25))) if ho else True,
        "coverage_80_calibrated": lo <= cov <= hi,
        "beats_ridge": (cv["segment_mae_s"] < metrics.get("ridge", {}).get("segment_mae_s", np.inf))
                       if a.get("beat_ridge", True) else True,
    }
    m = a.get("monotonic", {})
    probes = metrics.get("probes", {})
    if m.get("tyre_life_up_slower") and "tyre_life" in probes:
        out["tyre_life_up_slower"] = probes["tyre_life"]["mean_delta_s"] > 0
    if m.get("lap_number_up_faster_in_race") and "lap_number" in probes:
        out["lap_number_up_faster_in_race"] = probes["lap_number"]["mean_delta_s"] < 0
    return out
