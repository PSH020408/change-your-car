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
        Xp[c] = Xp[c].sample(frac=1.0, random_state=int(rng.integers(1 << 30))).to_numpy()
        mae = float(np.mean(np.abs(y - model.predict(Xp)["q50"].to_numpy())))
        rows.append({"feature": c, "mae_increase_s": mae - base})
    return pd.DataFrame(rows).sort_values("mae_increase_s", ascending=False).reset_index(drop=True)


def gates(metrics: dict, cfg: dict) -> dict[str, bool]:
    a = cfg["acceptance"]
    cv, ho = metrics["cv"], metrics.get("holdout", {})
    lo, hi = a.get("coverage_80", [0.7, 0.9])
    out = {
        "segment_mae": cv["segment_mae_s"] <= float(a["segment_mae_s"]),
        "lap_mae": cv["lap_mae_s"] <= float(a["lap_mae_s"]),
        "unseen_track_lap_mae": (ho.get("lap_mae_s", 0.0) <= float(a["unseen_track_lap_mae_s"])) if ho else True,
        "coverage_80": lo <= cv["coverage_80"] <= hi,
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
