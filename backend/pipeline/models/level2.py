"""P4 level 2 — what the session reference itself depends on.

Level 1 predicts a lap against ITS OWN session's best. That erases anything
constant within the session: a 30 C Bahrain qualifying and a 40 C one both
have a delta of zero at their best lap. Level 2 puts it back: for every
(circuit, segment) the session reference is compared with the best reference
that circuit has produced in any session we hold, and that gap is regressed
on the things that differ between sessions — track and air temperature,
qualifying vs race, season.

It is a small linear model on purpose. With 19 sessions over 14 circuits
there are a few dozen session-pairs to learn from; a tree model would
memorise them. Ridge with a session-level bootstrap gives coefficients WITH
intervals, and an interval that includes zero is reported as exactly that:
"not distinguishable from zero with this many sessions".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SECTORS = [1, 2, 3]


def session_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, event_slug, session, sector).

    Sectors, not segments: a segment index means nothing across seasons (each
    season's circuit is segmented on its own ensemble, and 2022 Bahrain's
    segment 7 is not 2024's), so the first version compared unrelated
    pieces of track and printed a 59% "race penalty". Official sectors are
    the same physical stretch every year, so a sector's reference time is
    comparable across seasons and sessions.
    """
    keep = df.dropna(subset=["segment_reference_s", "segment_sector"])
    keep = keep.drop_duplicates(["season", "event_slug", "session", "segment_index"])
    g = keep.groupby(["season", "event_slug", "session", "segment_sector"])
    t = g.agg(reference_s=("segment_reference_s", "sum"),
              track_temp_c=("track_temp_c", "median"), air_temp_c=("air_temp_c", "median")).reset_index()
    t = t.rename(columns={"segment_sector": "sector"})
    t["sector"] = t["sector"].astype(int)
    # temperatures per session (the segment rows carried the lap's values)
    temps = df.groupby(["season", "event_slug", "session"])[["track_temp_c", "air_temp_c"]].median().reset_index()
    t = t.drop(columns=["track_temp_c", "air_temp_c"]).merge(temps, on=["season", "event_slug", "session"], how="left")
    best = t.groupby(["event_slug", "sector"])["reference_s"].transform("min")
    t["target_pct"] = 100.0 * (t["reference_s"] - best) / best
    # centre the temperatures within the circuit: the model must not learn
    # "Bahrain is hot" as "hot is slow"
    for c in ("track_temp_c", "air_temp_c"):
        t[f"{c}_centred"] = t[c] - t.groupby("event_slug")[c].transform("mean")
    t["is_race"] = (t["session"] == "R").astype(float)
    t["season_idx"] = t["season"].astype(float) - 2022.0
    n_sessions = t.groupby("event_slug")["session"].transform(lambda s: s.nunique())
    n_seasons = t.groupby("event_slug")["season"].transform(lambda s: s.nunique())
    t["informative"] = (n_sessions > 1) | (n_seasons > 1)
    return t


def design(t: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols, names = [np.ones(len(t))], ["intercept"]
    cols.append(t["is_race"].to_numpy(float)); names.append("is_race")
    cols.append(t["season_idx"].to_numpy(float)); names.append("season_idx")
    # ONE temperature term. Air and track temperature move together across
    # sessions; fitted separately they came out +0.71 and -0.21 % per C,
    # i.e. a track-only change had the wrong sign. Track temperature is the
    # one the tyre feels.
    cols.append(t["track_temp_c_centred"].to_numpy(float)); names.append("track_temp_per_c")
    return np.column_stack(cols), names


@dataclass
class Level2Model:
    coef: dict[str, float]
    coef_lo: dict[str, float] = field(default_factory=dict)
    coef_hi: dict[str, float] = field(default_factory=dict)
    n_sessions: int = 0
    n_rows: int = 0
    alpha: float = 1.0

    def reference_pct_shift(self, kind: str = "", d_track_temp_c: float = 0.0, d_air_temp_c: float = 0.0,
                            to_race: float = 0.0, d_season: float = 0.0) -> tuple[float, float, float]:
        """% change of a reference time for a change in conditions.
        Returns (nominal, lo, hi) from the coefficient intervals. `kind` is
        accepted for API stability; the sector-level fit has one temperature
        coefficient for every segment kind."""
        def pick(d: dict, k: str) -> float:
            return float(d.get(k, self.coef.get(k, 0.0)))
        terms = [("is_race", to_race), ("season_idx", d_season), ("air_temp_per_c", d_air_temp_c),
                 ("track_temp_per_c", d_track_temp_c)]
        nom = sum(self.coef.get(k, 0.0) * x for k, x in terms)
        lo = sum(min(pick(self.coef_lo, k) * x, pick(self.coef_hi, k) * x) for k, x in terms)
        hi = sum(max(pick(self.coef_lo, k) * x, pick(self.coef_hi, k) * x) for k, x in terms)
        return float(nom), float(lo), float(hi)

    def to_dict(self) -> dict:
        return {"coef": self.coef, "coef_lo": self.coef_lo, "coef_hi": self.coef_hi,
                "n_sessions": self.n_sessions, "n_rows": self.n_rows, "alpha": self.alpha}

    @classmethod
    def from_dict(cls, d: dict) -> "Level2Model":
        return cls(d["coef"], d.get("coef_lo", {}), d.get("coef_hi", {}), d.get("n_sessions", 0),
                   d.get("n_rows", 0), d.get("alpha", 1.0))


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = X.shape[1]
    pen = alpha * np.eye(p)
    pen[0, 0] = 0.0                                   # never shrink the intercept
    return np.linalg.solve(X.T @ X + pen, X.T @ y)


def fit_level2(df: pd.DataFrame, alpha: float = 1.0, n_boot: int = 200, seed: int = 0) -> tuple[Level2Model, pd.DataFrame]:
    t = session_table(df)
    t = t[t["informative"]].reset_index(drop=True)
    if len(t) < 12 or t["is_race"].nunique() < 2 and t["season_idx"].nunique() < 2:
        return Level2Model({}, n_sessions=0, n_rows=int(len(t)), alpha=alpha), t
    X, names = design(t)
    y = t["target_pct"].to_numpy(float)
    beta = _ridge(X, y, alpha)
    sess = t[["season", "event_slug", "session"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    keys = t["season"].astype(str) + "|" + t["event_slug"] + "|" + t["session"]
    for _ in range(n_boot):
        pick = rng.choice(len(sess), len(sess), replace=True)
        chosen = ["|".join(map(str, sess[i])) for i in pick]
        idx = np.concatenate([np.flatnonzero(keys.to_numpy() == c) for c in chosen])
        boots.append(_ridge(X[idx], y[idx], alpha))
    b = np.asarray(boots)
    lo, hi = np.percentile(b, [10, 90], axis=0)
    model = Level2Model(coef=dict(zip(names, map(float, beta))), coef_lo=dict(zip(names, map(float, lo))),
                        coef_hi=dict(zip(names, map(float, hi))), n_sessions=int(len(sess)), n_rows=int(len(t)), alpha=alpha)
    return model, t


def describe(model: Level2Model) -> pd.DataFrame:
    rows = []
    for k, v in model.coef.items():
        lo, hi = model.coef_lo.get(k, np.nan), model.coef_hi.get(k, np.nan)
        rows.append({"term": k, "coef_pct": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                     "distinguishable_from_0": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    return pd.DataFrame(rows)
