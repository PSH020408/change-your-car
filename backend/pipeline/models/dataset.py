"""P4 — training table from the gold feature store, with the leaks kept out.

The gold store carries everything the lap produced: its speeds, brake points,
its lap time, its pace ratio to the session best. All of it is the ANSWER in
different units. A model fed `speed_min_kph` learns "slow apex -> slow
segment" perfectly and predicts nothing about a lap that has not been driven
yet. So the feature list is a whitelist of what is known before the lap —
tyre, fuel proxy, driver, car, session, temperature, and the segment's own
static geometry — and FeatureSpec refuses anything on the leak list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that ARE the target, or were measured on the lap being predicted.
LEAKAGE_PREFIXES = ("speed_", "brake_", "throttle_", "drs_", "bias_", "pace_ratio", "z_")
LEAKAGE_EXACT = {"lap_time_s", "segment_time_s", "segment_delta_s", "segment_time_gap_s",
                 "is_personal_best", "aggression_index", "lap_telemetry_coverage",
                 "lap_distance_scale", "sector1_s", "sector2_s", "sector3_s", "position"}

GROUP_COLS = ("season", "event_slug")


def is_leak(col: str) -> bool:
    return col in LEAKAGE_EXACT or col.startswith(LEAKAGE_PREFIXES)


# ------------------------------------------------------------ row selection
def select_training_rows(df: pd.DataFrame, cfg: dict, unstable: set[tuple] | None = None,
                         verbose: bool = True) -> pd.DataFrame:
    """Apply the training population rules and say how many rows each one cost."""
    d = cfg["data"]
    n0 = len(df)
    steps: list[tuple[str, int]] = []

    def keep(mask: pd.Series, why: str) -> None:
        nonlocal df
        before = len(df)
        df = df[mask]
        steps.append((why, before - len(df)))

    keep(df["segment_delta_s"].notna(), "no target (no reference for that segment)")
    if "condition" in df:
        keep(df["condition"].isin(d.get("train_conditions", ["dry"])), "not dry")
    if "lap_effort_class" in df:
        keep(df["lap_effort_class"].isin(d.get("effort_classes", ["push", "moderate"])), "cruise lap")
    if "segment_time_gap_s" in df:
        gap = pd.to_numeric(df["segment_time_gap_s"], errors="coerce").fillna(0).abs()
        keep(gap <= float(d.get("gap_tolerance_s", 0.05)), "telemetry hole inside the segment")
    if unstable:
        key = list(zip(df["season"].astype(int), df["event_slug"].astype(str)))
        keep(pd.Series([k not in unstable for k in key], index=df.index), "unstable race (SC / red flag)")
    # a delta of many seconds is a spin, a pit entry or a crash, not a setup question
    keep(df["segment_delta_s"] < 5.0, "delta >= 5 s (incident)")

    if verbose:
        print(f"training rows : {n0:,} -> {len(df):,}")
        for why, n in steps:
            if n:
                print(f"  -{n:>8,}  {why}")
    return df.reset_index(drop=True)


def unstable_races(df: pd.DataFrame) -> set[tuple[int, str]]:
    """(season, event_slug) of races physics-check #2 could not trust."""
    from pipeline.physics.calibrate import fuel_slope
    try:
        t = fuel_slope(df, n_boot=30)
    except Exception:            # noqa: BLE001 - the gate is advisory, never fatal
        return set()
    if not len(t):
        return set()
    return {(int(r.season), str(r.event)) for r in t.itertuples() if not r.trusted}


def group_key(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype(str) + "|" + df["event_slug"].astype(str)


# -------------------------------------------------------------- feature spec
@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = [c for c in self.numeric + self.categorical if is_leak(c)]
        if bad:
            raise ValueError(f"leakage columns refused as features: {bad}")

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def fit(self, df: pd.DataFrame) -> "FeatureSpec":
        for c in self.categorical:
            vals = df[c].dropna().astype(str).unique().tolist() if c in df else []
            self.vocab[c] = sorted(vals)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numeric as float; categorical as pandas 'category' with the FITTED
        vocabulary, so a driver unseen in training becomes NaN, never a new code."""
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
        for c in self.categorical:
            s = df[c].astype(str).where(df[c].notna(), None) if c in df else pd.Series(None, index=df.index)
            out[c] = pd.Categorical(s, categories=self.vocab.get(c, []))
        return out

    def codes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Integer codes (NaN for unknown) — for backends without category support."""
        out = X.copy()
        for c in self.categorical:
            codes = X[c].cat.codes.astype(float)
            out[c] = codes.where(codes >= 0, np.nan)
        return out

    def categorical_mask(self) -> list[bool]:
        return [c in self.categorical for c in self.columns]

    def to_json(self) -> str:
        return json.dumps({"numeric": self.numeric, "categorical": self.categorical, "vocab": self.vocab})

    @classmethod
    def from_json(cls, s: str) -> "FeatureSpec":
        d = json.loads(s)
        return cls(d["numeric"], d["categorical"], d.get("vocab", {}))

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureSpec":
        f = cfg["features"]
        return cls(list(f["numeric"]), list(f["categorical"]))


def load_gold(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
