"""P4 — inference entry point the API (P6) calls.

    pred = Predictor.load(Path("../data/artifacts/models"))
    q = pred.predict_segments(rows)                 # q10 / q50 / q90 per segment, seconds
    pct = pred.reference_shift("high_speed_corner", d_track_temp_c=+8)   # level 2, % of time
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.models import registry as R
from pipeline.models.level2 import Level2Model
from pipeline.models.quantile import QuantileSet


@dataclass
class Predictor:
    level1: QuantileSet
    level2: Level2Model
    metrics: dict

    @classmethod
    def load(cls, root: Path, version: str = "latest") -> "Predictor":
        l1, l2, m = R.load(root, version)
        return cls(l1, l2, m)

    @property
    def feature_columns(self) -> list[str]:
        return self.level1.spec.columns

    def predict_segments(self, rows: pd.DataFrame) -> pd.DataFrame:
        """`rows` = one row per segment with the level-1 feature columns
        (see spec.columns); unknown drivers / chassis are allowed (NaN category)."""
        X = self.level1.spec.transform(rows)
        return self.level1.predict(X)

    def reference_shift(self, kind: str = "", d_track_temp_c: float = 0.0, d_air_temp_c: float = 0.0,
                        to_race: float = 0.0, d_season: float = 0.0) -> tuple[float, float, float]:
        return self.level2.reference_pct_shift(kind, d_track_temp_c, d_air_temp_c, to_race, d_season)
