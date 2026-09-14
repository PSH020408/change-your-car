"""P1-5 — driver / team / chassis / power-unit metadata.

FastF1 tells us which TEAM ran a lap, never which chassis. The docking screen
selects a chassis (RB20, SF-24, MCL38), so the mapping has to come from
somewhere — `configs/chassis.yaml`, kept as data and validated against every
season actually ingested.

Validation is strict on purpose: an unmapped team writes a null chassis into
bronze, and a null chassis becomes a missing one-hot column at training time,
which is the kind of defect that shows up as unexplained model error weeks
later rather than as an import failure now.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


class ChassisMap:
    def __init__(self, table: dict):
        self._raw = table
        self._index: dict[tuple[int, str], dict] = {}
        for season, teams in table.items():
            for team, spec in teams.items():
                names = [team, *(spec.get("aliases") or [])]
                for n in names:
                    self._index[(int(season), self._norm(n))] = {
                        "team": team,
                        "chassis": spec["chassis"],
                        "pu": spec["pu"],
                    }

    @staticmethod
    def _norm(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def lookup(self, season: int, team: str) -> dict | None:
        return self._index.get((int(season), self._norm(team)))

    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self._raw)

    def validate_coverage(self, season: int, teams: list[str]) -> list[str]:
        """Return the teams this season's map cannot resolve."""
        return [t for t in teams if self.lookup(season, t) is None]


def load_chassis_map(path: Path) -> ChassisMap:
    return ChassisMap(yaml.safe_load(Path(path).read_text()))


def annotate(laps: pd.DataFrame, season: int, cmap: ChassisMap) -> pd.DataFrame:
    """Attach chassis / power-unit / canonical team to every lap."""
    out = laps.copy()
    if "Team" not in out:
        out["chassis"] = None
        out["power_unit"] = None
        out["team_canonical"] = None
        return out

    resolved = out["Team"].map(lambda t: cmap.lookup(season, t) or {})
    out["team_canonical"] = resolved.map(lambda d: d.get("team"))
    out["chassis"] = resolved.map(lambda d: d.get("chassis"))
    out["power_unit"] = resolved.map(lambda d: d.get("pu"))
    return out


def driver_table(laps: pd.DataFrame, season: int, cmap: ChassisMap) -> pd.DataFrame:
    """One row per driver in this session — the docking screen's catalogue."""
    cols = [c for c in ("Driver", "DriverNumber", "Team") if c in laps]
    if not cols:
        return pd.DataFrame()
    t = laps[cols].drop_duplicates(subset=["Driver"]).copy()
    t["season"] = season
    res = t["Team"].map(lambda x: cmap.lookup(season, x) or {}) if "Team" in t else {}
    t["chassis"] = res.map(lambda d: d.get("chassis")) if len(t) else None
    t["power_unit"] = res.map(lambda d: d.get("pu")) if len(t) else None
    return t.rename(columns={"Driver": "driver", "DriverNumber": "driver_number",
                             "Team": "team"}).reset_index(drop=True)
