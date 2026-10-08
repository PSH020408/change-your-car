"""P11 — tyre-strategy mode: score a race strategy lap by lap from one real lap.

The unit is the driver's representative RACE lap (compound c0, age a0, lap n0,
time T0). Everything a strategy changes is a difference from that lap:

    t_i = T0 + fuel(n_i) - fuel(n0)            the engine's fuel slope (s/kg) x the
                                               fuel the car carries on lap i
         + ML(c_i, a_i) - ML(c0, a0)           the conditions model: compound and
                                               tyre age, session temperatures fixed
    race = sum_i t_i + stops x pit_loss        pit loss measured on this race

Nothing is invented: the compound cards are the strategies teams actually ran
this weekend, a stint may not be longer than the longest stint anyone ran on
that compound (the tree model cannot extrapolate past the data), and the pit
loss is the median of this race's own stops. The race-pace management that a
qualifying lap would miss is already inside T0, because T0 is a race lap.

Ignored, and said so in the response: safety cars, traffic, out-lap warm-up,
in-lap push, the first lap, and that a different stint plan would change the
driver's pace management.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas import domain as S
from pipeline.physics import modifiers as M

FUEL_PROBE_KG = 10.0
SEQ_RE = re.compile(r"\s*(SOFT|MEDIUM|HARD)\s+(\d+)\s*")


@dataclass
class RaceInfo:
    race_laps: int
    pit_loss_s: float | None
    pit_loss_n: int
    green_share: float
    stint_max: dict[str, int]


class StrategyService:
    def __init__(self, engine):
        self.engine = engine
        self._race: dict[str, RaceInfo | None] = {}
        self._ml_cache: dict[tuple, float] = {}

    # ------------------------------------------------------------ data
    def race_info(self, season: int, event: str) -> RaceInfo | None:
        key = f"{season}/{event}"
        if key not in self._race:
            p = Path(self.engine.baselines_dir) / str(season) / event / "race.json"
            if not p.exists():
                self._race[key] = None
            else:
                d = json.loads(p.read_text())
                pl = d.get("pit_loss") or {}
                self._race[key] = RaceInfo(race_laps=int(d["race_laps"]), pit_loss_s=pl.get("median_s"),
                                           pit_loss_n=int(pl.get("n") or 0), green_share=float(d.get("green_share", 0.0)),
                                           stint_max={k: int(v) for k, v in (d.get("stint_max") or {}).items()})
        return self._race[key]

    @staticmethod
    def parse_sequence(seq: str) -> list[tuple[str, int]]:
        return [(m.group(1), int(m.group(2))) for m in SEQ_RE.finditer(seq.replace("→", " "))]

    @staticmethod
    def fit_to_race(stints: list[tuple[str, int]], race_laps: int) -> list[tuple[str, int]]:
        """Median stint lengths rarely sum to the race distance: stretch or trim the last stint."""
        if not stints:
            return stints
        total = sum(n for _, n in stints)
        c, n = stints[-1]
        return stints[:-1] + [(c, max(1, n + (race_laps - total)))]

    # ------------------------------------------------------------ pieces
    def _ml_lap(self, b, segs, compound: str, tyre_life: int) -> float:
        """Sum of q50 over the segments for these tyre conditions (session temps)."""
        key = (b.lap["lap_uid"], compound, int(tyre_life))
        if key in self._ml_cache:
            return self._ml_cache[key]
        if self.engine.predictor is None:
            self._ml_cache[key] = 0.0
            return 0.0
        env = S.Environment(compound=S.Compound(compound), tyre_life=int(tyre_life))
        rows = self.engine._ml_rows(b, segs, env, new=True)
        v = float(self.engine.predictor.predict_segments(rows)["q50"].sum())
        self._ml_cache[key] = v
        return v

    def _fuel_slope(self, b, total_laps: int) -> tuple[float, str]:
        """Seconds per kg of fuel on this lap: from the engine when the circuit has a line, else P3."""
        season, event = int(b.doc["season"]), b.doc["event"]
        if self.engine.qss.available(season, event):
            fuel0 = M.baseline_fuel_kg("R", b.lap.get("lap_number"), total_laps, self.engine.physics)
            q = self.engine.qss.lap(season, event, b.lap, b.trace, fuel0, b.lap.get("air_temp_c"))
            from pipeline.physics import qss as Q
            heavier = Q.solve(q.d, q.k, q.g, q.drs, q.car0.with_setup(fuel_delta_kg=FUEL_PROBE_KG))
            return (heavier.lap_time_s - q.prof0.lap_time_s) / FUEL_PROBE_KG, "engine"
        return float(self.engine.physics.coeff("fuel", "s_per_kg_per_lap").value), "table"

    # ------------------------------------------------------------ score
    def score(self, req: S.StrategyRequest, race_laps: int | None = None) -> S.StrategyResponse:
        """`race_laps` overrides the race distance (the back-test scores lapped cars on the laps they ran)."""
        b = self.engine.load_baseline(req.baseline)
        if str(b.doc["session"]).upper() != "R":
            raise ValueError("strategy mode needs a race baseline: the unit lap must already be a race lap")
        info = self.race_info(int(b.doc["season"]), b.doc["event"])
        if info is None or info.pit_loss_s is None:
            raise ValueError("no measured pit loss for this race yet (run make race-timing)")
        if race_laps is not None:
            info = RaceInfo(race_laps=int(race_laps), pit_loss_s=info.pit_loss_s, pit_loss_n=info.pit_loss_n,
                            green_share=info.green_share, stint_max=info.stint_max)
        segs = self.engine.segment_table(b)
        envelope = b.doc.get("tyre_envelope") or {}
        lap = b.lap
        t0, n0 = float(lap["lap_time_s"]), int(lap.get("lap_number") or 1)
        c0, a0 = str(lap.get("compound") or "MEDIUM"), int(lap.get("tyre_life") or 1)
        slope, slope_source = self._fuel_slope(b, info.race_laps)
        ml0 = self._ml_lap(b, segs, c0, a0)
        fuel_n0 = M.baseline_fuel_kg("R", n0, info.race_laps, self.engine.physics)

        def cap_of(c: str) -> int | None:
            a, b = (envelope.get(c) or {}).get("max_laps"), info.stint_max.get(c)
            vals = [x for x in (a, b) if x is not None]
            return max(vals) if vals else None

        def one(stints: list[tuple[str, int]], label: str) -> S.StrategyScore:
            total_laps = sum(n for _, n in stints)
            refused = []
            for c, n in stints:
                cap = cap_of(c)
                if cap is not None and n > cap:
                    refused.append(f"{c} {n} laps: nobody ran {c} past {cap} laps this weekend")
            if total_laps != info.race_laps:
                refused.append(f"{total_laps} laps planned for a {info.race_laps}-lap race")
            laps_out: list[S.StrategyLap] = []
            lap_no = 0
            for k, (c, n) in enumerate(stints):
                for age in range(1, n + 1):
                    lap_no += 1
                    fuel = M.baseline_fuel_kg("R", lap_no, info.race_laps, self.engine.physics)
                    d_fuel = slope * (fuel - fuel_n0)
                    d_ml = (self._ml_lap(b, segs, c, min(age, int(cap_of(c) or age))) - ml0) if not refused else 0.0
                    laps_out.append(S.StrategyLap(lap=lap_no, compound=c, tyre_life=age, fuel_kg=round(fuel, 1),
                                                  predicted_s=round(t0 + d_fuel + d_ml, 3), fuel_delta_s=round(d_fuel, 3),
                                                  tyre_delta_s=round(d_ml, 3), pit_in=(age == n and k < len(stints) - 1)))
            stops = max(len(stints) - 1, 0)
            driving = float(sum(l.predicted_s for l in laps_out))
            return S.StrategyScore(label=label, stints=[S.Stint(compound=c, laps=n) for c, n in stints],
                                   stops=stops, driving_s=round(driving, 3), pit_s=round(stops * info.pit_loss_s, 3),
                                   race_s=round(driving + stops * info.pit_loss_s, 3), laps=laps_out,
                                   refused=refused)

        user = one([(s.compound.value, s.laps) for s in req.stints], "yours")
        cards = []
        for st in b.doc.get("strategies") or []:
            stints = self.fit_to_race(self.parse_sequence(st["sequence"]), info.race_laps)
            if not stints:
                continue
            sc = one(stints, st["sequence"])
            sc.count = st.get("count")
            sc.winner = bool(st.get("winner"))
            sc.drivers = st.get("drivers") or []
            cards.append(sc)
        cards.sort(key=lambda x: (bool(x.refused), x.race_s))
        best = next((c for c in cards if not c.refused), None)
        return S.StrategyResponse(
            race_laps=info.race_laps, pit_loss_s=round(info.pit_loss_s, 2), pit_loss_n=info.pit_loss_n,
            green_share=round(info.green_share, 3),
            unit_lap=S.StrategyUnitLap(lap_uid=lap["lap_uid"], lap_number=n0, lap_time_s=t0, compound=c0, tyre_life=a0,
                                       fuel_kg=round(fuel_n0, 1), fuel_slope_s_per_kg=round(slope, 4), fuel_slope_source=slope_source),
            yours=user, cards=cards,
            delta_to_best_s=round(user.race_s - best.race_s, 3) if (best and not user.refused) else None,
            ignored=["safety cars and red flags", "traffic and dirty air", "out-lap warm-up and in-lap push",
                     "the first lap (start, turn 1)", "pace management changing with the plan"])
