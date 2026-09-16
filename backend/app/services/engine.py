"""P6 — the simulation engine: one request in, the HUD's whole payload out.

    baseline lap (store)  ->  segment table
    ML (P4)               ->  q(new conditions) - q(baseline conditions)   per segment, with band
    level 2 (P4)          ->  session-level temperature / session-type shift
    physics (P3)          ->  setup deltas with declared bands
    total                 ->  reconstruct (P5) -> trace, per-segment audit
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas import domain as S
from app.services import engineer_log as LOG
from pipeline.models.predict import Predictor
from pipeline.physics import modifiers as M, segment_delta as D
from pipeline.reconstruct import trace as T

DISPLAY_STEP_M = 20.0
CLEAN_AIR_GAP_S = 10.0
PUSH_EFFORT_INDEX = 0.75


class NotFound(Exception):
    pass


@dataclass
class LoadedBaseline:
    doc: dict
    driver: dict
    lap: dict
    trace: pd.DataFrame          # raw grid, distance rescaled
    segments: list[dict]


class Engine:
    def __init__(self, baselines_dir: Path, model_dir: Path, physics_cfg: M.PhysicsConfig | None = None):
        self.baselines_dir = Path(baselines_dir)
        self.index = json.loads((self.baselines_dir / "index.json").read_text())
        self.predictor: Predictor | None = None
        self.model_version = "none"
        try:
            self.predictor = Predictor.load(Path(model_dir))
            self.model_version = str(self.predictor.metrics.get("version", "?"))
        except FileNotFoundError:
            pass
        self.physics = physics_cfg or M.default_config()
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------ catalog
    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self.index["seasons"])

    def events(self, season: int) -> list[dict]:
        evs = self.index["seasons"].get(str(season), {})
        return [{"event": slug, "event_name": v["event_name"], "circuit": v["circuit"],
                 "sessions": sorted(v["sessions"].keys())} for slug, v in sorted(evs.items())]

    def drivers(self, season: int, event: str, session: str) -> list[dict]:
        doc = self._doc(season, event, session)
        return [{"driver": d, "team": v.get("team"), "chassis": v.get("chassis"), "power_unit": v.get("power_unit"),
                 "laps": len(v["available"]),
                 "representative_lap_time_s": (v["laps"].get("representative") or {}).get("lap_time_s")}
                for d, v in sorted(doc["drivers"].items())]

    def chassis(self, season: int) -> list[dict]:
        seen: dict[str, dict] = {}
        for slug, ev in self.index["seasons"].get(str(season), {}).items():
            for ses in ev["sessions"].values():
                for c in ses["chassis"]:
                    seen.setdefault(c, {"chassis": c, "events": 0})["events"] += 1
        return sorted(seen.values(), key=lambda r: r["chassis"])

    # ------------------------------------------------------------ baseline
    def _doc(self, season: int, event: str, session: str) -> dict:
        key = f"{season}/{event}/{session}"
        if key not in self._cache:
            p = self.baselines_dir / str(season) / event / f"{session}.json"
            if not p.exists():
                raise NotFound(f"no baseline store for {key}")
            self._cache[key] = json.loads(p.read_text())
        return self._cache[key]

    def load_baseline(self, ref: S.BaselineRef) -> LoadedBaseline:
        doc = self._doc(ref.season, ref.event, ref.session)
        drv = doc["drivers"].get(ref.driver.upper())
        if drv is None:
            raise NotFound(f"driver {ref.driver} not in {ref.season}/{ref.event}/{ref.session}; "
                           f"available: {sorted(doc['drivers'])}")
        laps = drv["laps"]
        choice = ref.lap if ref.lap in ("representative", "fastest") else None
        if choice is None:
            hit = [k for k, v in laps.items() if v.get("lap_uid") == ref.lap]
            if not hit:
                raise NotFound(f"lap {ref.lap} is not one of the stored laps "
                               f"({', '.join(v['lap_uid'] for v in laps.values())}); only the representative "
                               f"and fastest laps carry telemetry in the store")
            choice = hit[0]
        lap = laps[choice]
        if "alias_of" in lap:
            lap = laps[lap["alias_of"]]
        tr = pd.DataFrame(lap["trace"])
        tr["drs_raw"] = np.where(tr["drs_open"], 12, 8)
        return LoadedBaseline(doc=doc, driver=drv, lap=lap, trace=tr, segments=doc["segments"])

    # ------------------------------------------------------------ helpers
    @staticmethod
    def segment_table(b: LoadedBaseline) -> pd.DataFrame:
        d = b.trace["distance_m"].to_numpy(float); v = b.trace["speed_kph"].to_numpy(float)
        rows = []
        for (i0, i1), s in zip(T._segment_slices(d, b.segments), b.segments):
            vv = v[i0:i1]
            t0, t1 = T._tiled(i0, i1, len(d))
            t = T.integrate_lap_time(v[t0:t1], d[t0:t1]) if i1 - i0 >= 3 else 0.0
            rows.append({"segment_index": s["index"], "segment_kind": s["kind"], "segment_time_s": t,
                         "speed_min_kph": float(vv.min()) if len(vv) else np.nan,
                         "speed_mean_kph": float(vv.mean()) if len(vv) else np.nan,
                         "speed_max_kph": float(vv.max()) if len(vv) else np.nan,
                         "segment_length_m": float(s["length_m"]), "segment_min_radius_m": s.get("min_radius_m"),
                         "segment_sector": s.get("sector"), "segment_is_kink": s["kind"] == "kink"})
        return pd.DataFrame(rows)

    def _ml_rows(self, b: LoadedBaseline, segs: pd.DataFrame, env: S.Environment, new: bool) -> pd.DataFrame:
        lap = b.lap
        tyre_life = (env.tyre_life if (new and env.tyre_life is not None) else lap.get("tyre_life")) or 1
        compound = (env.compound.value if (new and env.compound is not None) else lap.get("compound")) or "MEDIUM"
        rows = segs.copy()
        rows["tyre_life"] = float(tyre_life)
        rows["lap_number"] = float(lap.get("lap_number") or 1)
        rows["track_temp_c"] = float((env.track_temp_c if (new and env.track_temp_c is not None) else lap.get("track_temp_c")) or 30.0)
        rows["air_temp_c"] = float((env.air_temp_c if (new and env.air_temp_c is not None) else lap.get("air_temp_c")) or 22.0)
        rows["lap_effort_index"] = PUSH_EFFORT_INDEX
        rows["gap_ahead_s"] = CLEAN_AIR_GAP_S
        rows["segment_reference_s"] = rows["segment_time_s"]
        rows["session"] = b.doc["session"]
        rows["compound"] = compound
        rows["fresh_tyre"] = bool(tyre_life <= 1) if (new and env.tyre_life is not None) else bool(lap.get("fresh_tyre") or False)
        rows["driver"] = lap["driver"]
        rows["chassis"] = lap.get("chassis") or "?"
        rows["season"] = int(b.doc["season"])
        return rows

    def _ml_lap_halfwidth(self) -> float:
        """80% half-width of a predicted lap CHANGE, from the registered
        model's measured counterfactual MAE on clean-air push laps (P4)."""
        if self.predictor is None:
            return 0.0
        m = self.predictor.metrics or {}
        cp = ((m.get("holdout") or {}).get("clean_push") or (m.get("cv") or {}).get("clean_push") or {})
        mae = float(cp.get("cf_lap_mae_s", 0.5) or 0.5)
        return float(np.log(5.0) * mae)

    @staticmethod
    def resample(trace: pd.DataFrame, lap_len: float, step: float = DISPLAY_STEP_M) -> S.TelemetryTrace:
        d = trace["distance_m"].to_numpy(float)
        grid = np.arange(0.0, lap_len + 1e-6, step)
        grid = grid[grid <= d.max()] if d.max() < lap_len else grid
        def lin(col):
            return np.interp(grid, d, pd.to_numeric(trace[col], errors="coerce").ffill().bfill().to_numpy(float))
        idx = np.abs(grid[:, None] - d[None, :]).argmin(axis=1)
        nearest_gap = np.abs(grid - d[idx])
        def near(col, cast):
            return [cast(x) for x in trace[col].to_numpy()[idx]]
        return S.TelemetryTrace(
            distance_m=[round(float(x), 1) for x in grid],
            time_s=[round(float(x), 3) for x in lin("time_s")],
            speed_kph=[round(float(x), 1) for x in lin("speed_kph")],
            throttle_pct=[round(float(x), 1) for x in lin("throttle_pct")],
            brake_on=near("brake_on", bool), gear=near("gear", int), drs_open=near("drs_open", bool),
            interpolated=[bool(g > 1.5 * step) for g in nearest_gap])

    def baseline_response(self, ref: S.BaselineRef) -> S.BaselineResponse:
        b = self.load_baseline(ref)
        r0 = T.reconstruct(b.trace, b.segments, [0.0] * len(b.segments), cfg=self.physics,
                           official_lap_time_s=b.lap["lap_time_s"])
        lap_len = float(b.doc["track"]["lap_length_m"])
        return S.BaselineResponse(
            lap=self._lap_meta(b), trace=self.resample(r0.trace, lap_len),
            segments=[S.SegmentInfo(**{k: s.get(k) for k in S.SegmentInfo.model_fields}) for s in b.segments],
            track=S.TrackMap(**b.doc["track"]),
            available_laps=b.driver["available"],
            integration_note=(f"sampled trace integrates {r0.lap_time_baseline_s:.3f} s vs official "
                              f"{b.lap['lap_time_s']:.3f} s; time axis scaled by {r0.time_scale:.4f} "
                              f"(the two partial 240 ms intervals at the line)"))

    def _lap_meta(self, b: LoadedBaseline) -> S.LapMeta:
        lap, doc, drv = b.lap, b.doc, b.driver
        return S.LapMeta(lap_uid=lap["lap_uid"], driver=lap["driver"], team=drv.get("team"), chassis=drv.get("chassis"),
                         power_unit=drv.get("power_unit"), season=doc["season"], event=doc["event"],
                         event_name=doc["event_name"], session=doc["session"], circuit=doc.get("circuit"),
                         lap_number=lap.get("lap_number"), lap_time_s=lap["lap_time_s"], compound=lap.get("compound"),
                         tyre_life=lap.get("tyre_life"), fresh_tyre=lap.get("fresh_tyre"),
                         track_temp_c=lap.get("track_temp_c"), air_temp_c=lap.get("air_temp_c"),
                         telemetry_quality=lap.get("telemetry_quality"), effort_class=lap.get("effort_class"),
                         gap_ahead_s=lap.get("gap_ahead_s"), sector_times_s=lap.get("sector_times_s") or [None] * 3)

    # ------------------------------------------------------------ simulate
    def simulate(self, req: S.SimulationRequest) -> S.SimulationResponse:
        t0 = time.perf_counter()
        b = self.load_baseline(req.baseline)
        segs = self.segment_table(b)
        n = len(segs)
        env, setup, lap = req.environment, req.setup, b.lap

        # --- ML level 1: conditions
        ml = np.zeros(n); ml_lo = np.zeros(n); ml_hi = np.zeros(n)
        env_changed = any(x is not None for x in (env.track_temp_c, env.air_temp_c, env.compound, env.tyre_life))
        if self.predictor is not None and env_changed:
            base_rows, new_rows = self._ml_rows(b, segs, env, new=False), self._ml_rows(b, segs, env, new=True)
            qb, qn = self.predictor.predict_segments(base_rows), self.predictor.predict_segments(new_rows)
            ml = qn["q50"].to_numpy() - qb["q50"].to_numpy()
            # Band for a DIFFERENCE of two predictions: the per-segment q10/q90
            # bands summed over 23 segments assume every segment errs the same
            # way and gave +-2 s for a 0.3 s tyre effect. The honest width is
            # the model's MEASURED accuracy on lap changes (counterfactual MAE
            # on unseen events, P4): for a Laplace error the 80% half-width is
            # ln(5) * MAE. Shared over segments by their baseline time.
            half = self._ml_lap_halfwidth()
            share = segs["segment_time_s"].to_numpy() / max(float(segs["segment_time_s"].sum()), 1e-9)
            ml_lo, ml_hi = ml - half * share, ml + half * share

        # --- level 2: session-level shift, % of segment time. INFORMATIONAL:
        # reported per segment, NOT added to the total. With 57 sessions the
        # air/track coefficients are collinear (+0.71 / -0.21 % per C) and a
        # track-only +10 C came out 1.9 s FASTER; level 1 already carries the
        # in-range temperature effect. Applied once a refit with one
        # temperature term passes a sign-and-size check.
        l2 = np.zeros(n)
        if self.predictor is not None and env_changed:
            d_track = (env.track_temp_c - lap["track_temp_c"]) if (env.track_temp_c is not None and lap.get("track_temp_c") is not None) else 0.0
            d_air = (env.air_temp_c - lap["air_temp_c"]) if (env.air_temp_c is not None and lap.get("air_temp_c") is not None) else 0.0
            pct, _, _ = self.predictor.reference_shift(d_track_temp_c=d_track, d_air_temp_c=d_air)
            l2 = segs["segment_time_s"].to_numpy() * pct / 100.0

        # --- physics: setup
        sin = M.SetupInput(front_wing=setup.front_wing, rear_wing=setup.rear_wing, ride_height=setup.ride_height,
                           suspension=setup.suspension, suspension_split=setup.suspension_split,
                           fuel_kg=setup.fuel_kg, weather=env.weather.value,
                           track_temp_c=env.track_temp_c, compound=env.compound.value if env.compound else lap.get("compound"),
                           stint_lap=int(env.tyre_life or lap.get("tyre_life") or 1))
        # the ML already owns tyre temperature in-range: hand physics only the setup + weather
        sin_phys = M.SetupInput(**{**sin.__dict__, "track_temp_c": None})
        phys = D.lap_physics_delta(segs, sin_phys, session=b.doc["session"], baseline_track_temp_c=lap.get("track_temp_c"),
                                   roughness=M.circuit_roughness(b.doc["event"], self.physics), cfg=self.physics)
        state = M.physics_state(sin_phys, session=b.doc["session"], roughness=M.circuit_roughness(b.doc["event"], self.physics),
                                cfg=self.physics)
        ph, ph_lo, ph_hi = (phys[c].to_numpy() for c in ("physics_delta_s", "physics_delta_lo_s", "physics_delta_hi_s"))

        total, total_lo, total_hi = ml + ph, ml_lo + ph_lo, ml_hi + ph_hi          # level 2 not applied (see above)
        rec = T.reconstruct(b.trace, b.segments, {int(i): float(t) for i, t in zip(segs["segment_index"], total)},
                            cfg=self.physics, grip_scale=state.grip_multiplier, official_lap_time_s=lap["lap_time_s"])
        base_rec = T.reconstruct(b.trace, b.segments, [0.0] * n, cfg=self.physics, official_lap_time_s=lap["lap_time_s"])
        audit = rec.segments.set_index("segment_index")
        lap_len = float(b.doc["track"]["lap_length_m"])

        seg_out = []
        for k, row in segs.iterrows():
            i = int(row["segment_index"])
            seg_out.append(S.SegmentDelta(
                index=i, kind=row["segment_kind"], sector=row.get("segment_sector"),
                baseline_time_s=round(float(row["segment_time_s"]), 3),
                ml_s=round(float(ml[k]), 4), ml_lo_s=round(float(ml_lo[k]), 4), ml_hi_s=round(float(ml_hi[k]), 4),
                level2_s=round(float(l2[k]), 4),
                physics_s=round(float(ph[k]), 4), physics_lo_s=round(float(ph_lo[k]), 4), physics_hi_s=round(float(ph_hi[k]), 4),
                total_s=round(float(total[k]), 4), total_lo_s=round(float(min(total_lo[k], total_hi[k])), 4),
                total_hi_s=round(float(max(total_lo[k], total_hi[k])), 4),
                achieved_s=round(float(audit.loc[i, "achieved_s"]), 4),
                refused_s=round(float(audit.loc[i, "residual_s"]) if bool(audit.loc[i, "clamped"]) else 0.0, 4)))
        sectors = [float(sum(s.total_s for s in seg_out if s.sector == k)) for k in (1, 2, 3)]
        refused = float(sum(s.refused_s for s in seg_out))
        summary = S.LapSummary(
            baseline_lap_time_s=round(lap["lap_time_s"], 3),
            simulated_lap_time_s=round(lap["lap_time_s"] + rec.achieved_delta_s, 3),
            delta_s=round(rec.achieved_delta_s, 3),
            delta_lo_s=round(float(np.minimum(total_lo, total_hi).sum()), 3),
            delta_hi_s=round(float(np.maximum(total_lo, total_hi).sum()), 3),
            ml_s=round(float(ml.sum()), 3), level2_s=round(float(l2.sum()), 3), level2_applied=False,
            physics_s=round(float(ph.sum()), 3),
            refused_s=round(refused, 3), sector_deltas_s=[round(x, 3) for x in sectors])
        pstate = S.PhysicsState(downforce_pct=state.downforce_pct, drag_pct=state.drag_pct, mech_grip_pct=state.mech_grip_pct,
                                grip_multiplier=state.grip_multiplier, thermal_grip_pct=state.thermal_grip_pct,
                                fuel_delta_kg=state.fuel_delta_kg, balance_index=state.balance_index, warning=state.warning)
        notes = LOG.build(req, b, seg_out, summary, pstate, self.predictor, env_changed)
        return S.SimulationResponse(
            lap=summary, segments=seg_out,
            baseline=self.resample(base_rec.trace, lap_len), simulated=self.resample(rec.trace, lap_len),
            physics=pstate, grades=LOG.grades(self.physics), engineer_log=notes,
            model_version=self.model_version, physics_version=str(self.physics.path.name if self.physics.path else "physics.yaml"),
            computed_ms=round((time.perf_counter() - t0) * 1000, 1))
