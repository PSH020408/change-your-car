"""P1-2 / P1-3 — the lap filter chain.

Every step reports how many laps it removed. That is not decoration: the
reconnaissance pass lost 46% of laps without anyone being able to say which
filter did it, and one session (2025 Silverstone R) silently dropped to 14%
because of a defective gate. Silent data loss is the failure mode this
module exists to prevent.

Order is deliberate (docs/recon/DECISIONS.md D3, D4):

    pit -> deleted -> accurate -> TRACK STATUS -> telemetry gap -> PACE GATE

Track status runs BEFORE the pace gate so its effect is observable. In the
recon run it came after, removed exactly zero laps, and there was no way to
tell whether safety-car laps had already been caught by the pace gate or
whether the status filter was simply broken.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pipeline.ingest.loader import LapTelemetry


# FastF1 track status codes. A lap's TrackStatus is the concatenation of
# every code that was active during it, e.g. "14" = green then safety car.
TRACK_STATUS_MEANING = {
    "1": "AllClear",
    "2": "Yellow",
    "3": "Unknown/3",
    "4": "SafetyCar",
    "5": "RedFlag",
    "6": "VirtualSafetyCar",
    "7": "VSCEnding",
}


@dataclass
class Step:
    name: str
    kept: int
    removed: int
    note: str = ""


@dataclass
class FilterReport:
    key: str
    raw: int
    steps: list[Step] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def add(self, name: str, kept: int, note: str = "") -> None:
        before = self.steps[-1].kept if self.steps else self.raw
        self.steps.append(Step(name=name, kept=kept, removed=before - kept, note=note))

    @property
    def final(self) -> int:
        return self.steps[-1].kept if self.steps else self.raw

    @property
    def yield_pct(self) -> float:
        return round(100.0 * self.final / self.raw, 2) if self.raw else 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "raw": self.raw,
            "final": self.final,
            "yield_pct": self.yield_pct,
            "steps": [asdict(s) for s in self.steps],
            "diagnostics": self.diagnostics,
        }

    def render(self) -> str:
        w = max((len(s.name) for s in self.steps), default=10)
        lines = [f"  {'raw':<{w}}  {self.raw:>5}"]
        for s in self.steps:
            pct = 100.0 * s.kept / self.raw if self.raw else 0.0
            flag = "  <-- removed nothing" if s.removed == 0 else ""
            lines.append(f"  {s.name:<{w}}  {s.kept:>5}  (-{s.removed:<4}) {pct:5.1f}%{flag}"
                         + (f"  {s.note}" if s.note else ""))
        lines.append(f"  {'YIELD':<{w}}  {self.final:>5}          {self.yield_pct:5.1f}%")
        return "\n".join(lines)


# --------------------------------------------------------------------- seconds
def _secs(col: pd.Series) -> pd.Series:
    """timedelta column -> float seconds, NaN-safe."""
    return pd.to_timedelta(col, errors="coerce").dt.total_seconds()


# ------------------------------------------------------------------- 1-3 cheap
def drop_pit_laps(laps: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=laps.index)
    if "PitInTime" in laps:
        m &= laps["PitInTime"].isna()
    if "PitOutTime" in laps:
        m &= laps["PitOutTime"].isna()
    return laps[m]


def drop_deleted(laps: pd.DataFrame) -> pd.DataFrame:
    if "Deleted" not in laps:
        return laps
    return laps[~laps["Deleted"].fillna(False).astype(bool)]


def require_accurate(laps: pd.DataFrame) -> pd.DataFrame:
    if "IsAccurate" not in laps:
        return laps
    return laps[laps["IsAccurate"].fillna(False).astype(bool)]


def require_laptime(laps: pd.DataFrame) -> pd.DataFrame:
    if "LapTime" not in laps:
        return laps
    return laps[_secs(laps["LapTime"]).notna()]


# ---------------------------------------------------------------- 4. D4 status
def exclude_track_status(
    laps: pd.DataFrame, codes: list[str], report: FilterReport, fastf1_laps=None
) -> pd.DataFrame:
    """Drop laps run under any of `codes`.

    Implemented directly on the string rather than through FastF1's
    `pick_track_status`, so the result is inspectable. FastF1's own answer is
    computed alongside and recorded: if the two disagree, that is a finding,
    not something to paper over.
    """
    if "TrackStatus" not in laps:
        report.diagnostics["track_status"] = {"present": False}
        return laps

    ts = laps["TrackStatus"].fillna("").astype(str)
    hit = pd.Series(False, index=laps.index)
    per_code = {}
    for c in codes:
        m = ts.str.contains(c, regex=False)
        per_code[f"{c}:{TRACK_STATUS_MEANING.get(c, '?')}"] = int(m.sum())
        hit |= m

    diag = {
        "present": True,
        "excluded_codes": codes,
        "laps_matching_each_code": per_code,
        "laps_removed_ours": int(hit.sum()),
        "observed_values": {str(k): int(v) for k, v in ts.value_counts().head(12).items()},
    }

    # Cross-check against the library, on the same input.
    if fastf1_laps is not None:
        try:
            ff1_kept = len(fastf1_laps.pick_track_status("".join(codes), how="none"))
            diag["laps_removed_fastf1"] = int(len(laps) - ff1_kept)
            diag["agrees_with_fastf1"] = bool(diag["laps_removed_fastf1"] == diag["laps_removed_ours"])
        except Exception as exc:                            # noqa: BLE001
            diag["fastf1_crosscheck_error"] = f"{type(exc).__name__}: {exc}"[:120]

    # If nothing matched, say whether the session simply never went yellow.
    if not hit.any():
        non_green = int((~ts.isin(["1", ""])).sum())
        diag["laps_with_any_non_green_code"] = non_green
        diag["verdict"] = ("session ran green throughout — filter correctly removed nothing"
                           if non_green == 0 else
                           "NON-GREEN LAPS EXIST BUT NONE MATCHED — investigate code set")

    report.diagnostics["track_status"] = diag
    return laps[~hit]


# -------------------------------------------------------------- 5. D2 gap
def max_telemetry_gap(
    laps: pd.DataFrame,
    telemetry: dict[str, "LapTelemetry"],
    uids: pd.Series,
    max_gap_m: float,
    report: FilterReport,
) -> pd.DataFrame:
    """Reject laps whose telemetry has a hole.

    A 98 m spacing at 300 km/h is ~1.2 s of missing samples — a transmission
    dropout, not sampling resolution. Interpolating across it would teach the
    model a stretch of driving that was never recorded (DECISIONS.md D2).
    """
    keep, reasons = [], {"no_telemetry": 0, "gap": 0, "ok": 0}
    gaps = []
    for idx, uid in zip(laps.index, uids):
        t = telemetry.get(uid)
        if t is None or not t.ok:
            reasons["no_telemetry"] += 1
            continue
        gaps.append(t.max_gap_m)
        if t.max_gap_m > max_gap_m:
            reasons["gap"] += 1
            continue
        reasons["ok"] += 1
        keep.append(idx)

    report.diagnostics["telemetry_gap"] = {
        "threshold_m": max_gap_m,
        "removed_no_telemetry": reasons["no_telemetry"],
        "removed_gap_exceeded": reasons["gap"],
        "max_gap_m_p50": round(float(np.median(gaps)), 2) if gaps else None,
        "max_gap_m_p95": round(float(np.quantile(gaps, 0.95)), 2) if gaps else None,
        "max_gap_m_max": round(float(np.max(gaps)), 2) if gaps else None,
    }
    return laps.loc[keep]


# ------------------------------------------------------------- 6. D3 pace gate
def pace_gate(
    laps: pd.DataFrame,
    cfg: dict,
    report: FilterReport,
) -> pd.DataFrame:
    """Keep laps within `threshold_pct` of a CONDITIONS-MATCHED reference.

    The classic 107% rule compares against the session best, which is only
    valid if the track stayed the same all session. It does not: a drying
    race sets its best in the last ten minutes, and everything earlier is
    binned even though those laps were the fastest possible *at the time*.
    That is precisely how 2025 Silverstone R fell to a 14.4% yield.

    The reference here is local in time instead — the median of the fastest
    `reference_top_n` laps set by ANY driver within +/- a rolling window. Using
    all drivers is deliberate: the field collectively defines what the track
    was capable of at that moment.
    """
    threshold = float(cfg.get("threshold_pct", 1.07))
    mode = str(cfg.get("reference", "rolling_window"))
    window_laps = float(cfg.get("window_laps", 5))
    top_n = int(cfg.get("reference_top_n", 3))

    lt = _secs(laps["LapTime"]).to_numpy(dtype=float)
    valid = np.isfinite(lt)
    if not valid.any():
        report.diagnostics["pace_gate"] = {"error": "no valid lap times"}
        return laps.iloc[0:0]

    if mode == "session_best":
        ref = np.full(len(lt), np.nanmin(lt))
        window_s = None
    else:
        # Window in SECONDS, derived from the session's own median lap time,
        # so the same setting works for a 1:20 qualifying lap and a 1:45 race
        # lap, and for qualifying where lap NUMBERS are not comparable between
        # drivers running different programmes.
        median_lap = float(np.nanmedian(lt[valid]))
        window_s = window_laps * median_lap

        start = _secs(laps["LapStartTime"]).to_numpy(dtype=float) \
            if "LapStartTime" in laps else np.arange(len(lt), dtype=float) * median_lap
        if not np.isfinite(start).all():
            fill = np.nanmedian(start[np.isfinite(start)]) if np.isfinite(start).any() else 0.0
            start = np.where(np.isfinite(start), start, fill)

        ref = np.full(len(lt), np.nan)
        lt_valid = np.where(valid, lt, np.inf)
        for i in range(len(lt)):
            near = np.abs(start - start[i]) <= window_s
            cand = lt_valid[near]
            cand = cand[np.isfinite(cand)]
            if not len(cand):
                ref[i] = lt[i]
                continue
            k = min(top_n, len(cand))
            ref[i] = float(np.median(np.partition(cand, k - 1)[:k]))

    ratio = np.where(np.isfinite(ref) & (ref > 0), lt / ref, np.inf)
    keep = valid & (ratio <= threshold)

    report.diagnostics["pace_gate"] = {
        "reference": mode,
        "threshold_pct": threshold,
        "window_seconds": round(window_s, 1) if window_s else None,
        "reference_top_n": top_n,
        "session_best_s": round(float(np.nanmin(lt[valid])), 3),
        "reference_spread_s": [round(float(np.nanmin(ref)), 3),
                               round(float(np.nanmax(ref)), 3)] if np.isfinite(ref).any() else None,
        "removed": int((~keep).sum()),
    }

    out = laps[keep].copy()
    out["pace_reference_s"] = ref[keep]
    out["pace_ratio"] = ratio[keep]
    return out


# ----------------------------------------------------------------- conditions
def tag_conditions(laps: pd.DataFrame, cfg: dict) -> pd.Series:
    """Label each lap dry / inter / wet from the compound actually fitted.

    Conditions are TAGGED, never filtered away (DECISIONS.md D3): the dry
    model trains on dry laps, and the wet laps stay in bronze for a later
    conditions model rather than being thrown out at ingest.
    """
    inter = set(cfg.get("inter_compounds", ["INTERMEDIATE"]))
    wet = set(cfg.get("wet_compounds", ["WET"]))
    if "Compound" not in laps:
        return pd.Series("unknown", index=laps.index)
    c = laps["Compound"].fillna("").astype(str).str.upper()
    return pd.Series(
        np.where(c.isin(wet), "wet", np.where(c.isin(inter), "inter", "dry")),
        index=laps.index,
    )


# --------------------------------------------------------------------- driver
def apply_chain(
    laps: pd.DataFrame,
    telemetry: dict,
    uids_for,
    cfg: dict,
    key: str,
    fastf1_laps=None,
) -> tuple[pd.DataFrame, FilterReport]:
    """Run the configured chain in order, reporting every step."""
    rep = FilterReport(key=key, raw=int(len(laps)))
    cur = laps

    order = cfg.get("order") or [
        "drop_pit_laps", "drop_deleted", "require_accurate",
        "exclude_track_status", "max_telemetry_gap", "pace_gate",
    ]

    for step in order:
        if step == "drop_pit_laps":
            cur = drop_pit_laps(cur); rep.add("pit in/out", len(cur))
        elif step == "drop_deleted":
            cur = drop_deleted(cur); rep.add("deleted", len(cur))
        elif step == "require_accurate":
            cur = require_accurate(cur); rep.add("accurate", len(cur))
            cur = require_laptime(cur); rep.add("has lap time", len(cur))
        elif step == "exclude_track_status":
            cur = exclude_track_status(
                cur, cfg.get("exclude_track_status", ["4", "5", "6", "7"]), rep,
                fastf1_laps=fastf1_laps)
            d = rep.diagnostics.get("track_status", {})
            rep.add("track status", len(cur), d.get("verdict", ""))
        elif step == "max_telemetry_gap":
            cur = max_telemetry_gap(
                cur, telemetry, uids_for(cur),
                float(cfg.get("max_telemetry_gap_m", 40.0)), rep)
            rep.add("telemetry gap", len(cur))
        elif step == "pace_gate":
            cur = pace_gate(cur, cfg.get("pace_gate", {}), rep)
            rep.add("pace gate", len(cur))

    return cur, rep
