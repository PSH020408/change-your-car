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

    # Codes that appear in the data but are NOT in the exclusion set. These
    # are informational: a yellow-flag lap is not a safety-car lap, and
    # treating "not all-clear" as "should have been removed" is what made the
    # first version of this diagnostic cry wolf.
    other_codes: dict[str, int] = {}
    for val, n in ts.value_counts().items():
        for ch in str(val):
            if ch not in codes and ch != "1":
                other_codes[f"{ch}:{TRACK_STATUS_MEANING.get(ch, '?')}"] = \
                    other_codes.get(f"{ch}:{TRACK_STATUS_MEANING.get(ch, '?')}", 0) + int(n)

    diag = {
        "present": True,
        "excluded_codes": codes,
        "laps_matching_each_code": per_code,
        "laps_removed_ours": int(hit.sum()),
        "codes_present_but_not_excluded": other_codes,
        "observed_values": {str(k): int(v) for k, v in ts.value_counts().head(12).items()},
    }

    # Cross-check against the library — on THIS frame, not on the unfiltered
    # session. Comparing a filtered subset's count against the raw session's
    # count produced a meaningless negative number in the first run.
    target = laps if hasattr(laps, "pick_track_status") else fastf1_laps
    if target is not None and hasattr(target, "pick_track_status"):
        try:
            ff1_removed = int(len(target) - len(target.pick_track_status("".join(codes), how="none")))
            diag["laps_removed_fastf1"] = ff1_removed
            diag["crosscheck_population"] = int(len(target))
            diag["agrees_with_fastf1"] = bool(ff1_removed == diag["laps_removed_ours"])
        except Exception as exc:                            # noqa: BLE001
            diag["fastf1_crosscheck_error"] = f"{type(exc).__name__}: {exc}"[:120]
    else:
        diag["fastf1_crosscheck_error"] = "filtered frame is not a fastf1 Laps object"

    # "Removed nothing" has two very different causes. Separate them.
    if not hit.any():
        present_excluded = sum(per_code.values())
        if present_excluded == 0:
            extra = (f" (present but not excluded: "
                     f"{', '.join(other_codes)})" if other_codes else "")
            diag["verdict"] = ("no lap ran under any excluded code — "
                               "removing nothing is correct" + extra)
            diag["broken"] = False
        else:
            diag["verdict"] = (f"BROKEN — {present_excluded} lap(s) carry an excluded "
                               f"code but none were removed")
            diag["broken"] = True

    report.diagnostics["track_status"] = diag
    return laps[~hit]


def tag_telemetry_quality(laps: pd.DataFrame, telemetry, uids) -> pd.Series:
    """Band each lap by its worst telemetry gap, keeping the lap either way.

    Third time this project has reached the same conclusion — conditions,
    yellow flags, now feed gaps: destroying rows at ingest throws away the
    evidence needed to decide whether they should have been destroyed.
    """
    out = []
    for uid in uids:
        t = telemetry.get(uid)
        out.append("missing" if (t is None or not t.ok)
                   else telemetry_quality_band(t.max_gap_s))
    return pd.Series(out, index=laps.index)


def tag_track_status(laps: pd.DataFrame, codes: list[str]) -> pd.Series:
    """Flag laps that ran under a non-excluded, non-green code (yellow).

    A yellow flag forces a lift somewhere on track, so the lap is compromised
    — but it is still a real lap driven with a real setup, and dropping 20% of
    a qualifying session for it is too expensive. Tag it, keep it in bronze,
    and let the training config decide (same principle as wet conditions).
    """
    if "TrackStatus" not in laps:
        return pd.Series("unknown", index=laps.index)
    ts = laps["TrackStatus"].fillna("").astype(str)
    excluded = set(codes)
    def classify(v: str) -> str:
        marks = {c for c in v if c != "1" and c not in excluded}
        return "green" if not marks else "flagged_" + "".join(sorted(marks))
    return ts.map(classify)


# -------------------------------------------------------------- 5. D2 gap
SWEEP_SECONDS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

# Measured on 2022 Australian GP R (813 laps): worst-gap per lap has
# p50 0.88 s, p95 1.24 s, max 1.32 s against a 0.24 s nominal period. A
# ~3-sample dropout therefore happens on the MEDIAN lap — it is the feed's
# normal behaviour, not an anomaly, and any threshold inside that range
# slices a unimodal distribution arbitrarily. So the gate became a safety net
# and the real signal is carried as a tag.
QUALITY_BANDS = [("clean", 0.5), ("normal", 1.0), ("gappy", 1.5)]


def telemetry_quality_band(max_gap_s: float) -> str:
    for name, limit in QUALITY_BANDS:
        if max_gap_s <= limit:
            return name
    return "holed"


def max_telemetry_gap(
    laps: pd.DataFrame,
    telemetry: dict[str, "LapTelemetry"],
    uids: pd.Series,
    max_gap_s: float,
    report: FilterReport,
) -> pd.DataFrame:
    """Reject laps whose telemetry has a hole, measured in TIME.

    The first version of this filter gated on distance at 40 m and removed 88%
    of a qualifying session. The measured distribution explains why: the
    median lap's worst spacing is 67 m, because at 300 km/h a single skipped
    sample already covers 20 m. Distance measures how fast the car was going
    as much as it measures whether data went missing.

    Time is the honest axis. The feed's nominal period is 240 ms, so a 1.0 s
    gap means roughly three consecutive samples were dropped, at any speed.

    The diagnostic sweeps candidate thresholds and reports what each would
    cost, so this number gets set from the distribution rather than from an
    argument about it (DECISIONS.md D2, revised).
    """
    keep, reasons = [], {"no_telemetry": 0, "gap": 0, "ok": 0}
    gaps_s, gaps_m, missed, where = [], [], [], []
    implied, neg_steps, bands = [], 0, {}

    for idx, uid in zip(laps.index, uids):
        t = telemetry.get(uid)
        if t is None or not t.ok:
            reasons["no_telemetry"] += 1
            continue
        gaps_s.append(t.max_gap_s)
        gaps_m.append(t.max_gap_m)
        missed.append(t.missed_samples_worst)
        where.append(t.worst_gap_at_frac)
        implied.append(getattr(t, "implied_speed_kph", 0.0))
        neg_steps += int(getattr(t, "negative_distance_steps", 0))
        b = telemetry_quality_band(t.max_gap_s)
        bands[b] = bands.get(b, 0) + 1
        if t.max_gap_s > max_gap_s:
            reasons["gap"] += 1
            continue
        reasons["ok"] += 1
        keep.append(idx)

    def q(arr, p):
        return round(float(np.quantile(arr, p)), 3) if arr else None

    n = len(gaps_s)
    sweep = {f"{th}s": int(sum(1 for g in gaps_s if g <= th)) for th in SWEEP_SECONDS} \
        if gaps_s else {}
    sweep_pct = {k: (round(100.0 * v / n, 1) if n else 0.0) for k, v in sweep.items()}

    report.diagnostics["telemetry_gap"] = {
        "threshold_s": max_gap_s,
        "laps_evaluated": n,
        "removed_no_telemetry": reasons["no_telemetry"],
        "removed_gap_exceeded": reasons["gap"],
        "max_gap_s_p50": q(gaps_s, 0.50),
        "max_gap_s_p90": q(gaps_s, 0.90),
        "max_gap_s_p95": q(gaps_s, 0.95),
        "max_gap_s_max": q(gaps_s, 1.0),
        "missed_samples_p50": int(np.median(missed)) if missed else None,
        "missed_samples_max": int(np.max(missed)) if missed else None,
        "worst_gap_position_p50": q(where, 0.50),
        "max_gap_m_p50": q(gaps_m, 0.50),
        "max_gap_m_p95": q(gaps_m, 0.95),
        "threshold_sweep_laps_kept": sweep,
        "threshold_sweep_pct_kept": sweep_pct,
        "quality_bands": bands,
        # Distance-axis integrity. The speed implied by (distance covered /
        # time elapsed) across each lap's worst gap must be physically
        # possible; an F1 car has never exceeded ~380 km/h.
        "implied_speed_kph_p50": q(implied, 0.50),
        "implied_speed_kph_max": q(implied, 1.0),
        "laps_with_impossible_implied_speed": int(sum(1 for v in implied if v > 400)),
        "negative_distance_steps_total": neg_steps,
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
        "exclude_track_status", "drop_pit_laps", "drop_deleted",
        "require_accurate", "max_telemetry_gap", "pace_gate",
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
                float(cfg.get("max_telemetry_gap_s", 1.0)), rep)
            d = rep.diagnostics.get("telemetry_gap", {})
            rep.add("telemetry gap", len(cur),
                    f"p50={d.get('max_gap_s_p50')}s p95={d.get('max_gap_s_p95')}s")
        elif step == "pace_gate":
            cur = pace_gate(cur, cfg.get("pace_gate", {}), rep)
            rep.add("pace gate", len(cur))

    return cur, rep
