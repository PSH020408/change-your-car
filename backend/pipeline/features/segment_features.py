"""P2-5 — per (lap x segment) aggregation. This is the model's row.

The target is per-segment time, so the row has to be per segment. Everything
here is integrated over the RAW samples inside the segment's distance window —
never a resampled grid (DECISIONS.md D1). Segment time in particular is the
label: computing it from interpolated samples would mean training on times
nobody drove.

Segment time is obtained by integrating dt over the samples in the window
rather than differencing the window's endpoints, so a sample gap inside the
segment does not silently become extra time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def align_distance(tel: pd.DataFrame, reference_length_m: float) -> pd.DataFrame:
    """Put this lap's distance axis onto the reference lap's scale.

    Every lap gets its own distance axis from add_distance(), which integrates
    speed x dt and therefore accumulates its own error — the reference lap
    came out 1.8% short of the published circuit length. Segments are defined
    on the REFERENCE lap's axis but applied to every lap's own, so a 1-2%
    disagreement puts a boundary 50-90 m away from where it belongs. On a
    380 m corner that is a quarter of the segment: one lap's "corner" includes
    part of the preceding straight and another's does not, and the difference
    lands in the target as if the driver had caused it.

    Rescaling proportionally aligns the axes end to end. It does not fix
    within-lap drift — matching each sample to the nearest point on the
    reference centreline would — but it removes the dominant term for a
    fraction of the cost.
    """
    out = tel.copy()
    d = pd.to_numeric(out["distance_m"], errors="coerce").to_numpy(dtype=float)
    total = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    scale = (reference_length_m / total) if total > 0 else 1.0
    out["distance_m"] = d * scale
    out["distance_scale"] = scale
    return out


def segment_slice(tel: pd.DataFrame, start_m: float, end_m: float,
                  lap_length_m: float) -> pd.DataFrame:
    """Samples inside a segment, handling one that wraps start-finish."""
    d = tel["distance_m"].to_numpy(dtype=float)
    if end_m > lap_length_m:                       # wrapped segment
        m = (d >= start_m) | (d < end_m - lap_length_m)
    else:
        m = (d >= start_m) & (d < end_m)
    return tel[m]


def boundary_times(tel: pd.DataFrame, segments: list[dict],
                   lap_length_m: float) -> dict[int, float]:
    """Each segment's duration, measured on the FULL lap's time axis.

    This has to be done at lap level, not inside the segment. A segment's own
    slice ends at its last interior sample, so interpolating within it loses
    the stretch from that sample to the boundary — which is how the first fix
    still shed 3% of the lap. Interpolating on the whole lap makes adjacent
    segments share a boundary exactly, so the durations tile by construction.
    """
    d = pd.to_numeric(tel["distance_m"], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(tel.get("session_time_s"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(d) & np.isfinite(t)
    out: dict[int, float] = {}
    if ok.sum() < 2:
        return out
    dd, tt = d[ok], t[ok]
    order = np.argsort(dd)
    dd, tt = dd[order], tt[order]

    lap_span = float(tt[-1] - tt[0])
    for seg in segments:
        a, b = float(seg["start_m"]), float(seg["end_m"])
        if b > lap_length_m:                 # wraps through start-finish
            tail = lap_span - float(np.interp(a, dd, tt) - tt[0])
            head = float(np.interp(b - lap_length_m, dd, tt) - tt[0])
            out[seg["index"]] = tail + head
        else:
            out[seg["index"]] = float(np.interp(b, dd, tt) - np.interp(a, dd, tt))
    return out


def features_for_segment(seg_tel: pd.DataFrame, seg: dict,
                         lap_length_m: float,
                         time_span_s: float | None = None) -> dict:
    out: dict = {
        "segment_index": seg["index"],
        "segment_kind": seg["kind"],
        "segment_sector": seg.get("sector"),
        "segment_length_m": seg["length_m"],
        "segment_min_radius_m": seg.get("min_radius_m"),
        "segment_direction": seg.get("direction"),
        "segment_is_kink": seg["kind"] == "kink",
        "n_samples": int(len(seg_tel)),
    }
    if not len(seg_tel):
        return out

    v = pd.to_numeric(seg_tel.get("speed_kph"), errors="coerce").to_numpy(dtype=float)
    thr = pd.to_numeric(seg_tel.get("throttle_pct"), errors="coerce").to_numpy(dtype=float)
    brk = pd.Series(seg_tel.get("brake_on")).fillna(False).astype(bool).to_numpy()
    t = pd.to_numeric(seg_tel.get("session_time_s"), errors="coerce").to_numpy(dtype=float)
    d = seg_tel["distance_m"].to_numpy(dtype=float)

    # --- the label -------------------------------------------------------
    # Summing the INTERIOR intervals loses the one that crosses each segment's
    # opening boundary: N samples give N-1 intervals, so every segment sheds
    # about one sampling period. Across 35 segments that is ~8 s of a 80 s lap,
    # and the invariant caught it at a 7% median error.
    #
    # The time is therefore measured at the boundaries themselves, interpolated
    # on the distance axis, which makes adjacent segments tile the lap exactly.
    # The gap contribution is still computed from the interior intervals and
    # reported separately, so a dropout inside a segment stays visible even
    # though it is no longer silently excluded from the total.
    out["segment_time_s"] = round(time_span_s, 4) if time_span_s is not None else np.nan
    out["segment_time_gap_s"] = np.nan
    ok = np.isfinite(t)
    if ok.sum() >= 2:
        intervals = np.diff(t[ok])
        # The gap can no longer be subtracted — the total has to tile the lap —
        # so it is reported instead. A lap that spent 3 s in a dropout inside
        # this segment is still visible as one, and P4 can weight it down.
        out["segment_time_gap_s"] = round(float(intervals[intervals >= 2.0].sum()), 4)

    # --- speed -----------------------------------------------------------
    vf = _finite(v)
    if len(vf):
        out.update({
            "speed_mean_kph": round(float(vf.mean()), 2),
            "speed_min_kph": round(float(vf.min()), 2),
            "speed_max_kph": round(float(vf.max()), 2),
            "speed_entry_kph": round(float(vf[0]), 2),
            "speed_exit_kph": round(float(vf[-1]), 2),
        })

    # --- pedals ----------------------------------------------------------
    if np.isfinite(thr).any():
        out["throttle_mean_pct"] = round(float(np.nanmean(thr)), 2)
        out["throttle_full_frac"] = round(float((thr >= 95).mean()), 4)
        # Where in the segment the driver got back to full throttle: the
        # single most setup-sensitive thing a driver does in a corner.
        full = np.flatnonzero(thr >= 95)
        out["throttle_on_frac"] = (round(float((d[full[0]] - d[0])
                                               / max(seg["length_m"], 1e-6)), 4)
                                   if len(full) else np.nan)
    if brk.any():
        out["brake_frac"] = round(float(brk.mean()), 4)
        first = int(np.argmax(brk))
        out["brake_point_frac"] = round(float((d[first] - d[0])
                                              / max(seg["length_m"], 1e-6)), 4)
    else:
        out["brake_frac"] = 0.0
        out["brake_point_frac"] = np.nan

    if "drs_raw" in seg_tel:
        drs = pd.to_numeric(seg_tel["drs_raw"], errors="coerce").to_numpy(dtype=float)
        # FastF1 DRS codes: 10, 12, 14 mean the flap is open.
        out["drs_open_frac"] = round(float(np.isin(drs, (10, 12, 14)).mean()), 4)

    # --- loads -----------------------------------------------------------
    r = seg.get("min_radius_m") or np.inf
    if len(vf) and np.isfinite(r) and r > 0:
        out["lateral_g_peak"] = round(float((vf.max() / 3.6) ** 2 / r / G), 3)
        out["lateral_g_apex"] = round(float((vf.min() / 3.6) ** 2 / r / G), 3)
    if np.isfinite(t).sum() >= 3 and len(vf) >= 3:
        ms = v / 3.6
        dv = np.diff(ms)
        dtt = np.diff(t)
        ok = np.isfinite(dv) & np.isfinite(dtt) & (dtt > 0) & (dtt < 2.0)
        if ok.any():
            acc = dv[ok] / dtt[ok]
            out["long_g_accel_max"] = round(float(acc.max() / G), 3)
            out["long_g_brake_max"] = round(float(-acc.min() / G), 3)
    return out


def features_for_lap(tel: pd.DataFrame, segments: list[dict],
                     lap_length_m: float) -> pd.DataFrame:
    spans = boundary_times(tel, segments, lap_length_m)
    rows = [features_for_segment(
        segment_slice(tel, s["start_m"], s["end_m"], lap_length_m),
        s, lap_length_m, time_span_s=spans.get(s["index"]))
        for s in segments]
    return pd.DataFrame(rows)
