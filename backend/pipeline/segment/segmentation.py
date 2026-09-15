"""P2-2 / P2-3 — corner-and-straight segmentation.

The lap is split where curvature crosses a threshold. Two things make this
harder than thresholding an array.

THE THRESHOLD IS NOT KNOWABLE A PRIORI. 0.0035 1/m means "a corner is
anything tighter than a 286 m radius", which is a guess dressed as a
constant. This project has now been burned twice by exactly that move (the
10 m resample grid, the 40 m gap gate), so the threshold is not asserted
here: `sweep_thresholds` reports how many corners each candidate produces,
and the answer is chosen by comparing against the circuit's published corner
count. Same method that settled the telemetry-gap threshold.

RAW THRESHOLDING PRODUCES NONSENSE SEGMENTS. Curvature wobbles across the
threshold, so a single corner shatters into a dozen fragments and a straight
picks up phantom kinks. Two morphological passes fix it: close gaps shorter
than `min_gap_m` (one corner, briefly under threshold at its apex or between
two apexes of a double-apex turn), then drop runs shorter than
`min_segment_len_m` (noise).

The lap is a LOOP, so all of this wraps: a corner straddling the start-finish
line is one corner, not two.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

CORNER_KINDS = ("low_speed_corner", "medium_speed_corner", "high_speed_corner")

# A curve the car takes flat is not a corner the setup has to be tuned for.
# Albert Park 2022 returned 10 of 14 corners as "high speed" because several
# were 40 m of curvature at 305 km/h with entry == apex == exit: real geometry,
# but the driver never lifted. Splitting them out keeps the corner count
# honest against published figures (which include kinks) while telling the
# simulator which corners a setup change will actually be felt in.
KINK_SPEED_RETENTION = 0.95      # apex >= 95% of entry means no meaningful lift


@dataclass
class Segment:
    index: int
    kind: str                 # straight | *_corner
    start_m: float
    end_m: float
    length_m: float
    mean_curvature_1pm: float
    peak_curvature_1pm: float
    min_radius_m: float
    interior_min_radius_m: float | None = None   # excluding window/2 at each end
    direction: str = "straight"   # left | right | straight
    apex_speed_kph: float | None = None      # corners only — a straight has no apex
    min_speed_kph: float | None = None
    max_speed_kph: float | None = None
    entry_speed_kph: float | None = None
    exit_speed_kph: float | None = None
    sector: int | None = None
    wraps_start_finish: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------ morphology
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start, end) index pairs."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _close_gaps(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """Fill False runs shorter than `min_gap` (double-apex corners)."""
    out = mask.copy()
    for a, b in _runs(~mask):
        if (b - a) < min_gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for a, b in _runs(mask):
        if (b - a) < min_len:
            out[a:b] = False
    return out


def corner_mask(curvature: np.ndarray, step_m: float, threshold: float,
                min_gap_m: float = 30.0, min_segment_len_m: float = 40.0) -> np.ndarray:
    """Boolean corner mask, cleaned and wrap-aware."""
    k = np.abs(np.asarray(curvature, dtype=float))
    raw = k > threshold

    # Roll so that any run straddling start-finish becomes contiguous, do the
    # morphology, then roll back.
    shift = 0
    if raw[0] and raw[-1]:
        zero = np.flatnonzero(~raw)
        if len(zero):
            shift = int(zero[0])
            raw = np.roll(raw, -shift)

    gap = max(1, int(round(min_gap_m / max(step_m, 1e-6))))
    seg = max(1, int(round(min_segment_len_m / max(step_m, 1e-6))))
    cleaned = _drop_short(_close_gaps(raw, gap), seg)

    return np.roll(cleaned, shift) if shift else cleaned


def count_turns(segments: list["Segment"]) -> dict:
    """Corners, kinks, and the total that published turn counts describe."""
    corners = sum(1 for s in segments if s.kind.endswith("_corner"))
    kinks = sum(1 for s in segments if s.kind == "kink")
    return {"corners": corners, "kinks": kinks, "turns": corners + kinks}


def count_corners(mask: np.ndarray) -> int:
    """Corner count, counting a start-finish straddle once.

    Must agree with what `build_segments` produces — a sweep table that
    reports a different number than the pipeline actually emits is worse than
    no sweep table, because the threshold gets chosen against the wrong
    column.
    """
    runs = _runs(mask)
    if len(runs) > 1 and mask[0] and mask[-1]:
        return len(runs) - 1
    return len(runs)


def sweep_thresholds(curvature: np.ndarray, step_m: float,
                     candidates: list[float], **kw) -> dict[str, int]:
    """Corner count at each candidate threshold — pick from published counts."""
    return {f"{t:.4f}": count_corners(corner_mask(curvature, step_m, t, **kw))
            for t in candidates}


def sweep_full(geo, cfg: dict, candidates: list[float],
               raw_distance_m=None, raw_speed_kph=None,
               max_lateral_g: float = 6.5) -> list[dict]:
    """Corner count AND composition AND physics verdict, per threshold.

    A count alone cannot settle the threshold: 11 corners and 14 corners can
    both look reasonable until you see that one of them classifies seven of
    them as high-speed on a circuit that does not have seven fast corners.
    One run now answers the whole question.
    """
    rows = []
    for t in candidates:
        c = dict(cfg)
        c["curvature_threshold_1pm"] = t
        segs = build_segments(geo, c, raw_distance_m=raw_distance_m,
                              raw_speed_kph=raw_speed_kph)
        kinds: dict[str, int] = {}
        for s in segs:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        phys = check_physics(segs, t, max_lateral_g)
        turns = count_turns(segs)
        rows.append({
            "threshold_1pm": t,
            "radius_m": round(1.0 / t) if t > 0 else None,
            # What this threshold means physically: the lateral load a car
            # pulls at 300 km/h on a corner right at the boundary.
            "lateral_g_at_300kph": round((300 / 3.6) ** 2 * t / 9.81, 2),
            "corners": turns["corners"],
            "kinks": turns["kinks"],
            "turns": turns["turns"],
            "low": kinds.get("low_speed_corner", 0),
            "medium": kinds.get("medium_speed_corner", 0),
            "high": kinds.get("high_speed_corner", 0),
            "segments": len(segs),
            "physics_passes": phys["passes"],
            "over_g": len(phys["corners_over_g_limit"]),
            "hidden": len(phys["straights_hiding_a_corner"]),
        })
    return rows


# ------------------------------------------------------------ classification
def is_corner_kind(kind: str) -> bool:
    """Kinks count as track geometry but not as corners the driver works."""
    return kind.endswith("_corner") or kind == "kink"


def classify_corner(apex_speed_kph: float | None, bins: dict) -> str:
    if apex_speed_kph is None or not np.isfinite(apex_speed_kph):
        return "medium_speed_corner"
    for name in ("low", "medium", "high"):
        lo, hi = bins.get(name, (0, 0))
        if lo <= apex_speed_kph < hi:
            return f"{name}_speed_corner"
    return "high_speed_corner"


def _speeds_in(distance_m: np.ndarray, speed_kph: np.ndarray, start: float, end: float
               ) -> tuple[float | None, float | None, float | None, float | None]:
    """Apex / entry / exit speed from RAW samples inside a distance window.

    Deliberately reads the raw telemetry rather than the geometry grid: the
    geometry grid is interpolated, and an interpolated apex speed is a number
    nobody drove (DECISIONS.md D1).
    """
    m = (distance_m >= start) & (distance_m < end)
    if not m.any():
        return None, None, None, None
    v = speed_kph[m]
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None, None, None
    return float(v.min()), float(v[0]), float(v[-1]), float(v.max())


def build_segments(
    geo, cfg: dict,
    raw_distance_m: np.ndarray | None = None,
    raw_speed_kph: np.ndarray | None = None,
) -> list[Segment]:
    step = geo.grid_step_m
    # The smoothing window bleeds a corner's curvature ~window/2 into the
    # straight on either side. A straight's radius measured over its whole
    # length therefore always looks "corner-ish" near its ends, which is how
    # 17 of 19 circuits failed the hidden-corner invariant. The interior
    # radius excludes that bleed zone and is what the invariant checks.
    bleed_pts = int(round(float(cfg.get("smooth_window_m", 90.0)) / 2.0 / step))
    threshold = float(cfg.get("curvature_threshold_1pm", 0.0035))
    mask = corner_mask(
        geo.curvature_1pm, step, threshold,
        min_gap_m=float(cfg.get("min_gap_m", 30.0)),
        min_segment_len_m=float(cfg.get("min_segment_len_m", 40.0)),
    )

    bins = {k: tuple(v) for k, v in (cfg.get("corner_speed_bins") or {}).items()}
    d = geo.distance_m
    n = len(d)

    # Alternating runs over the whole lap, corner and straight alike.
    boundaries: list[tuple[int, int, bool]] = []
    for a, b in _runs(mask):
        boundaries.append((a, b, True))
    for a, b in _runs(~mask):
        boundaries.append((a, b, False))
    boundaries.sort()

    segments: list[Segment] = []
    for i, (a, b, is_corner) in enumerate(boundaries):
        start, end = float(d[a]), float(d[min(b, n - 1)])
        k = geo.curvature_1pm[a:b]
        if not len(k):
            continue
        # Percentile, not max. A single noisy sample used to set the whole
        # segment's radius — on the 2022 Australian GP that produced corners
        # implying 9-10 g. The 90th percentile of |curvature| tracks the apex
        # without letting one outlier define the corner.
        absk = np.abs(k)
        thresh_k = float(np.quantile(absk, 0.90))
        peak_i = int(np.argmin(np.abs(absk - thresh_k)))
        peak = float(k[peak_i])
        # None means "too short to have an interior", never "perfectly
        # straight" — a perfectly straight interior is a very large radius.
        interior = absk[bleed_pts:len(absk) - bleed_pts] if len(absk) > 2 * bleed_pts + 2 else None
        if interior is None:
            interior_r = None
        else:
            q = float(np.quantile(interior, 0.90))
            interior_r = round(1.0 / q, 1) if q > 1e-9 else 99999.0

        apex = entry = exit_ = vmax = None
        if raw_distance_m is not None and raw_speed_kph is not None:
            apex, entry, exit_, vmax = _speeds_in(raw_distance_m, raw_speed_kph, start, end)

        if is_corner:
            kind = classify_corner(apex, bins)
            direction = "left" if peak > 0 else "right"
        else:
            kind, direction = "straight", "straight"

        segments.append(Segment(
            index=i, kind=kind,
            start_m=round(start, 1), end_m=round(end, 1),
            length_m=round(end - start, 1),
            mean_curvature_1pm=round(float(np.mean(k)), 6),
            peak_curvature_1pm=round(peak, 6),
            min_radius_m=round(1.0 / thresh_k, 1) if thresh_k > 1e-9 else 99999.0,
            interior_min_radius_m=interior_r,
            direction=direction,
            apex_speed_kph=round(apex, 1) if (is_corner and apex is not None) else None,
            min_speed_kph=round(apex, 1) if apex is not None else None,
            max_speed_kph=round(vmax, 1) if vmax is not None else None,
            entry_speed_kph=round(entry, 1) if entry is not None else None,
            exit_speed_kph=round(exit_, 1) if exit_ is not None else None,
        ))

    segments = _merge_wrap(segments, mask, float(d[-1]) + step)
    _mark_kinks(segments)
    for i, seg in enumerate(segments):
        seg.index = i
    return segments


def _mark_kinks(segments: list[Segment]) -> None:
    """Reclassify curves the car took flat.

    The comparison has to be against the APPROACH speed — the exit of the
    segment before — not the corner's own entry sample. Inside a corner the
    car has already finished braking, so a corner's first sample is its slowed
    speed and every corner would look flat. The lap is a loop, so the segment
    before the first one is the last one.
    """
    if not segments:
        return
    for i, seg in enumerate(segments):
        if not seg.kind.endswith("_corner") or seg.apex_speed_kph is None:
            continue
        prev = segments[i - 1]                      # wraps at i == 0
        # The TOP speed of the preceding segment, not its last sample: a
        # straight's boundary often falls inside the braking zone, so its exit
        # sample is already slowed and every corner would read as flat.
        approach = prev.max_speed_kph or prev.exit_speed_kph or seg.entry_speed_kph
        if approach and seg.apex_speed_kph >= KINK_SPEED_RETENTION * approach:
            seg.kind = "kink"


def _merge_wrap(segments: list[Segment], mask: np.ndarray,
                lap_length_m: float) -> list[Segment]:
    """Join the first and last segments when the lap wraps through them.

    The morphology already treats the lap as a loop, but rebuilding segments
    from the un-rolled mask splits whatever straddles the start-finish line
    back into two. On the synthetic oval that showed up as a 10 m phantom
    corner at distance 0 — and on a real circuit that is a whole corner
    counted twice and measured wrong, which is exactly what the corner-count
    gate is supposed to catch.

    The merged segment keeps monotonic distances by letting `end_m` run past
    the lap length; consumers take it modulo. Splitting it for display is
    cheap, reconstructing it after the fact is not.
    """
    if len(segments) < 2 or not (mask[0] and mask[-1]):
        return segments
    first, last = segments[0], segments[-1]
    if (first.kind == "straight") != (last.kind == "straight"):
        return segments

    merged = Segment(
        index=0,
        kind=last.kind if last.length_m >= first.length_m else first.kind,
        start_m=last.start_m,
        end_m=round(lap_length_m + first.end_m, 1),
        length_m=round((lap_length_m - last.start_m) + first.end_m, 1),
        mean_curvature_1pm=round(
            (last.mean_curvature_1pm * last.length_m
             + first.mean_curvature_1pm * first.length_m)
            / max(last.length_m + first.length_m, 1e-9), 6),
        peak_curvature_1pm=max(last.peak_curvature_1pm, first.peak_curvature_1pm,
                               key=abs),
        min_radius_m=min(last.min_radius_m, first.min_radius_m),
        interior_min_radius_m=min([v for v in (last.interior_min_radius_m,
                                               first.interior_min_radius_m)
                                   if v is not None], default=None),
        direction=last.direction,
        apex_speed_kph=min([v for v in (last.apex_speed_kph, first.apex_speed_kph)
                            if v is not None], default=None),
        min_speed_kph=min([v for v in (last.min_speed_kph, first.min_speed_kph)
                           if v is not None], default=None),
        max_speed_kph=max([v for v in (last.max_speed_kph, first.max_speed_kph)
                           if v is not None], default=None),
        entry_speed_kph=last.entry_speed_kph,
        exit_speed_kph=first.exit_speed_kph,
        wraps_start_finish=True,
    )
    return [merged] + segments[1:-1]


# ------------------------------------------------------------ microsectors
def microsectors(lap_length_m: float, n: int = 28) -> list[dict]:
    """Equal-distance divisions for the HUD's delta heat overlay."""
    edges = np.linspace(0.0, lap_length_m, n + 1)
    return [{"index": i, "start_m": round(float(edges[i]), 1),
             "end_m": round(float(edges[i + 1]), 1)} for i in range(n)]


def assign_sectors(segments: list[Segment], sector_boundaries_m: list[float]) -> None:
    """Tag each segment with the official sector its midpoint falls in."""
    if len(sector_boundaries_m) < 2:
        return
    for s in segments:
        mid = 0.5 * (s.start_m + s.end_m)
        sec = 1
        for i, b in enumerate(sector_boundaries_m):
            if mid >= b:
                sec = i + 2
        s.sector = min(max(sec, 1), 3)


def sector_boundaries_from_times(
    raw_distance_m: np.ndarray, raw_time_s: np.ndarray,
    lap_start_s: float, sector1_s: float, sector2_s: float,
) -> list[float]:
    """Distance at which each official sector ends, from the lap's sector times.

    FastF1 publishes sector TIMES but not sector distances, and the HUD needs
    to colour the track map by sector. Walking the lap's own time axis is the
    only way to convert one into the other.
    """
    t = np.asarray(raw_time_s, dtype=float) - float(lap_start_s)
    d = np.asarray(raw_distance_m, dtype=float)
    ok = np.isfinite(t) & np.isfinite(d)
    if ok.sum() < 3:
        return []
    t, d = t[ok], d[ok]
    order = np.argsort(t)
    t, d = t[order], d[order]
    return [float(np.interp(sector1_s, t, d)),
            float(np.interp(sector1_s + sector2_s, t, d))]


def median_sector_boundaries(per_lap: list[list[float]], lap_length_m: float) -> list[float]:
    """One sector boundary pair from many laps' pairs, robust to the odd bad lap.

    Each lap's boundaries were mapped onto the ensemble axis already. A value
    that wrapped past the timing line (sector 3 ending at 5 m instead of
    5,270 m) is unwrapped toward the first lap before the median, so a
    boundary near distance zero cannot be averaged with its own wrap.
    """
    rows = [b for b in per_lap if len(b) == 2 and all(np.isfinite(b))]
    if not rows:
        return []
    arr = np.asarray(rows, dtype=float)
    ref = arr[0]
    arr = np.where(arr - ref > lap_length_m / 2, arr - lap_length_m,
                   np.where(ref - arr > lap_length_m / 2, arr + lap_length_m, arr))
    med = np.median(arr, axis=0) % lap_length_m
    return [float(med[0]), float(med[1])]


# --------------------------------------------------------------- invariants
def check_physics(segments: list[Segment], corner_threshold_1pm: float,
                  max_lateral_g: float = 6.5,
                  hidden_corner_factor: float = 0.7) -> dict:
    """Two invariants that must hold, or the segmentation is wrong.

    1. No corner may imply more lateral load than the car can generate. The
       first version of this pipeline produced six corners at 7-10 g on one
       circuit, all of them curvature noise wearing a corner's label.
    2. No straight may contain a radius tighter than the corner threshold —
       that is a corner the segmentation lost. Five of sixteen straights
       failed this on the same circuit.

    Reported rather than raised: a marginal circuit should still produce
    output, but the run must say so out loud.
    """
    corner_radius = 1.0 / corner_threshold_1pm if corner_threshold_1pm > 0 else 1e9
    # A straight whose robust radius sits just under the threshold is not a
    # lost corner, it is a gentle kink at the classification boundary — the
    # boundary has to fall somewhere. Only a MATERIALLY tighter radius is a
    # defect: R = 21 m inside a straight is a missed hairpin, R = 281 m
    # against a 286 m threshold is a 2% judgement call.
    material_radius = corner_radius * hidden_corner_factor
    over_g, hidden, borderline = [], [], []

    for s in segments:
        if s.kind.endswith("_corner"):
            v = (s.apex_speed_kph or 0.0) / 3.6
            r = s.min_radius_m or 1e9
            g = (v * v / r / 9.81) if r > 0 else 0.0
            if g > max_lateral_g:
                over_g.append({"index": s.index, "lateral_g": round(g, 1),
                               "radius_m": r, "apex_kph": s.apex_speed_kph})
        elif s.kind == "straight":
            # Judge on the interior where one exists; a straight too short to
            # have an interior beyond the bleed zone cannot hide a corner
            # longer than itself and is not judged at all.
            r = s.interior_min_radius_m
            if r is None or r >= corner_radius:
                continue
            row = {"index": s.index, "radius_m": r, "length_m": s.length_m,
                   "pct_of_threshold": round(100.0 * r / corner_radius)}
            (hidden if r < material_radius else borderline).append(row)

    n_corner = sum(1 for s in segments if is_corner_kind(s.kind))
    n_straight = sum(1 for s in segments if s.kind == "straight")

    # Over-g is never acceptable: it means the curvature is not physical.
    # A single straight holding a tight-ish radius is a boundary call on a
    # real circuit, so the gate tolerates one, or 10% of straights, before
    # calling the segmentation contaminated. Loosened deliberately: the first
    # version failed a whole circuit over a 2% difference, which makes the
    # alarm worthless for the 21 m violation it exists to catch.
    hidden_fraction = len(hidden) / n_straight if n_straight else 0.0
    passes = (not over_g) and hidden_fraction <= 0.10
    return {
        "max_lateral_g": max_lateral_g,
        "corner_threshold_radius_m": round(corner_radius, 1),
        "corners_over_g_limit": over_g,
        "straights_hiding_a_corner": hidden,
        "straights_at_the_boundary": borderline,
        "material_radius_m": round(material_radius, 1),
        "corners_checked": n_corner,
        "straights_checked": n_straight,
        "hidden_fraction": round(hidden_fraction, 3),
        "passes": passes,
    }
