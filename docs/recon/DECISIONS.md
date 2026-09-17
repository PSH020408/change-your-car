# R gate — reconnaissance decisions

_Settled 2026-09-14 from `recon_report.md` (10 sessions) and the warm-cache ledger
(23 sessions). This document **is** the R gate: the values fixed here are the inputs
to P1 ingestion and P2 feature design. Two revisions, written after the first two
ingest runs, follow at the end — they are kept because they correct this document._

---

## D1 — Resampling grid: the 10 m grid is dropped

**Measured:** car telemetry arrives at a fixed `dt = 240 ms`. Spatial spacing is
therefore purely a function of speed.

| Speed | Distance covered in 240 ms |
|---|---:|
| 60 km/h (hairpin) | 4.0 m |
| 100 km/h | 6.7 m |
| 300 km/h | 20.0 m |
| 350 km/h (Monza straight) | 23.3 m |

The report confirms it: the median spacing per circuit follows the circuit's average
speed exactly (Monaco 9.9 m, Hungaroring 12.5 m, Bahrain 14.0–14.7 m, Silverstone
16.2–16.4 m, Monza 17.8–18.4 m).

So the p95 warnings of 28–36 m were overstated: the wide spacings are on straights,
where the speed profile is close to linear, while corners are naturally sampled at
3–7 m. Samples are densest where the signal is complex — a favourable distribution.

**Decisions.**
1. No uniform grid for model features. The target is a per-segment time anyway, so raw
   samples are integrated inside each segment; no interpolation means no invented detail.
2. A uniform grid only for the HUD overlay, at 20 m (the real spacing at 300 km/h), with
   an `interpolated` mask so the UI never claims resolution it does not have.
3. `scope.yaml`: `feature_resampling: none`, `display_step_m: 20.0`.

---

## D2 — The maximum gap is a dropout detector, not a resolution figure

A maximum spacing of 97.96 m was observed in the 2025 Silverstone race. At 300 km/h
that is 1.2 s — four or five samples missing in a row. Bahrain (78 m), Spa (69 m) and
Hungaroring (61 m) show the same pattern.

**Decision (superseded — see Revision 1 and 2).** A new lap filter: laps with any
telemetry gap over 40 m are excluded, and the share of laps removed is measured in P1.

---

## D3 — "Low lap yield" is two different phenomena

**(a) Qualifying at 26–37 % is healthy.** Most qualifying laps are out-laps, cool-down
laps and in-laps; a driver has three to five real push laps, and the 107 % gate removes
exactly the rest. No action.

**(b) 2025 Silverstone race at 14 % is a real defect.** The race had an intermediate
phase (961 INTERMEDIATE laps in the sample). A 107 % gate relative to the *session
best* fails when the session best was set in different grip: everything before or after
the crossover is killed.

**Decisions.**
1. The 107 % reference becomes a conditions-matched one: the rolling median of the top
   N laps within a ±5-lap window.
2. Wet and intermediate laps are **tagged** (`condition: dry | inter | wet`), not
   filtered; the dry model trains on dry laps only.
3. Every filter logs how many laps it removed — no more silent data loss.

---

## D4 — The green-flag filter removed nothing (to be verified)

`within 107 %: 3883 → green flag only: 3883`. The likely explanation is that safety-car
laps are far outside 107 % and were already gone; but a broken `pick_track_status`
match is equally possible.

**Decision.** Move track status *before* the pace gate and log each step. If it still
removes nothing, it is broken and gets fixed; guessing here would let safety-car laps
into the training data unnoticed.

---

## D5 — Two contract bugs (fixed before P7)

**(a) `Brake` is boolean.** FastF1 gives brake on/off, not pressure. The schema's
`brake_pct: list[float]` promised data that does not exist → `brake_on: list[bool]`,
and the HUD draws a band, not a curve.

**(b) DRS is a race-situation variable, not a setup one.** In the sample lap the DRS
channel had cardinality 1 — the car led and never opened it. Whether DRS opens depends
on being within one second of the car ahead. → In the simulator DRS is a track property
(detection/activation zones) and the simulation assumes "open in the zone".

---

## D6 — Setup proxies: the lap table already carries speed traps

| Column | null % | Proxy |
|---|---:|---|
| `SpeedST` | 13.4 % | drag — longest straight |
| `SpeedFL` | 5.4 % | drag + traction (finish line) |
| `SpeedI1` | 20.8 % | downforce — mid-sector |
| `SpeedI2` | 0.2 % | downforce (cleanest) |

Derived: `aero_balance ≈ SpeedST / high-speed-corner apex speed`,
`traction ≈ low-speed-corner exit acceleration`,
`braking_stability ≈ deceleration length / entry speed²`.

**Decision.** Impute missing `SpeedI1` / `SpeedST` from the telemetry maximum within
the trap's segment, and carry a `speed_trap_imputed` flag so the model can tell.
`SpeedI2` is the primary downforce proxy.

---

## D7 — Drivers without telemetry are mostly DNFs, not data holes

2024 Monaco R — HUL, MAG, OCO, PER: a multi-car crash on lap 1. Legitimate absence.
**Decision.** The survey distinguishes "no completed lap" from "laps but no telemetry";
only the latter is an issue.

---

## D8 — Disk budget, measured

Median per session (23-session ledger): FP 52–74 MB, Q 99 MB, R 110 MB, S 64 MB.
The era is 92 events (21 sprint weekends): 460 sessions, 36.9 GB in full; 24.3 GB
without practice.

**Decision.** Collect broadly, train narrowly: keep the background download running
(raw cache is schema-independent, re-downloading is rate-limited and expensive); train
on Q + R only (practice fuel loads are unknown, so fuel-corrected pace is unreliable).
_Later narrowed further: the warm scope itself became Q + R (2026-09-16)._

---

## D9 — Cache path bug

The cache was created under `backend/data/cache` instead of `data/cache`:
`Path("configs/scope.yaml").parent.parent.parent` saturates to `.` on a relative path.
The download was running, so the cache was not moved; the paths were made explicit in
`scope.yaml` and the code reads them from there.

---

## Checklist

- [x] Resampling grid → none for features, 20 m + mask for display (D1)
- [x] Sessions entering training → Q + R (D8)
- [x] Filter chain order and thresholds → track status first, conditions-matched 107 %, gap filter, per-step logging (D2–D4)
- [x] Setup-proxy feature list → SpeedI2 primary, SpeedST, SpeedFL, SpeedI1 + imputation flags (D6)
- [x] Full-era download projection → 36.9 GB / 24.3 GB (D8)

**R gate passed. P1 ingestion may start.**

---

# Revision 1 — 2026-09-14, after the first ingest run

The first ingest (2022 Australian GP qualifying) exposed two wrong decisions. They are
recorded so the same mistake is not repeated.

## D2 corrected — the gap filter was on the wrong axis (distance → time)

**Symptom:** 146 of 165 laps (88 %) removed by the gap filter; final yield 4.4 %.
**Measured:** `max_gap_m` p50 67.4 m, p95 319.8 m, max 434.3 m.

Two causes. First, the threshold (40 m) sat below the median of the data — the survey
already showed 7 of 10 sample laps above 40 m, and D2 had set the number by reasoning
("two samples missing at high speed") right after D1 had said *measure, don't estimate*.
Second, and more fundamentally, distance is the wrong axis: with a fixed 240 ms period,
a distance gap cannot separate "data is missing" from "the car was fast". At 300 km/h
one missing sample is already 20 m; 67 m on a straight is three samples in a nearly
linear stretch, harmless; 67 m at 80 km/h is twelve samples and, in a corner, fatal.

**Decisions.** Gate moved to the time axis (`max_telemetry_gap_s: 1.0`, ≈ 3 samples);
distance metrics kept for reporting; the number and position of the worst gap stored per
lap; and `ingest.run` now prints a threshold sweep table so the next threshold is chosen
from the data, not from an argument.

## D4 corrected — the filter was fine, the diagnosis was wrong

Observed track-status codes in that session: `1` (green) and `2` (yellow) only. No
safety car, no VSC — removing zero laps was correct. Two bugs in the *diagnostic*:
it counted yellow as excludable (it is deliberately not), and it cross-checked against
FastF1 on a different population (338 raw laps vs 165 filtered), producing a negative
"laps removed". Both fixed; the check now only flags BROKEN when an excludable code is
present and nothing was removed.

Side fixes: red flag (code 5) was missing from the exclusion list; yellow is tagged
(`track_status_flag`), never dropped — 20 % of qualifying laps touch a yellow sector.

**Lesson.** Reconnaissance caught assumptions; the first run caught my reading of the
reconnaissance. A report is not enough — thresholds have to be laid against the printed
distribution, so the sweep table is now a permanent output.

---

# Revision 2 — 2026-09-14, after the second ingest run

The second run (2022 Australian GP race, 1045 laps) produced the sweep table, and the
table showed that the gap filter's premise was wrong — and that D4 was still untested.

## D2 again — not a threshold problem; not a filter problem at all

Measured on 813 laps (nominal period 0.24 s): worst gap per lap p50 0.88 s, p90 1.20 s,
p95 1.24 s, max 1.32 s; missing samples p50 3, max 5. The distribution is unimodal and
tight: a three-sample dropout is *normal feed behaviour*. A 1.0 s threshold removed 294
laps (36 %) — a filter that fires on the median lap is a coin toss, not a defect
detector; 1.5 s removed none. There are no two peaks to cut between.

**Decisions.** Demote to a safety net (`max_telemetry_gap_s: 2.0`, outside the observed
range, catches genuinely broken laps elsewhere) and carry the real signal as a tag:
`telemetry_quality ∈ {clean ≤ 0.5 s, normal ≤ 1.0 s, gappy ≤ 1.5 s, holed > 1.5 s}`.

> This is the third time the project reached the same conclusion — wet conditions,
> yellow flags, now feed gaps. **Destroying a row at ingest also destroys the evidence
> for whether it should have been destroyed.** From here on: *ingest tags; downstream
> decides.*

## D4 re-examined — the filter was positioned where it could not answer

The 2022 Australian GP had three cautions (laps 2, 23, 39). The filter removed zero laps
and saw only codes 1 and 2. The chain explains it: `require_accurate` removed 189 laps
first, and the caution laps were among them. The filter was not broken; it was
inspecting an already-emptied set. But nothing guarantees `IsAccurate` removes caution
laps in every session, and relying on that silently is a risk.

**Decision.** Track status moves to the front of the chain — it is a pure property of
the lap, needing neither telemetry nor timing validity, and its count only means
something against the full population:

```
exclude_track_status → drop_pit_laps → drop_deleted → require_accurate
                     → max_telemetry_gap (safety net) → pace_gate
```

## New — distance-axis integrity

The qualifying run had reported a maximum distance gap of 434 m; the race run a maximum
time gap of 1.32 s. Covering 434 m in 1.32 s needs 1183 km/h. My diagnostic had compared
the maximum of time with the maximum of distance — two different samples. This matters
because `Distance` is the axis every feature integrates over from P2 onward.

**Decision.** Time and distance are measured on the same sample; the implied speed
(`distance / time × 3.6`), the count of laps implying > 400 km/h (the F1 record is
~372) and the count of negative distance steps are printed as a `DISTANCE AXIS
integrity` block on every ingest run. Any hit blocks P2.
