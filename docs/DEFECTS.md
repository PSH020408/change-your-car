# Defects and lessons

_Every mistake that changed the design, in the order it was found. "Symptom" is what
the pipeline printed; "cause" is what it turned out to be; "change" is what the code or
the config does differently now. Kept because the pattern of mistakes says more about a
project than its final numbers._

| # | Stage | Symptom | Cause | Change |
|---|---|---|---|---|
| 1 | Recon | Planned a uniform 10 m telemetry grid | Car telemetry arrives at a fixed 240 ms, so spacing is a function of speed (4 m in a hairpin, 20 m at 300 km/h); a 10 m grid would interpolate every straight | No grid for features (raw samples integrated per segment); 20 m grid for display only, with an `interpolated` mask |
| 2 | Recon | A 40 m "telemetry gap" filter was proposed to catch dropouts | Chosen by reasoning, not from the distribution; the sample's median worst gap was 67 m | Threshold sweep table printed on every ingest run; thresholds are set from it |
| 3 | Recon | Same filter, now 1.0 s on the time axis, removed 36 % of a race | The worst-gap distribution is unimodal (p50 0.88 s, max 1.32 s): a 3-sample dropout is normal feed behaviour, there is no clean/broken split to threshold | Demoted to a 2.0 s safety net; the real signal is a `telemetry_quality` tag (clean / normal / gappy / holed) kept in bronze |
| 4 | Recon | The 107 % pace gate kept 14 % of a wet-to-dry race | The reference was the session best, set in different grip | Conditions-matched reference: rolling median of the top laps in a ±5-lap window; wet/inter laps tagged, not dropped |
| 5 | Recon | Track-status filter removed 0 laps in a session with three safety cars | It ran *after* `require_accurate`, which had already removed those laps; the diagnostic also miscounted yellow flags as excludable | Track status runs first on the full population; every filter step logs its removals; yellow is tagged, never dropped |
| 6 | Recon | Red flag (code 5) missing from the exclusion list | Copied list | `["4","5","6","7"]` |
| 7 | Recon | Diagnostic compared the max time gap with the max distance gap — implied 1183 km/h | The two maxima came from different samples | Time and distance measured on the same sample; implied speed and negative-distance-step counters printed as a `DISTANCE AXIS` block |
| 8 | Recon | Schema promised `brake_pct: float` | FastF1 brake is on/off | `brake_on: bool`; the HUD draws a band, not a curve |
| 9 | Recon | DRS treated as a setup variable | DRS opening depends on being within 1 s of the car ahead, a race situation | DRS is a track property (zones); the simulator assumes "open in the zone" |
| 10 | Recon | FastF1 cache landed in `backend/data/cache` instead of `data/cache` | `Path(...).parent.parent.parent` saturates to `.` on a relative path | Paths made explicit in `scope.yaml` and read from there |
| 11 | Warm | 426 sessions marked failed in ten minutes | FastF1 swallows its rate-limit error and raises `DataNotLoadedError` | Treated as the rate-limit signal; 60 s pacing; retry failed sessions; stop after 5 consecutive failures |
| 12 | Segment | Sector boundaries 100 % null | The ensemble line has no session-time axis | Per-lap sector boundaries mapped onto the ensemble axis, median taken |
| 13 | Physics check | Fuel slope came out at −0.8 s/kg | Per-driver regression: lap number and tyre life are perfectly collinear within a stint | Field-pooled regression with driver fixed effects (recovers 0.030 on synthetic data; 0.0294 measured) |
| 14 | Physics check | Aero-efficiency slope had the wrong sign | Raw slope confounded by car quality (fast cars are fast everywhere) | Partial slope holding pace fixed: 8 of 9 circuits negative, as the physics predicts |
| 15 | ML | LightGBM "categorical feature does not match" during permutation importance | Shuffling a column dropped its category dtype | `iloc[perm].set_axis(index)` preserves dtype |
| 16 | ML | Model config paths broke when run from `backend/` | Relative paths written from the repo root | Paths relative to `backend/` throughout |
| 17 | ML | Level 2 penalised cross-season comparisons by 59 % | It compared segment *indices* across seasons, and segments are re-cut per season | Level 2 works on sectors, which are stable |
| 18 | ML | Lap gate: absolute lap MAE 1.5 s against a 0.30 s target | Lap-specific offsets accumulate over 28 segments; Ridge scored the same | Gate redefined on the counterfactual quantity the HUD shows |
| 19 | ML | Counterfactual error 1.32 s | Traffic | `gap_ahead_s` feature; gate population restricted to clean-air push laps |
| 20 | ML | Gate compared against the driver's best lap while the noise floor compared two ordinary laps | Inconsistent baselines | Baseline = the driver's representative (median) clean push lap; 0.71 → 0.509 s |
| 21 | ML | Track-evolution feature ranked 4th in importance but did not move the gate and blurred the fuel probe (−0.35 → −0.09) | It absorbed the lap-number signal | Removed from the model, kept in gold |
| 22 | Reconstruction | The *real* lap was clamped by its own envelope | Anchor used central differences at the edges | Two-point v² acceleration anchor |
| 23 | Reconstruction | Integrated trace 0.5 % short of the official lap time | The two partial 240 ms intervals at the start/finish line | Time axis scaled to the official lap time (factor ≈ 1.0053) |
| 24 | Reconstruction | Requested +1.200 s, achieved +1.256 s | Gaps between segments belonged to nobody | Segment intervals tiled to cover the lap |
| 25 | Reconstruction / ML | "No usable lap for VER" in qualifying | The clean-air gap rule was applied to qualifying, where there is no car ahead | Gap rule is race-only (also fixed the ML gate population) |
| 26 | API | Tyre-age band ±2 s for a 0.3 s effect | Summing 23 per-segment bands assumes every segment errs the same way | Lap band = ln(5) × measured counterfactual MAE, shared over segments by their time |
| 27 | API | Track +10 °C produced −1.9 s through level 2 | Air and track temperature collinear in a 57-session fit | Level 2 reported, not applied; refit with one temperature term on 165 sessions: coefficient indistinguishable from 0 |
| 28 | HUD | Headline delta (+1.230) ≠ physics sum (+1.200); +4.355 vs +4.171 in intermediate conditions | The headline integrates the rebuilt trace; the rebuild lands past the requested sum, more so under low grip | Shown as a fourth "trace rebuild" term so the split adds up; backend fix pending |
| 29 | Warm | Three sessions marked "done" had laps but no telemetry in the cache | Download cut mid-session, ledger written anyway (6.8 MB delta vs ~110 MB normal) | Entries reset to failed and re-warmed; cache size delta is now a sanity signal |
| 30 | Ingest | 2024 Baku R: first 0 laps, then 16,120 laps (18× the real 878) | FastF1 shipped a "minimal driver list" — team blank, driver abbreviation blank — so first the rows were dropped, then every driver's lap N shared one `lap_uid` and the telemetry merge multiplied | Driver and team borrowed from sibling sessions of the same weekend; a session with duplicate `lap_uid`s is refused outright |
| 31 | Segment | Seven circuits failed the hidden-corner check (a 78 m radius inside a "straight" at Zandvoort) | The auto-scaled smoothing window is too wide for some circuits; and the calibration grid was swept with a different `min_gap` than the real run used | Per-circuit overrides in `circuits.yaml` (window, threshold, `min_gap` pinned); 20/20 circuit-seasons pass |
| 32 | HUD / ML | Tyre age 2 → 20 laps on a qualifying baseline predicted only +0.13 s | Qualifying data never contains tyres past ~8 laps; trees cannot extrapolate | First an engineer-log warning; then (2026-09-22) the slider itself is bounded by the longest stint run on that compound that weekend, compounds nobody raced are marked, and the log warns only past that bound |

## Patterns

- **Tag, don't drop, at ingest.** Reached three times independently (wet laps, yellow
  flags, feed gaps): destroying a row at ingest also destroys the evidence for whether
  it should have been destroyed.
- **A threshold is a claim about a distribution.** Print the distribution first (the
  sweep tables in ingest, segment and train exist for this reason).
- **Compare like with like.** Three defects (#7, #17, #20) were comparisons between
  quantities from different populations or axes.
- **Show the residual.** When a stage cannot honour a request exactly (#24, #26, #28),
  the honest move is a visible term, not a hidden correction.
