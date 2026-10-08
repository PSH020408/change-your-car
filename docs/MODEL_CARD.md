# Model card — segment delta model `v2026.09.17-1`

## What it predicts

For one segment of one lap: the time delta (seconds) versus the session's reference
time for that segment, as three quantiles (q10 / q50 / q90). The HUD uses the
*difference* of two predictions — q50 under the new conditions minus q50 under the
baseline lap's conditions — so a setup-only change leaves the ML term at exactly 0.

## Data

- 184 sessions: every qualifying and race of 2022–2025 (92 events × Q/R). Sprint and
  practice sessions excluded by decision (few sessions, rules changed yearly, unknown
  fuel loads).
- 2,611,636 segment rows after feature building; 2,311,785 enter training after removing
  rows with no target, non-dry laps, cruise laps, telemetry holes inside the segment and
  laps under safety car / red flag.
- Hold-out circuit: Miami (never in training, never in calibration).

## Features — only what is known before the lap is driven

numeric: `tyre_life`, `lap_number`, `track_temp_c`, `air_temp_c`, `lap_effort_index`,
`gap_ahead_s`, `segment_length_m`, `segment_min_radius_m`, `segment_reference_s`
categorical: `segment_kind`, `session`, `compound`, `fresh_tyre`, `driver`, `chassis`,
`season`, `segment_sector`, `segment_is_kink`

Refused by the feature specification (leakage): anything measured *during* the lap
(speeds, brake points, throttle, DRS, pace ratios, lap time), driver-bias features
computed over the whole dataset, stint, and a track-evolution feature that was tried and
removed (it blurred the fuel probe without improving the gate metric).

## Method

Three LightGBM regressors with pinball loss (600 trees, learning rate 0.03, 31 leaves,
min 200 samples per leaf), non-crossing enforced, conformal margin from out-of-fold
residuals. GroupKFold by `season|event` (5 folds). A Ridge model is trained as the
linear floor the GBM must beat; a naive median as the sanity floor.

Level 2 (session-level): a sector-level Ridge with session bootstrap explaining a
session's reference sector time versus the circuit's best, from session type, season
and one temperature term. Result on 165 sessions: race +4.75 %, season −0.93 %/year,
track temperature +0.0015 %/°C (not distinguishable from 0). **Reported, not applied.**

## Results

| | CV (out-of-group) | Miami (unseen) |
|---|---|---|
| Segment MAE q50 | 0.062 s (median 0.032) | 0.078 s |
| Absolute lap MAE | 0.720 s | 0.580 s |
| Counterfactual lap MAE, clean-air push laps vs representative lap | **0.474 s** (median 0.345; "no change" 0.659) | 0.443 s (0.589) |
| Skill | 28 % | 25 % |
| Noise floor / skill ceiling | 0.339 s / 49 % | — |
| Coverage of the 80 % band, raw → calibrated | 0.74 → 0.80 | 0.75 → 0.81 |
| Ridge (must be beaten) | 0.076 s segment, 0.639 s counterfactual | — |

By segment kind (MAE): kink 0.029 · high-speed corner 0.056 · straight 0.058 ·
medium-speed corner 0.085 · **low-speed corner 0.111**. By session: Q 0.049, R 0.063.

Monotonicity probes: tyre life +5 laps → +0.0044 s per segment, slower in 79 % of rows;
lap number +10 (fuel burn) → −0.013 s, faster in 96 % of rows.

Permutation importance (top): segment reference time, segment length, minimum radius,
lap number, segment kind, chassis, gap ahead, tyre life.

## Gates

| Gate | Threshold | Result |
|---|---|---|
| segment_mae | ≤ 0.15 s | **pass** (0.062) |
| clean_push_lap_within_noise_floor | ≤ 1.25 × floor (0.424 s) | fail (0.474) |
| clean_push_lap_skill | ≥ 35 % | fail (28 %) |
| unseen_track_clean_push_lap_skill | ≥ 25 % | fail (24.8 %) |
| coverage_80_calibrated | 0.70–0.90 | **pass** (0.81) |
| beats_ridge | — | **pass** |
| tyre_life_up_slower | — | **pass** |
| lap_number_up_faster_in_race | — | **pass** |

5 of 8. The three failing gates are lap-level targets that were frozen as aspirational
after the gate history documented in `DESIGN.md`. The model was accepted manually with
that reason recorded in `data/artifacts/models/latest.json`; the previous model
(66 sessions, 33 events) had scored 26 % skill and had never seen most of the events
and the 2024–25 chassis the HUD now serves.

## Known limitations

- **Skill ceiling.** Half of lap-to-lap variation is not predictable from pre-lap
  information. Do not read the band as narrower than it is.
- **Extrapolation on tree models.** Qualifying data never contains tyres older than ~8
  laps; asking for a 20-lap tyre on a qualifying baseline is answered from the nearest
  thing the model saw and understates degradation. The HUD warns and suggests a race
  baseline.
- **Dry only.** Trained on dry laps; wet and intermediate conditions are handled by a
  grade-C physics grip multiplier, not by the model.
- **No setup in the data.** The ML part cannot know anything about wings, ride height
  or suspension; those are physics, graded separately.
- **Low-speed corners** are the weakest kind; traction was the suspected missing
  signal — tested below, it is not enough.

## Experiments that did not replace the model

### v2026.10.08-1 — compound hardness and physics-engine segment features (rejected)

Two additions, both legitimate pre-lap information:

- `compound_hardness`: the C-compound Pirelli nominated for each label that weekend
  (`configs/tyres.yaml`, 92 events from Pirelli press releases), because SOFT is a C3 at
  Bahrain and a C5 at Monaco.
- eight segment features from the calibrated point-mass engine (`docs/PHYSICS_ENGINE.md`):
  the share of the segment spent traction-, brake-, grip- and power-limited, gradient,
  climb, DRS share and the simulated apex speed.

The acceptance rule was fixed before training: adopt only if the clean-air
counterfactual lap MAE improves on 0.474 s **and** the low-speed-corner segment MAE
improves on 0.111 s.

| Metric | v2026.09.17-1 (kept) | v2026.10.08-1 |
|---|---|---|
| Segment MAE, q50 (CV) | 0.062 s | **0.059 s** |
| Low-speed corner MAE | 0.111 s | 0.110 s |
| Counterfactual lap MAE, clean-air push | 0.474 s | 0.478 s |
| Skill | 28 % | 28 % |
| Unseen circuit (Miami) counterfactual MAE | 0.443 s | **0.436 s** |
| Coverage 80 %, calibrated | 0.80 | 0.80 |
| Gates | 5/8 | 6/9 |

The physics features ranked 5th–7th in permutation importance (power share, brake
share, apex speed) and lowered the per-segment error; the lap-level change the HUD
shows did not move, and the low-speed corner did not either. Compound hardness ranked
outside the top ten. The new sign gate — "+10 °C track temperature should hurt C4+
more than C1–C2" — **failed with the opposite sign** (soft +0.0001 s/segment, hard
+0.0021 s): in this data the hard end of the range is the temperature-sensitive one,
which is consistent with hard compounds needing heat to work and graining when cold,
and inconsistent with the assumption the gate encoded. The gate stays as written and
the model was not registered. The features remain in gold and in `model.yaml` so the
experiment reproduces with `make train`; the API serves them to whichever model is
registered, and the registered model ignores them.
