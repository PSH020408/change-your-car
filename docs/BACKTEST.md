# Back-test — qualifying lap → race lap through the sliders

_Run 2026-09-22 with model `v2026.09.17-1`; `make backtest` reproduces it from the
baseline store in about 70 s. Raw per-lap results in `data/artifacts/backtest/laps.parquet`._

## Question

Every component of the simulator was validated on its own — the segment model on a
held-out circuit, the physics coefficients by regression. Nobody had tested the **sum**.
The one natural experiment the data offers: the same driver drives the same circuit in
qualifying (light, fresh tyres, flat out) and in the race (heavy, older tyres). Take the
qualifying representative lap as the baseline, set only what the race lap changes —
fuel for that lap number, the tyre it was on, the race session's temperatures — and
compare the simulated lap time with the race lap actually driven.

## Population

2,447 laps · 75 events · 30 drivers. Dry qualifying lap → dry race first-stint
laps; clean push laps in clean air (≥ 2.5 s to the car ahead), telemetry clean or normal,
lap ≥ 3. Wet weekends are excluded on purpose: a wet qualifying against a dry race (2022
Canada, 2025 Las Vegas) or the reverse (2024 Canada, 2025 Australia) produced 10–20 s
errors in a first run and is a different question, not a hard case of this one.

## Result

| Predictor | MAE (s) | Bias (s) | Median abs error (s) |
|---|---|---|---|
| Qualifying lap unchanged | 5.63 | -5.54 | 5.55 |
| Qualifying + physics fuel term only | 3.67 | -3.50 | 3.51 |
| **Simulator** (fuel physics + tyre/temperature ML) | **3.27** | **-3.08** | 3.08 |
| Qualifying × (1 + 4.75 % race shift from level 2) | 2.00 | -1.54 | 1.83 |
| **Simulator + measured race-pace offset** (leave-one-event-out) | **1.21** | **+0.00** | **0.91** |
| Simulator + circuit's own offset from other seasons | 1.32 | +0.03 | 0.76 |
| Oracle: event's own median Q→R gap (upper bound, not a competitor) | 0.59 | +0.06 | 0.40 |

**Race-pace offset** (race lap − simulated flat-out lap): median **+3.05 s**, p10–p90
+1.57 … +4.89 s. By compound the simulator is 2.8 s optimistic on hard,
3.1 s on medium and 3.6 s on soft — the softer the tyre, the more it is
managed in a race.

## What it means

1. A race first-stint lap is ~5.5 s slower than the same driver's qualifying lap. The
   simulator explains about 2.5 s of it — 2.0 s of fuel (physics, grade B) and 0.4 s of
   tyre age and temperature (ML, grade A) — and is therefore **3.1 s optimistic**: it
   answers "this lap, in these conditions, driven flat out", which is what the sliders
   ask, not "this lap at race pace".
2. The remaining 3.0 s is **race-pace management** — engine modes, lift-and-coast,
   tyre saving. No slider represents it and no pre-lap feature predicts its size for a
   single lap, but its **median is stable across circuits** (p10–p90 within ±1.7 s),
   so it can be *measured* and applied as an explicit term. With that term the
   simulator predicts the race lap to **1.21 s MAE with no bias**, 40 % better than the
   best naive rule and within 0.6 s of the oracle.
3. The 80 % band the HUD shows covers the race lap in only 4 % of these cases: it is a
   band for the flat-out question, not for race pace. A race-pace mode would carry the
   offset's own spread (±1.6 s) in addition.
4. The oracle at 0.59 s says that once the event's race-pace level is known, driver-to-driver
   variation is small. The gap between 0.76 s (circuit offset) and 0.59 s is what a
   per-event feature (tyre allocation, track evolution, temperature that weekend) could
   still buy.

## Consequences for the product

- The HUD's setup and conditions answers are unchanged: they are flat-out counterfactuals
  and this test does not contradict them.
- A **race-pace term** (grade A — measured, `+3.05 s` median, per-circuit where three seasons
  exist) is now justified for any race-level feature, first of all the planned
  tyre-strategy mode. It will be labelled as what it is: measured, not simulated.
- The simulator now has a whole-system accuracy figure to quote next to the segment MAE.

## Per event (n ≥ 10, simulator without offset; sorted by MAE)

| Event | Laps | MAE (s) | Bias (s) |
|---|---|---|---|
| 2023 qatar | 20 | 1.07 | -0.99 |
| 2025 italian | 107 | 1.68 | -1.68 |
| 2022 sao paulo | 30 | 1.87 | +0.24 |
| 2025 azerbaijan | 59 | 2.00 | -2.00 |
| 2025 canadian | 63 | 2.05 | -2.05 |
| 2022 miami | 68 | 2.05 | -2.05 |
| 2023 british | 76 | 2.14 | -1.13 |
| 2024 qatar | 103 | 2.17 | -2.17 |
| 2025 sao paulo | 39 | 2.22 | -2.20 |
| 2024 dutch | 59 | 2.31 | -2.31 |
| 2023 las vegas | 44 | 2.34 | -2.34 |
| 2023 austrian | 30 | 2.47 | -2.47 |
| 2024 british | 49 | 2.56 | -2.21 |
| 2023 miami | 41 | 2.58 | -2.58 |
| 2022 austrian | 46 | 2.60 | -2.60 |
| 2024 miami | 36 | 2.64 | -2.64 |
| 2022 dutch | 31 | 2.73 | -2.73 |
| 2024 italian | 50 | 2.78 | -2.78 |
| 2023 saudi arabian | 24 | 2.85 | -2.85 |
| 2024 australian | 30 | 2.86 | -2.86 |
| 2023 italian | 32 | 2.89 | -2.89 |
| 2024 las vegas | 17 | 2.89 | -2.89 |
| 2022 australian | 51 | 2.90 | -2.90 |
| 2023 australian | 11 | 3.05 | -3.05 |
| 2024 austrian | 82 | 3.13 | -3.13 |
| 2022 saudi arabian | 51 | 3.16 | -3.16 |
| 2022 italian | 64 | 3.16 | -3.16 |
| 2025 emilia romagna | 41 | 3.21 | -3.21 |
| 2024 saudi arabian | 48 | 3.27 | -3.27 |
| 2025 japanese | 54 | 3.27 | -3.27 |
| 2025 qatar | 11 | 3.32 | -3.32 |
| 2024 azerbaijan | 113 | 3.33 | -3.33 |
| 2022 hungarian | 20 | 3.35 | -3.35 |
| 2025 austrian | 36 | 3.51 | -3.51 |
| 2023 azerbaijan | 14 | 3.71 | -3.71 |
| 2024 abu dhabi | 59 | 3.77 | -3.77 |
| 2025 dutch | 15 | 3.79 | -3.79 |
| 2022 abu dhabi | 33 | 3.90 | -3.90 |
| 2024 emilia romagna | 100 | 3.94 | -3.94 |
| 2023 spanish | 21 | 3.99 | -3.99 |
| 2022 azerbaijan | 41 | 4.06 | -4.06 |
| 2023 abu dhabi | 14 | 4.13 | -4.13 |
| 2025 saudi arabian | 48 | 4.14 | -4.14 |
| 2025 hungarian | 35 | 4.18 | -4.18 |
| 2024 united states | 51 | 4.19 | -4.19 |
| 2025 united states | 19 | 4.40 | -4.40 |
| 2024 bahrain | 23 | 4.49 | -4.49 |
| 2022 united states | 10 | 4.54 | -4.54 |
| 2022 french | 45 | 4.57 | -4.57 |
| 2025 abu dhabi | 14 | 4.58 | -4.58 |
| 2025 singapore | 49 | 4.94 | -4.94 |
| 2024 hungarian | 14 | 4.98 | -4.98 |
| 2022 bahrain | 35 | 5.27 | -5.27 |
| 2024 singapore | 20 | 5.81 | -5.81 |
| 2024 spanish | 12 | 6.26 | -6.26 |
| 2024 japanese | 18 | 6.31 | -6.31 |
| 2022 belgian | 17 | 6.80 | -6.80 |
| 2023 canadian | 13 | 7.97 | +7.97 |
