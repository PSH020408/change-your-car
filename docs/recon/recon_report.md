# Data Reconnaissance Report

_Generated 2026-09-14T15:05:46+00:00_

Sessions surveyed: **10 / 10** · Era: **ground_effect** [2022, 2023, 2024, 2025]
Cache: `/Users/jules/Claude/F1 Virtual Sim/backend/data/cache` — **1.3 GB**

## 1. Coverage matrix

| Season | Event | Ses | Laps raw | Usable | Yield | Drivers | No telem | Weather | Car dt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | Bahrain Grand Prix | R | 1125 | 750 | 66.7% | 20 | 0 | 163 | 240 ms |
| 2022 | Monza | Q | 260 | 78 | 30.0% | 20 | 0 | 78 | 240 ms |
| 2023 | Silverstone | Q | 386 | 100 | 25.9% | 20 | 0 | 94 | 240 ms |
| 2023 | Hungaroring | R | 1252 | 1044 | 83.4% | 20 | 2 | 164 | 240 ms |
| 2024 | Bahrain Grand Prix | Q | 267 | 83 | 31.1% | 20 | 0 | 77 | 240 ms |
| 2024 | Monaco | R | 1237 | 561 | 45.4% | 20 | 4 | 200 | 240 ms |
| 2024 | Spa-Francorchamps | Q | 348 | 127 | 36.5% | 20 | 0 | 79 | 240 ms |
| 2025 | Silverstone | R | 826 | 119 | 14.4% | 20 | 2 | 155 | 240 ms |
| 2025 | Monza | Q | 315 | 99 | 31.4% | 20 | 0 | 78 | 241 ms |
| 2025 | Bahrain Grand Prix | R | 1128 | 922 | 81.7% | 20 | 0 | 158 | 240 ms |

## 2. Sampling resolution

Does the raw sample rate support a 10 m resample grid? The grid is only
defensible where sample spacing stays below it — check the p95 column.

| Season | Event | Lap length | Samples | Spacing median | p95 | max | Top speed |
|---|---|---:|---:|---:|---:|---:|---:|
| 2022 | Bahrain Grand Prix | 5341.7 m | 351 | 14.11 m | 28.08 m | 56.1 m | 299.0 km/h |
| 2022 | Monza | 5755.3 m | 306 | 17.83 m | 36.1 m | 45.73 m | 343.0 km/h |
| 2023 | Silverstone | 5812.6 m | 337 | 16.2 m | 30.08 m | 36.54 m | 328.0 km/h |
| 2023 | Hungaroring | 4320.7 m | 308 | 12.53 m | 26.02 m | 60.9 m | 301.0 km/h |
| 2024 | Bahrain Grand Prix | 5369.5 m | 335 | 14.67 m | 30.66 m | 73.49 m | 319.0 km/h |
| 2024 | Monaco | 3280.7 m | 284 | 9.92 m | 23.63 m | 31.44 m | 288.0 km/h |
| 2024 | Spa-Francorchamps | 6953.4 m | 420 | 15.28 m | 33.44 m | 68.93 m | 319.0 km/h |
| 2025 | Silverstone | 5807.4 m | 329 | 16.44 m | 32.62 m | 97.96 m | 317.0 km/h |
| 2025 | Monza | 5753.1 m | 289 | 18.44 m | 34.59 m | 62.09 m | 348.0 km/h |
| 2025 | Bahrain Grand Prix | 5361.9 m | 348 | 14.0 m | 27.65 m | 78.06 m | 298.0 km/h |

Worst p95 spacing: **36.1 m**. **WARNING — raw spacing exceeds the 10 m grid at speed; interpolation will invent detail**

## 3. Lap yield through the filter chain

Cumulative, in pipeline order. The last row is the real training-set size.

| Filter step | Laps remaining | % of raw |
|---|---:|---:|
| raw | 7144 | 100.0% |
| drop pit in/out | 5831 | 81.6% |
| drop deleted | 5743 | 80.4% |
| accurate only | 5232 | 73.2% |
| within 107% | 3883 | 54.4% |
| green flag only | 3883 | 54.4% |

## 4. Data dictionary

### Laps

_from 2022 Bahrain Grand Prix R_

| Column | dtype | null % | cardinality | example |
|---|---|---:|---:|---|
| `Time` | timedelta64[ns] | 0.0 | 1125 | 0 days 01:04:15.422000 |
| `Driver` | object | 0.0 | 20 | VER |
| `DriverNumber` | object | 0.0 | 20 | 1 |
| `LapTime` | timedelta64[ns] | 2.4 | 1041 | 0 days 00:01:37.880000 |
| `LapNumber` | float64 | 0.0 | 57 | 1.0 |
| `Stint` | float64 | 0.0 | 4 | 1.0 |
| `PitOutTime` | timedelta64[ns] | 94.84 | 58 | 0 days 01:26:08.393000 |
| `PitInTime` | timedelta64[ns] | 94.76 | 59 | 0 days 01:25:43.479000 |
| `Sector1Time` | timedelta64[ns] | 1.96 | 884 | 0 days 00:00:31.285000 |
| `Sector2Time` | timedelta64[ns] | 0.18 | 983 | 0 days 00:00:42.325000 |
| `Sector3Time` | timedelta64[ns] | 0.18 | 866 | 0 days 00:00:24.389000 |
| `Sector1SessionTime` | timedelta64[ns] | 2.13 | 1101 | 0 days 01:04:46.662000 |
| `Sector2SessionTime` | timedelta64[ns] | 0.18 | 1123 | 0 days 01:03:51.046000 |
| `Sector3SessionTime` | timedelta64[ns] | 0.18 | 1123 | 0 days 01:04:15.427000 |
| `SpeedI1` | float64 | 20.8 | 78 | 230.0 |
| `SpeedI2` | float64 | 0.18 | 102 | 254.0 |
| `SpeedFL` | float64 | 5.42 | 81 | 274.0 |
| `SpeedST` | float64 | 13.42 | 123 | 250.0 |
| `IsPersonalBest` | object | 0.18 | 2 | False |
| `Compound` | object | 0.0 | 3 | SOFT |
| `TyreLife` | float64 | 0.0 | 24 | 4.0 |
| `FreshTyre` | bool | 0.0 | 2 | False |
| `Team` | object | 0.0 | 10 | Red Bull Racing |
| `LapStartTime` | timedelta64[ns] | 0.0 | 1106 | 0 days 01:02:34.872000 |
| `LapStartDate` | datetime64[ns] | 0.18 | 1104 | 2022-03-20 15:03:34.889000 |
| `TrackStatus` | object | 0.0 | 9 | 1 |
| `Position` | float64 | 0.18 | 20 | 2.0 |
| `Deleted` | bool | 0.0 | 2 | False |
| `DeletedReason` | object | 0.18 | 2 |  |
| `FastF1Generated` | bool | 0.0 | 2 | False |
| `IsAccurate` | bool | 0.0 | 2 | False |

### Car telemetry

_from 2022 Bahrain Grand Prix R_

| Column | dtype | null % | cardinality | example |
|---|---|---:|---:|---|
| `Date` | datetime64[ns] | 0.0 | 351 | 2022-03-20 16:30:01.322000 |
| `RPM` | int64 | 0.0 | 326 | 11361 |
| `Speed` | int64 | 0.0 | 174 | 282 |
| `nGear` | int64 | 0.0 | 7 | 7 |
| `Throttle` | int64 | 0.0 | 57 | 100 |
| `Brake` | bool | 0.0 | 2 | False |
| `DRS` | int64 | 0.0 | 1 | 1 |
| `Source` | object | 0.0 | 1 | car |
| `Time` | timedelta64[ns] | 0.0 | 351 | 0 days 00:00:00.185000 |
| `SessionTime` | timedelta64[ns] | 0.0 | 351 | 0 days 02:29:01.305000 |

### Position telemetry

_from 2022 Bahrain Grand Prix R_

| Column | dtype | null % | cardinality | example |
|---|---|---:|---:|---|
| `Date` | datetime64[ns] | 0.0 | 353 | 2022-03-20 16:30:01.200000 |
| `Status` | object | 0.0 | 1 | OnTrack |
| `X` | int64 | 0.0 | 341 | -375 |
| `Y` | int64 | 0.0 | 347 | 1379 |
| `Z` | int64 | 0.0 | 135 | -159 |
| `Source` | object | 0.0 | 1 | pos |
| `Time` | timedelta64[ns] | 0.0 | 353 | 0 days 00:00:00.063000 |
| `SessionTime` | timedelta64[ns] | 0.0 | 353 | 0 days 02:29:01.183000 |

### Weather

_from 2022 Bahrain Grand Prix R_

| Column | dtype | null % | cardinality | example |
|---|---|---:|---:|---|
| `Time` | timedelta64[ns] | 0.0 | 163 | 0 days 00:01:03.204000 |
| `AirTemp` | float64 | 0.0 | 35 | 25.6 |
| `Humidity` | float64 | 0.0 | 20 | 17.0 |
| `Pressure` | float64 | 0.0 | 5 | 1010.2 |
| `Rainfall` | bool | 0.0 | 1 | False |
| `TrackTemp` | float64 | 0.0 | 51 | 32.3 |
| `WindDirection` | int64 | 0.0 | 92 | 346 |
| `WindSpeed` | float64 | 0.0 | 9 | 0.5 |

## 5. Tyre compounds observed

| Compound | Laps |
|---|---:|
| SOFT | 2272 |
| HARD | 2024 |
| MEDIUM | 1887 |
| INTERMEDIATE | 961 |

## 6. Issues to resolve before P2 (feature design)

- `2022 Monza Q` — low lap yield (30.0%); filters may be too aggressive here
- `2023 Silverstone Q` — low lap yield (25.9%); filters may be too aggressive here
- `2023 Hungaroring R` — 2 driver(s) without usable telemetry: GAS, OCO
- `2024 Bahrain Grand Prix Q` — low lap yield (31.1%); filters may be too aggressive here
- `2024 Monaco R` — 4 driver(s) without usable telemetry: HUL, MAG, OCO, PER
- `2024 Spa-Francorchamps Q` — low lap yield (36.5%); filters may be too aggressive here
- `2025 Silverstone R` — 2 driver(s) without usable telemetry: COL, LAW
- `2025 Silverstone R` — low lap yield (14.4%); filters may be too aggressive here
- `2025 Monza Q` — low lap yield (31.4%); filters may be too aggressive here

## 7. Decisions this report should settle

- [ ] 10 m resample grid — confirmed or revised to the measured spacing
- [ ] Which sessions enter training (Q only / Q+R / +FP)
- [ ] Filter chain final order and thresholds
- [ ] Setup-proxy feature list (which observable channels stand in for setup)
- [ ] Full-era download size projection and whether it fits the disk budget
