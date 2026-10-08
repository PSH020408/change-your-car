# Tyre-strategy mode

_Added 2026-10-08. Scores a race plan — a sequence of compounds and stint lengths —
with the same lap model the HUD already uses, against the strategies teams really ran.
Back-tested on 677 drivers over 40 green-flag races: race total within 0.39 % (median)._

## 1. What it answers

"If this driver had run SOFT 15 → HARD 21 → SOFT 21 instead, how long would the race have
taken, and how does that compare with what the field actually did?" The mode is visible
on the HUD only for race sessions, and only for weekends where the race timing was
measured (`make race-timing`). The default plan is the winner's real strategy; the
cards below it are every distinct sequence run that day, scored with the same model, so
the user's plan is compared like with like.

## 2. The lap model

One race lap is the baseline's representative race lap plus two explicit changes:

```
t_i = T0 + slope · (fuel(n_i) − fuel(n0)) + [ ML(c_i, a_i) − ML(c0, a0) ]
race = Σ t_i + stops × pit_loss
```

- `T0`, `n0`, `c0`, `a0` — lap time, lap number, compound and tyre age of the
  representative lap (already a race lap, so the race-pace offset measured in
  `docs/BACKTEST.md` is inside it and is not added again).
- `fuel(n)` — the same fuel-per-lap model as the fuel slider; `slope` comes from the
  physics engine (a +10 kg probe on the lap's calibrated solver) and falls back to the
  P3 coefficient table where a circuit has no racing line.
- `ML(c, a)` — the registered model's q50 delta summed over the lap's segments for
  compound `c` at age `a`, with the session's temperatures. Only the *difference* to the
  representative lap enters, so the model's absolute bias cancels.
- `pit_loss` — measured for that race, not assumed (§3).

A plan is refused, not approximated, when a stint exceeds the longest stint any driver
ran on that compound that weekend (the larger of the tyre envelope and the measured
stint maximum), or when the stint lengths do not add up to the race distance.

## 3. The race-timing sidecar

`backend/pipeline/ingest/race_timing.py` reads each race from the FastF1 cache (timing
only, no telemetry, offline) and writes `race_timing.json` per race plus a small
`race.json` into the baseline store for the API:

| Field | How it is measured |
|---|---|
| `race_laps` | Leader's lap count |
| `neutralised` (per lap) | Track status contains safety car, VSC or red flag (codes 4–7) |
| `green_share` | Share of laps run with no neutralisation |
| `pit_loss` | For every stop made under green: in-lap + out-lap − 2 × median of up to three clean green laps on each side of the stop (median, p10, p90 and count over the race) |
| `stint_max` | Longest stint per compound that day, so the envelope can be widened to what was actually driven |

Pit loss over 90 races: median **22.3 s**, from 18.2 s (Zandvoort) to 42.9 s (Singapore
2022, a 2022 pit-lane speed limit and layout). Those numbers are what the HUD prints
next to "pit loss … measured (n stops)".

## 4. Back-test (`make strategy-backtest`)

Population: every classified finisher in every race with green share ≥ 0.90, whose laps
are all timed (lap 1 never is), whose stints add up to the laps they ran (lapped cars
are scored on their own lap count), and whose laps 2..N are at least 80 % clean. Laps
run under a safety car, VSC or red flag are removed from **both** the real and the
predicted total — a neutralised lap is 40 s slow on the real side and nothing on the
predicted side — and a stop made under neutralisation is not charged its pit loss.

Gates, written before the first run:

| Gate | Criterion | Result | |
|---|---|---|---|
| S1 race total | median \|error\| ≤ 0.5 %, p90 ≤ 1.5 % | **0.389 %** (20.7 s) median, p90 1.03 %, bias −0.29 % | PASS |
| S2 strategy order | within a race, the predicted order of distinct sequences (group medians) agrees with the real order in ≥ 70 % of pairs | **74.4 %** of 172 pairs | PASS |
| S3 pit loss | reported for ≥ 80 races, median in 18–30 s | 90 races, median 22.3 s | PASS |

By number of stops the median error is flat: 0.39 % (1-stop), 0.39 % (2), 0.47 % (3),
0.36 % (4), 0.39 % (5). 44 driver-races were skipped, almost all for a green share
below 0.90 (most of 2022) — and 2024 Baku, whose final lap carries no time in the feed.

The 0.39 % is not a lap-model accuracy claim. Fuel, tyre age and compound are the
large, slow terms of a race; the model gets those from data, and the rest is listed on
the HUD as ignored.

## 5. What the mode ignores — and says so

Safety cars and red flags · traffic and dirty air · out-lap warm-up and in-lap push ·
the first lap (start, turn 1) · pace management that changes with the plan (a driver on
a one-stop manages more than one on a three-stop; the representative lap carries one
level of management, the driver's own). The HUD prints this list under every score.

## 6. Files

| Path | Role |
|---|---|
| `backend/pipeline/ingest/race_timing.py` | Sidecar: per-lap timing, neutralisation, pit loss, stint maxima |
| `backend/app/services/strategy.py` | Lap model, refusals, cards from the real strategies, `POST /api/strategy` |
| `backend/pipeline/eval/strategy_backtest.py` | Population, gates S1–S3, `data/artifacts/strategy/strategy_backtest_report.json` |
| `frontend/src/components/hud/StrategyPanel.tsx` | Stint editor, race time, gap to the best card, lap-time chart, the real strategies |
| `backend/tests/test_strategy.py` | Parsing and fitting; three end-to-end tests on the stored Bahrain race |
