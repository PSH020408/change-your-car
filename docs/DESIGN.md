# Design — from idea to deployment

_This is the narrative version of the project: what I wanted, what the data allowed, the
decisions that followed, and what I would do differently. The numbers are the ones the
pipeline printed; each has a log or a test behind it._

## 1. The idea

Every F1 broadcast has the moment when a driver is told the car will not be changed and
the radio goes quiet. I wanted the opposite: a page where anyone can take a real lap from
the ground-effect era (2022–2025), *change the car*, and see — on the circuit, in the
telemetry, in the replay — where that decision pays and where it costs.

Two constraints shaped everything:

- **No setup data exists in public.** Teams never publish wing angles, ride heights or
  spring rates. FastF1 gives laps, telemetry, tyres, weather and speed traps — not setups.
- **It has to run in a browser, for free, fast.** No 3D, no GPU, no heavy chart library.

## 2. The concept that survived the data

A first sketch would have had a machine-learning model "learn the car". The
reconnaissance (see `recon/DECISIONS.md`) killed that in a day: there is nothing to learn
setup effects *from*. What the data does contain, with real variation, is tyre compound
and age, track and air temperature, fuel (through lap number), the driver, the chassis
and the season.

So the architecture split along that line:

| Axis | Owner | Why |
|---|---|---|
| Front/rear wing, ride height, suspension, fuel, weather grip | **Physics** — explicit formulas | No data; but the physics is well known and the coefficient *sizes* can be checked against our own laps |
| Tyre age, compound, temperatures | **ML** — learned from 2.6 M segment rows | The data varies along these axes every session |

Both produce a per-segment time delta with an interval and are added. The HUD then shows
the sum *and* the two parts, plus the two corrections the reconstruction introduces.

A second decision followed from the first: **every physics coefficient carries a grade**.
A — learned from data. B — physics, but the coefficient was checked on our own laps
(fuel: literature 0.030 s/kg/lap, measured 0.0294; aero crossover speed 150 km/h,
fitted 145; mechanical friction 1.6, fitted 1.5). C — literature only, cannot be
verified with public data (ride height, suspension stiffness, wet grip multipliers).
The grade is on the slider, so a user knows how much to trust each axis before touching it.

## 3. Why segments

A wing change is not "0.1 s per lap"; it is a gain in every corner and a loss on every
straight, in proportions that depend on speed. Aerodynamic share of grip goes as
α = v² / (v² + v0²): about 0.18 in a hairpin, 0.72 in a fast sweeper. On a straight,
top speed goes as Cd^(−1/3). So the lap is cut into segments (straight / kink /
low-, medium-, high-speed corner) and both the physics and the ML work per segment.
That is also what makes the circuit map meaningful: the colours are real per-segment
predictions, not a lap total smeared along the track.

Segmentation itself is physics-checked: the curvature threshold was settled from a sweep
against published corner counts (0.0025 m⁻¹ ≈ 400 m radius ≈ 1.8 g at 300 km/h), and
every cut circuit must pass two invariants — no corner may hide inside a segment
labelled "straight", and no corner may imply more than 6.5 g lateral. Seven circuits
failed that check with the automatic smoothing window and received per-circuit
overrides chosen from a calibration grid (Zandvoort, Baku, COTA, Red Bull Ring, Suzuka).

## 4. Why quantile GBMs, and why not deep learning

The target is tabular: a handful of pre-lap features per segment, 2.6 M rows. Gradient
boosted trees are the right tool for that shape of data, and three quantile models
(q10/q50/q90) with conformal calibration give an honest band instead of a point.

The stronger argument against a bigger model is the **noise floor**. Two consecutive
clean push laps by the same driver on the same tyres differ by 0.339 s. Nothing known
before the lap can explain that difference, which caps the achievable skill at 49 %.
The model sits at 28 % (58 % of the ceiling). The remaining gap is information, not
capacity — more model would not move the floor. (More *features* might: compound ×
temperature interactions, driver × circuit history, low-speed-corner specifics.)

Two levels: level 1 predicts the segment delta versus the session reference and cannot,
by construction, see anything constant within a session (the weather). Level 2 was
meant to explain the session reference itself from temperature, session type and
season. Refit with a single temperature term on 165 sessions, the temperature
coefficient came out at +0.0015 % per °C — indistinguishable from zero. So the
session-wide temperature effect is not measurable in this data; in-range temperature
effects live in level 1 (which has temperature features), and level 2 is reported in
the HUD but never added.

## 5. Gates, and the honest history of the lap gate

Every stage has a gate; a model that fails one is not registered as latest. The
segment-level gate (MAE ≤ 0.15 s) was easy. The lap-level gate was rewritten four times,
and the rewrites are part of the record:

1. Absolute lap MAE ≤ 0.30 s — chosen by feel. Result 1.5 s: lap-specific offsets
   (traffic, management) accumulate over 28 segments. Ridge scored the same, so the
   information was simply not there.
2. Counterfactual error (the quantity the HUD actually shows) ≤ 0.30 s — 1.32 s.
   Traffic → a gap-to-car-ahead feature and a clean-air push population → 0.71 s.
3. The floor had never been measured. Measured it: 0.352 s (later 0.339 s on the full
   data). Gate = 1.25 × floor, skill ≥ 35 %.
4. The floor compared two ordinary laps; the gate compared against the driver's *best*
   lap, which is lucky by definition. Baseline = the representative (median) lap.
   0.71 → 0.509 s.
5. Frozen. 66 → 184 sessions improved it to 0.474 s; the three lap-level gates still
   fail and are kept as aspirational, and the acceptance reason is written into the
   registry.

## 6. Reconstruction: from numbers back to a trace

The HUD does not draw a bar chart; it draws a speed trace. Each segment's real speed is
warped (bisection on a shape-weighted multiplier — apex for corners, top speed for
straights) until its integrated time matches the requested delta; the result is clamped
to a g-g envelope (lateral, traction, braking) anchored on the real lap with 15 %
headroom; throttle, brake, gear and DRS are re-synthesised from it. Where the envelope
refuses part of a requested gain, that amount is shown as "envelope refused". Where the
rebuild lands past the requested sum (it does: +2.5 % on a fuel change, +4 % in
intermediate conditions), that residual is shown as "trace rebuild". The four parts add
up to the headline delta exactly; nothing is silently absorbed.

## 7. The HUD

One screen, three columns: change the car on the left, see *where* it changed in the
middle (circuit map, replay of real vs simulated car with the live gap, telemetry with
Δspeed and running Δtime), read *how much and why* on the right (headline delta with
band, the four-way split, sectors, physics state, the largest movers, the engineer log).
Everything is hand-drawn SVG; the map colours segments with a `pathLength` dash trick so
no geometry is computed in the browser. Colours passed a colour-vision-deficiency check
on the dark surface. First load: 121 kB.

## 8. Deployment

One container serves the API and the statically exported HUD on the same origin (no
CORS, one URL). The baselines (184 session JSONs, 93 MB) and the registered model
(5.5 MB) are committed so the service builds straight from the repository; a push to
`main` redeploys. It is live at https://change-your-car.onrender.com on Render's free
tier. The trade-off is explicit: the instance sleeps after 15 minutes without visitors
and the next request waits 30–60 s for the container to start; on its shared CPU a
simulation then takes ~300 ms (15 ms on a laptop). I considered three hosts — Hugging Face Spaces (Docker spaces now need a paid
plan), Google Cloud Run (effectively free at this traffic, but needs a card on file) and
Render (free, no card, sleeps). For a portfolio prototype the visible cold start was the
cheapest honest option; `make deploy` keeps the Cloud Run path ready for the day the
traffic justifies it.

## 9. What I would do next

_Done since first deployment (2026-09-22): the tyre-age slider is bounded per weekend
and compound by the longest stint any team ran, compounds nobody raced are marked, and
the race's real strategies are shown next to the controls. This replaced the plan to fit
a degradation curve: extrapolating past the data would have produced numbers nobody has
driven, and a race in which no team chose the medium is a fact worth showing, not a gap
to fill._

- Low-speed corners are the weakest segment kind (MAE 0.111 s vs 0.058 s on straights):
  a traction/exit feature is the obvious candidate.
- Compound × temperature interaction, driver × circuit history.
- A back-test of the simulator as a whole: predict a race lap from the same driver's
  qualifying lap by moving only the fuel slider, and score it per circuit.
- Fix, rather than expose, the reconstruction residual under low grip.
- A proper front-end test layer; today the HUD is verified by hand and by the
  contract tests on the API.
