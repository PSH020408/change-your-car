# The physics engine (P9, 2026-09-29)

A quasi-steady-state point-mass lap simulation that replaces the coefficient table
(P3) and the warp-based trace reconstruction (P5) wherever a circuit has a racing line.
This document records what it is, how it was calibrated, the gates it was held to,
which of them it failed, and why it is used in *differential* form.

## 1. What it computes

The car is a point of mass *m* on the circuit's racing line. At every 5 m the line
gives a curvature κ(s) and a road gradient dz/ds; the car brings a wheel power *P*,
a drag area C<sub>d</sub>A, a downforce area C<sub>l</sub>A and a friction coefficient μ.
The speed profile follows in three sweeps and nothing else:

1. **Grip limit in every corner** — μ (m g cos θ + ½ρC<sub>l</sub>A v²) ≥ m v² |κ|, so
   v² = μ g cos θ / (|κ| − μρC<sub>l</sub>A / 2m); a corner whose aero load alone would
   hold it at any speed is flat-out.
2. **Backward sweep** from every limit: braking with the longitudinal grip left after
   the cornering load (friction ellipse), plus drag, plus gravity.
3. **Forward sweep**: accelerating on the lesser of traction (rear axle only, 60 % of the
   grip) and power minus drag, minus gravity on a climb.

v(s) is the point-wise minimum; lap time is ∫ ds / v. DRS is open exactly where the
baseline lap opened it (drag −12 %, downforce −8 %). One solve of a 1,100-point line
takes ~4 ms in plain Python.

**Elevation** comes from the timing system's Z channel, the median of the same 16 laps
that form the line, smoothed with the same 75 m window as the curvature. It is gated:
the lap must return to its starting height within 3 m and the laps must agree to 2 m.
All 92 circuit-seasons pass (worst closure 0.3 m, worst spread 0.3 m); Spa climbs 102 m,
Austria 64 m, Bahrain 17 m.

## 2. What is fitted, what is fixed

Three numbers are fitted **per lap**, to the measured speed trace and never to the lap
time — so the lap-time error below is a genuine measure, not a residual of the fit:

| Parameter | Prior box | Median over 552 laps |
|---|---|---|
| μ — effective grip (tyre and driver together) | 1.2 – 2.8 | 1.92 |
| C<sub>l</sub>A — downforce area | 2.5 – 6.5 m² | 4.73 m² |
| C<sub>d</sub>A — drag area | 0.8 – 1.8 m² | 1.16 m² |

Everything else is one constant for every lap, and the constants were **set by a sweep,
not by a fit**: eight circuits (Bahrain, Monaco, Monza, Las Vegas, Hungaroring, Spa,
Suzuka, Singapore), one lap each, over power × traction share × braking share ×
curvature window × tyre load sensitivity. The sweep log is `data/logs/qss-sweep.log`.

| Constant | Value | How it was chosen |
|---|---|---|
| Wheel power | **480 kW** | 780 → 600 → 480: every step lowered the speed RMS and moved the mean error toward zero (+7.1 → +0.3 km/h); at 480 the fitted drag area leaves its upper bound (88 % → 31 % of laps at a bound) and lands in the published F1 range. 480 kW is not a peak figure: it is the lap-average power a flat-throttle point mass needs to match real acceleration, with ERS deployment limits, lift-and-coast, gear shifts and driveline losses folded in. |
| Driven-axle share | **0.60** | Traction out of slow corners was 20–30 km/h too strong with four-wheel grip; 0.6 (static rear weight plus the rear-biased floor) removed most of it. |
| Braking share | **1.00** | 0.8 was tried: worse RMS and a faster lap (the fit raised μ to compensate). |
| Curvature window | **75 m** | 53 (the segmenter's auto window) / 75 / 100 m: 75 m gave the lowest RMS on 8 of 8 circuits; 100 m smears hairpins. |
| Tyre load sensitivity | **0** | 0.15 and 0.30 were worse on every circuit and pinned C<sub>l</sub>A at its bound. |

Fuel comes from the P3 estimate (8 kg in qualifying, linear burn in the race), air
density from the session's air temperature.

## 3. Gates and results (552 laps = 184 sessions × 3 fastest drivers, 25 circuits)

The gates were written before the first run (design note P9). Two things changed
afterwards and both are recorded here rather than hidden: the model gained the
driven-axle and power constants after the first result, and gate **G1b** was added
after the residual analysis explained why G1 cannot pass.

| Gate | Target | Result | |
|---|---|---|---|
| G1 speed RMS vs the real trace | median ≤ 8 km/h | **12.8 km/h** (p90 15.8) | FAIL |
| G1b segment-time error | median ≤ 0.10 s | **0.051 s** (2.6 %), p90 0.27 s, bias 0.000 | PASS (added after the analysis below) |
| G2 lap-time error, not a fit target | median ≤ 0.5 s, p90 ≤ 1.5 s | **0.59 s** (0.6 %), p90 1.39 s, bias −0.36 s | FAIL (median), PASS (p90) |
| G3 parameters plausible | ≥ 90 % inside the box; Monaco/Hungary/Singapore rank high, Monza/Baku/Las Vegas low | 76 % inside, 25 % at a bound; Monaco and Singapore rank 1–2 but so do Spa and Monza | FAIL |
| G4 the fuel effect the engine was never told | 0.0294 s/kg ± 25 % | **0.0242 s/kg** (−18 %), p10–p90 0.019–0.029 | PASS |
| G5 sign checks | rear wing lowers top speed and speeds up grip-limited corners | 99.8 % / 100 % of laps | PASS |

Qualifying and race laps behave alike (RMS 12.4 vs 13.3 km/h; lap error −0.40 vs
−0.31 s). Median fit time 123 ms.

### Where the 12.8 km/h comes from

Splitting the 565,000 grid points of all 552 laps by driving phase:

| Phase | Share of points | Bias | RMS | Share of squared error |
|---|---|---|---|---|
| steady (straight, flat-out corner) | 48 % | −2.0 km/h | 8.4 | 20 % |
| accelerating | 33 % | +4.3 | 13.4 | 34 % |
| braking | 19 % | −2.4 | 20.5 | **46 %** |

Braking zones are one fifth of the lap and half the error, and the shape is always the
same: the point mass brakes 10–20 m later and harder than the driver, reaches its apex
earlier and at the geometric limit, and accelerates sooner. A real driver brakes a
little earlier and less than the limit, trail-brakes to a lower and later apex, and
picks the throttle up later. At 3–5 g, a 10 m phase shift is a 30–40 km/h speed error
that lasts a few tenths of a second — and nearly cancels in time. That is why the
segment-time error is 0.05 s while the speed RMS is 12.8 km/h, and why G1b is the
figure that describes what the simulator actually outputs. G1 is left in the table as
failed because the model does not reproduce the driver's braking shape, and no
parameter of a point mass will.

By segment kind (15,570 segments): straights 0.039 s, kinks 0.023 s, high-speed corners
0.064 s, medium-speed 0.167 s (bias +0.11 s, the engine is slower), low-speed 0.203 s
(bias −0.07 s, the engine is faster). Low-speed corners are the weak point of the ML
model as well.

### On G3

A quarter of the fits touch a bound (mostly C<sub>l</sub>A = 6.5 on high-downforce
street circuits). The circuit ranking of C<sub>l</sub>A is half right: Monaco and
Singapore lead, Qatar and Saudi Arabia (fast, flowing) trail — but Spa and Monza also
rank high. On circuits with few slow corners the three parameters are weakly
identified: with C<sub>d</sub>A set by the top speed, extra downforce is the only way
to lift the fast-corner speeds, and the fit takes it. A per-circuit aero prior would fix
the ranking at the cost of another fitted quantity; it was not done.

## 4. Why the engine is used in differential form

Because of §3, the engine's own speed trace is not shown. Every request runs the solver
three times on the fitted car:

    prof0   the baseline car
    prof1   the baseline car with the setup sliders applied (P3 modifiers → C_lA, C_dA, μ, m)
    prof2   prof1 with one grip multiplier chosen so its lap-time change equals the
            ML's conditions estimate (tyre age, compound, temperature)

and reports differences: setup = prof1 − prof0, conditions = prof2 − prof1, per
segment. The simulated telemetry is the **real trace plus (prof2 − prof0)**, so it keeps
the driver's braking shape while carrying the physics of the change; the per-segment
times are then integrated on the real trace's own samples, so setup, conditions and
headline add up exactly and the HUD's fourth "trace rebuild" term is rounding only. The
structural error of §3 appears in both terms of every difference and largely cancels.
This is how lap simulations are used for setup work in practice.

Circuits without a line fall back to the P3 table and the P5 warp. Today every circuit
has a line, so the fallback exists for robustness only. The first request for a lap
carries the 3-parameter fit (~120 ms); after that a simulation is 3–10 solves,
15–45 ms on a laptop.

## 5. What this changed

- **Defect #28 is closed:** the headline and the components come from the same
  integration; `refused` is identically zero because the solver never exceeds the grip.
- The wing sliders now act on C<sub>l</sub>A and C<sub>d</sub>A of a car that reproduces
  the lap's own top speed and corner speeds; a braking zone that crosses a segment
  boundary is resolved by the sweeps, not by a per-segment formula.
- The fuel effect is emergent (0.0242 s/kg against 0.0294 measured) instead of a
  coefficient.

## 6. Limitations

- Point mass: no weight transfer, no yaw, no tyre thermal model, no gear ratios, no
  ERS strategy; the racing line is the weekend's median line and does not move with
  the setup.
- The three fitted numbers are effective quantities (μ includes the driver), valid for
  the lap they were fitted to.
- The ML conditions effect is redistributed by the engine as a uniform grip change; a
  tyre that is weak only in slow corners is not represented as such in the trace.
- Elevation enters the longitudinal balance only; banking and camber are ignored.

## 7. Reproduce

    make line             # 92 racing lines (curvature, gradient, DRS) into data/silver and the baseline store
    make qss-calibrate    # 552-lap calibration, gates, data/artifacts/qss/calibration_report.json
    make qss-sweep        # the constant sweep on 8 circuits
    make test             # 183 tests incl. solver invariants and the engine on the stored Bahrain lap
