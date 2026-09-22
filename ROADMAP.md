# Roadmap — the plan, the gates, and what actually happened

> Ten stages, each with a **gate** that had to pass before the next one started. The
> left column is the plan as written before the work; the "outcome" lines were added as
> each gate was passed, failed or rewritten. Task IDs are stable.
>
> **Scope: 2022–2025, qualifying and race only.** The ground-effect floor arrived with
> the 2022 regulations and left with the 2026 ones; 2021 cars belong to a different aero
> era and were excluded by decision. Sprint and practice sessions were excluded during
> the data expansion (few sessions, yearly rule changes, unknown fuel loads).

## Dependency flow

```
P0 Foundation
      │
      ▼
R  Data reconnaissance ─────────┐   ◄── before design: answer "what do we actually have?"
      │  (cache warming starts here and keeps running in the background until P8)
      ▼                         │
P1 Ingestion ───────────────────┤
      │                         │
      ▼                         │
P2 Segmentation & features      │
      │                         │
      ├──────────► P3 Physics layer
      │                    │
      ▼                    ▼
P4 ML delta model ◄────────┘
      │
      ▼
P5 Telemetry reconstruction
      │
      ▼
P6 Backend API ◄──── contract published early so P7 could start against the schema
      │                         │
      ▼                         ▼
      └────────► P8 Integration & deployment ◄──── P7 Frontend HUD
```

---

## P0 — Foundation
*One command brings the stack up.*

| ID | Task |
|---|---|
| P0-1 | Monorepo layout (`backend/ frontend/ data/ docs/`), git |
| P0-2 | Python environment, pinned requirements |
| P0-3 | Next.js + Tailwind + design tokens |
| P0-4 | FastF1 cache and lake paths |
| P0-5 | Scope frozen: era 2022–25, pilot 2023–24 |

**Gate:** `make setup`, `/health` 200, frontend renders. **Outcome:** passed.

---

## R — Data reconnaissance
*Know what the data contains before designing a single feature.*

Two tracks at once: raw downloads are slow and rate-limited but independent of design;
design needs only a survey of ~10 sessions.

| ID | Task |
|---|---|
| R-1 | Resumable, rate-limit-aware cache warming job |
| R-2 | Download the era (later narrowed to Q + R: 184 sessions, 16.8 GB) |
| R-3 | Survey script |
| R-4 | Survey report: channel dictionary, coverage, lap yield, sample rate |
| R-5 | Decide the resampling grid → **no grid for features** (dt = 240 ms fixed; spacing is a function of speed) |
| R-6 | Choose setup-proxy channels → speed traps in the lap table |

**Gate:** nine decisions written down (`docs/recon/DECISIONS.md`). **Outcome:** passed
2026-09-14, then **revised twice** after the first two ingest runs (gap filter axis and
threshold; track-status filter order). Both revisions are in the same document.

---

## P1 — Ingestion → `data/bronze`

| ID | Task |
|---|---|
| P1-1 | Session loader on the warm cache (no network on re-run) |
| P1-2 | Lap filters: track status → pit in/out → deleted → accurate → gap safety net → pace gate |
| P1-3 | Conditions-matched pace gate (rolling window, 107 %) |
| P1-4 | Weather and track-status merge |
| P1-5 | Driver / team / chassis / power-unit metadata (`chassis.yaml`) |
| P1-6 | Bronze parquet + manifest, with a per-session filter report |

**Gate:** every filter step logs its removals; the track-status filter is *proven* to
remove laps when caution codes exist. **Outcome:** passed after moving track status to
the front of the chain (defect #5). Later additions: gap-to-neighbours feature, sibling
lookup for sessions with a blank driver list (defect #30).

---

## P2 — Segmentation & features → `data/silver` → `data/gold`

| ID | Task |
|---|---|
| P2-1 | Ensemble racing line per circuit (16 phase-locked laps) |
| P2-2 | Curvature → straight / kink / low-, medium-, high-speed corner |
| P2-3 | Sector mapping (from per-lap sector times, median on the ensemble axis) |
| P2-4 | SVG path for the HUD (`pathLength` in metres) |
| P2-5 | Per-segment features by raw-sample integration |
| P2-6 | Driver bias features (HUD only — leakage for the model) |
| P2-7 | Setup proxies from speed traps, with imputation flags |

**Gate:** corner count within ±2 of the published figure where one is verified; physics
invariants (no hidden corner in a straight, no corner over 6.5 g). **Outcome:** passed
on the calibration set; on the full era seven circuits failed the invariants and got
per-circuit overrides (defect #31). COTA remains under-counted (esses merge).

---

## P3 — Physics layer

| ID | Task |
|---|---|
| P3-1 | Wings → ΔCl / ΔCd (rear = drag-dominated, front = balance) |
| P3-2 | Ride height → ground effect (non-monotonic: bottoming) |
| P3-3 | Suspension stiffness → mechanical grip, weighted by circuit roughness |
| P3-4 | Fuel mass, tyre thermal window, weather grip multipliers |
| P3-5 | Balance index; monotonicity unit tests (29) |

**Gate:** all monotonicity tests pass; coefficient signs agree with regressions on our
own data. **Outcome:** passed 3/3 in `make physics-check` — aero-efficiency partial
slope negative on 8/9 circuits, fuel 0.0294 s/kg measured vs 0.030 set, corner grip fit
μ 1.5 / v0 145 km/h vs 1.6 / 150 — after fixing two flaws in the check itself
(defects #13, #14). Coefficients graded A/B/C in `physics.yaml`.

---

## P4 — ML delta model → `data/artifacts/models`

| ID | Task |
|---|---|
| P4-1 | Target: segment delta vs session reference; leakage-refusing feature spec |
| P4-2 | Ridge baseline (must be beaten) |
| P4-3 | LightGBM quantile models q10/q50/q90, GroupKFold by event, conformal calibration |
| P4-4 | Permutation importance, monotonicity probes |
| P4-5 | Gates, including a measured noise floor and skill ceiling |
| P4-6 | Registry with versions, metrics and manual acceptance with reason |

**Gate (as planned):** sector MAE < 0.15 s, lap MAE < 0.30 s, unseen circuit < 0.45 s.
**Outcome:** segment gate passed (0.062 s). The lap gate was rewritten four times and
then frozen (see `docs/DESIGN.md` §5); the final model passes 5 of 8 gates and was
accepted manually with the reason recorded. Skill 28 % against a measured 49 % ceiling.

---

## P5 — Telemetry reconstruction

| ID | Task |
|---|---|
| P5-1 | Segment delta → warped speed trace (bisection on a shape-weighted multiplier) |
| P5-2 | Throttle / brake / gear / DRS synthesis |
| P5-3 | g-g envelope clamp anchored on the real lap |
| P5-4 | Integration consistency vs the requested delta |

**Gate:** no envelope violation; integrated lap time matches the requested delta.
**Outcome:** passed after four fixes on real telemetry (defects #22–25); a residual
remains under low grip and is exposed in the HUD as its own term (#28).

---

## P6 — Backend API

| ID | Task |
|---|---|
| P6-1 | App, settings, engine singleton |
| P6-2 | Pydantic contract (`app/schemas/domain.py`) — frozen early for P7 |
| P6-3 | Catalogue: `GET /api/meta/*` |
| P6-4 | `GET /api/baseline` from precomputed per-session JSON |
| P6-5 | `POST /api/simulate` — one call returns everything |
| P6-6 | Rule-based engineer log |
| P6-7 | Contract tests, smoke suite |

**Gate:** contract frozen; p95 < 400 ms. **Outcome:** passed — baseline 8 ms,
simulate 15 ms, 31 kB; 165 tests; 8 smoke cases.

---

## P7 — HUD

| ID | Task |
|---|---|
| P7-1 | Design tokens, one-screen console layout |
| P7-2 | Docking bar (custom dropdowns — native ones spilled off-screen on macOS) |
| P7-3 | 2D car schematic bound to the sliders |
| P7-4 | Debounced simulate with in-flight cancellation |
| P7-5 | Environment panel |
| P7-6 | SVG track map coloured per segment; replay of real vs simulated car |
| P7-7 | Telemetry chart with Δspeed and running Δtime rows |
| P7-8 | Delta panel (four-way split that adds up), engineer log |
| P7-9 | State (zustand) and data (SWR) |

**Gate:** no perceived lag; build clean. **Outcome:** passed — 121 kB first load, tsc
and lint clean; numbers verified against the smoke suite on screen.

---

## Data expansion (between P7 and P8)

66 → 184 sessions through `make expand`; model retrained (66 → 184 sessions,
33 → 81 events). Three defects surfaced and were fixed (#29–31).

---

## P8 — Integration & deployment

| ID | Task |
|---|---|
| P8-1 | End-to-end check on real data |
| P8-2 | Bundle budget (< 300 kB gzip) — met at 121 kB |
| P8-3 | Latency budget (p95 < 400 ms) — met at ~15 ms |
| P8-4 | Deploy: one container, API + static HUD, same origin — **live at https://change-your-car.onrender.com** (Render free tier, cold start 30–60 s after idle) |
| P8-5 | Back-test the simulator as a whole (qualifying → race lap) — **done 2026-09-22**: flat-out answer -3.1 s optimistic; 1.21 s MAE after a measured race-pace offset (`docs/BACKTEST.md`) |
| P8-6 | Bound the tyre controls by the weekend's real stints; show the race's actual strategies — **done 2026-09-22** |

**Gate:** live URL, budgets met, retrained model re-gated. **Outcome:** passed 2026-09-17 — first deploy built in 1 m 58 s from the repository; P8-5 back-test done 2026-09-22.

---

## Budgets

| Metric | Target | Actual |
|---|---|---|
| First-load JavaScript | < 300 kB gzip | 121 kB |
| Simulate round-trip | p95 < 400 ms | ~15 ms |
| Segment delta MAE | < 0.15 s | 0.062 s |
| Lap counterfactual MAE | ≤ 1.25 × noise floor (0.424 s) | 0.474 s — not met, documented |
| Unseen-circuit skill | ≥ 25 % | 24.8 % — borderline, documented |
