# F1 Virtual Sim — Build Pipeline

> 10 stages · 61 tasks. Each stage has a **gate**: until the gate passes, the
> next stage is not started. Task IDs (`P3-2`) are stable and match the
> progress dashboard.
>
> **Scope: 2022–2025 only.** The ground-effect floor arrived with the 2022
> regulations and was replaced again in 2026. 2021 cars are a different aero
> era — including them would teach the model to average across two
> philosophies, so they are excluded by decision, not by oversight.

## Dependency flow

```
P0 Foundation
      │
      ▼
R  Data Reconnaissance ─────────┐   ◄── 설계보다 먼저. "무엇을 갖고 있는가"에 답한다
      │  (캐시 워밍은 여기서 시작해 P8까지 백그라운드로 계속 돈다)
      ▼                         │
P1 Data Ingestion ──────────────┤
      │                         │
      ▼                         │
P2 Segmentation & Features      │
      │                         │
      ├──────────► P3 Physics Layer
      │                    │
      ▼                    ▼
P4 ML Delta Model ◄────────┘
      │
      ▼
P5 Telemetry Reconstruction
      │
      ▼
P6 Backend API ◄──── (contract published early: P7 can start in parallel)
      │                         │
      ▼                         ▼
      └────────► P8 Integration & Optimization ◄──── P7 Frontend HUD
```

**Critical path:** P0 → R → P1 → P2 → P4 → P5 → P6 → P8.
P3 runs parallel to P2 (it only needs the scope config).
P7 starts as soon as the P6 contract stub is published — it develops against
the OpenAPI schema, not a working model.

---

## P0 — Foundation & Environment
*Goal: one command brings the whole stack up. Nothing else starts until it does.*

| ID | Task | Output |
|----|------|--------|
| P0-1 | Monorepo 구조 확정 + git init | `backend/ frontend/ data/ docs/` |
| P0-2 | Python 환경 + 의존성 고정 | `backend/requirements.txt`, `.venv` |
| P0-3 | Next.js + Tailwind + 디자인 토큰 초기화 | `frontend/tailwind.config.ts` |
| P0-4 | FastF1 캐시 · 데이터 레이크 경로 설정 | `.env`, `app/config.py` |
| P0-5 | 스코프 고정 (era 2022–25 / 파일럿 2023–24) | `configs/scope.yaml` |

**Gate:** `make setup && make dev-be && make dev-fe` — `/health` 200, 프론트 렌더.

---

## R — Data Reconnaissance
*Goal: know what the data actually contains before a single feature is designed.*

두 트랙이 **동시에** 돕니다. 다운로드는 느리고 rate limit이 걸리지만 설계와 무관하고,
설계는 정찰 리포트 한 장만 있으면 시작할 수 있습니다.

**Track A — 백그라운드 수집 (지금 시작, P8까지 계속)**
FastF1은 raw GET 응답을 캐시에 저장하고, **캐시 히트는 rate limit에 포함되지 않습니다.**
즉 "일단 캐시를 채우는 일"은 설계를 기다릴 이유가 전혀 없습니다.

**Track B — 정찰 (앞단, 세션 10개면 충분)**
전량이 아니라 표본으로 답하는 질문들입니다.

| ID | Task | Track | Note |
|----|------|-------|------|
| R-1 | 캐시 워밍 잡 구현 (resumable · 백오프 · 용량 실측) | A | `pipeline/ingest/warm_cache.py` |
| R-2 | 2022–2025 전 세션 다운로드 착수 + 용량 실측 | A | `make warm-bg` → 원장(ledger)에 세션당 용량 기록 |
| R-3 | 정찰 서베이 스크립트 구현 | B | `pipeline/recon/survey.py` |
| R-4 | 정찰 리포트 생성 | B | 채널 사전 · 커버리지 매트릭스 · 랩 수율 · 샘플레이트 |
| R-5 | ~~10 m 그리드 타당성 판정~~ ✅ | B | dt=240 ms 고정 → 간격은 속도의 함수. 그리드 폐기 (D1) |
| R-6 | ~~셋업 프록시 후보 채널 확정~~ ✅ | B | Laps 테이블의 SpeedI2/ST/FL/I1 트랩 사용 (D6) |

**Gate:** ✅ **통과 (2026-09-14)** — 판정 근거는 [`docs/recon/DECISIONS.md`](docs/recon/DECISIONS.md).

| 결정 | 결과 |
|---|---|
| 리샘플 그리드 | 10 m 폐기 → 피처는 그리드 없음(세그먼트 raw 적분), 표시만 20 m + 보간 마스크 |
| 학습 세션 | 수집은 460세션 전부(36.9 GB), 학습은 Q+R+S+SQ만. FP3는 확장 후보 |
| 필터 체인 | track status 선행 · 조건정합 107%(롤링 윈도) · 40 m 텔레메트리 갭 필터 신설 |
| 셋업 프록시 | SpeedI2(주·결측 0.2%) · SpeedST · SpeedFL · SpeedI1 + 임퓨테이션 플래그 |
| 디스크 예산 | 실측 median: FP 52–74 MB · Q 99 MB · R 110 MB → 전량 36.9 GB |

> 왜 "전량 수집 후 설계"가 아닌가: 스키마를 모르는 채로 전량을 수집하면, 스키마가
> 바뀌는 순간 전량을 다시 받아야 합니다. FastF1은 rate limit이 있어 재수집이 싸지
> 않습니다. 반대로 raw 캐시는 스키마와 무관하므로, **원본은 지금 받고 해석은 나중에**
> 하는 편이 순서상 옳습니다.

---

## P1 — Data Ingestion Pipeline → `data/bronze`
*Goal: a reproducible, filtered, distance-normalised lap store.*

| ID | Task | Note |
|----|------|------|
| P1-1 | FastF1 세션 로더 + 캐시 워밍 | 세션 1회만 다운로드, 재실행 시 캐시 히트 |
| P1-2 | 유효 랩 필터 | 인/아웃랩, 삭제랩, 107% 초과, SC/VSC 구간 제거 |
| P1-3 | 조건정합 페이스 게이트 + 40 m 갭 필터 | D2·D3 — 균일 리샘플링은 하지 않음. 세그먼트 raw 적분 |
| P1-4 | Weather / track status 병합 | 트랙·에어 온도, 강수, 트랙 상태 플래그 |
| P1-5 | 메타데이터 테이블 (드라이버·팀·섀시·PU 매핑) | 2021–25 섀시 코드 사전 |
| P1-6 | Bronze parquet + 매니페스트        | `writer.py` — laps/telemetry parquet + session.json + manifest.json |

**Gate:** 파일럿 세션 ingest 성공 · 재실행 시 네트워크 호출 0 (`offline_mode`) ·
**D4 판정 완료** (track status 필터가 실제로 랩을 제거하는지, 또는 제거 0이 정당한지) ·
필터 단계별 제거 수가 전부 기록됨.

---

## P2 — Track Segmentation & Feature Engineering → `data/silver` → `data/gold`
*Goal: turn a lap into a feature vector the model can reason about.*

| ID | Task | Note |
|----|------|------|
| P2-1 | 트랙 중심선 추출 + 스무딩 | X/Y GPS → Savitzky-Golay |
| P2-2 | 곡률 계산 & 코너/직선 분할 | `k = |x'y" − y'x"| / (x'²+y'²)^1.5` |
| P2-3 | 코너 등급 분류 + 마이크로섹터(28) 정의 | Low/Med/High-speed, 섹터 1·2·3 매핑 |
| P2-4 | SVG 트랙 path 생성 | 프론트 트랙맵이 그대로 소비 |
| P2-5 | 세그먼트 집계 피처 | apex/entry/exit 속도, 브레이킹 포인트, DRS |
| P2-6 | 드라이버 성향 바이어스 피처 | 브레이킹·스로틀 공격성 z-score |
| P2-7 | **셋업 프록시 피처** | R-6에서 확정: SpeedI2(주) · SpeedST · SpeedFL · SpeedI1 + 임퓨테이션 플래그 |

**Gate:** 세그먼트 수가 해당 서킷 공식 코너 수와 ±2 이내 · 피처 결측률 < 1% · `feature_spec.json` 확정.

> ⚠️ **핵심 리스크:** F1 팀 셋업 데이터는 공개되지 않습니다. P2-7의 프록시 품질이
> 이 프로젝트 전체의 정확도 상한을 결정합니다. 여기서 시간을 아끼면 안 됩니다.

---

## P3 — Physics Modifier Layer *(P2와 병렬)*
*Goal: sliders → coefficient deltas, analytically and testably.*

| ID | Task | Note |
|----|------|------|
| P3-1 | 윙 각도 → ΔCl / ΔCd 매핑 | 리어=드래그 지배, 프론트=밸런스 지배 |
| P3-2 | Ride height → 그라운드 이펙트 | 낮을수록 이득, 바닥 아래선 붕괴 (**비단조**) |
| P3-3 | 서스펜션 강성 → 기계적 그립 | 트랙 노면 거칠기 가중 |
| P3-4 | 타이어 열 모델 + 날씨 계수 | 컴파운드별 온도 윈도 종형 곡선 |
| P3-5 | 밸런스 지수 + 단조성 단위 테스트 | `tests/test_physics_monotonicity.py` |

**Gate:** 모든 단조성 테스트 통과 · 계수가 실측 트랩속도 회귀와 부호 일치.

---

## P4 — ML Delta Model → `data/artifacts`
*Goal: predict per-segment time delta, and prove it generalises.*

| ID | Task | Note |
|----|------|------|
| P4-1 | 타깃 정의 & 학습 데이터셋 빌드 | 세그먼트 통과시간 Δ (기준랩 대비) |
| P4-2 | 베이스라인 모델 (Ridge) | 반드시 이겨야 하는 하한선 |
| P4-3 | XGBoost / LightGBM 학습 파이프라인 | GroupKFold (season·event·driver) |
| P4-4 | 하이퍼파라미터 튜닝 + SHAP 검증 | 피처 중요도가 물리적으로 말이 되는지 |
| P4-5 | **합격 게이트 + 단조성 프로브** | 섹터 MAE < 0.15 s, 랩 MAE < 0.30 s |
| P4-6 | 모델 레지스트리 (버전 + 메트릭 동봉) | `artifacts/{version}/model.json` |

**Gate:** 미학습 트랙(Monza) 홀드아웃 MAE < 0.45 s · 물리 단조성 프로브 전부 통과.
게이트 미달 모델은 레지스트리에 등록하지 않습니다.

---

## P5 — Telemetry Reconstruction
*Goal: turn scalar deltas back into a trace the HUD can draw.*

| ID | Task | Note |
|----|------|------|
| P5-1 | 세그먼트 Δ → 연속 속도 트레이스 워핑 | 적분 시간이 예측 Δ와 일치하도록 |
| P5-2 | Throttle / Brake / Gear / DRS 채널 합성 | DRS는 실제 활성화 존에서만 |
| P5-3 | 물리 제약 클램핑 (g-g 다이어그램) | 종·횡 가속 한계, 마찰 타원 |
| P5-4 | 랩타임 적분 정합성 검증 | 재구성 랩타임 vs 델타 합 오차 < 0.02 s |

**Gate:** 재구성 트레이스가 물리 한계를 위반하지 않고, 적분 랩타임이 예측과 일치.

---

## P6 — FastAPI Backend Engine
*Goal: a stable contract, fast enough to feel live.*

| ID | Task | Endpoint |
|----|------|----------|
| P6-1 | 앱 구조 + 설정 + DI | `app/main.py`, `app/config.py` |
| P6-2 | Pydantic 스키마 확정 (계약 동결) | `app/schemas/domain.py` |
| P6-3 | 카탈로그 API | `GET /api/meta/*` |
| P6-4 | 베이스라인 API | `GET /api/baseline` |
| P6-5 | 시뮬레이션 API | `POST /api/simulate` |
| P6-6 | AI 엔지니어 로그 룰 엔진 | `POST /api/engineer-log` |
| P6-7 | 캐싱 + 응답 압축 + API 테스트 | 베이스라인 프리컴퓨트 |

**Gate:** OpenAPI 스키마 동결 · `/api/simulate` p95 < 400 ms · 계약 테스트 통과.

> P6-2가 끝나는 즉시 P7을 병렬 착수합니다. 프론트는 모델이 아니라 **계약**에 의존합니다.

---

## P7 — 2D HUD Frontend *(P6-2 이후 병렬)*

| ID | Task | Note |
|----|------|------|
| P7-1 | 디자인 토큰 + HUD 셸 레이아웃 | Neon/Slate, 네온은 **데이터에만** |
| P7-2 | Driver & Chassis 도킹 화면 | 시즌 → 드라이버 → 섀시 |
| P7-3 | 2D SVG Car Schematic | 윙·차고·서스펜션 조작부 |
| P7-4 | 셋업 슬라이더 바인딩 + 디바운스 | 조작 중 과도한 API 호출 방지 |
| P7-5 | Environment 패널 | 온도·날씨·트랙 에볼루션 |
| P7-6 | SVG 트랙맵 + 섹터 델타 컬러 오버레이 | 이득=라임 / 손실=로즈 |
| P7-7 | Canvas 텔레메트리 오버레이 차트 | 실측 vs 시뮬 (Speed/Throttle/Brake/DRS) |
| P7-8 | Delta 요약 HUD + AI Engineering Log 패널 | 오버/언더스티어 경고 |
| P7-9 | API 클라이언트 + 상태관리 + 로딩/에러 | zustand + SWR |

**Gate:** 슬라이더 조작 → 차트 갱신까지 체감 지연 없음 · 텔레메트리 리렌더 < 16 ms.

---

## P8 — Integration, QA & Optimization

| ID | Task | Note |
|----|------|------|
| P8-1 | E2E 통합 + 계약 검증 | 프론트↔백엔드 스키마 드리프트 탐지 |
| P8-2 | 번들 최적화 | 초기 JS < 300 KB gzip |
| P8-3 | 추론 레이턴시 최적화 + 캐싱 | p95 < 400 ms |
| P8-4 | 반응형 · 접근성 점검 | 대비비, 키보드 조작 |
| P8-5 | 배포 (프론트 Vercel / 백엔드 컨테이너) | 모델 아티팩트 동봉 |
| P8-6 | **데이터 스코프 전체 확장 (2022–2025)** | 파일럿 파이프라인 그대로 재실행 |

**Gate:** 성능 예산 전부 충족 · 전체 시즌 재학습 후 P4 게이트 재통과.

---

## Performance budgets (P8 기준)

| 지표 | 목표 |
|------|------|
| 초기 JS 번들 | < 300 KB gzip |
| 시뮬레이션 왕복 p95 | < 400 ms |
| 텔레메트리 리렌더 | < 16 ms/frame |
| 섹터 델타 MAE | < 0.15 s |
| 랩 델타 MAE | < 0.30 s |
| 미학습 트랙 MAE | < 0.45 s |
