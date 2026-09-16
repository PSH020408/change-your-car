# Change Your F***ing Car — F1 Virtual Sim

2022–2025 Ground Effect ERA 실제 F1 텔레메트리(FastF1) 기반의 **2D 엔지니어링
셋업 & 조건 시뮬레이터**. 예선·레이스 184세션(92이벤트 × Q/R) 위에서 돌아갑니다.

드라이버와 실제 랩을 고르고, 윙·차고·서스펜션·연료를 조절하고, 타이어 나이와
온도·날씨를 바꾸면 — 물리 모델과 ML이 **세그먼트별 시간 델타(±80 % 밴드)**,
**재구성된 텔레메트리**, 실제 vs 시뮬 **리플레이**를 돌려줍니다.

```
FastF1 ──► warm_cache ──► ingest ──► segment ──► features ──► LightGBM q10/q50/q90 ──► FastAPI ──► Next.js HUD
                          bronze     silver       gold         artifacts/models        baselines    (hand-drawn SVG)
                                                                     ▲                      ▲
                                            physics modifiers (sliders → ΔCl/ΔCd/Δgrip)     reconstruct (delta → trace)
```

> **Scope: 2022–2025, Q + R only.** 그라운드 이펙트 플로어는 2022년 규정과 함께
> 도입되어 2026년 규정으로 교체되었습니다. 스프린트·연습 세션은 제외(결정, 2026-09-16).

## Quick start

```bash
make setup          # python venv + npm install  (Node.js 필요)
make api            # FastAPI  :8000   (별도 터미널)
make dev-fe         # Next.js  :3000   → http://localhost:3000
```

## Pipeline

```bash
make warm-bg        # FastF1 raw 캐시 (백그라운드, 재개 가능) · make warm-status
make expand         # ingest → segment → features → baselines  (캐시된 Q/R 전부)
make train          # LightGBM 분위수 모델 + 게이트 → data/artifacts/models/<version>
                    # 게이트 미달 시 latest로 등록되지 않음: registry accept 로 수동 승인 (사유 기록)
make test           # 165 pytest (물리 단조성 · 세그먼트 · 모델 · 재구성 · API 계약)
make api-smoke      # 8케이스 스모크 (2024 Bahrain Q VER)
make fe-check       # tsc + lint + next build
```

## Layout

| 경로 | 역할 |
|------|------|
| `backend/pipeline/ingest` `segment` `features` | 배치 데이터 파이프라인 (bronze → silver → gold) |
| `backend/pipeline/physics/` | 셋업 슬라이더 → 물리 계수 (등급 A/B/C, `configs/physics.yaml`) |
| `backend/pipeline/models/` | 분위수 GBM · 컨포멀 밴드 · 반사실 평가 · 레지스트리 |
| `backend/pipeline/reconstruct/` | 세그먼트 델타 → 연속 텔레메트리 (g-g 엔벨로프 클램프) |
| `backend/app/` | FastAPI: `/api/meta/*` `/api/baseline` `POST /api/simulate` |
| `frontend/src/` | Next.js 15 HUD — 차트 라이브러리 없이 SVG 직접 그림 (첫 로드 121 kB) |
| `backend/configs/` | `scope.yaml` `physics.yaml` `model.yaml` `circuits.yaml` `chassis.yaml` |
| `data/` | 메달리온 레이크 + 베이스라인 JSON(184세션) + 모델 아티팩트 (git 제외) |

## 설계 원칙

1. **3D 없음.** 트랙맵·차량 도식·텔레메트리 전부 손으로 그린 SVG.
2. **물리는 셋업을, ML은 조건을.** 슬라이더→계수는 해석 가능한 수식(등급 표기), 타이어·온도는 실데이터 학습.
   둘은 가산적으로 합치고, 재구성 잔차·엔벨로프 거절까지 HUD에 그대로 노출.
3. **밴드 없는 숫자 없음.** 모든 델타에 80 % 구간, 보간 구간은 빗금, 검증 안 된 계수는 등급 C.
4. **게이트 없는 단계 진행 없음.** 미달은 숨기지 않고 사유와 함께 기록 (결함 목록 28개, `ROADMAP.md`).

## How this was built

Solo project by **PARK, SEHO** (Sept 2026). Every design decision, every acceptance
gate, every pipeline run and every verification of the numbers in this repo is mine;
I used **Claude (Anthropic) as a pair programmer** — it drafted code and read logs
with me, and each commit that it touched carries a `Co-Authored-By` trailer for that
reason. The defect list (31 items and counting), the frozen-then-documented gate
history and the C-grade markings on unverifiable physics coefficients are there
because I would rather show what the model cannot do than hide it.

Data: FastF1 (public F1 timing/telemetry). No team setup data exists publicly; every
setup axis is expressed as a change relative to the lap actually driven.
