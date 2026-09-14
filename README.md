# F1 Virtual Sim

2021–2025 Ground Effect ERA 실제 F1 텔레메트리 기반의 **2D 엔지니어링
텔레메트리 & 셋업 시뮬레이터**.

드라이버·섀시를 고르고, 윙 각도와 차고·서스펜션을 조절하고, 트랙 온도와
날씨를 바꾸면 — ML 추론이 실제 레이스 기록 대비 **섹터별 델타**와
**재구성된 텔레메트리**를 돌려줍니다.

```
FastF1 ──► ingest ──► segment ──► features ──► XGBoost ──► FastAPI ──► 2D HUD
           bronze     silver       gold         artifacts
                                      ▲
                              physics modifiers
                          (sliders → ΔCl/ΔCd/Δgrip)
```

## Quick start

```bash
cp .env.example .env
make setup          # python venv + npm install
make dev-be         # FastAPI  :8000
make dev-fe         # Next.js  :3000
```

## Pipeline

```bash
make ingest         # FastF1 → data/bronze
make segment        # 코너/직선 분할 → data/silver
make features       # 피처 스토어 → data/gold
make train          # XGBoost → data/artifacts
make test           # 물리 단조성 + 계약 테스트
```

## Layout

| 경로 | 역할 |
|------|------|
| `backend/pipeline/` | 배치 데이터·ML 파이프라인 (Phase 1–4) |
| `backend/pipeline/physics/` | 셋업 슬라이더 → 물리 계수 변환 (추론 시점) |
| `backend/pipeline/reconstruct/` | 세그먼트 델타 → 연속 텔레메트리 |
| `backend/app/` | FastAPI 엔진 API |
| `frontend/src/` | Next.js 2D HUD |
| `data/` | 메달리온 레이크 (bronze/silver/gold) + 모델 아티팩트 |
| `configs/` | 파일럿 스코프, 모델 설정, 합격 기준 |

## 설계 원칙

1. **3D 라이브러리 금지.** 트랙맵·차량 도식은 손으로 쓴 SVG, 텔레메트리는 Canvas 2D.
2. **물리는 명시적으로, 응답은 학습으로.** 슬라이더→계수 매핑은 해석 가능한 수식이
   소유하고, ML은 "그 계수 변화에 차가 어떻게 반응하는가"만 학습합니다.
3. **게이트 없는 단계 진행 없음.** 각 단계의 합격 기준은 `ROADMAP.md`에 있고,
   기준 미달 모델은 레지스트리에 등록되지 않습니다.

전체 단계·태스크·진행도: **[ROADMAP.md](./ROADMAP.md)**
