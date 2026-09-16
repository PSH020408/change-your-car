.PHONY: expand setup setup-be setup-fe fe-check dev-be dev-fe recon warm warm-bg warm-status ingest segment features physics-check train test lint clean

setup: setup-be setup-fe

setup-be:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

setup-fe:
	@mkdir -p data/logs
	cd frontend && npm install 2>&1 | tee ../data/logs/setup-fe.log

# Type-check + lint + production build of the HUD. First real verification of P7.
fe-check:
	@mkdir -p data/logs
	cd frontend && (npx tsc --noEmit && npm run lint && npm run build) 2>&1 | tee ../data/logs/fe-check.log

dev-be:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-fe:
	cd frontend && npm run dev -- --port 3000

# --- P0.5 Reconnaissance -----------------------------------------------------
# Answer "what do we actually have?" before designing features.
recon:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.recon.survey --scope configs/scope.yaml --out ../docs/recon 2>&1 | tee ../data/logs/recon.log

# Every heavy target above/below writes its full output to data/logs/<target>.log
# as well as the terminal, so a run can be read back (by Claude, via the
# project folder) instead of pasted.
# Background cache warming — resumable, rate-limit aware. Run it and forget it.
warm-bg:
	cd backend && PYTHONUNBUFFERED=1 nohup .venv/bin/python -m pipeline.ingest.warm_cache \
		--scope configs/scope.yaml > ../data/cache/warm.log 2>&1 & \
		echo "warming started -> data/cache/warm.log"

warm:
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.ingest.warm_cache --scope configs/scope.yaml

warm-status:
	@pgrep -fl warm_cache >/dev/null && echo "warm_cache: RUNNING" || echo "warm_cache: NOT running"
	@python3 -c "import json,collections;d=json.load(open('backend/data/cache/_warm_ledger.json'));print('ledger  :',dict(collections.Counter(v['status'] for v in d.values())))" 2>/dev/null || true
	@tail -n 8 data/cache/warm.log 2>/dev/null || echo "no warm.log yet"

# --- P1 Ingestion ------------------------------------------------------------
# Reads only what warm_cache has already downloaded, so it makes ZERO network
# calls and can run while the background download is still going.
# Data expansion (2026-09-16): every cached Q/R session of the era through the
# lake. Each step tees its own log; make stops at the first failure.
#   ingest    ~118 new sessions -> bronze     (skips sessions already there)
#   segment   new events -> silver track.json (skips existing)
#   features  gold/features.parquet rebuilt   (~2.6 M rows expected)
#   baselines data/artifacts/baselines/*      (HUD picks these up on restart)
# Then `make train` separately, and read the gates before accepting.
expand: ingest segment features baselines
	@echo "expand done -> restart 'make api' to serve the new baselines, then 'make train'"

ingest-force:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.ingest.run --scope configs/scope.yaml --force --verbose 2>&1 | tee ../data/logs/ingest-force.log

ingest:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.ingest.run --scope configs/scope.yaml 2>&1 | tee ../data/logs/ingest.log

ingest-pilot:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.ingest.run --scope configs/scope.yaml --pilot --verbose 2>&1 | tee ../data/logs/ingest-pilot.log

ingest-one:
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.ingest.run --scope configs/scope.yaml --limit 1 --verbose

bronze-status:
	@python3 -c "import json,sys;\
d=json.load(open('data/bronze/manifest.json'));t=d['totals'];\
print(f\"sessions {t['sessions']}  laps {t['laps']:,}  samples {t['telemetry_rows']:,}  {t['bytes']/1e9:.2f} GB\")" \
	2>/dev/null || echo "no bronze manifest yet — run 'make ingest'"

# --- P2 Segmentation ---------------------------------------------------------
segment:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.segment.run --scope configs/scope.yaml 2>&1 | tee ../data/logs/segment.log

segment-all:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.segment.run --scope configs/scope.yaml --verbose 2>&1 | tee ../data/logs/segment-all.log

segment-calibrate:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.segment.run --scope configs/scope.yaml --force --verbose --calibrate 2>&1 | tee ../data/logs/segment-calibrate.log

segment-one:
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.segment.run --scope configs/scope.yaml --limit 1 --force --verbose

track-status:
	@python3 -c "import json,glob;\
fs=sorted(glob.glob('data/silver/*/*/track.json'));\
print(f'{len(fs)} circuit(s) segmented');\
[print(f\"  {d['season']} {d['event']:<30s} {d['counts']['turns']:>3} turns  {d['geometry']['lap_length_m']:>7.0f} m  ref {d['reference_lap']['session']}\") for d in (json.load(open(f)) for f in fs)]" \
	2>/dev/null || echo "no silver tracks yet — run 'make segment'"

silver-clean:
	rm -rf data/silver

# --- P2-5..P2-7 Feature store ------------------------------------------------
features:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.features.run --scope configs/scope.yaml --verbose 2>&1 | tee ../data/logs/features.log

physics-check:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.physics.calibrate --gold ../data/gold/features.parquet 2>&1 | tee ../data/logs/physics-check.log

features-one:
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.features.run --scope configs/scope.yaml --limit 1 --verbose

train:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.models.train --config configs/model.yaml 2>&1 | tee ../data/logs/train.log

reconstruct-demo:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.reconstruct.demo --season 2024 --event bahrain_grand_prix --session Q --driver VER 2>&1 | tee ../data/logs/reconstruct-demo.log

baselines:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m app.services.baselines build 2>&1 | tee ../data/logs/baselines.log

api-smoke:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m app.smoke 2>&1 | tee ../data/logs/api-smoke.log

api:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

train-quick:
	@mkdir -p data/logs
	cd backend && PYTHONUNBUFFERED=1 .venv/bin/python -m pipeline.models.train --config configs/model.yaml --quick 2>&1 | tee ../data/logs/train-quick.log

test:
	@mkdir -p data/logs
	cd backend && .venv/bin/pytest -q 2>&1 | tee ../data/logs/test.log

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint

clean:
	rm -rf data/bronze/* data/silver/* data/gold/*
