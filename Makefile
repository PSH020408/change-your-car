.PHONY: setup setup-be setup-fe dev-be dev-fe recon warm warm-bg warm-status ingest segment features train test lint clean

setup: setup-be setup-fe

setup-be:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

setup-fe:
	cd frontend && npm install

dev-be:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-fe:
	cd frontend && npm run dev

# --- P0.5 Reconnaissance -----------------------------------------------------
# Answer "what do we actually have?" before designing features.
recon:
	cd backend && .venv/bin/python -m pipeline.recon.survey --scope configs/scope.yaml --out ../docs/recon

# Background cache warming — resumable, rate-limit aware. Run it and forget it.
warm-bg:
	cd backend && nohup .venv/bin/python -m pipeline.ingest.warm_cache \
		--scope configs/scope.yaml > ../data/cache/warm.log 2>&1 & \
		echo "warming started -> data/cache/warm.log"

warm:
	cd backend && .venv/bin/python -m pipeline.ingest.warm_cache --scope configs/scope.yaml

warm-status:
	@tail -n 20 data/cache/warm.log 2>/dev/null || echo "no warm.log yet"

# --- P1-P4 pipeline ----------------------------------------------------------
ingest:
	cd backend && .venv/bin/python -m pipeline.ingest.run --scope configs/scope.yaml

segment:
	cd backend && .venv/bin/python -m pipeline.segment.run --scope configs/scope.yaml

features:
	cd backend && .venv/bin/python -m pipeline.features.run --scope configs/scope.yaml

train:
	cd backend && .venv/bin/python -m pipeline.models.train --config configs/model.yaml

test:
	cd backend && .venv/bin/pytest -q

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint

clean:
	rm -rf data/bronze/* data/silver/* data/gold/*
