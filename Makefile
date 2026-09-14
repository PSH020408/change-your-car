.PHONY: setup setup-be setup-fe dev-be dev-fe ingest segment features train test lint clean

setup: setup-be setup-fe

setup-be:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

setup-fe:
	cd frontend && npm install

dev-be:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-fe:
	cd frontend && npm run dev

# --- Pipeline stages (Phase 1-4) ---
ingest:
	cd backend && .venv/bin/python -m pipeline.ingest.run --scope configs/pilot_scope.yaml

segment:
	cd backend && .venv/bin/python -m pipeline.segment.run --scope configs/pilot_scope.yaml

features:
	cd backend && .venv/bin/python -m pipeline.features.run --scope configs/pilot_scope.yaml

train:
	cd backend && .venv/bin/python -m pipeline.models.train --config configs/model.yaml

test:
	cd backend && .venv/bin/pytest -q

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint

clean:
	rm -rf data/bronze/* data/silver/* data/gold/*
