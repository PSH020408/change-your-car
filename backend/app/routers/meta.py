"""Catalog endpoints: seasons, events, drivers, chassis."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/seasons")
def seasons() -> list[int]:
    raise NotImplementedError("Phase 2 — backed by pipeline.ingest metadata table")


@router.get("/events/{season}")
def events(season: int) -> list[dict]:
    raise NotImplementedError("Phase 2")


@router.get("/drivers/{season}")
def drivers(season: int) -> list[dict]:
    raise NotImplementedError("Phase 2")


@router.get("/chassis/{season}")
def chassis(season: int) -> list[dict]:
    raise NotImplementedError("Phase 2 — chassis/PU mapping table")
