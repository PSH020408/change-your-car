"""Catalog: what the docking screen can choose from (from the baseline store index)."""
from fastapi import APIRouter, Depends, HTTPException

from app.core.state import get_engine
from app.services.engine import Engine, NotFound

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/seasons")
def seasons(e: Engine = Depends(get_engine)) -> list[int]:
    return e.seasons()


@router.get("/events/{season}")
def events(season: int, e: Engine = Depends(get_engine)) -> list[dict]:
    return e.events(season)


@router.get("/drivers/{season}/{event}/{session}")
def drivers(season: int, event: str, session: str, e: Engine = Depends(get_engine)) -> list[dict]:
    try:
        return e.drivers(season, event, session)
    except NotFound as exc:
        raise HTTPException(404, str(exc))


@router.get("/chassis/{season}")
def chassis(season: int, e: Engine = Depends(get_engine)) -> list[dict]:
    return e.chassis(season)
