"""Reference lap: real telemetry + track geometry for the HUD."""
from fastapi import APIRouter, Depends, HTTPException

from app.core.state import get_engine
from app.schemas.domain import BaselineRef, BaselineResponse
from app.services.engine import Engine, NotFound

router = APIRouter(prefix="/api/baseline", tags=["baseline"])


@router.get("", response_model=BaselineResponse)
def get_baseline(season: int, event: str, session: str = "Q", driver: str = "VER", lap: str = "representative",
                 e: Engine = Depends(get_engine)) -> BaselineResponse:
    try:
        return e.baseline_response(BaselineRef(season=season, event=event, session=session, driver=driver, lap=lap))
    except NotFound as exc:
        raise HTTPException(404, str(exc))
