"""Race baseline + a stint plan -> lap-by-lap race time, against the strategies teams really ran."""
from fastapi import APIRouter, Depends, HTTPException

from app.core.state import get_engine
from app.schemas.domain import StrategyRequest, StrategyResponse
from app.services.engine import Engine, NotFound

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


@router.post("", response_model=StrategyResponse)
def strategy(req: StrategyRequest, e: Engine = Depends(get_engine)) -> StrategyResponse:
    try:
        return e.strategy.score(req)
    except NotFound as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
