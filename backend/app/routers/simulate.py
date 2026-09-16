"""Setup + environment -> physics + ML -> deltas, trace, engineer log."""
from fastapi import APIRouter, Depends, HTTPException

from app.core.state import get_engine
from app.schemas.domain import SimulationRequest, SimulationResponse
from app.services.engine import Engine, NotFound

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


@router.post("", response_model=SimulationResponse)
def simulate(req: SimulationRequest, e: Engine = Depends(get_engine)) -> SimulationResponse:
    try:
        return e.simulate(req)
    except NotFound as exc:
        raise HTTPException(404, str(exc))
