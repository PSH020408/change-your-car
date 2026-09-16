"""The engineer log alone (same computation as /simulate, log only)."""
from fastapi import APIRouter, Depends, HTTPException

from app.core.state import get_engine
from app.schemas.domain import EngineerNote, SimulationRequest
from app.services.engine import Engine, NotFound

router = APIRouter(prefix="/api/engineer-log", tags=["engineer-log"])


@router.post("", response_model=list[EngineerNote])
def engineer_log(req: SimulationRequest, e: Engine = Depends(get_engine)) -> list[EngineerNote]:
    try:
        return e.simulate(req).engineer_log
    except NotFound as exc:
        raise HTTPException(404, str(exc))
