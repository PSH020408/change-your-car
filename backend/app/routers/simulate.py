"""Setup + environment -> physics modifiers -> ML delta -> telemetry."""
from fastapi import APIRouter

from app.schemas.domain import SimulationRequest, SimulationResponse

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


@router.post("", response_model=SimulationResponse)
def simulate(req: SimulationRequest) -> SimulationResponse:
    raise NotImplementedError("Phase 2 — wires physics layer + model registry")
