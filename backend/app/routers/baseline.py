"""Reference lap: real telemetry + track geometry for the HUD."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/baseline", tags=["baseline"])


@router.get("")
def get_baseline(season: int, event: str, session: str, driver: str) -> dict:
    """Returns TelemetryTrace + SVG path + segment table for the reference lap."""
    raise NotImplementedError("Phase 2 — reads data/gold baseline store")
