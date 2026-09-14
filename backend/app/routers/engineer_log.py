"""Rule-based race-engineer commentary on a setup."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/engineer-log", tags=["engineer"])


@router.post("")
def engineer_log(payload: dict) -> list[dict]:
    raise NotImplementedError("Phase 2 — balance/tyre/aero rule engine")
