"""F1 Virtual Sim — engine API.

Phase 2 deliverable. Routers are mounted as stubs so the frontend can
develop against a stable contract before the ML layer lands.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import baseline, engineer_log, meta, simulate

settings = get_settings()

app = FastAPI(
    title="F1 Virtual Sim Engine API",
    version="0.1.0",
    description="Setup + environment -> ML delta inference -> telemetry stream",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router)
app.include_router(baseline.router)
app.include_router(simulate.router)
app.include_router(engineer_log.router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
