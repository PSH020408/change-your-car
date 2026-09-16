"""F1 Virtual Sim — engine API.

P6. One Engine per process (baseline store + registered model + physics
config), four routers, one contract: app/schemas/domain.py.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import baseline, engineer_log, meta, simulate

settings = get_settings()

app = FastAPI(
    title="F1 Virtual Sim Engine API",
    version="0.6.0",
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
def health() -> dict:
    from app.core.state import get_engine
    e = get_engine()
    return {"status": "ok", "version": app.version, "model_version": e.model_version,
            "seasons": e.seasons()}
