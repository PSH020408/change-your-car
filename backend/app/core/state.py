"""One Engine per process, built at startup from Settings."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.engine import Engine


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    return Engine(baselines_dir=s.model_registry_dir / "baselines", model_dir=s.model_registry_dir / "models")
