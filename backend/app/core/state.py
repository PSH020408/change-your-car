"""One Engine per process, built at startup from Settings."""
from __future__ import annotations

from functools import lru_cache
from threading import Lock

from app.config import get_settings
from app.services.engine import Engine


_initialization_lock = Lock()


@lru_cache(maxsize=1)
def _load_engine() -> Engine:
    s = get_settings()
    return Engine(baselines_dir=s.model_registry_dir / "baselines", model_dir=s.model_registry_dir / "models")


def get_engine() -> Engine:
    # lru_cache alone may execute the factory concurrently before the first
    # result is cached. Hold this lock around the cached call, not inside it.
    with _initialization_lock:
        return _load_engine()
