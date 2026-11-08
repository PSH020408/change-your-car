"""Serving regressions for simultaneous cold requests and long browsing sessions."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
import time
from types import SimpleNamespace

import pytest

from app.core.cache import BoundedCache
from app.core import state
from app.services.engine import Engine


def test_recent_entries_survive_and_evicted_entries_reload():
    cache = BoundedCache(2)
    loads = []

    def load(key):
        return cache.get_or_create(key, lambda: loads.append(key) or object())

    first = load("first")
    load("second")
    assert load("first") is first
    load("third")
    assert len(cache) == 2
    load("second")
    assert loads == ["first", "second", "third", "second"]


def test_failed_load_does_not_poison_or_evict_a_successful_entry():
    cache = BoundedCache(1)
    existing = cache.get_or_create("good", object)
    with pytest.raises(ValueError):
        cache.get_or_create("bad", lambda: (_ for _ in ()).throw(ValueError("unavailable")))
    assert cache.get_or_create("good", object) is existing
    assert cache.get_or_create("bad", lambda: 42) == 42


def test_parallel_first_requests_construct_one_engine(monkeypatch, tmp_path):
    state._load_engine.cache_clear()
    calls = []
    gate = Barrier(4)

    def slow_engine(**kwargs):
        value = object()
        calls.append(value)
        time.sleep(0.05)
        return value

    monkeypatch.setattr(state, "Engine", slow_engine)
    monkeypatch.setattr(state, "get_settings", lambda: SimpleNamespace(model_registry_dir=tmp_path))

    def request(_):
        gate.wait()
        return state.get_engine()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(request, range(4)))
        assert len(calls) == 1
        assert all(result is results[0] for result in results)
    finally:
        state._load_engine.cache_clear()


def test_session_browsing_is_bounded_and_reload_keeps_original_data(tmp_path):
    (tmp_path / "index.json").write_text('{"seasons": {}}')
    for index in range(12):
        path = tmp_path / "2024" / f"event_{index}"
        path.mkdir(parents=True)
        (path / "Q.json").write_text(json.dumps({"marker": index}))
    engine = Engine(tmp_path, tmp_path / "missing-model")
    for index in range(12):
        assert engine._doc(2024, f"event_{index}", "Q") == {"marker": index}
        assert len(engine._cache) <= 4
    assert engine._doc(2024, "event_0", "Q") == {"marker": 0}
    assert len(engine._cache) == 4


def test_startup_loads_the_engine_before_accepting_requests(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    calls = []
    engine = SimpleNamespace(model_version="test", seasons=lambda: [], _cache=[])
    monkeypatch.setattr(state, "get_engine", lambda: calls.append(1) or engine)
    with TestClient(app) as client:
        assert calls == [1]
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["process_peak_memory_mb"] > 0


def test_invalid_model_prevents_a_successful_startup(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    def fail():
        raise RuntimeError("model could not load")

    monkeypatch.setattr(state, "get_engine", fail)
    with pytest.raises(RuntimeError, match="model could not load"):
        with TestClient(app):
            pass
