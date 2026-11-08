# Serving within a small instance

The API and static HUD run in one container. The serving changes below reduce
resource growth; they do not guarantee that a hosting platform will wake or route
requests correctly.

## Startup and concurrency

- Construct the engine once under a process-wide lock. `lru_cache` alone can run
  its factory more than once during simultaneous first requests.
- Load the engine during ASGI lifespan startup. Model loading failures now fail
  startup instead of appearing only after the HUD opens.
- Keep one Uvicorn worker, one inference thread per saved model, and one OpenMP /
  BLAS thread in the serving container. Training configuration and saved model
  artifacts are unchanged.

## Memory bounds

Recent entries stay in process-local LRU caches:

| Cache | Maximum entries |
|---|---:|
| Session JSON documents | 4 |
| Circuit line frames | 4 |
| Fitted laps | 8 |
| Strategy prediction values | 512 |

Eviction discards a cached copy, not a source file. Selecting an older session
loads the same committed data again; an evicted lap may need to be fitted again.
Factories run under their cache's lock so parallel requests do not build duplicate
copies. Smaller race metadata and per-segment feature tables have finite keys
derived from the committed event catalog.

## Verification on 2026-11-08

A local macOS / Python 3.11 process read all 184 committed session documents:

| Measure | Previous serving code | Bounded serving code |
|---|---:|---:|
| Documents retained after visiting all sessions | 184 | 4 |
| Process RSS after the traversal | 580.3 MiB | 190.7 MiB |
| Factory calls from four simultaneous cold requests, using a delayed test factory | 4 | 1 |

These are local process measurements, not Render's Linux memory measurements or
a concurrency capacity guarantee. A same-environment comparison of neutral and
changed-setup simulations produced identical response fields except `computed_ms`:
0.000 s and -0.101 s respectively, with 23 segments.

The serving/resource, API, QSS, strategy, physics and tyre checks passed 61 tests.
The frontend's separate six transport checks cover deadlines and cancellation.

## Operations

`/health` includes `process_peak_memory_mb` (the process high-water RSS, in MiB)
and `cached_sessions`. This high-water mark will not fall after eviction. It is
not the entire container's current memory use.

Check public requests and startup/application logs in addition to deployment
status. A Live deployment is not sufficient evidence of public availability.
The HUD's 90-second request deadline and Retry baseline control let visitors
recover from a failed request; they cannot repair a stalled platform instance.
