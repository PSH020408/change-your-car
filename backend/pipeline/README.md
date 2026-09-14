# Pipeline (Phase 1 – 4)

Medallion data lake. Each stage reads the previous layer and writes the next;
every stage is re-runnable and idempotent, keyed by `(season, event, session)`.

```
FastF1 API
   │  pipeline/ingest        session loader, lap filter, telemetry resample
   ▼
data/bronze/   raw per-session parquet  (laps, telemetry, weather, meta)
   │  pipeline/segment       centreline, curvature, corner/straight split
   ▼
data/silver/   segmented + cleaned      (laps, segments, telemetry_seg)
   │  pipeline/features      aggregation, driver-style bias, setup proxies
   ▼
data/gold/     model-ready feature store + baseline HUD payloads
   │  pipeline/models        train / evaluate / register
   ▼
data/artifacts/  model.json + metrics.json + feature_spec.json
```

`pipeline/physics` and `pipeline/reconstruct` are inference-time modules, not
batch stages: physics converts UI sliders into coefficient deltas, reconstruct
turns per-segment model deltas back into a continuous telemetry trace.
