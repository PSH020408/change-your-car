"""P2-5..P2-7 orchestrator — bronze + silver -> gold feature store.

One row per (lap x segment). Everything the model sees is assembled here, and
the run reports what it built rather than just that it finished: coverage,
imputation rates, effort distribution, and the target's spread.

Usage
-----
    python -m pipeline.features.run --scope configs/scope.yaml --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest import paths
from pipeline.features import driver_style, effort as effort_mod, segment_features, setup_proxy

LAP_CONTEXT = [
    "lap_uid", "season", "event", "event_slug", "session", "circuit",
    "driver", "team", "chassis", "power_unit", "lap_number", "stint",
    "lap_time_s", "compound", "tyre_life", "fresh_tyre", "condition",
    "track_status_flag", "telemetry_quality", "pace_ratio",
    "pace_ratio_session_best", "air_temp_c", "track_temp_c", "rainfall",
    "wind_speed_kph", "humidity_pct",
]


def build_session(bronze_dir: Path, silver_dir: Path, verbose: bool = False) -> pd.DataFrame:
    laps = pd.read_parquet(bronze_dir / "laps.parquet")
    tel = pd.read_parquet(bronze_dir / "telemetry.parquet")
    track = json.loads((silver_dir / "track.json").read_text())

    segments = track["segments"]
    lap_len = float(track["geometry"]["lap_length_m"])
    bounds = track.get("sector_boundaries_m") or []

    by_lap = dict(tuple(tel.groupby("lap_uid")))
    rows: list[pd.DataFrame] = []

    for _, lap in laps.iterrows():
        uid = str(lap["lap_uid"])
        lt = by_lap.get(uid)
        if lt is None or len(lt) < 20:
            continue
        lt = lt.sort_values("distance_m")

        seg_df = segment_features.features_for_lap(lt, segments, lap_len)
        if not len(seg_df):
            continue

        # Lap-level context, broadcast onto every segment row.
        for c in LAP_CONTEXT:
            if c in laps.columns:
                seg_df[c] = lap.get(c)

        # Effort: whether the driver was working the car on this lap at all.
        e = effort_mod.lap_effort(lt)
        for k, v in e.items():
            seg_df[f"lap_{k}"] = v
        seg_df["lap_effort_class"] = effort_mod.classify_effort(e["effort_index"])

        # Setup proxies, now that segments exist to locate the traps.
        for k, v in setup_proxy.impute_traps(lap, lt, segments, bounds, lap_len).items():
            seg_df[k] = v

        rows.append(seg_df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment delta against the session's best time in that segment.

    The reference is per (season, event, session, segment_index) so a delta
    always compares like with like. Only PUSH laps set the reference — a
    procession's cruise laps would otherwise define "best" and every genuine
    lap would look fast (2024 Monaco, DECISIONS.md D3).
    """
    if "segment_time_s" not in df:
        return df
    key = ["season", "event", "session", "segment_index"]
    push = df[df["lap_effort_class"].isin(["push", "moderate"])]
    ref = (push if len(push) else df).groupby(key)["segment_time_s"].min()
    df = df.merge(ref.rename("segment_reference_s"), on=key, how="left")
    df["segment_delta_s"] = df["segment_time_s"] - df["segment_reference_s"]
    return df


def run(scope_path: Path, limit: int | None, verbose: bool) -> int:
    scope = paths.load_scope(scope_path)
    bronze = paths.bronze_dir(scope_path, scope)
    silver = paths.lake_dir(scope_path, scope) / "silver"
    gold = paths.lake_dir(scope_path, scope) / "gold"

    sessions = []
    for track_json in sorted(silver.rglob("track.json")):
        rel = track_json.parent.relative_to(silver)
        b = bronze / rel
        if (b / "laps.parquet").exists():
            sessions.append((b, track_json.parent))
    if limit:
        sessions = sessions[:limit]

    print(f"bronze  : {bronze}")
    print(f"silver  : {silver}")
    print(f"gold    : {gold}")
    print(f"sessions: {len(sessions)}")
    print()

    frames, failures = [], []
    for i, (b, s) in enumerate(sessions, 1):
        label = "/".join(b.parts[-3:])
        try:
            df = build_session(b, s, verbose)
            if not len(df):
                failures.append({"session": label, "error": "no usable laps"})
                continue
            frames.append(df)
            print(f"[{i:>3}/{len(sessions)}] {label:<40s} {len(df):>7} rows  "
                  f"{df['lap_uid'].nunique():>4} laps x "
                  f"{df['segment_index'].nunique():>3} segments")
        except Exception as exc:                            # noqa: BLE001
            failures.append({"session": label, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"[{i:>3}/{len(sessions)}] {label:<40s} FAILED  {exc}"[:150],
                  file=sys.stderr)
            if verbose:
                traceback.print_exc()

    if not frames:
        print("\nno features built", file=sys.stderr)
        return 1

    feat = add_targets(pd.concat(frames, ignore_index=True))
    bias = driver_style.driver_bias(feat)

    gold.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(gold / "features.parquet", index=False, compression="zstd")
    if len(bias):
        bias.to_parquet(gold / "driver_bias.parquet", index=False, compression="zstd")

    # ---- what did we actually build? ------------------------------------
    print()
    print(f"FEATURE STORE   {len(feat):,} rows, {len(feat.columns)} columns")
    print(f"  laps          : {feat['lap_uid'].nunique():,}")
    print(f"  drivers       : {feat['driver'].nunique()}")
    print(f"  segment kinds : " + ", ".join(
        f"{k}={v}" for k, v in feat["segment_kind"].value_counts().items()))

    print()
    print("EFFORT  (the 2024 Monaco problem — cruise laps look normal to the pace gate)")
    for k, v in feat.groupby("lap_effort_class")["lap_uid"].nunique().items():
        print(f"  {k:<10} {v:>6} laps")

    print()
    print("SETUP PROXY imputation  (P1 flagged, P2 fills — measured null rates)")
    for t in setup_proxy.TRAPS:
        col = f"speed_{t}_imputed"
        if col in feat:
            per_lap = feat.groupby("lap_uid")[col].first()
            filled = float(per_lap.mean()) * 100
            present = float(feat.groupby("lap_uid")[f"speed_{t}_kph"].first().notna().mean()) * 100
            print(f"  speed_{t:<3} imputed {filled:>5.1f}% of laps, "
                  f"{present:>5.1f}% now present")

    print()
    print("TARGET  segment_delta_s")
    dlt = pd.to_numeric(feat.get("segment_delta_s"), errors="coerce").dropna()
    if len(dlt):
        print(f"  rows with a target : {len(dlt):,} ({100*len(dlt)/len(feat):.1f}%)")
        print(f"  median / p90 / max : {dlt.median():.3f} / "
              f"{dlt.quantile(0.9):.3f} / {dlt.max():.3f} s")
        neg = int((dlt < -1e-9).sum())
        print(f"  negative deltas    : {neg}  "
              + ("(should be 0 — the reference is a per-segment minimum)"
                 if neg else "(correct)"))

    thin = feat.isna().mean().sort_values(ascending=False).head(6)
    print()
    print("SPARSEST columns  (a feature this empty is not a feature)")
    for c, frac in thin.items():
        print(f"  {c:<32} {frac*100:>5.1f}% null")

    (gold / "feature_report.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(feat)), "columns": list(feat.columns),
        "laps": int(feat["lap_uid"].nunique()),
        "failures": failures,
        "null_fraction": {c: round(float(v), 4) for c, v in feat.isna().mean().items()},
    }, indent=1))
    print()
    print(f"gold -> {gold / 'features.parquet'}  ({len(failures)} failure(s))")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Build the gold feature store.")
    p.add_argument("--scope", type=Path, default=Path("configs/scope.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args()
    try:
        sys.exit(run(a.scope, a.limit, a.verbose))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
