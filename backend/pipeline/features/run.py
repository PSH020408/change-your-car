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

REFERENCE_QUANTILE = 0.02

LAP_CONTEXT = [
    "lap_uid", "season", "event", "event_slug", "session", "circuit",
    "driver", "team", "chassis", "power_unit", "lap_number", "stint",
    "lap_time_s", "compound", "tyre_life", "fresh_tyre", "condition",
    "track_status_flag", "telemetry_quality", "pace_ratio",
    "pace_ratio_session_best", "air_temp_c", "track_temp_c", "rainfall",
    "wind_speed_kph", "humidity_pct",
]


# A lap whose telemetry covers materially less (or more) of the circuit than
# the reference is not a lap, it is a fragment. Rescaling a half-lap to full
# length stretches half a circuit across the whole and every segment on it is
# fiction: the first 19-session build carried rescale factors up to 1.82 and a
# 92 s per-segment delta because of exactly this.
COVERAGE_MIN, COVERAGE_MAX = 0.95, 1.05


def build_session(bronze_dir: Path, track_json: Path, verbose: bool = False
                  ) -> tuple[pd.DataFrame, dict]:
    laps = pd.read_parquet(bronze_dir / "laps.parquet")
    tel = pd.read_parquet(bronze_dir / "telemetry.parquet")
    track = json.loads(track_json.read_text())

    segments = track["segments"]
    lap_len = float(track["geometry"]["lap_length_m"])
    bounds = track.get("sector_boundaries_m") or []

    by_lap = dict(tuple(tel.groupby("lap_uid")))
    rows: list[pd.DataFrame] = []
    skipped = {"too_few_samples": 0, "coverage": 0}

    for _, lap in laps.iterrows():
        uid = str(lap["lap_uid"])
        lt = by_lap.get(uid)
        if lt is None or len(lt) < 20:
            skipped["too_few_samples"] += 1
            continue
        raw_len = float(pd.to_numeric(lt["distance_m"], errors="coerce").max())
        coverage = raw_len / lap_len if lap_len else 0.0
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            skipped["coverage"] += 1
            continue
        lt = segment_features.align_distance(lt.sort_values("distance_m"), lap_len)

        seg_df = segment_features.features_for_lap(lt, segments, lap_len)
        seg_df["lap_distance_scale"] = float(lt["distance_scale"].iloc[0])
        seg_df["lap_telemetry_coverage"] = round(coverage, 4)
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

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), skipped


def add_targets(df: pd.DataFrame, gap_tolerance_s: float = 0.05,
                reference_quantile: float = REFERENCE_QUANTILE) -> pd.DataFrame:
    """Per-segment delta against a clean, robust reference for that segment.

    Three things the first version got wrong, all of them the same mistake in
    different clothes: taking `min` of a quantity we know is corrupted.

    1. A lap with a telemetry gap inside a segment has its time UNDERSTATED —
       the gap interval is excluded from the integration on purpose. Such a
       lap then wins the minimum and every honest lap is measured against a
       time nobody drove. Only laps with a clean segment may set the
       reference.
    2. `min` over 88 laps is the most outlier-sensitive statistic available.
       A low quantile keeps the same meaning with none of the fragility.
    3. Only push laps set it, so a procession cannot define "best"
       (2024 Monaco, DECISIONS.md D3).
    """
    if "segment_time_s" not in df:
        return df
    key = ["season", "event", "session", "segment_index"]

    eligible = df
    if "lap_effort_class" in df:
        push = df[df["lap_effort_class"].isin(["push", "moderate"])]
        eligible = push if len(push) else df
    if "segment_time_gap_s" in eligible:
        clean = eligible[pd.to_numeric(eligible["segment_time_gap_s"],
                                       errors="coerce").fillna(0).abs() <= gap_tolerance_s]
        eligible = clean if len(clean) else eligible

    ref = eligible.groupby(key)["segment_time_s"].quantile(reference_quantile)
    df = df.merge(ref.rename("segment_reference_s"), on=key, how="left")
    df["segment_delta_s"] = df["segment_time_s"] - df["segment_reference_s"]
    return df


def check_segment_times_sum_to_the_lap(df: pd.DataFrame) -> dict:
    """The invariant that would have caught a broken target immediately.

    A lap's segments tile it exactly once, so their times must add up to the
    lap time. If they do not, either the segmentation is leaking samples or
    the integration is wrong — and every delta built on top is meaningless.
    """
    if not {"segment_time_s", "lap_time_s", "lap_uid"} <= set(df.columns):
        return {"checked": False}
    per_lap = df.groupby("lap_uid").agg(
        summed=("segment_time_s", "sum"), lap=("lap_time_s", "first"))
    per_lap = per_lap[per_lap["lap"].notna() & (per_lap["summed"] > 0)]
    if not len(per_lap):
        return {"checked": False}
    err = (per_lap["summed"] - per_lap["lap"]).abs()
    rel = err / per_lap["lap"]
    return {
        "checked": True,
        "laps": int(len(per_lap)),
        "median_abs_error_s": round(float(err.median()), 4),
        "p95_abs_error_s": round(float(err.quantile(0.95)), 4),
        "median_rel_error_pct": round(float(rel.median()) * 100, 3),
        "laps_over_1pct": int((rel > 0.01).sum()),
        "passes": bool(rel.median() <= 0.01),
    }


def run(scope_path: Path, limit: int | None, verbose: bool) -> int:
    scope = paths.load_scope(scope_path)
    bronze = paths.bronze_dir(scope_path, scope)
    silver = paths.lake_dir(scope_path, scope) / "silver"
    gold = paths.lake_dir(scope_path, scope) / "gold"

    # Tracks are per circuit (silver/{season}/{event}); sessions are per
    # session (bronze/{season}/{event}/{session}). Every session of a weekend
    # reads the same track definition.
    sessions = []
    for sj in sorted(bronze.rglob("session.json")):
        meta = json.loads(sj.read_text())
        tj = silver / str(meta["season"]) / meta["event_slug"] / "track.json"
        if tj.exists():
            sessions.append((sj.parent, tj))
    if limit:
        sessions = sessions[:limit]

    print(f"bronze  : {bronze}")
    print(f"silver  : {silver}")
    print(f"gold    : {gold}")
    print(f"sessions: {len(sessions)}")
    print()

    frames, failures = [], []
    skipped_total = {"too_few_samples": 0, "coverage": 0}
    for i, (b, tj) in enumerate(sessions, 1):
        label = "/".join(b.parts[-3:])
        try:
            df, skipped = build_session(b, tj, verbose)
            for k, v in skipped.items():
                skipped_total[k] += v
            if not len(df):
                failures.append({"session": label, "error": "no usable laps"})
                continue
            frames.append(df)
            print(f"[{i:>3}/{len(sessions)}] {label:<40s} {len(df):>7} rows  "
                  f"{df['lap_uid'].nunique():>4} laps x "
                  f"{df['segment_index'].nunique():>3} segments"
                  + (f"  (skipped {skipped['coverage']} partial-coverage laps)"
                     if skipped["coverage"] else ""))
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
    print(f"  laps skipped  : {skipped_total['coverage']} partial telemetry coverage "
          f"(outside {COVERAGE_MIN:.0%}-{COVERAGE_MAX:.0%} of the circuit), "
          f"{skipped_total['too_few_samples']} too few samples")
    print(f"  drivers       : {feat['driver'].nunique()}")
    print(f"  segment kinds : " + ", ".join(
        f"{k}={v}" for k, v in feat["segment_kind"].value_counts().items()))

    print()
    print("EFFORT  (the 2024 Monaco problem — cruise laps look normal to the pace gate)")
    for k, v in feat.groupby("lap_effort_class")["lap_uid"].nunique().items():
        print(f"  {k:<10} {v:>6} laps")

    print()
    print("SETUP PROXY imputation  (P1 flagged, P2 fills)")
    print("  Reconnaissance measured these null on RAW laps: i1 20.8%, st 13.4%,")
    print("  fl 5.4%, i2 0.2%. A low fill rate here is expected, not a bug — a")
    print("  trap goes unrecorded on out-laps and cool-downs, which the filter")
    print("  chain has already removed.")
    for t in setup_proxy.TRAPS:
        col = f"speed_{t}_imputed"
        if col in feat:
            per_lap = feat.groupby("lap_uid")[col].first()
            filled = float(per_lap.mean()) * 100
            present = float(feat.groupby("lap_uid")[f"speed_{t}_kph"].first().notna().mean()) * 100
            print(f"  speed_{t:<3} imputed {filled:>5.1f}% of laps -> {present:>5.1f}% present")

    print()
    print("TARGET  segment_delta_s")
    dlt = pd.to_numeric(feat.get("segment_delta_s"), errors="coerce").dropna()
    if len(dlt):
        print(f"  rows with a target : {len(dlt):,} ({100*len(dlt)/len(feat):.1f}%)")
        print(f"  median / p90 / max : {dlt.median():.3f} / "
              f"{dlt.quantile(0.9):.3f} / {dlt.max():.3f} s")
        # The reference is a low QUANTILE, not a minimum, so a small share of
        # laps sitting below it is the definition working, not a defect. Only
        # an excess is worth reporting.
        neg = int((dlt < -1e-9).sum())
        expected = REFERENCE_QUANTILE * len(dlt)
        verdict = ("as expected for a q={:.2f} reference"
                   .format(REFERENCE_QUANTILE) if neg <= 2.5 * expected
                   else "MORE than the quantile explains — check the reference")
        print(f"  below reference    : {neg} ({100*neg/len(dlt):.1f}%)  {verdict}")

    # ---- the invariant --------------------------------------------------
    inv = check_segment_times_sum_to_the_lap(feat)
    print()
    print("INVARIANT  segment times must sum to the lap time")
    if inv.get("checked"):
        verdict = "PASS" if inv["passes"] else "FAIL"
        print(f"  {verdict}  median error {inv['median_abs_error_s']:.3f} s "
              f"({inv['median_rel_error_pct']:.2f}%), p95 {inv['p95_abs_error_s']:.3f} s, "
              f"{inv['laps_over_1pct']} lap(s) over 1%")
        if not inv["passes"]:
            print("  Segments are leaking samples or the integration is wrong — every")
            print("  delta built on top of this is meaningless until it passes.")
    else:
        print("  not checkable (missing columns)")

    if "lap_telemetry_coverage" in feat:
        cov = feat.groupby("lap_uid")["lap_telemetry_coverage"].first()
        print()
        print("DISTANCE AXIS alignment  (each lap integrates its own; rescaled to the circuit)")
        print(f"  coverage of surviving laps : {cov.min():.3f} - {cov.max():.3f} "
              f"(gate {COVERAGE_MIN}-{COVERAGE_MAX})")
        print(f"  rescale applied            : up to {max(abs(1-cov.min()), abs(1-cov.max()))*100:.1f}%")

    # ---- sparsity, but only where the column is supposed to exist ---------
    print()
    print("SPARSITY  (structural nulls excluded — a straight has no brake point)")
    kind = feat.get("segment_kind", pd.Series(dtype=str))
    n_s = pd.to_numeric(feat.get("n_samples", pd.Series(dtype=float)), errors="coerce")
    applicable = {
        # a brake point needs braking to have happened
        "brake_point_frac": feat.get("brake_frac", pd.Series(dtype=float)) > 0,
        # "where did full throttle return" only means something where it can:
        # a slow corner feeding another corner never reaches it, by design
        "throttle_on_frac": kind.isin(["straight", "kink", "high_speed_corner"]),
        # a longitudinal-g extreme needs at least three samples to difference
        "long_g_accel_max": n_s >= 3,
        "long_g_brake_max": n_s >= 3,
    }
    rows = []
    for c in feat.columns:
        mask = applicable.get(c)
        sub = feat[mask] if mask is not None and mask.any() else feat
        if not len(sub):
            continue
        rows.append((c, float(sub[c].isna().mean()), mask is not None))
    for c, frac, conditioned in sorted(rows, key=lambda r: -r[1])[:6]:
        tag = "  (where applicable)" if conditioned else ""
        print(f"  {c:<32} {frac*100:>5.1f}% null{tag}")

    (gold / "feature_report.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(feat)), "columns": list(feat.columns),
        "laps": int(feat["lap_uid"].nunique()),
        "failures": failures,
        "invariant_segment_times": inv,
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
