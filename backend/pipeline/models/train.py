"""P4 — train, evaluate out-of-group, gate, register.

    make train            # full run (~10 min on 250k rows)
    make train-quick      # 3 folds, 150 trees — for iterating on features

Level 1: three GBMs (q10 / q50 / q90) on segment_delta_s vs the session
reference; GroupKFold on season|event plus one event held out entirely.
Level 2: ridge on the session reference vs the circuit's best.
Baseline: ridge on the same level-1 features — the GBM must beat it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.models import dataset as D, evaluate as E, level2 as L2, quantile as Q, registry as R


def group_folds(groups: pd.Series, n_folds: int) -> list[np.ndarray]:
    """GroupKFold without sklearn: greedy size-balanced assignment of groups."""
    sizes = groups.value_counts()
    fold_of: dict[str, int] = {}
    load = [0] * n_folds
    for g, n in sizes.items():
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += int(n)
    f = groups.map(fold_of).to_numpy()
    return [np.flatnonzero(f == k) for k in range(n_folds)]


def _report(title: str, m: dict) -> None:
    print(f"  {title:<10} segment MAE {m['segment_mae_s']:.3f} s  (median {m['segment_median_ae_s']:.3f})"
          f"   lap MAE {m['lap_mae_s']:.3f} s  (median {m['lap_median_ae_s']:.3f})"
          f"   coverage80 {m['coverage_80']:.2f}   width {m['interval_width_median_s']:.2f} s"
          f"   n={m['n']:,} / {m['laps']:,} laps")


def run(cfg_path: Path, quick: bool = False, register: bool = True) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(cfg_path.read_text())
    if quick:
        cfg["level1"]["n_folds"] = 3
        cfg["level1"]["params"]["n_estimators"] = 150
        cfg["level2"]["bootstrap"] = 50
    # relative paths in model.yaml are relative to backend/ (where make runs), not to configs/
    root_dir = cfg_path.resolve().parent.parent
    gold = Path(cfg["data"]["gold"])
    gold = gold if gold.is_absolute() else (root_dir / gold).resolve()
    df = D.load_gold(gold)
    print(f"gold      : {gold}  ({len(df):,} rows)")

    unstable = D.unstable_races(df) if cfg["data"].get("exclude_unstable_races", True) else set()
    if unstable:
        print(f"unstable  : {sorted(unstable)}")
    df = D.select_training_rows(df, cfg, unstable)

    hold_slug = cfg["data"].get("holdout_event")
    is_hold = df["event_slug"].astype(str) == str(hold_slug)
    train, hold = df[~is_hold].reset_index(drop=True), df[is_hold].reset_index(drop=True)
    print(f"holdout   : {hold_slug}  {len(hold):,} rows / {hold['lap_uid'].nunique():,} laps   "
          f"train {len(train):,} rows / {train['lap_uid'].nunique():,} laps, "
          f"{D.group_key(train).nunique()} events")

    spec = D.FeatureSpec.from_config(cfg).fit(train)
    X, y = spec.transform(train), train["segment_delta_s"].to_numpy(float)
    backend = Q.resolve_backend(cfg["level1"].get("backend", "auto"))
    params, quantiles = cfg["level1"]["params"], [float(q) for q in cfg["level1"]["quantiles"]]
    print(f"backend   : {backend}   features {len(spec.columns)}   quantiles {quantiles}")

    # ------------------------------------------------------------- CV
    folds = group_folds(D.group_key(train), int(cfg["level1"]["n_folds"]))
    oof = pd.DataFrame(np.nan, index=X.index, columns=["q10", "q50", "q90"])
    oof_ridge = np.full(len(X), np.nan)
    try:
        ridge_ok = True
        Q.make_ridge(spec)
    except ImportError:
        ridge_ok = False
    for k, val in enumerate(folds, 1):
        tr = np.setdiff1d(np.arange(len(X)), val)
        m = Q.QuantileSet(spec, quantiles, backend, params).fit(X.iloc[tr], y[tr])
        oof.iloc[val] = m.predict(X.iloc[val]).to_numpy()
        if ridge_ok:
            rg = Q.make_ridge(spec).fit(Q.ridge_frame(spec, X.iloc[tr]), y[tr])
            oof_ridge[val] = rg.predict(Q.ridge_frame(spec, X.iloc[val]))
        print(f"  fold {k}/{len(folds)}  val {len(val):>7,} rows   {time.time() - t0:5.0f}s")

    cv = {**E.segment_metrics(y, oof), **E.lap_metrics(train["lap_uid"], y, oof)}
    metrics: dict = {"cv": cv, "backend": backend, "n_train_rows": int(len(X)),
                     "n_train_laps": int(train["lap_uid"].nunique()), "quick": quick}
    print("\nOUT-OF-GROUP (GroupKFold on season|event)")
    _report("GBM q50", cv)
    if ridge_ok:
        rq = pd.DataFrame({"q10": oof_ridge, "q50": oof_ridge, "q90": oof_ridge})
        ridge = {**E.segment_metrics(y, rq), **E.lap_metrics(train["lap_uid"], y, rq)}
        metrics["ridge"] = ridge
        print(f"  {'Ridge':<10} segment MAE {ridge['segment_mae_s']:.3f} s   lap MAE {ridge['lap_mae_s']:.3f} s"
              f"   (the linear floor the GBM must beat)")
    naive = pd.DataFrame({c: np.full(len(y), np.median(y)) for c in ("q10", "q50", "q90")})
    metrics["naive_median"] = E.segment_metrics(y, naive)
    print(f"  {'Naive':<10} segment MAE {metrics['naive_median']['segment_mae_s']:.3f} s   (predict the median delta)")

    print("\n  by segment kind (OOF):")
    bk = E.by_kind(train["segment_kind"], y, oof)
    print("   " + bk.round(3).to_string().replace("\n", "\n   "))
    metrics["by_kind"] = bk.round(4).to_dict(orient="index")
    bs = E.by_kind(train["session"], y, oof)
    print("  by session (OOF):")
    print("   " + bs.round(3).to_string().replace("\n", "\n   "))
    metrics["by_session"] = bs.round(4).to_dict(orient="index")

    # ------------------------------------------------------- final + holdout
    final = Q.QuantileSet(spec, quantiles, backend, params).fit(X, y)
    if len(hold):
        Xh, yh = spec.transform(hold), hold["segment_delta_s"].to_numpy(float)
        ph = final.predict(Xh)
        ho = {**E.segment_metrics(yh, ph), **E.lap_metrics(hold["lap_uid"], yh, ph)}
        metrics["holdout"] = {**ho, "event": str(hold_slug)}
        print(f"\nUNSEEN TRACK  ({hold_slug}, never in training)")
        _report("GBM q50", ho)
        unseen_drivers = int((~hold["driver"].astype(str).isin(spec.vocab.get("driver", []))).sum())
        if unseen_drivers:
            print(f"  ({unseen_drivers:,} rows from drivers unseen in training -> category NaN)")

    # ------------------------------------------------------------- probes
    print("\nMONOTONICITY PROBES  (mean change of q50; + = slower)")
    probes = {}
    probes["tyre_life"] = E.monotonic_probe(final, X, "tyre_life", +5.0)
    is_race = (train["session"] == "R")
    if is_race.any():
        probes["lap_number"] = E.monotonic_probe(final, X, "lap_number", +10.0, subset=is_race)
    for k, p in probes.items():
        print(f"  {k:<12} {p['shift']:+.0f} -> {p['mean_delta_s']:+.4f} s   ({p['share_slower']:.0%} of rows slower, n={p['n']:,})")
    metrics["probes"] = probes

    print("\nPERMUTATION IMPORTANCE  (MAE increase when shuffled, 20k-row train sample)")
    imp = E.permutation_importance(final, X, y)
    print("   " + imp.head(10).round(4).to_string(index=False).replace("\n", "\n   "))
    metrics["importance"] = imp.round(5).to_dict(orient="records")

    # ------------------------------------------------------------- level 2
    print("\nLEVEL 2  (session reference vs the circuit's best, % of segment time)")
    l2, t2 = L2.fit_level2(df, alpha=float(cfg["level2"].get("ridge_alpha", 1.0)),
                           n_boot=int(cfg["level2"].get("bootstrap", 200)))
    if l2.coef:
        desc = L2.describe(l2)
        print(f"  {l2.n_sessions} sessions on circuits with >1 session/season, {l2.n_rows} rows")
        print("   " + desc.to_string(index=False).replace("\n", "\n   "))
        metrics["level2"] = desc.to_dict(orient="records")
    else:
        print(f"  not enough informative sessions ({l2.n_rows} rows) — level 2 is identity until more data lands")
        metrics["level2"] = []

    # ------------------------------------------------------------- gates
    g = E.gates(metrics, cfg)
    metrics["gates"] = g
    passed = all(g.values())
    print("\nGATES")
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  -> {'ALL PASSED' if passed else 'NOT REGISTERED AS LATEST'}")

    if register:
        root = Path(cfg["registry"])
        root = root if root.is_absolute() else (root_dir / root).resolve()
        version = R.register(root, final, l2, metrics, cfg, passed)
        print(f"\nregistry  : {root / version}   {'(latest)' if passed else '(kept for inspection)'}")
    print(f"total     : {time.time() - t0:.0f}s")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    run(a.config, quick=a.quick, register=not a.no_register)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
