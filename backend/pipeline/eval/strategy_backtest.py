"""P11-3 — does the strategy model add up to a real race?

For every race with measured timing, every classified finisher whose laps are
all timed gets scored on the strategy they ACTUALLY ran (their real stint
lengths from the timing sidecar), from their own representative race lap, and
the predicted race total is compared with the real one (laps 2..N, so the
start and turn 1 are out on both sides). Only green-flag races count: a
safety car compresses every strategy alike and says nothing about the model.

Gates (design P11, written before the first run):
  S1  |predicted - real| / real: median <= 0.5 %, p90 <= 1.5 %   (classified, fully timed, green >= 0.90)
  S2  within a race, the model's ordering of strategy GROUPS (drivers sharing a
      compound sequence) agrees in sign with the real ordering of their mean race
      times in >= 70 % of group pairs
  S3  pit-loss distribution across circuits (reported)

Usage
-----
    python -m pipeline.eval.strategy_backtest
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas import domain as S
from app.services.engine import Engine, NotFound

GREEN_MIN = 0.90


def real_stints(laps: list[dict]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for l in sorted(laps, key=lambda x: x["lap_number"]):
        c = l.get("compound")
        if not c or c not in ("SOFT", "MEDIUM", "HARD"):
            return []
        if out and out[-1][0] == c and not l.get("pit_out"):
            out[-1] = (c, out[-1][1] + 1)
        else:
            out.append((c, 1))
    return out


def run(bronze: Path, baselines: Path, models: Path, out_dir: Path, limit: int | None, verbose: bool) -> int:
    eng = Engine(baselines, models)
    rows, groups_rows, pit_rows, skipped = [], [], [], []
    files = sorted(bronze.glob("*/*/R/race_timing.json"))
    for i, p in enumerate(files, 1):
        doc = json.loads(p.read_text())
        season, event = int(doc["season"]), doc["event_slug"]
        label = f"{season}/{event}"
        pl = doc.get("pit_loss") or {}
        if pl.get("median_s") is not None:
            pit_rows.append({"season": season, "event": event, "pit_loss_s": pl["median_s"], "n": pl["n"]})
        if doc["green_share"] < GREEN_MIN:
            skipped.append({"race": label, "reason": f"green share {doc['green_share']:.2f}"})
            continue
        n_ok = 0
        for drv, d in doc["drivers"].items():
            t = d["totals"]
            if not (t["classified_finisher"] and t["timed_laps"] >= t["laps"] - 1):
                continue
            stints = real_stints(d["laps"])
            if not stints or sum(n for _, n in stints) != t["laps"]:
                continue
            laps_sorted = sorted(d["laps"], key=lambda x: x["lap_number"])
            # Laps 2..N that ran under green (or a local yellow). A safety-car lap is 40 s
            # slow on the real side and nothing on the predicted side; it is removed from
            # BOTH, and a stop made under it is not charged its pit loss either.
            clean = [l for l in laps_sorted[1:] if l["lap_time_s"] is not None and not l.get("neutralised")]
            clean_nums = {l["lap_number"] for l in clean}
            real_2n = sum(l["lap_time_s"] for l in clean)
            n_neutral = len(laps_sorted) - 1 - len(clean)
            if not real_2n or len(clean) < 0.8 * (len(laps_sorted) - 1):
                continue
            try:
                req = S.StrategyRequest(baseline=S.BaselineRef(season=season, event=event, session="R", driver=drv, lap="representative"),
                                        stints=[S.Stint(compound=S.Compound(c), laps=n) for c, n in stints])
                res = eng.strategy.score(req, race_laps=t["laps"])      # lapped cars: the laps they ran
            except (NotFound, ValueError) as exc:
                skipped.append({"race": label, "driver": drv, "reason": str(exc)[:120]})
                continue
            if res.yours.refused:
                skipped.append({"race": label, "driver": drv, "reason": "; ".join(res.yours.refused)[:120]})
                continue
            charged_stops = sum(1 for l in res.yours.laps if l.pit_in and l.lap in clean_nums and (l.lap + 1) in clean_nums)
            pred_2n = sum(l.predicted_s for l in res.yours.laps if l.lap in clean_nums) + charged_stops * res.pit_loss_s
            rows.append({"season": season, "event": event, "driver": drv, "stints": " → ".join(f"{c} {n}" for c, n in stints),
                         "sequence": "→".join(c for c, _ in stints), "stops": res.yours.stops, "laps": t["laps"],
                         "neutral_laps": n_neutral, "charged_stops": charged_stops,
                         "real_s": round(real_2n, 3), "pred_s": round(pred_2n, 3), "err_s": round(pred_2n - real_2n, 3),
                         "rel_err": (pred_2n - real_2n) / real_2n, "unit_lap": res.unit_lap.lap_number,
                         "fuel_slope": res.unit_lap.fuel_slope_s_per_kg})
            n_ok += 1
        if verbose or True:
            print(f"[{i:>3}/{len(files)}] {label:<36s} green {doc['green_share']:.2f}  scored {n_ok:>2} drivers", flush=True)
        if limit and len(rows) >= limit:
            break

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "strategy_backtest.parquet", index=False)
    df.to_csv(out_dir / "strategy_backtest.csv", index=False)

    # S2: strategy groups inside a race
    agree, pairs = 0, 0
    for (season, event), g in df.groupby(["season", "event"]):
        gr = g.groupby("sequence").agg(real=("real_s", "mean"), pred=("pred_s", "mean"), n=("driver", "size"))
        gr = gr[gr["n"] >= 2]
        for a, b in itertools.combinations(gr.index, 2):
            dr, dp = gr.loc[a, "real"] - gr.loc[b, "real"], gr.loc[a, "pred"] - gr.loc[b, "pred"]
            if abs(dr) < 1.0:
                continue
            pairs += 1
            agree += int(np.sign(dr) == np.sign(dp))
            groups_rows.append({"season": season, "event": event, "a": a, "b": b, "real_gap_s": round(dr, 2), "pred_gap_s": round(dp, 2)})
    rel = df["rel_err"].abs()
    pit = pd.DataFrame(pit_rows)
    gates = {
        "S1_race_total": {"median_abs_pct": round(100 * float(rel.median()), 3), "p90_abs_pct": round(100 * float(rel.quantile(0.9)), 3),
                          "bias_pct": round(100 * float(df["rel_err"].median()), 3), "median_abs_s": round(float(df["err_s"].abs().median()), 1),
                          "drivers": int(len(df)), "races": int(df.groupby(["season", "event"]).ngroups),
                          "passes": bool(rel.median() <= 0.005 and rel.quantile(0.9) <= 0.015)},
        "S2_group_order": {"pairs": pairs, "agree_share": round(agree / pairs, 3) if pairs else None,
                           "passes": bool(pairs and agree / pairs >= 0.70)},
        "S3_pit_loss": {"races": int(len(pit)), "median_s": round(float(pit["pit_loss_s"].median()), 2) if len(pit) else None,
                        "min": (pit.sort_values("pit_loss_s").iloc[0][["event", "pit_loss_s"]].to_dict() if len(pit) else None),
                        "max": (pit.sort_values("pit_loss_s").iloc[-1][["event", "pit_loss_s"]].to_dict() if len(pit) else None),
                        "passes": True},
    }
    by_stops = df.groupby("stops")["rel_err"].agg(lambda e: float(np.median(np.abs(e)) * 100))
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "green_min": GREEN_MIN,
              "gates": gates, "by_stops_median_abs_pct": {int(k): round(v, 3) for k, v in by_stops.items()},
              "skipped": len(skipped), "skipped_examples": skipped[:10], "group_pairs": groups_rows[:50]}
    (out_dir / "strategy_backtest_report.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print()
    print(f"STRATEGY BACK-TEST  {len(df)} drivers / {gates['S1_race_total']['races']} green-flag races  (skipped {len(skipped)})")
    for k, g in gates.items():
        flag = "PASS" if g["passes"] else "FAIL"
        print(f"  [{flag}] {k:<16s} " + "  ".join(f"{kk}={vv}" for kk, vv in g.items() if kk != "passes"))
    print("  by stops: " + "  ".join(f"{k}-stop {v:.2f}%" for k, v in by_stops.items()))
    print(f"  -> {out_dir / 'strategy_backtest_report.json'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bronze", type=Path, default=Path("../data/bronze"))
    ap.add_argument("--baselines", type=Path, default=Path("../data/artifacts/baselines"))
    ap.add_argument("--models", type=Path, default=Path("../data/artifacts/models"))
    ap.add_argument("--out", type=Path, default=Path("../data/artifacts/strategy"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    sys.exit(run(a.bronze, a.baselines, a.models, a.out, a.limit, a.verbose))


if __name__ == "__main__":
    main()
