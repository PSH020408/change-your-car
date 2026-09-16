"""P6 smoke on the real store, no server needed:  make api-smoke"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    c = TestClient(app)
    h = c.get("/health").json()
    print(f"health    : {h}")
    seasons = c.get("/api/meta/seasons").json()
    print(f"seasons   : {seasons}")
    ev = c.get(f"/api/meta/events/{seasons[-1]}").json()
    print(f"events    : {len(ev)} in {seasons[-1]}  e.g. {[e['event'] for e in ev[:4]]}")
    season, event, session, driver = 2024, "bahrain_grand_prix", "Q", "VER"
    drv = c.get(f"/api/meta/drivers/{season}/{event}/{session}").json()
    print(f"drivers   : {len(drv)} in {season} {event} {session}  e.g. {drv[0]}")
    t0 = time.perf_counter()
    b = c.get("/api/baseline", params=dict(season=season, event=event, session=session, driver=driver))
    print(f"baseline  : {b.status_code}  {(time.perf_counter() - t0) * 1000:.0f} ms  {len(b.content) // 1024} KB")
    bj = b.json()
    print(f"  lap {bj['lap']['lap_uid']}  {bj['lap']['lap_time_s']} s  {len(bj['trace']['distance_m'])} display samples, "
          f"{sum(bj['trace']['interpolated'])} interpolated  segments {len(bj['segments'])}")
    print(f"  {bj['integration_note']}")
    cases = [
        ("baseline", {}),
        ("rear wing 0.9", {"setup": {"rear_wing": 0.9}}),
        ("low ride height 0.1", {"setup": {"ride_height": 0.1}}),
        ("+40 kg fuel", {"setup": {"fuel_kg": 48}}),
        ("tyre 2 -> 20 laps", {"environment": {"tyre_life": 20}}),
        ("track +10 C", {"environment": {"track_temp_c": float(bj["lap"]["track_temp_c"] or 30) + 10}}),
        ("inter", {"environment": {"weather": "inter"}}),
        ("understeer", {"setup": {"front_wing": 0.0, "rear_wing": 1.0, "suspension_split": 1.0}}),
    ]
    for name, body in cases:
        t0 = time.perf_counter()
        r = c.post("/api/simulate", json={"baseline": dict(season=season, event=event, session=session, driver=driver), **body})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            print(f"  {name:<22} HTTP {r.status_code}: {r.text[:200]}"); continue
        j = r.json(); lap = j["lap"]
        print(f"  {name:<22} {lap['delta_s']:+.3f} s [{lap['delta_lo_s']:+.3f}, {lap['delta_hi_s']:+.3f}]  "
              f"setup {lap['physics_s']:+.3f}  ml {lap['ml_s']:+.3f}  l2 {lap['level2_s']:+.3f}  refused {lap['refused_s']:.3f}  "
              f"sectors {lap['sector_deltas_s']}  {ms:.0f} ms  {len(r.content) // 1024} KB")
        for n in j["engineer_log"][:3]:
            print(f"      [{n['severity']}/{n['channel']}] {n['message'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
