#!/usr/bin/env python3
"""PIPELINE TEST ONLY: write a synthetic field-test folder (node logs + flight log) from the
simulator, so field_analysis.py can be checked before real flights.
These files are NOT measurements and must never be reported as field results.

Usage: python3 make_test_dataset.py out_folder            (simulated flight, log in DJI export format)
       python3 make_test_dataset.py out_folder --drive    (ground drive-through, GPS track CSV)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from aerofence.sim import GENUINE, Params, receiver_positions, simulate_drone  # noqa: E402

LAT0, LON0 = 26.2185, 50.0560      # arbitrary reference point (PMU campus area)
R_E = 6378137.0


def to_ll(x, y):
    return LAT0 + np.degrees(y / R_E), LON0 + np.degrees(x / (R_E * np.cos(np.radians(LAT0))))


def main(out: Path):
    rng = np.random.default_rng(42)
    p = Params(L=200, L_warn=300, Nr=4, h_r=1.5, T_obs=240, Tb=1.0, eps=0.05, sigma_gnss=2.0,
               Pt_dbm=18, n_pl=2.3, shadow_db=4)
    rx = receiver_positions(p) * np.array([1.3, 1.3, 1])      # nodes 30 m outside the corners
    # a straight pass through the fence (scaled-down transit)
    d = None
    while d is None or len(d.t_arr) < 50:
        d = simulate_drone(rng, p, rx, GENUINE, "transit")
    t0 = 1_800_000_000.0
    (out / "logs").mkdir(parents=True, exist_ok=True)
    names = [f"N{i + 1}" for i in range(len(rx))]
    rows = {n: [] for n in names}
    for i in range(len(d.t_tx)):
        lat, lon = to_ll(d.rp[i, 0], d.rp[i, 1])
        sp = float(np.hypot(*d.rv[i, :2])); di = float(np.degrees(np.arctan2(d.rv[i, 0], d.rv[i, 1])) % 360)
        tx = t0 + d.t_tx[i]
        for j, n in enumerate(names):
            if d.ok[i, j]:
                rxt = tx + 0.02 + rng.exponential(0.03)
                rows[n].append({"node": n, "rx_time_unix": rxt, "bearer": "wifi_beacon",
                                "src_mac": "60:60:1f:00:00:01", "rssi_dbm": round(d.rssi[i, j]),
                                "msg_type": 1, "uas_id": "TEST-ONLY-SYNTHETIC", "lat": lat, "lon": lon,
                                "alt_geo_m": d.rp[i, 2], "speed_h_ms": sp, "speed_v_ms": 0.0,
                                "direction_deg": di, "rid_ts_unix": tx,
                                "rid_ts_tenths": int(round((tx % 3600) * 10))})
    for n in names:
        pd.DataFrame(rows[n]).to_csv(out / "logs" / f"rid_{n}_SYNTHETIC.csv", index=False)
    lat, lon = to_ll(d.true_xy[:, 0], d.true_xy[:, 1])
    t = t0 + d.t_grid
    fl = pd.DataFrame({"datetime(utc)": pd.to_datetime(np.floor(t), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                       "time(millisecond)": np.round((t - t0) * 1000).astype(int),
                       "latitude": lat, "longitude": lon, "height_above_takeoff(feet)": 60 / 0.3048,
                       "isFlying": d.t_grid >= d.t_on})
    fl.to_csv(out / "flight.csv", index=False)
    nl = {n: dict(zip(("lat", "lon"), to_ll(rx[j, 0], rx[j, 1])), h=1.5) for j, n in enumerate(names)}
    (out / "nodes.json").write_text(json.dumps(nl, indent=1))
    (out / "test.json").write_text(json.dumps({
        "fence_center": [LAT0, LON0], "fence_side_m": 200, "warn_side_m": 300,
        "uas_id": "TEST-ONLY-SYNTHETIC", "flight_log": "flight.csv",
        "power_on_utc": pd.Timestamp(t0 + d.t_on, unit="s", tz="UTC").isoformat(),
        "note": "SYNTHETIC PIPELINE TEST, NOT FIELD DATA"}, indent=1))
    print("wrote", out)


def drive(out: Path):
    """Synthetic drive-through: a car-mounted beacon crosses a 200-m virtual zone 6 times."""
    rng = np.random.default_rng(7)
    p = Params(L=200, L_warn=300, Nr=4, h_r=1.5)
    rx = receiver_positions(p) * np.array([1.3, 1.3, 1])
    names = [f"N{i + 1}" for i in range(len(rx))]
    # back-and-forth passes along y = 20 m, from x = -350 to +350 m at 10 m/s, 15-s turnarounds
    seg = []
    t = 0.0
    for k in range(6):
        xs = np.arange(-350, 350, 1.0) * (1 if k % 2 == 0 else -1)
        seg.append(np.column_stack([t + np.arange(len(xs)) / 10.0, xs, np.full(len(xs), 20.0)]))
        t = seg[-1][-1, 0] + 15.0
        seg.append(np.column_stack([np.arange(seg[-1][-1, 0] + 0.1, t, 0.1),
                                    np.full(len(np.arange(seg[-1][-1, 0] + 0.1, t, 0.1)), xs[-1]),
                                    np.full(len(np.arange(seg[-1][-1, 0] + 0.1, t, 0.1)), 20.0)]))
    tr = np.vstack(seg)
    t0 = 1_800_000_000.0
    out.mkdir(parents=True, exist_ok=True)
    # 1-Hz GPS track in the template format
    g = tr[np.isclose(tr[:, 0] % 1.0, 0, atol=0.05) | np.isclose(tr[:, 0] % 1.0, 1, atol=0.05)]
    lat, lon = to_ll(g[:, 1], g[:, 2])
    vx = np.gradient(g[:, 1], g[:, 0]); vy = np.gradient(g[:, 2], g[:, 0])
    pd.DataFrame({"timestamp": pd.to_datetime(t0 + g[:, 0], unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                  "latitude": lat, "longitude": lon, "altitude_m": 12.0,
                  "speed_mps": np.round(np.hypot(vx, vy), 2),
                  "heading_deg": np.round(np.degrees(np.arctan2(vx, vy)) % 360, 1),
                  "data_label": "SYNTHETIC_EXAMPLE_NOT_MEASURED"}).to_csv(out / "gps_track.csv", index=False)
    # beacon broadcasts at 1 Hz from the true position, received with the simulator's link model
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rows = {n: [] for n in names}
    for tb in np.arange(0, tr[-1, 0], 1.0):
        x, y = np.interp(tb, tr[:, 0], tr[:, 1]), np.interp(tb, tr[:, 0], tr[:, 2])
        vx_ = (np.interp(tb + 0.5, tr[:, 0], tr[:, 1]) - np.interp(tb - 0.5, tr[:, 0], tr[:, 1]))
        vy_ = (np.interp(tb + 0.5, tr[:, 0], tr[:, 2]) - np.interp(tb - 0.5, tr[:, 0], tr[:, 2]))
        ex, ey = rng.normal(0, 2.0, 2)
        la, lo = to_ll(x + ex, y + ey)
        for j, n in enumerate(names):
            d = max(np.hypot(x - rx[j, 0], y - rx[j, 1]), 1.0)
            rssi = 18 - (40.05 + 10 * 2.6 * np.log10(d)) + rng.normal(0, 5)
            if rssi < -94 or rng.random() < 0.05:
                continue
            rows[n].append({"node": n, "rx_time_unix": t0 + tb + 0.02 + rng.exponential(0.03),
                            "bearer": "wifi_beacon", "src_mac": "60:60:1f:00:00:02", "rssi_dbm": round(rssi),
                            "msg_type": 1, "uas_id": "TEST-ONLY-SYNTHETIC-DRIVE", "lat": la, "lon": lo,
                            "alt_geo_m": 12.0, "speed_h_ms": float(np.hypot(vx_, vy_)), "speed_v_ms": 0.0,
                            "direction_deg": float(np.degrees(np.arctan2(vx_, vy_)) % 360),
                            "rid_ts_unix": t0 + tb, "rid_ts_tenths": int(round(((t0 + tb) % 3600) * 10))})
    for n in names:
        pd.DataFrame(rows[n]).to_csv(out / "logs" / f"rid_{n}_SYNTHETIC.csv", index=False)
    # transmitter-side log: the independent denominator of the packet reception ratio
    pd.DataFrame({"rid_ts_unix": t0 + np.arange(0, tr[-1, 0], 1.0),
                  "data_label": "SYNTHETIC_EXAMPLE_NOT_MEASURED"}).to_csv(out / "tx_log.csv", index=False)
    nl = {n: dict(zip(("lat", "lon"), to_ll(rx[j, 0], rx[j, 1])), h=1.5) for j, n in enumerate(names)}
    (out / "nodes.json").write_text(json.dumps(nl, indent=1))
    (out / "test.json").write_text(json.dumps({
        "fence_center": [LAT0, LON0], "fence_side_m": 200, "warn_side_m": 300,
        "uas_id": "TEST-ONLY-SYNTHETIC-DRIVE", "flight_log": "gps_track.csv", "tx_height_m": 1.5,
        "tx_log": "tx_log.csv", "clock_offset_s": {n: 0.0 for n in names},
        "note": "SYNTHETIC PIPELINE TEST, NOT FIELD DATA"}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    o = Path(sys.argv[1] if len(sys.argv) > 1 else "test_dataset")
    drive(o) if "--drive" in sys.argv else main(o)
