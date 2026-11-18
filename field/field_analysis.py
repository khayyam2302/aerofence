#!/usr/bin/env python3
"""Analyse an AeroFence field test: receiver logs + drone flight log -> KPIs and figures.

Inputs (one folder per test day):
  logs/*.csv        files written by rid_receiver.py on each node
  nodes.json        {"N1": {"lat": .., "lon": .., "h": 1.5}, ...}   (surveyed node positions)
  test.json         {"fence_center": [lat, lon], "fence_side_m": 200, "warn_side_m": 300,
                     "uas_id": "<serial number shown in the app>",
                     "power_on_utc": "2026-12-01T08:15:02Z",            (optional)
                     "flight_log": "flight.csv", "flight_log_utc_offset_h": 3,
                     "clock_offset_s": {"N1": 0.012, "N2": -0.004},     (optional, see below)
                     "tx_log": "tx.csv" | "reference_node": "N1" | "broadcast_hz": 1}
                                                                           (optional, see below)
  flight.csv        ground truth, one of:
                    - drone flight log as CSV: DJI Airdata or PhantomHelp Log Viewer export, or the GPS-track format
                    - GPS track CSV for ground tests (drive-through or carried beacon), columns
                      timestamp (ISO 8601, UTC), latitude, longitude, altitude_m, speed_mps,
                      heading_deg, data_label. Set "tx_height_m" in test.json (default 1.5 m).
                      Rows whose data_label contains "SYNTHETIC" mark the whole run as synthetic.

Time quantities. Three clocks are involved and are kept apart:
  rid_ts_unix   the time stamp the drone puts in each Location message (GNSS time);
  rx_time_unix  the node's own clock when the frame was captured;
  clock offset  how far each node's clock is from UTC ("System time" in `chronyc tracking`,
                positive when the node clock is fast), entered per node as clock_offset_s.
  The receive delay rx_time_unix - rid_ts_unix therefore contains the node's clock offset,
  the drone's own time-stamping error, and the capture and processing delay. It is reported
  as measured and, when clock_offset_s is given, also with the offset removed. It is not a
  network latency: no backhaul is involved in these logs.

Packet reception ratio. A receiver network cannot see broadcasts that every node missed, so
the denominator must come from an independent record. In order of preference:
  tx_log          transmitter-side log (CSV with column rid_ts_unix, or rid_ts_tenths plus
                  tx_time_unix), e.g. the serial log of an ESP32 ArduRemoteID beacon;
  reference_node  a node placed within a few metres of the transmitter, whose reception is
                  taken as the transmitted set;
  broadcast_hz    the nominal Location message rate (default 1 Hz). This is an assumption:
                  if the aircraft sends more often, the ratio is overstated (and may exceed 1).
  The method used is written to summary.json as prr_denominator.

Outputs (folder results/):
  summary.json      TTFF, receive delay, reception, position error, path-loss fit, alert latency
  per_node.csv      the same per node and for the network
  fig_*.pdf         receive-delay CDF, reception vs distance, RSSI vs distance, track vs truth

The offline fusion uses exactly the AeroFence code of the simulation study
(aerofence/fusion.py), so simulated and measured KPIs are computed the same way.

Usage:  python3 field_analysis.py path/to/test_folder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from aerofence.fusion import epoch_features, run_fusion, track  # noqa: E402
from aerofence.metrics import drone_kpis  # noqa: E402
from aerofence.sim import GENUINE, Params, zone_of  # noqa: E402

R_E = 6378137.0


def _rid_tenths_to_unix(tenths: int, ref_unix: float) -> float:
    """Tenths of a second after the UTC hour, placed in the hour nearest to ref_unix."""
    hour0 = ref_unix - (ref_unix % 3600)
    t = hour0 + tenths / 10.0
    return t - 3600 if t - ref_unix > 1800 else t + 3600 if ref_unix - t > 1800 else t


def enu(lat, lon, lat0, lon0):
    x = np.radians(np.asarray(lon) - lon0) * R_E * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_E
    return x, y


# --------------------------------------------------------------------------- inputs
def load_flight_log(path: Path, utc_offset_h: float, tx_height_m: float = 1.5) -> pd.DataFrame:
    """Return DataFrame with t_unix, lat, lon, h_m (height above ground), flying flag.
    The attribute .attrs["synthetic"] is True if the truth file is labelled synthetic."""
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    synthetic = False
    if "timestamp" in cols and "latitude" in cols and "longitude" in cols:   # GPS track (ground test)
        t = pd.to_datetime(df[cols["timestamp"]], utc=True, format="ISO8601")
        lat, lon = df[cols["latitude"]], df[cols["longitude"]]
        h = pd.Series(tx_height_m, index=df.index)                 # transmitter height above ground
        fly = pd.Series(True, index=df.index)                      # whole track is the test window
        if "data_label" in cols:
            synthetic = bool(df[cols["data_label"]].astype(str).str.upper().str.contains("SYNTHETIC").any())
    elif "datetime(utc)" in cols:                                   # Airdata export
        t = pd.to_datetime(df[cols["datetime(utc)"]], utc=True)
        if "time(millisecond)" in cols:
            t = t.dt.floor("s") + pd.to_timedelta(df[cols["time(millisecond)"]] % 1000, unit="ms")
        lat, lon = df[cols["latitude"]], df[cols["longitude"]]
        hcol = next(c for c in df.columns if c.lower().startswith("height_above_takeoff"))
        h = df[hcol] * (0.3048 if "feet" in hcol.lower() else 1.0)
        fly = df[cols["isflying"]].astype(bool) if "isflying" in cols else h > 0.5
    else:                                                         # PhantomHelp Log Viewer
        d = df[cols["custom.date [local]"]].astype(str)
        tt = df[cols["custom.updatetime [local]"]].astype(str)
        t = pd.to_datetime(d + " " + tt, errors="coerce") - pd.Timedelta(hours=utc_offset_h)
        t = t.dt.tz_localize("UTC")
        lat, lon = df[cols["osd.latitude"]], df[cols["osd.longitude"]]
        hcol = next(c for c in df.columns if c.lower().startswith("osd.height"))
        h = df[hcol] * (0.3048 if "[ft]" in hcol.lower() else 1.0)
        fly = df[cols["osd.isflying"]].astype(str).str.lower().eq("true") if "osd.isflying" in cols else h > 0.5
    t_unix = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    out = pd.DataFrame({"t_unix": t_unix, "lat": lat, "lon": lon, "h_m": h,
                        "flying": fly}).dropna()
    out = out[(out.lat != 0) & (out.lon != 0)].sort_values("t_unix").drop_duplicates("t_unix")
    out = out.reset_index(drop=True)
    out.attrs["synthetic"] = synthetic
    return out


def load_logs(folder: Path, uas_id: str | None) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(folder.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    if uas_id:
        df = df[df.uas_id.astype(str).str.strip() == uas_id]
    return df


# --------------------------------------------------------------------------- KPIs
def analyse(folder: Path):
    cfg = json.loads((folder / "test.json").read_text())
    nodes = json.loads((folder / "nodes.json").read_text())
    names = sorted(nodes)
    lat0, lon0 = cfg["fence_center"]
    fl = load_flight_log(folder / cfg["flight_log"], cfg.get("flight_log_utc_offset_h", 0),
                         cfg.get("tx_height_m", 1.5))
    synthetic = fl.attrs.get("synthetic", False) or "SYNTHETIC" in str(cfg.get("note", "")).upper() \
        or "SYNTHETIC" in str(cfg.get("uas_id", "")).upper()
    if synthetic:
        print("WARNING: SYNTHETIC input. Results are a pipeline test, NOT measurements.")
    logs = load_logs(folder / "logs", cfg.get("uas_id"))
    loc = logs[(logs.msg_type == 1) & logs.lat.notna()].copy()
    if loc.empty:
        sys.exit("no Location messages for this UAS ID in the logs")
    out_dir = folder / "results"
    out_dir.mkdir(exist_ok=True)

    t_fly = fl.t_unix[fl.flying]
    t_start = float(pd.Timestamp(cfg["power_on_utc"]).timestamp()) if cfg.get("power_on_utc") \
        else float(t_fly.iloc[0])
    t_end = float(t_fly.iloc[-1])
    nx, ny = enu([nodes[n]["lat"] for n in names], [nodes[n]["lon"] for n in names], lat0, lon0)
    rx = np.column_stack([nx, ny, [nodes[n].get("h", 1.5) for n in names]])

    # truth at each message (interpolated flight log; flight-log height + small take-off offset)
    fx, fy = enu(fl.lat.values, fl.lon.values, lat0, lon0)
    loc["x_true"] = np.interp(loc.rid_ts_unix, fl.t_unix, fx)
    loc["y_true"] = np.interp(loc.rid_ts_unix, fl.t_unix, fy)
    loc["h_true"] = np.interp(loc.rid_ts_unix, fl.t_unix, fl.h_m)
    loc["x"], loc["y"] = enu(loc.lat.values, loc.lon.values, lat0, lon0)
    loc["pos_err_m"] = np.hypot(loc.x - loc.x_true, loc.y - loc.y_true)
    # receive delay: node clock minus the drone's broadcast time stamp (see module docstring)
    offs = cfg.get("clock_offset_s", {})
    loc["rx_delay_s"] = loc.rx_time_unix - loc.rid_ts_unix
    loc["rx_delay_corr_s"] = loc.rx_delay_s - loc.node.map(lambda n: offs.get(str(n), np.nan))
    ni = {n: i for i, n in enumerate(names)}
    loc["node_i"] = loc.node.map(ni)
    loc["dist_m"] = np.sqrt((loc.x_true - rx[loc.node_i, 0]) ** 2 + (loc.y_true - rx[loc.node_i, 1]) ** 2
                            + (loc.h_true - rx[loc.node_i, 2]) ** 2)
    flight = loc[(loc.rid_ts_unix >= t_start) & (loc.rid_ts_unix <= t_end)]
    # denominator of the packet reception ratio (see module docstring)
    sent = np.unique(np.round(flight.rid_ts_unix, 1))
    if cfg.get("tx_log"):
        tx = pd.read_csv(folder / cfg["tx_log"])
        if "rid_ts_unix" in tx:
            tts = tx.rid_ts_unix.astype(float)
        else:
            tts = pd.Series([_rid_tenths_to_unix(int(a), float(b))
                             for a, b in zip(tx.rid_ts_tenths, tx.tx_time_unix)], dtype=float)
        tts = np.unique(np.round(tts[(tts >= t_start) & (tts <= t_end)], 1))
        n_expected, prr_method = max(len(tts), 1), "transmitter log"
    elif cfg.get("reference_node"):
        ref = flight[flight.node == cfg["reference_node"]]
        n_expected = max(int(np.unique(np.round(ref.rid_ts_unix, 1)).size), 1)
        prr_method = f"reference node {cfg['reference_node']} next to the transmitter"
    else:
        hz = float(cfg.get("broadcast_hz", 1.0))
        n_expected = max(int((t_end - t_start) * hz), 1)
        prr_method = f"nominal broadcast rate {hz:g} Hz (assumption)"
        if len(sent) > n_expected:
            print(f"WARNING: {len(sent)} distinct broadcasts heard but only {n_expected} expected "
                  f"at {hz:g} Hz; set broadcast_hz or tx_log in test.json")

    rows = []
    for n in names + ["network"]:
        d = flight if n == "network" else flight[flight.node == n]
        if d.empty:
            rows.append({"node": n}); continue
        uniq = d.drop_duplicates("rid_ts_unix")
        first = d.rx_time_unix.min() if n != "network" else flight.rx_time_unix.min()
        rows.append({
            "node": n,
            "ttff_s": first - t_start,
            "rx_delay_median_ms": 1e3 * d.rx_delay_s.median(),
            "rx_delay_p95_ms": 1e3 * d.rx_delay_s.quantile(0.95),
            "rx_delay_corr_median_ms": 1e3 * d.rx_delay_corr_s.median(),
            "prr": len(uniq) / n_expected,
            "pos_err_median_m": d.pos_err_m.median(),
            "pos_err_p95_m": d.pos_err_m.quantile(0.95),
            "rssi_median_dbm": d.rssi_dbm.median() if n != "network" else np.nan,
            "bearers": ",".join(sorted(d.bearer.unique())),
        })
    per_node = pd.DataFrame(rows)
    per_node.to_csv(out_dir / "per_node.csv", index=False)

    # log-distance path-loss fit RSSI = P0 - 10 n log10(d)
    # Path-loss fit. Packets weaker than the receiver sensitivity are never logged, so the RSSI
    # sample is truncated from below and an ordinary least-squares fit underestimates the
    # exponent. We fit a truncated-normal (Tobit-type) model by maximum likelihood, with the
    # truncation point set to the weakest RSSI logged (or "sensitivity_dbm" in test.json).
    from scipy.optimize import minimize
    from scipy.stats import norm
    ok = flight.rssi_dbm.notna() & (flight.dist_m > 5)
    yv = flight.rssi_dbm[ok].values.astype(float)
    xv = -10 * np.log10(flight.dist_m[ok].values)
    s_cut = float(cfg.get("sensitivity_dbm", yv.min() - 0.5))
    A = np.column_stack([np.ones(ok.sum()), xv])
    (P0, n_pl), *_ = np.linalg.lstsq(A, yv, rcond=None)          # OLS start / fallback

    def nll(th):
        mu = th[0] + th[1] * xv; sg = np.exp(th[2])
        return -np.sum(norm.logpdf(yv, mu, sg) - norm.logsf(s_cut, mu, sg))
    r = minimize(nll, [P0, n_pl, np.log(np.std(yv - A @ np.array([P0, n_pl])) + 1e-3)], method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    ols = (float(P0), float(n_pl))
    if r.success:
        P0, n_pl = float(r.x[0]), float(r.x[1])
    resid = flight.rssi_dbm[ok] - A @ np.array([P0, n_pl])
    shadow_sd = float(np.exp(r.x[2])) if r.success else float(resid.std())

    # offline AeroFence on the measured messages
    al = offline_fusion(flight, fl, rx, names, cfg, t_start, t_end, lat0, lon0, P0, n_pl,
                        shadow_sd)
    # broadcast matching across nodes: diversity and inter-node arrival spread. Propagation
    # differences are microseconds, so the spread measures clock offset plus capture jitter
    # and serves as a check of the time synchronization.
    fk = flight.assign(key=np.round(flight.rid_ts_unix, 1))
    mb = fk.groupby("key").agg(first_rx_unix=("rx_time_unix", "min"),
                               spread_ms=("rx_time_unix", lambda x: 1e3 * (x.max() - x.min())),
                               n_nodes=("node", "nunique"),
                               nodes=("node", lambda x: ",".join(sorted(set(map(str, x))))),
                               pos_err_m=("pos_err_m", "first")).reset_index()
    mb.to_csv(out_dir / "matched_broadcasts.csv", index=False)
    multi = mb[mb.n_nodes >= 2]
    matching = {"broadcasts": int(len(mb)), "mean_nodes_per_broadcast": float(mb.n_nodes.mean()),
                "share_heard_by_2plus": float((mb.n_nodes >= 2).mean()),
                "share_heard_by_3plus": float((mb.n_nodes >= 3).mean()),
                "arrival_spread_ms_median": float(multi.spread_ms.median()) if len(multi) else None,
                "arrival_spread_ms_p95": float(np.percentile(multi.spread_ms, 95)) if len(multi) else None}
    by_bearer = {b: {"n": int(len(d)), "rx_delay_ms_median": float(1e3 * d.rx_delay_s.median()),
                     "rx_delay_ms_p95": float(1e3 * np.percentile(d.rx_delay_s, 95))}
                 for b, d in flight.groupby("bearer")} if "bearer" in flight else {}
    summary = {"data_status": "SYNTHETIC PIPELINE TEST, NOT MEASURED" if synthetic else "measured",
               "t_start_utc": pd.Timestamp(t_start, unit="s", tz="UTC").isoformat(),
               "duration_s": t_end - t_start, "n_nodes": len(names), "n_messages": int(len(flight)),
               "per_node": per_node.to_dict("records"),
               "prr_denominator": {"method": prr_method, "expected_broadcasts": int(n_expected),
                                   "distinct_broadcasts_heard": int(len(sent))},
               "clock_offset_s": offs or "not provided: receive delays include clock offsets",
               "path_loss": {"P0_dbm_at_1m": float(P0), "n": float(n_pl), "shadow_std_db": shadow_sd,
                             "method": "truncated-normal ML" if r.success else "OLS",
                             "truncation_dbm": s_cut, "ols_P0_n": ols},
               "matching": matching, "by_bearer": by_bearer,
               "aerofence_offline": al}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figures(flight, per_node, out_dir, P0, n_pl, fx, fy, rx, names, cfg, synthetic)
    print(per_node.to_string(index=False))
    print(json.dumps(summary["path_loss"], indent=1), json.dumps(al, indent=1, default=float))


def offline_fusion(flight, fl, rx, names, cfg, t_start, t_end, lat0, lon0, P0, n_pl, shadow):
    """Run the AeroFence fusion server on the measured messages (one drone)."""
    t0 = t_start
    g = flight.groupby("rid_ts_unix")
    ts = np.array(sorted(g.groups))
    n = len(ts)

    class Log:  # same fields as the simulator's DroneLog
        pass

    lg = Log()
    lg.kind, lg.t_on, lg.t_off = GENUINE, 0.0, np.inf
    lg.t_tx = ts - t0
    lg.t_arr = np.array([g.get_group(t).rx_time_unix.min() for t in ts]) - t0
    order = np.argsort(lg.t_arr, kind="stable")
    ts, lg.t_tx, lg.t_arr = ts[order], lg.t_tx[order], lg.t_arr[order]
    lg.rp = np.zeros((n, 3)); lg.rv = np.zeros((n, 3))
    lg.ok = np.zeros((n, len(names)), bool); lg.rssi = np.full((n, len(names)), np.nan)
    for i, t in enumerate(ts):
        d = g.get_group(t)
        r = d.iloc[0]
        lg.rp[i] = [r.x, r.y, r.alt_geo_m if pd.notna(r.alt_geo_m) else 0.0]
        sp = r.speed_h_ms if pd.notna(r.speed_h_ms) else 0.0
        di = np.radians(r.direction_deg if pd.notna(r.direction_deg) else 0.0)
        lg.rv[i] = [sp * np.sin(di), sp * np.cos(di), r.speed_v_ms if pd.notna(r.speed_v_ms) else 0.0]
        for _, rr in d.iterrows():
            j = names.index(rr.node)
            lg.ok[i, j] = True
            lg.rssi[i, j] = rr.rssi_dbm
    T = t_end - t0
    grid = np.arange(0, T, 0.1)
    fx, fy = enu(fl.lat.values, fl.lon.values, lat0, lon0)
    lg.t_grid = grid
    lg.true_xy = np.column_stack([np.interp(grid + t0, fl.t_unix, fx), np.interp(grid + t0, fl.t_unix, fy)])
    p = Params(L=cfg["fence_side_m"], L_warn=cfg.get("warn_side_m", 1.4 * cfg["fence_side_m"]),
               Nr=len(names), T_obs=T, Tb=float(np.median(np.diff(np.sort(lg.t_tx)))) if n > 2 else 1.0,
               sigma_gnss=3.0, Pt_dbm=P0 - 40.05, n_pl=n_pl, shadow_db=shadow)
    if (HERE.parent / "results" / "learn.json").exists():
        L = json.loads((HERE.parent / "results" / "learn.json").read_text())
        b = np.array(L["beta4"]); B = np.maximum(b[1:], 0).sum()
        pon = L["thr"]["C_on_L"]
        c_on = (np.log(pon / (1 - pon)) - b[0]) / B
        p = Params(**{**p.__dict__, "w": tuple(np.maximum(b[1:], 0) / B), "C_on": c_on, "C_off": c_on - 0.1})
    te = np.arange(0, p.T_obs, p.dt_eval)
    tr = track(lg, rx, p)
    ft = epoch_features(lg, tr, p, te)
    res = run_fusion(lg, rx, p, "aerofence", tr, ft)
    k = drone_kpis(lg, res, p)
    zt = zone_of(lg.true_xy, p)
    crossings = int(np.sum(np.diff(zt) != 0))
    # every true entry into the protected zone, matched to the next entry alert
    ent = lg.t_grid[1:][(zt[1:] == 2) & (zt[:-1] < 2)]
    ea = np.array([a[0] for a in res.alerts if a[1] == "entry"])
    per_entry = []
    for i, te_ in enumerate(ent):
        nxt = ent[i + 1] if i + 1 < len(ent) else np.inf
        m = ea[(ea >= te_ - 5) & (ea < nxt)]
        per_entry.append(round(float(m[0] - te_), 2) if len(m) else None)
    got = [v for v in per_entry if v is not None]
    return {"alert_latency_s": k.get("latency"), "entry_label": k.get("entry_label"),
            "true_entries": len(ent), "entry_latency_each_s": per_entry,
            "entry_latency_median_s": float(np.median(got)) if got else None,
            "entries_missed": int(sum(v is None for v in per_entry)),
            "verified_fraction": k.get("verified"), "continuity": k.get("continuity"),
            "flips_per_min": k.get("flips_per_min"), "true_zone_changes": crossings,
            "alerts": [(round(a[0], 1), a[1], a[2]) for a in res.alerts],
            "cues": [(round(c[0], 1), c[1]) for c in res.cues]}


def _save(fig, path, synthetic):
    import matplotlib.pyplot as plt
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT MEASURED", ha="center", va="center", rotation=25,
                 fontsize=16, color="#d03030", alpha=0.35, weight="bold")
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


def figures(flight, per_node, out, P0, n_pl, fx, fy, rx, names, cfg, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a5cc4", "#555555"]
    # latency CDF
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, n in enumerate(names):
        v = np.sort(1e3 * flight[flight.node == n].rx_delay_s.values)
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=cols[i % 6], lw=1.6, label=n)
    ax.set_xlabel("Receive delay: node clock minus Remote ID time stamp (ms)"); ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=6.5); ax.grid(color="#ddd", lw=0.5)
    _save(fig, out / "fig_rx_delay_cdf.pdf", synthetic)
    # reception vs distance (1-s bins of transmitted timestamps)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = np.arange(0, flight.dist_m.max() + 50, 50)
    for i, n in enumerate(names):
        d = flight[flight.node == n]
        got = set(np.round(d.rid_ts_unix, 1))
        allts = flight.drop_duplicates("rid_ts_unix")
        dist_all = np.sqrt((allts.x_true - rx[i, 0]) ** 2 + (allts.y_true - rx[i, 1]) ** 2 + allts.h_true ** 2)
        rec = np.array([np.round(t, 1) in got for t in allts.rid_ts_unix])
        idx = np.digitize(dist_all, bins)
        pr = [rec[idx == b].mean() if (idx == b).sum() >= 5 else np.nan for b in range(1, len(bins))]
        ax.plot(bins[:-1] + 25, pr, "o-", ms=2.5, color=cols[i % 6], lw=1.2, label=n)
    ax.set_xlabel("Distance from node (m)"); ax.set_ylabel("Packet reception ratio")
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=6.5); ax.grid(color="#ddd", lw=0.5)
    _save(fig, out / "fig_prr_distance.pdf", synthetic)
    # RSSI vs distance with fit
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.scatter(flight.dist_m, flight.rssi_dbm, s=2, color="#2a78d6", alpha=0.3, lw=0)
    dd = np.linspace(max(flight.dist_m.min(), 5), flight.dist_m.max(), 100)
    ax.plot(dd, P0 - 10 * n_pl * np.log10(dd), color="#eb6834", lw=1.6, label=f"fit: n = {n_pl:.2f}")
    ax.set_xscale("log"); ax.set_xlabel("Distance (m)"); ax.set_ylabel("RSSI (dBm)")
    ax.legend(frameon=False); ax.grid(color="#ddd", lw=0.5)
    _save(fig, out / "fig_rssi_distance.pdf", synthetic)
    # map: flight log vs Remote ID positions, nodes and fence
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    a = cfg["fence_side_m"] / 2
    ax.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a], color="#555", lw=1, ls="--", label="virtual fence")
    ax.plot(fx, fy, color="#999", lw=1, label="ground truth (flight log or GPS track)")
    u = flight.drop_duplicates("rid_ts_unix")
    ax.scatter(u.x, u.y, s=3, color="#2a78d6", label="Remote ID positions", zorder=3)
    ax.scatter(rx[:, 0], rx[:, 1], marker="^", s=40, color="#eb6834", zorder=4, label="receiver nodes")
    for i, n in enumerate(names):
        ax.annotate(n, rx[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.legend(frameon=False, fontsize=6, loc="best")
    _save(fig, out / "fig_map.pdf", synthetic)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    analyse(Path(sys.argv[1]))
