"""Unit tests of the tracker, geofence hysteresis, verification and escalation rules.

Each test builds a small, noise-free Remote ID log by hand (no random draws), so the expected
behaviour can be stated exactly. Run with:  python3 -m pytest tests
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerofence.fusion import run_fusion  # noqa: E402
from aerofence.sim import GENUINE, Params, path_loss_db, receiver_positions, zone_of  # noqa: E402
from aerofence.sim import DroneLog  # noqa: E402


def learned_params(**kw):
    """Nominal parameters with the learned weights and threshold of results/learn.json."""
    L = json.loads((ROOT / "results" / "learn.json").read_text())
    b = np.array(L["beta4"]); B = np.maximum(b[1:], 0).sum(); p_on = L["thr"]["C_on_L"]
    c_on = (np.log(p_on / (1 - p_on)) - b[0]) / B
    return replace(Params(), w=tuple(np.maximum(b[1:], 0) / B), C_on=float(c_on),
                   C_off=float(c_on - 0.1), **kw)


def make_log(p, path_xy, t_on=5.0, t_off=None, tx_xy=None, heard=None):
    """Broadcast every p.Tb from t_on along path_xy(t); received by every receiver in `heard`
    with RSS from the path-loss model at the physical transmitter position tx_xy(t)."""
    rx = receiver_positions(p)
    t_end = p.T_obs if t_off is None else t_off
    t_tx = np.arange(t_on, t_end, p.Tb)
    pos = np.array([path_xy(t) for t in t_tx]).reshape(-1, 2)
    vel = np.gradient(pos, p.Tb, axis=0) if len(t_tx) > 1 else np.zeros_like(pos)
    tx = np.array([(tx_xy or path_xy)(t) for t in t_tx]).reshape(-1, 2)
    tx3 = np.column_stack([tx, np.full(len(t_tx), 60.0)])
    d = np.linalg.norm(tx3[:, None, :] - rx[None, :, :], axis=2)
    rssi = p.Pt_dbm - path_loss_db(d, p)
    ok = np.ones_like(d, bool) if heard is None else np.isin(np.arange(len(rx)), heard)[None, :] & np.ones_like(d, bool)
    tg = np.arange(0, p.T_obs + 0.1, 0.1)
    log = DroneLog(GENUINE, tg, np.array([path_xy(t) for t in tg]), t_on=t_on,
                   t_off=np.inf if t_off is None else t_off)
    log.claim_xy = log.true_xy
    log.t_tx, log.t_arr = t_tx, t_tx + 0.05
    log.rp = np.column_stack([pos, np.full(len(t_tx), 60.0)])
    log.rv = np.column_stack([vel, np.zeros(len(t_tx))])
    log.ok, log.rssi = ok, np.where(ok, rssi, np.nan)
    log.n_sent = len(t_tx)
    return log, rx


def straight(x0, y0, vx, vy):
    return lambda t: np.array([x0 + vx * t, y0 + vy * t])


def test_zone_margins():
    p = Params()
    edge = p.L / 2
    inside = np.array([[edge - p.m_h - 1.0, 0.0]])
    shallow = np.array([[edge - p.m_h + 1.0, 0.0]])
    assert zone_of(inside, p, margin=-p.m_h)[0] == 2
    assert zone_of(shallow, p, margin=-p.m_h)[0] == 1        # inside, but not by m_h
    assert zone_of(shallow, p)[0] == 2


def test_entry_alert_waits_for_margin_and_dwell():
    p = learned_params()
    # straight transit from the west at 10 m/s along y = 0
    log, rx = make_log(p, straight(-800.0, 0.0, 10.0, 0.0))
    res = run_fusion(log, rx, p)
    entries = [a for a in res.alerts if a[1] == "entry"]
    assert len(entries) == 1
    t_cross = (800.0 - p.L / 2) / 10.0                      # true boundary crossing
    t_margin = (800.0 - p.L / 2 + p.m_h) / 10.0             # m_h inside
    assert entries[0][0] >= t_margin + p.T_dwell - p.dt_eval
    assert entries[0][0] > t_cross
    assert entries[0][2] == "confirm"                         # consistent track, 8 receivers


def test_shallow_track_does_not_enter():
    p = learned_params()
    depth = p.m_h - 5.0                                       # inside, but less than m_h
    log, rx = make_log(p, straight(p.L / 2 - depth, -600.0, 0.0, 5.0))
    assert not [a for a in run_fusion(log, rx, p).alerts if a[1] == "entry"]
    raw = run_fusion(log, rx, p, mode="raw")                  # message-level geofence alerts
    assert [a for a in raw.alerts if a[1] == "entry"]


def test_silent_cue_after_t_silent():
    p = learned_params()
    t_off = 70.0                                              # inside the zones at t = 70 s
    log, rx = make_log(p, straight(-800.0, 0.0, 10.0, 0.0), t_off=t_off)
    res = run_fusion(log, rx, p)
    silent = [c[0] for c in res.cues if c[1] == "silent"]
    assert silent, "no silent cue"
    last_rx = log.t_arr[-1]
    assert last_rx + p.T_silent <= silent[0] <= last_rx + p.T_silent + 2 * p.dt_eval


def test_no_silent_cue_outside_the_zones():
    p = learned_params()
    log, rx = make_log(p, straight(-900.0, 0.0, 1.0, 0.0), t_off=60.0)   # stays outside
    assert not [c for c in run_fusion(log, rx, p).cues if c[1] == "silent"]


def test_drone_that_never_broadcasts_has_no_alerts():
    p = learned_params()
    log, rx = make_log(p, straight(-800.0, 0.0, 10.0, 0.0), t_on=400.0)  # after T_obs
    assert len(log.t_arr) == 0
    res = run_fusion(log, rx, p)
    assert res.alerts == [] and res.cues == []


def test_geometry_term_separates_a_fixed_transmitter():
    """A transmitter that stays outside the site while claiming a transit produces an RSS
    pattern that does not fit the claimed positions, so S_geo is lower than for a genuine
    drone flying the same path. (Whether the track is then flagged depends on the threshold
    calibrated with shadowing and fading, which this noise-free test does not model.)"""
    p = learned_params()
    claim = straight(-800.0, 0.0, 10.0, 0.0)
    genuine, rx = make_log(p, claim)
    spoof, _ = make_log(p, claim, tx_xy=lambda t: np.array([0.0, -900.0]))
    g = run_fusion(genuine, rx, p).S[:, 3]
    s_ = run_fusion(spoof, rx, p).S[:, 3]
    window = (np.arange(len(g)) * p.dt_eval > 20) & (np.arange(len(g)) * p.dt_eval < 120)
    assert g[window].mean() > 0.95
    assert s_[window].mean() < g[window].mean() - 0.2


def test_single_receiver_score_is_bounded():
    """With one receiver, S_div <= 1/3 and S_geo = 0.5, so C cannot exceed
    w_div/3 + w_con + w_mot + 0.5 w_geo (0.68 with the learned weights, Section V)."""
    p = learned_params()
    log, rx = make_log(p, straight(-800.0, 0.0, 10.0, 0.0), heard=[0])
    res = run_fusion(log, rx, p)
    w = np.asarray(p.w)
    bound = w[0] / 3 + w[1] + w[2] + 0.5 * w[3]
    assert abs(bound - 0.68) < 0.01
    assert np.nanmax(res.C) <= bound + 1e-9
    assert np.allclose(res.S[np.isfinite(res.age), 3], 0.5)
