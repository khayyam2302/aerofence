"""AeroFence simulator: networked Remote ID receivers, fusion, confidence, geofencing.

Everything in Section VI of the paper is implemented here:
  * geometry: square protected zone (side L) inside a square warning zone (side L_warn),
    receivers on the protected-zone perimeter at height h_r;
  * UAS: transit, probe (turn back inside the warning ring) and loiter behaviours;
  * Remote ID broadcast every T_b with GNSS error (AR(1)-correlated, std sigma_gnss);
  * link: log-distance path loss, time-correlated log-normal shadowing, Rician fading,
    receiver sensitivity, and interference erasure with occupancy epsilon;
  * adversarial/abnormal classes: smooth location spoofer (fabricated trajectory sent
    from a fixed transmitter), inconsistent reporter (jumps/replays), and a drone that
    switches Remote ID off near the zone;
  * fusion server: de-duplication, gating (Algorithm 1), constant-velocity Kalman filter,
    confidence C = sum w_k S_k, verification hysteresis, layered geofence with
    hysteresis/dwell, alert labels, escalation (silent and low-confidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# --------------------------------------------------------------------------- parameters
@dataclass
class Params:
    L: float = 1000.0            # protected zone side (m)
    L_warn: float = 1400.0       # warning zone side (m)
    Nr: int = 8                  # receivers
    h_r: float = 15.0            # receiver height (m)
    Nd: int = 20                 # UAS per run
    T_obs: float = 300.0         # observation period (s)
    Tb: float = 1.0              # broadcast period (s)
    eps: float = 0.10            # interference occupancy (per-packet erasure prob.)
    eps_common: float = 0.0      # fraction of erasures shared by all receivers
    sigma_gnss: float = 3.0      # GNSS error std per horizontal axis (m)
    gnss_tau: float = 5.0        # GNSS error correlation time (s)
    sigma_v: float = 0.5         # reported-velocity error std (m/s)
    # link budget (2.4 GHz Wi-Fi beacon Remote ID)
    Pt_dbm: float = 18.0
    n_pl: float = 2.5
    sens_dbm: float = -94.0
    shadow_db: float = 6.0
    shadow_dcorr: float = 50.0
    rician_K: float = 3.0
    # robustness experiments (all zero in the main study)
    rx_gain_sd: float = 0.0      # uncalibrated per-receiver gain offset std (dB), fixed per scene
    tx_pattern_db: float = 0.0   # peak-to-peak azimuthal transmit-antenna gain variation (dB)
    spoof_pwr_db: float = 0.0    # spoofer transmit power varied uniformly within +/- this (dB)
    # backhaul / processing delay (s)
    delay_min: float = 0.02
    delay_mean: float = 0.06
    # fusion
    W: float = 2.0
    tau_t: float = 1.0
    tau_p: float = 60.0
    tau_v: float = 8.0
    q_acc: float = 1.0           # CV process-noise spectral density (m^2/s^3)
    v_max: float = 35.0
    a_max: float = 12.0
    # confidence
    H: float = 10.0
    n_target: int = 3
    tau_c: float = 3.0
    w: tuple = (0.34, 0.33, 0.33, 0.0)      # weights of (S_div, S_con, S_mot, S_geo)
    beta: tuple | None = None               # logistic model (b0, b1..b4); overrides w
    C_on: float = 0.6
    C_off: float = 0.5
    T_ver: float = 4.0           # = 2W
    # geofence and escalation
    use_hysteresis: bool = True
    m_h: float = 15.0
    T_dwell: float = 2.0
    T_low: float = 4.0
    T_silent: float = 3.0
    T_loiter: float = 60.0
    # population
    p_spoof_smooth: float = 0.10
    p_spoof_jump: float = 0.10
    p_silent: float = 0.10
    behaviours: tuple = (0.5, 0.3, 0.2, 0.0)   # transit, probe, loiter, boundary-skim
    dt_eval: float = 0.5
    seed: int = 0


GENUINE, SPOOF_SMOOTH, SPOOF_JUMP, GO_SILENT = 0, 1, 2, 3


# --------------------------------------------------------------------------- geometry
def receiver_positions(p: Params) -> np.ndarray:
    a = p.L / 2
    if p.Nr == 1:
        xy = [(0.0, -a)]
    else:
        corners = [(-a, -a), (a, -a), (a, a), (-a, a)]
        per_edge = (p.Nr - 4) // 4 if p.Nr >= 4 else 0
        xy = []
        for k in range(4):
            x0, y0 = corners[k]
            x1, y1 = corners[(k + 1) % 4]
            xy.append((x0, y0))
            for j in range(1, per_edge + 1):
                f = j / (per_edge + 1)
                xy.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
        xy = xy[: p.Nr]
    xy = np.array(xy, float)
    return np.column_stack([xy, np.full(len(xy), p.h_r)])


def zone_of(xy: np.ndarray, p: Params, margin: float = 0.0) -> np.ndarray:
    """0 outside, 1 warning ring, 2 protected zone (Chebyshev norm, square zones)."""
    d = np.max(np.abs(xy), axis=-1)
    z = np.zeros(d.shape, int)
    z[d <= p.L_warn / 2 + margin] = 1
    z[d <= p.L / 2 + margin] = 2
    return z


# --------------------------------------------------------------------------- trajectories
def make_trajectory(rng, p: Params, t: np.ndarray, kind: str):
    """Return true positions (T,3) and velocities (T,3) on time grid t."""
    dt = t[1] - t[0]
    ang = rng.uniform(0, 2 * np.pi)
    r0 = rng.uniform(900, 1200)
    start = np.array([r0 * np.cos(ang), r0 * np.sin(ang)])
    alt = rng.uniform(50, 120)
    speed = rng.uniform(6, 15)
    if kind == "transit":
        target = rng.uniform(-0.35, 0.35, 2) * p.L
        wps = [target, target + (target - start) * 1.5]
        loiter = None
    elif kind == "probe":
        target = rng.uniform(-0.3, 0.3, 2) * p.L
        u = (target - start) / np.linalg.norm(target - start)
        # turn back inside the warning ring, 30-170 m outside the protected boundary
        lo, hi = 0.0, np.linalg.norm(target - start)
        turn_d = rng.uniform(p.L / 2 + 30, p.L_warn / 2 - 30)
        for _ in range(40):
            mid = (lo + hi) / 2
            if np.max(np.abs(start + u * mid)) > turn_d:
                lo = mid
            else:
                hi = mid
        turn = start + u * lo
        side = np.array([-u[1], u[0]]) * rng.choice([-1, 1])
        wps = [turn, turn + side * 150 - u * 200, start]
        loiter = None
    elif kind == "boundary":
        # skim along the protected boundary at a small signed offset (worst case for jitter)
        a_ = p.L / 2 + rng.uniform(-20, 20)
        side = rng.integers(4)
        pts = np.array([[-a_ - 300, -a_], [a_ + 300, -a_]])
        rot = np.array([[np.cos(side * np.pi / 2), -np.sin(side * np.pi / 2)],
                        [np.sin(side * np.pi / 2), np.cos(side * np.pi / 2)]])
        pts = pts @ rot.T
        start = pts[0] + (pts[0] / np.linalg.norm(pts[0])) * 200
        wps = [pts[0], pts[1], pts[1] * 2]
        loiter = None
    else:  # loiter
        target = rng.uniform(-0.25, 0.25, 2) * p.L
        wps = [target, target]
        loiter = (target, rng.uniform(60, 90), rng.uniform(60, 120))
    pos = np.zeros((len(t), 2))
    vel = np.zeros((len(t), 2))
    x = start.copy()
    k = 0
    lat = 0.0
    t_loiter0 = None
    for i in range(len(t)):
        if loiter is not None and k >= 1:
            c, rad, dur = loiter
            if t_loiter0 is None:
                t_loiter0 = t[i]
            if t[i] - t_loiter0 < dur:
                w_ang = speed / rad
                th = w_ang * (t[i] - t_loiter0)
                newx = c + rad * np.array([np.cos(th), np.sin(th)]) - np.array([rad, 0])
                vel[i] = (newx - x) / dt if i else 0
                x = newx
                pos[i] = x
                continue
            else:
                loiter = None
                wps.append(start)
                k = 1
        goal = wps[min(k, len(wps) - 1)]
        dvec = goal - x
        dist = np.linalg.norm(dvec)
        if dist < speed * dt * 2 and k < len(wps) - 1:
            k += 1
        u = dvec / max(dist, 1e-6)
        lat = 0.98 * lat + rng.normal(0, 0.15)
        v = u * speed + np.array([-u[1], u[0]]) * lat
        x = x + v * dt
        pos[i], vel[i] = x, v
    z = alt + 3 * np.sin(2 * np.pi * t / rng.uniform(40, 80))
    return np.column_stack([pos, z]), np.column_stack([vel, np.gradient(z, dt)])


# --------------------------------------------------------------------------- link model
def path_loss_db(d, p: Params):
    return 40.05 + 10 * p.n_pl * np.log10(np.maximum(d, 1.0))


def rician_db(rng, size, K):
    s = np.sqrt(K / (K + 1))
    sig = np.sqrt(1 / (2 * (K + 1)))
    h = (s + sig * rng.normal(size=size)) + 1j * sig * rng.normal(size=size)
    return 20 * np.log10(np.abs(h))


# --------------------------------------------------------------------------- one drone
@dataclass
class DroneLog:
    kind: int
    t_grid: np.ndarray
    true_xy: np.ndarray
    t_on: float = 0.0
    t_off: float = np.inf


def simulate_drone(rng, p: Params, rx: np.ndarray, kind: int, behaviour: str, rx_off=None) -> DroneLog:
    dt = 0.1
    t = np.arange(0, p.T_obs + dt, dt)
    true_pos, true_vel = make_trajectory(rng, p, t, behaviour)
    t_on = rng.uniform(0, 30)
    log = DroneLog(kind, t, true_pos[:, :2], t_on=t_on)
    log.claim_xy = true_pos[:, :2]
    # transmitter (physical) and reported trajectories
    tx_pos = true_pos
    rep_pos, rep_vel = true_pos.copy(), true_vel.copy()
    if kind == SPOOF_SMOOTH:
        # fabricated flight sent from a fixed transmitter outside the zone
        fake_pos, fake_vel = make_trajectory(rng, p, t, "transit")
        rep_pos, rep_vel = fake_pos, fake_vel
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(750, 950)
        tx_pos = np.tile([r * np.cos(ang), r * np.sin(ang), 20.0], (len(t), 1))
        log.true_xy = tx_pos[:, :2]            # physical transmitter (never enters)
        log.claim_xy = fake_pos[:, :2]         # fabricated flight the messages claim
    t_tx = np.arange(t_on + rng.uniform(0, p.Tb), p.T_obs, p.Tb)
    t_tx = t_tx + rng.uniform(-0.05, 0.05, len(t_tx)) * p.Tb
    t_tx = t_tx[(t_tx > 0) & (t_tx < p.T_obs)]
    if kind == GO_SILENT:
        # switch off once inside the warning ring, 0-20 s after entering it
        zin = np.where(zone_of(true_pos[:, :2], p) >= 1)[0]
        if len(zin):
            log.t_off = t[zin[0]] + rng.uniform(0, 20)
            t_tx = t_tx[t_tx < log.t_off]
    idx = np.clip(np.round(t_tx / dt).astype(int), 0, len(t) - 1)
    n = len(idx)
    # GNSS error, AR(1)
    a = np.exp(-p.Tb / p.gnss_tau)
    e = np.zeros((n, 2))
    if n:
        e[0] = rng.normal(0, p.sigma_gnss, 2)
        for i in range(1, n):
            e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * rng.normal(0, p.sigma_gnss, 2)
    rp = rep_pos[idx].copy()
    rp[:, :2] += e
    rp[:, 2] += rng.normal(0, p.sigma_gnss, n) * 1.5
    rv = rep_vel[idx] + rng.normal(0, p.sigma_v, (n, 3))
    if kind == SPOOF_JUMP and n:
        i = 0
        while i < n:
            if rng.random() < 0.06:
                L_ = rng.integers(3, 9)
                off = rng.uniform(150, 400) * np.array([np.cos(th := rng.uniform(0, 6.283)), np.sin(th)])
                rp[i:i + L_, :2] += off
                i += L_
            i += 1
    # link: distance from physical transmitter to each receiver
    tp = tx_pos[idx]
    d = np.linalg.norm(tp[:, None, :] - rx[None, :, :], axis=2)
    # time-correlated shadowing per receiver (correlation by distance travelled)
    sh = np.zeros_like(d)
    if n:
        sh[0] = rng.normal(0, p.shadow_db, rx.shape[0])
        step = np.r_[0, np.linalg.norm(np.diff(tp, axis=0), axis=1)]
        for i in range(1, n):
            rho = np.exp(-step[i] / p.shadow_dcorr)
            sh[i] = rho * sh[i - 1] + np.sqrt(1 - rho * rho) * rng.normal(0, p.shadow_db, rx.shape[0])
    rssi = p.Pt_dbm - path_loss_db(d, p) + sh + rician_db(rng, d.shape, p.rician_K)
    if n and (p.rx_gain_sd > 0 or p.tx_pattern_db > 0 or p.spoof_pwr_db > 0):
        # separate generator so the traffic and channel draws stay identical (common random numbers)
        prng = np.random.default_rng(int(abs(t_tx[0]) * 1e6) % (2 ** 32))
        if rx_off is not None:
            rssi = rssi + rx_off[None, :]
        if p.tx_pattern_db > 0:
            vel = np.gradient(tp[:, :2], axis=0) if n > 1 else np.zeros((n, 2))
            psi = np.where(np.hypot(vel[:, 0], vel[:, 1]) > 0.1, np.arctan2(vel[:, 1], vel[:, 0]), prng.uniform(0, 2 * np.pi))
            az = np.arctan2(rx[None, :, 1] - tp[:, None, 1], rx[None, :, 0] - tp[:, None, 0])
            rssi = rssi + 0.5 * p.tx_pattern_db * np.cos(2 * (az - psi[:, None]))
        if p.spoof_pwr_db > 0 and kind in (SPOOF_SMOOTH, SPOOF_JUMP):
            rssi = rssi + prng.uniform(-p.spoof_pwr_db, p.spoof_pwr_db, n)[:, None]
    ok = rssi >= p.sens_dbm
    common = rng.random(n) < p.eps * p.eps_common
    indiv = rng.random(d.shape) < p.eps * (1 - p.eps_common)
    ok &= ~common[:, None] & ~indiv
    delay = p.delay_min + rng.exponential(p.delay_mean - p.delay_min, d.shape)
    got = ok.any(axis=1)
    t_arr = np.where(ok, t_tx[:, None] + delay, np.inf).min(axis=1)
    order = np.argsort(t_arr[got], kind="stable")
    sel = np.where(got)[0][order]
    log.t_tx, log.t_arr = t_tx[sel], t_arr[sel]
    log.rp, log.rv = rp[sel], rv[sel]
    log.ok, log.rssi = ok[sel], np.where(ok[sel], rssi[sel], np.nan)
    log.n_sent = n
    return log


def simulate_scene(p: Params, seed: int):
    rng = np.random.default_rng(seed)
    rx = receiver_positions(p)
    rx_off = np.random.default_rng(seed + 99991).normal(0, p.rx_gain_sd, len(rx)) if p.rx_gain_sd > 0 else None
    drones = []
    for j in range(p.Nd):
        u = rng.random()
        if u < p.p_spoof_smooth:
            kind = SPOOF_SMOOTH
        elif u < p.p_spoof_smooth + p.p_spoof_jump:
            kind = SPOOF_JUMP
        elif u < p.p_spoof_smooth + p.p_spoof_jump + p.p_silent:
            kind = GO_SILENT
        else:
            kind = GENUINE
        beh = rng.choice(["transit", "probe", "loiter", "boundary"], p=list(p.behaviours))
        if kind == GO_SILENT:
            beh = "transit"
        drones.append(simulate_drone(rng, p, rx, kind, beh, rx_off))
    return rx, drones
