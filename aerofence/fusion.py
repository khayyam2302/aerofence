"""Fusion server: tracking, confidence, verification, geofencing, escalation.

Methods compared on the same received messages (same scenes, same seeds):
  "aerofence"  proposed: gated CV-Kalman tracking, confidence C = sum_k w_k S_k,
               verification hysteresis, geofence hysteresis/dwell, graded alerts,
               silent and low-confidence escalation
  "raw"        message-level geofencing on the union of receivers, as in Remote ID
               apps and basic receiver products: the zone of the latest message is
               the alert state; no tracking, no verification, no escalation
  "kf"         multi-receiver CV-Kalman tracking with gating and geofence hysteresis;
               every track is treated as valid; escalation only on silence
  "innov"      Kalman innovation (NIS) consistency monitoring, the detector used for
               Remote ID spoofing by Bjorkman et al. (AIAA SciTech 2026)
  "rss"        RSS position-claim verification against the receiver geometry, the
               approach of GhostBuster (Keizer et al., IEEE CCNC 2024) and
               Dickerson et al. (2026)
The tracker is shared by all track-based methods; they differ only in how tracks are
verified, which isolates the effect of the verification logic.
"""
from __future__ import annotations

import numpy as np

from .sim import Params, path_loss_db, zone_of

SIG_REF_DB = 8.0       # expected RSS residual std (shadowing + fading)


# --------------------------------------------------------------------------- tracker
def track(log, rx, p: Params):
    n = len(log.t_arr)
    passed = np.ones(n, bool)
    nis = np.zeros(n)
    xs = np.zeros((n, 4))
    tu = np.zeros(n)
    R = np.diag([p.sigma_gnss ** 2] * 2 + [(2 * p.sigma_v) ** 2] * 2)
    x = P = None
    t_upd = None
    last = None
    streak = 0
    I4 = np.eye(4)
    for i in range(n):
        t_tx = log.t_tx[i]
        z = np.r_[log.rp[i, :2], log.rv[i, :2]]
        if x is None:
            x, P, t_upd = z.copy(), np.diag([p.sigma_gnss ** 2] * 2 + [4.0, 4.0]), t_tx
        else:
            dt = max(t_tx - t_upd, 1e-3)
            F = I4.copy(); F[0, 2] = F[1, 3] = dt
            q = p.q_acc
            Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                              [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
            xp = F @ x
            Pp = F @ P @ F.T + Q
            Si = np.linalg.inv(Pp + R)
            inn = z - xp
            nis[i] = float(inn @ Si @ inn)
            ok_t = abs(log.t_arr[i] - t_tx) <= p.tau_t and t_tx > last[0]
            ok_p = np.hypot(inn[0], inn[1]) <= p.tau_p + 2 * np.sqrt(Pp[0, 0] + Pp[1, 1])
            ok_v = np.hypot(inn[2], inn[3]) <= p.tau_v
            jump = np.hypot(*(log.rp[i, :2] - last[1])) - 3 * np.sqrt(2) * p.sigma_gnss
            ok_k = jump / max(t_tx - last[0], 1e-3) <= p.v_max
            if ok_t and ok_p and ok_v and ok_k:
                Kg = Pp @ Si
                x = xp + Kg @ inn
                P = (I4 - Kg) @ Pp
                t_upd = t_tx
                streak = 0
            else:
                passed[i] = False
                streak += 1
                if streak >= 3:      # persistent new evidence: re-initialise the track
                    x, P, t_upd = z.copy(), np.diag([p.sigma_gnss ** 2] * 2 + [4.0, 4.0]), t_tx
                    streak = 0
        last = (t_tx, log.rp[i, :2].copy())
        xs[i], tu[i] = x, t_upd
    # RSS residual of each message against its claimed position (unknown Tx power):
    # residuals of receivers that heard it, plus a penalty for receivers that should
    # have heard it at least 12 dB above sensitivity but did not.
    rr = np.full(n, np.nan)
    if n and rx.shape[0] >= 2:
        d = np.linalg.norm(log.rp[:, None, :] - rx[None, :, :], axis=2)
        pl = path_loss_db(d, p)
        ok = log.ok
        cnt = ok.sum(1)
        pt = np.where(ok, log.rssi + pl, 0.0).sum(1) / np.maximum(cnt, 1)
        res = np.where(ok, log.rssi + pl - pt[:, None], 0.0)
        excess = np.where(~ok, pt[:, None] - pl - p.sens_dbm, -np.inf)
        miss = np.where(excess > 12.0, excess, 0.0)
        tot = (res ** 2).sum(1) + (miss ** 2).sum(1)
        nn = cnt + (miss > 0).sum(1)
        rr = np.where(cnt >= 2, np.sqrt(tot / np.maximum(nn, 1)), np.nan)
    return {"passed": passed, "nis": nis, "x": xs, "t_upd": tu, "rr": rr}


# --------------------------------------------------------------------------- features
def epoch_features(log, tr, p: Params, te):
    """Sliding-horizon evidence at each evaluation epoch, computed with cumulative sums."""
    ta = log.t_arr
    hi = np.searchsorted(ta, te, "right")
    lo = np.searchsorted(ta, te - p.H, "left")
    cnt = (hi - lo).astype(float)
    zc = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)])
    K = len(te)
    if len(ta):
        cok = zc(log.ok.astype(float))
        div = ((cok[hi] - cok[lo]) > 0).sum(1)
        cpass = zc(tr["passed"].astype(float))
        npass = cpass[hi] - cpass[lo]
        valid = ~np.isnan(tr["rr"])
        rr0 = np.where(valid, tr["rr"], 0.0)
        g = np.where(valid, np.exp(-rr0 ** 2 / (2 * SIG_REF_DB ** 2)), 0.0)
        cg, cv, crr, cnis = zc(g), zc(valid.astype(float)), zc(rr0), zc(tr["nis"])
        ng = cv[hi] - cv[lo]
        s_geo = np.where(ng > 0, (cg[hi] - cg[lo]) / np.maximum(ng, 1), 0.5)
        rss_mean = np.where(ng > 0, (crr[hi] - crr[lo]) / np.maximum(ng, 1), np.nan)
        nis_mean = np.where(cnt > 0, (cnis[hi] - cnis[lo]) / np.maximum(cnt, 1), 0.0)
        last_arr = np.where(hi > 0, ta[np.maximum(hi - 1, 0)], -np.inf)
    else:
        div = np.zeros(K); npass = np.zeros(K); s_geo = np.full(K, 0.5)
        rss_mean = np.full(K, np.nan); nis_mean = np.zeros(K); last_arr = np.full(K, -np.inf)
    age = te - last_arr
    s_div = np.minimum(1.0, div / p.n_target)
    s_con = np.where(np.isfinite(age), np.exp(-np.maximum(age, 0) / p.tau_c), 0.0)
    s_mot = (npass + 1) / (cnt + 2)
    S = np.column_stack([s_div, s_con, s_mot, s_geo])
    est = np.full((K, 2), np.nan)
    raw = np.full((K, 2), np.nan)
    has = hi > 0
    if has.any():
        idx = hi[has] - 1
        x = tr["x"][idx]
        dt = np.clip(te[has] - tr["t_upd"][idx], 0, 10)
        est[has] = x[:, :2] + x[:, 2:] * dt[:, None]
        raw[has] = log.rp[idx, :2]
    return S, age, est, raw, nis_mean, rss_mean, cnt


# --------------------------------------------------------------------------- decisions
class Result:
    pass


def run_fusion(log, rx, p: Params, mode="aerofence", tr=None, feats=None,
               innov_thr=13.28, rss_thr=14.0):
    te = np.arange(0, p.T_obs, p.dt_eval)
    if tr is None:
        tr = track(log, rx, p)
    if feats is None:
        feats = epoch_features(log, tr, p, te)
    S, age, est, raw, nis_mean, rss_mean, cnt = feats
    if p.beta is not None:
        b = np.asarray(p.beta, float)
        C = 1.0 / (1.0 + np.exp(-(b[0] + S @ b[1:])))
    else:
        C = S @ np.asarray(p.w, float)
    K = len(te)
    out = Result()
    out.te, out.S, out.C, out.age, out.est = te, S, C, age, est
    out.first_arr = log.t_arr[0] if len(log.t_arr) else np.nan
    out.alerts, out.cues = [], []
    ver_arr = np.zeros(K, bool)
    zone_arr = np.zeros(K, int)
    if mode in ("innov", "aeroinnov"):
        suspect = (cnt >= 3) & (nis_mean > innov_thr)
    elif mode == "rss":
        suspect = ~np.isnan(rss_mean) & (rss_mean > rss_thr)
    else:
        suspect = np.zeros(K, bool)
    pos = raw if mode == "raw" else est
    pz = np.nan_to_num(pos, nan=1e9)
    z_plain = zone_of(pz, p)
    z_in = zone_of(pz, p, margin=-p.m_h)
    z_out = zone_of(pz, p, margin=+p.m_h)
    ver = False; above = None
    zs = 0; up = dn = None
    low_since = None; low_cued = False; sil_cued = False
    loit = None; loit_done = False
    hyst = p.use_hysteresis and mode != "raw"
    finite_pos = np.isfinite(pos[:, 0])
    for k in range(K):
        t = te[k]
        a = age[k]
        valid = a < 10
        # ---- verification
        if mode in ("aerofence", "aeroinnov"):
            if C[k] >= p.C_on:
                if above is None:
                    above = t
                if t - above >= p.T_ver:
                    ver = True
            else:
                above = None
            if C[k] < p.C_off:
                ver = False
            if mode == "aeroinnov" and suspect[k]:
                ver = False
        else:
            ver = valid and not suspect[k]
        ver_arr[k] = ver
        # ---- geofence
        prev = zs
        if mode == "raw" and not valid:
            zs = 0
        elif finite_pos[k]:
            if hyst:
                if z_in[k] > zs:
                    if up is None:
                        up = t
                    if t - up >= p.T_dwell:
                        zs, up = int(z_in[k]), None
                else:
                    up = None
                if z_out[k] < zs:
                    if dn is None:
                        dn = t
                    if t - dn >= p.T_dwell:
                        zs, dn = int(z_out[k]), None
                else:
                    dn = None
            else:
                zs = int(z_plain[k])
        zone_arr[k] = zs
        if zs != prev:
            lab = "confirm" if ver else "investigate"
            if zs == 2:
                out.alerts.append((t, "entry", lab))
            elif zs == 1 and prev == 0:
                out.alerts.append((t, "approach", lab))
        if zs == 2:
            if loit is None:
                loit = t
            if not loit_done and t - loit >= p.T_loiter:
                out.alerts.append((t, "loiter", "confirm" if ver else "investigate")); loit_done = True
        else:
            loit = None
        # ---- escalation (not part of the message-level baseline)
        if mode == "raw" or not np.isfinite(a):
            continue
        rel = zs >= 1
        if rel and a > p.T_silent:
            if not sil_cued:
                out.cues.append((t, "silent")); sil_cued = True
        elif a <= p.T_silent:
            sil_cued = False
        if mode == "aerofence":
            low = C[k] < p.C_on
        elif mode == "aeroinnov":
            low = (C[k] < p.C_on) or bool(suspect[k])
        elif mode in ("innov", "rss"):
            low = bool(suspect[k])
        else:
            low = False
        if rel and low and a <= p.T_silent:
            if low_since is None:
                low_since = t
            if not low_cued and t - low_since >= p.T_low:
                out.cues.append((t, "low_conf")); low_cued = True
        else:
            low_since = None
            if not low:
                low_cued = False
    out.verified, out.zone = ver_arr, zone_arr
    out.nis_mean, out.rss_mean = nis_mean, rss_mean
    return out
