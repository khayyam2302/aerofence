"""Reproduce every number, table and figure of the revised AeroFence paper.

Stages (run all, or one by name):
  learn    fit the confidence model on training scenes (seeds 1000-1059), bootstrap CIs,
           calibrate thresholds of every method at a 5 % genuine false-escalation rate
  roc      held-out ROC comparison (seeds 2000+)
  kpi      KPI table: Nr x sigma_gnss (seeds 0-99)
  tb       alert latency vs broadcast period
  esc      escalation vs Nr and interference occupancy
  con      verified-track rate vs C_on
  flip     boundary-jitter table (boundary-skimming scenario)
  compare  method comparison table
  sens     one-at-a-time and global (LHS) sensitivity analysis
Results are written to results/*.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from aerofence.analysis import (bootstrap_logistic, collect, fit_logistic, roc,  # noqa: E402
                                threshold_at, tpr_at)
from aerofence.run import run_setting  # noqa: E402
from aerofence.sim import Params  # noqa: E402

RES = ROOT / "results"
BASE = Params(Nr=8, sigma_gnss=3.0, eps=0.10, Tb=1.0)
# Optional channel override, e.g. propagation parameters fitted to measured data:
#   AF_CHANNEL=results/channel_measured.json AF_TAG=measured python3 experiments.py learn roc kpi compare combo
# The JSON holds Params fields (n_pl, shadow_db, shadow_dcorr, ...) under "params". Results go to
# results_<AF_TAG>/ so the nominal results are never overwritten. To test the nominal weights
# under the measured channel, copy results/learn.json and results/combo.json into that folder
# first and run only "kpi compare roc".
if os.environ.get("AF_CHANNEL"):
    _ch = json.loads(Path(os.environ["AF_CHANNEL"]).read_text())["params"]
    BASE = replace(BASE, **_ch)
    RES = ROOT / f"results_{os.environ.get('AF_TAG', 'channel')}"
    print("channel override:", _ch, "->", RES, flush=True)
RES.mkdir(exist_ok=True)
RUNS = 100
POOL = 2


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def _job(args):
    p, runs, modes, seed0, ov, kw = args
    return run_setting(p, runs, modes, seed0, ov, kw)


def par(jobs):
    with Pool(POOL) as pool:
        return pool.map(_job, jobs)


def split(p, runs, modes, seed0=0, ov=None, kw=None, chunks=POOL):
    """Split one setting over workers by seed range and merge means (equal chunk sizes)."""
    n = runs // chunks
    return [(p, n, modes, seed0 + i * n, ov, kw) for i in range(chunks)]


def merge(parts):
    out = {}
    for m in parts[0]:
        out[m] = {k: [float(np.nanmean([pp[m][k][0] for pp in parts])),
                      float(np.sqrt(np.nanmean([pp[m][k][1] ** 2 for pp in parts])))]
                  for k in parts[0][m]}
    return out


# --------------------------------------------------------------------------- learn
def _collect(args):
    p, seeds, beta = args
    return collect(p, seeds, beta)


def stage_learn():
    conds = [replace(BASE, Nr=nr, sigma_gnss=s) for nr in (4, 8, 12) for s in (3.0, 10.0)]
    jobs = [(c, range(1000 + 10 * i, 1010 + 10 * i), None) for i, c in enumerate(conds)]
    with Pool(POOL) as pool:
        parts = pool.map(_collect, jobs)
    X = np.vstack([pp[1][0] for pp in parts]); Y = np.concatenate([pp[1][1] for pp in parts])
    G = np.concatenate([pp[1][2] + 10 ** 6 * i for i, pp in enumerate(parts)])
    b4 = fit_logistic(X, Y)
    b3 = fit_logistic(X[:, :3], Y)
    boot = bootstrap_logistic(X, Y, G, B=200)
    w4 = np.maximum(b4[1:], 0) / np.maximum(b4[1:], 0).sum()
    # per-feature separability
    single = {}
    for j, nm in enumerate(["div", "con", "mot", "geo"]):
        bj = fit_logistic(X[:, [j]], Y)
        z = bj[0] + X[:, j] * bj[1]
        from aerofence.analysis import roc as _roc
        _, _, a = _roc(z[Y == 1], z[Y == 0])
        single[nm] = {"beta": bj.tolist(), "epoch_auc": float(a)}
    # thresholds at 5 % genuine false escalation, from training-scene drone scores
    dr_eq = [d for pp in parts for d in pp[0]]
    with Pool(POOL) as pool:
        parts_L = pool.map(_collect, [(c, s, b4) for c, s, _ in jobs])
    dr_L = [d for pp in parts_L for d in pp[0]]
    neg_eq = [d for d in dr_eq if d["kind"] == 0]
    neg_L = [d for d in dr_L if d["kind"] == 0]
    thr = {
        "innov": threshold_at([d["innov"] for d in neg_eq]),
        "rss": threshold_at([d["rss"] for d in neg_eq]),
        "C_on_L": 1 - threshold_at([d["aero"] for d in neg_L]),
        "C_on_eq_5pct": 1 - threshold_at([d["aero"] for d in neg_eq]),
    }
    save("learn", {"beta4": b4.tolist(), "beta3": b3.tolist(), "w4": w4.tolist(),
                   "boot_lo": np.percentile(boot, 2.5, 0).tolist(),
                   "boot_hi": np.percentile(boot, 97.5, 0).tolist(),
                   "n_epochs": int(len(Y)), "pos_frac": float(Y.mean()),
                   "single": single, "thr": thr})
    print("beta4", np.round(b4, 2), "w4", np.round(w4, 3), "thr", thr)


def learned():
    """Learned design in the paper's additive form: w = beta/sum(beta) and the logistic
    threshold P_on mapped to the additive scale, C_on = (logit(P_on) - b0) / sum(beta)."""
    L = load("learn")
    b = np.array(L["beta4"])
    B = np.maximum(b[1:], 0).sum()
    p_on = L["thr"]["C_on_L"]
    c_on = (np.log(p_on / (1 - p_on)) - b[0]) / B
    ov = {"beta": None, "w": tuple(np.maximum(b[1:], 0) / B), "C_on": float(c_on),
          "C_off": float(c_on - 0.1)}
    kw_b = {"innov": {"innov_thr": L["thr"]["innov"]}, "rss": {"rss_thr": L["thr"]["rss"]}}
    return ov, kw_b


# --------------------------------------------------------------------------- roc
def stage_roc():
    L = load("learn")
    out = {}
    for nr, s in ((8, 3.0), (8, 10.0), (4, 3.0), (12, 3.0)):
        p = replace(BASE, Nr=nr, sigma_gnss=s)
        seeds = list(range(2000, 2040))
        half = len(seeds) // 2
        with Pool(POOL) as pool:
            eqp = pool.map(_collect, [(p, seeds[:half], None), (p, seeds[half:], None)])
            Lp = pool.map(_collect, [(p, seeds[:half], L["beta4"]), (p, seeds[half:], L["beta4"])])
        eq = [d for pp in eqp for d in pp[0]]
        lw = [d for pp in Lp for d in pp[0]]
        res = {}
        for name, data, key in (("aerofence_eq", eq, "aero"), ("aerofence_L", lw, "aero"),
                                ("innov", eq, "innov"), ("rss", eq, "rss")):
            neg = [d[key] for d in data if d["kind"] == 0]
            r = {}
            for lab, kinds in (("all", (1, 2)), ("smooth", (1,)), ("jump", (2,))):
                pos = [d[key] for d in data if d["kind"] in kinds]
                f, t, a = roc(neg, pos)
                r[lab] = {"auc": a, "tpr5": tpr_at(f, t, 0.05),
                          "fpr": f[:: max(len(f) // 200, 1)].tolist(),
                          "tpr": t[:: max(len(t) // 200, 1)].tolist()}
            res[name] = r
        out[f"Nr{nr}_s{int(s)}"] = res
        print(nr, s, {k: round(v["all"]["auc"], 3) for k, v in res.items()})
    save("roc", out)


# --------------------------------------------------------------------------- kpi table
def stage_kpi():
    ov, _ = learned()
    out = {}
    jobs, keys = [], []
    for s in (3.0, 10.0):
        for nr in (1, 4, 8, 12):
            p = replace(BASE, Nr=nr, sigma_gnss=s)
            jobs += split(p, RUNS, ("aerofence:L", "aerofence:eq"), 0, {"aerofence:L": ov})
            keys.append(f"s{int(s)}_Nr{nr}")
    parts = par(jobs)
    for i, k in enumerate(keys):
        out[k] = merge(parts[POOL * i: POOL * (i + 1)])
        print(k, {m: round(v["ttff"][0], 2) for m, v in out[k].items()})
    save("kpi", out)


def stage_tb():
    ov, _ = learned()
    jobs, keys = [], []
    for tb in (0.5, 1.0, 2.0):
        for nr in (1, 4, 8, 12):
            p = replace(BASE, Nr=nr, Tb=tb, sigma_gnss=3.0)
            jobs += split(p, 50, ("aerofence:L",), 0, {"aerofence:L": ov})
            keys.append(f"Tb{tb}_Nr{nr}")
    parts = par(jobs)
    save("tb", {k: merge(parts[POOL * i: POOL * (i + 1)]) for i, k in enumerate(keys)})


def stage_esc():
    ov, _ = learned()
    jobs, keys = [], []
    for e in (0.05, 0.10, 0.15):
        for nr in (1, 4, 8, 12):
            for corr in (0.0, 0.5):
                p = replace(BASE, Nr=nr, eps=e, eps_common=corr, sigma_gnss=3.0)
                jobs += split(p, 50, ("aerofence:L",), 0, {"aerofence:L": ov})
                keys.append(f"e{e}_Nr{nr}_c{corr}")
    parts = par(jobs)
    save("esc", {k: merge(parts[POOL * i: POOL * (i + 1)]) for i, k in enumerate(keys)})


def stage_con():
    ov, _ = learned()
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = tuple(f"aerofence:c{c}" for c in grid)
    jobs, keys = [], []
    for nr in (1, 4, 8, 12):
        p = replace(BASE, Nr=nr, sigma_gnss=3.0)
        ovs = {f"aerofence:c{c}": {**ov, "C_on": c, "C_off": max(c - 0.1, 0)} for c in grid}
        jobs += split(p, 40, modes, 0, ovs)
        keys.append(f"Nr{nr}")
    parts = par(jobs)
    save("con", {k: merge(parts[POOL * i: POOL * (i + 1)]) for i, k in enumerate(keys)})


def stage_flip():
    ov, _ = learned()
    jobs, keys = [], []
    for nr in (1, 4, 8, 12):
        p = replace(BASE, Nr=nr, sigma_gnss=10.0, behaviours=(0, 0, 0, 1.0),
                    p_spoof_smooth=0, p_spoof_jump=0, p_silent=0)
        ovs = {"aerofence:L": ov, "aerofence:nohyst": {**ov, "use_hysteresis": False}}
        jobs += split(p, 50, ("aerofence:L", "aerofence:nohyst", "raw"), 0, ovs)
        keys.append(f"Nr{nr}")
    parts = par(jobs)
    save("flip", {k: merge(parts[POOL * i: POOL * (i + 1)]) for i, k in enumerate(keys)})


def stage_compare():
    ov, kw = learned()
    modes = ("raw", "kf", "innov", "rss", "aerofence:eq", "aerofence:L")
    out = {}
    jobs, keys = [], []
    for s in (3.0, 10.0):
        p = replace(BASE, sigma_gnss=s)
        jobs += split(p, RUNS, modes, 5000, {"aerofence:L": ov}, kw)
        keys.append(f"s{int(s)}")
    parts = par(jobs)
    for i, k in enumerate(keys):
        out[k] = merge(parts[POOL * i: POOL * (i + 1)])
    # processing cost per message for each verification step
    from aerofence.fusion import epoch_features, run_fusion, track
    from aerofence.sim import simulate_scene
    rx, drones = simulate_scene(BASE, 99)
    te = np.arange(0, BASE.T_obs, BASE.dt_eval)
    nmsg = sum(len(d.t_arr) for d in drones)
    t0 = time.perf_counter()
    trs = [track(d, rx, BASE) for d in drones]
    t_track = time.perf_counter() - t0
    fts = [epoch_features(d, t, BASE, te) for d, t in zip(drones, trs)]
    cost = {"track_us_per_msg": 1e6 * t_track / nmsg}
    for m in modes:
        t0 = time.perf_counter()
        for d, t, f in zip(drones, trs, fts):
            run_fusion(d, rx, replace(BASE, **(ov if m == "aerofence:L" else {})), m.split(":")[0], t, f)
        cost[m] = 1e6 * (time.perf_counter() - t0) / nmsg
    out["cost_us_per_msg"] = cost
    save("compare", out)


def _ttff(args):
    p, seeds = args
    from aerofence.sim import simulate_scene
    v = []
    for sd in seeds:
        _, drones = simulate_scene(p, sd)
        v += [d.t_arr[0] - d.t_on if len(d.t_arr) else np.inf for d in drones]
    return v


def stage_ttff():
    out = {}
    for nr in (1, 4, 8, 12):
        p = replace(BASE, Nr=nr)
        with Pool(POOL) as pool:
            parts = pool.map(_ttff, [(p, range(0, 50)), (p, range(50, 100))])
        out[f"Nr{nr}"] = [x for pp in parts for x in pp]
    save("ttff", out)



# --------------------------------------------------------------------------- combined OR rule
def _epochs(args):
    """Per-drone epoch arrays needed to evaluate an OR rule exactly: relevance mask,
    additive learned confidence and horizon-mean NIS (NaN-safe)."""
    p, seeds, ov = args
    from aerofence.fusion import epoch_features, run_fusion, track
    from aerofence.sim import simulate_scene, GO_SILENT
    te = np.arange(0, p.T_obs, p.dt_eval)
    pm = replace(p, **ov)
    out = []
    for sd in seeds:
        rx, drones = simulate_scene(p, sd)
        for d in drones:
            if d.kind == GO_SILENT:
                continue
            tr = track(d, rx, p); f = epoch_features(d, tr, p, te)
            res = run_fusion(d, rx, pm, "aerofence", tr=tr, feats=f)
            rel = (res.zone >= 1) & (res.age <= p.T_silent)
            if not rel.any():
                continue
            cnt = f[6]
            nis = np.where(cnt >= 3, np.nan_to_num(res.nis_mean, nan=-np.inf), -np.inf)
            out.append((int(d.kind), rel, res.C.astype(np.float32), nis.astype(np.float32)))
    return out


def _escalated(rel, low, n):
    run = 0
    for r, l in zip(rel, low):
        run = run + 1 if (r and l) else 0
        if run >= n:
            return True
    return False


def _rate(data, c, t, n, kinds):
    sel = [x for x in data if x[0] in kinds]
    return float(np.mean([_escalated(x[1], (x[2] < c) | (x[3] > t), n) for x in sel])), len(sel)


def stage_combo():
    ov, kw = learned()
    L = load("learn")
    n = max(int(round(BASE.T_low / BASE.dt_eval)), 1)
    conds = [replace(BASE, Nr=nr, sigma_gnss=s) for nr in (4, 8, 12) for s in (3.0, 10.0)]
    jobs = [(c, range(1000 + 10 * i, 1010 + 10 * i), ov) for i, c in enumerate(conds)]
    with Pool(POOL) as pool:
        train = [x for pp in pool.map(_epochs, jobs) for x in pp]
    # calibration on training scenes only: for each innovation threshold, the largest C_on
    # meeting 5 % genuine false escalation; keep the pair with the highest training detection
    grid_t = [float(L["thr"]["innov"]) * f for f in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)]
    best = None; table = []
    for t in grid_t:
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = (lo + hi) / 2
            fpr, _ = _rate(train, mid, t, n, (0,))
            lo, hi = (mid, hi) if fpr <= 0.05 else (lo, mid)
        c = lo
        fpr, _ = _rate(train, c, t, n, (0,)); tpr, _ = _rate(train, c, t, n, (1, 2))
        table.append({"innov_thr": t, "C_on": c, "train_fpr": fpr, "train_tpr": tpr})
        if best is None or tpr > best["train_tpr"]:
            best = table[-1]
    print("calibration", table)
    # held-out evaluation at fixed thresholds (same 40 scenes as the ROC study)
    p = BASE
    with Pool(POOL) as pool:
        test = [x for pp in pool.map(_epochs, [(p, range(2000, 2020), ov), (p, range(2020, 2040), ov)]) for x in pp]
    ops = {"aerofence_L": (ov["C_on"], np.inf), "innov": (-np.inf, L["thr"]["innov"]),
           "aeroinnov": (best["C_on"], best["innov_thr"])}
    held = {}
    for k, (c, t) in ops.items():
        held[k] = {"fpr": _rate(test, c, t, n, (0,))[0], "tpr_all": _rate(test, c, t, n, (1, 2))[0],
                   "tpr_smooth": _rate(test, c, t, n, (1,))[0], "tpr_jump": _rate(test, c, t, n, (2,))[0]}
    print("held-out", held)
    # operational run on the comparison seeds
    ov2 = dict(ov); ov2["C_on"] = best["C_on"]; ov2["C_off"] = best["C_on"] - 0.1
    modes = ("aerofence:L", "aeroinnov")
    parts = par(split(p, RUNS, modes, 5000, {"aerofence:L": ov, "aeroinnov": ov2},
                      {"aeroinnov": {"innov_thr": best["innov_thr"]}}))
    kpi = merge(parts)
    from aerofence.fusion import epoch_features, run_fusion, track
    from aerofence.sim import simulate_scene
    rx, drones = simulate_scene(BASE, 99)
    te = np.arange(0, BASE.T_obs, BASE.dt_eval)
    nmsg = sum(len(d.t_arr) for d in drones)
    trs = [track(d, rx, BASE) for d in drones]
    fts = [epoch_features(d, t, BASE, te) for d, t in zip(drones, trs)]
    t0 = time.perf_counter()
    for d, t, f in zip(drones, trs, fts):
        run_fusion(d, rx, replace(BASE, **ov2), "aeroinnov", t, f, innov_thr=best["innov_thr"])
    cost = 1e6 * (time.perf_counter() - t0) / nmsg
    save("combo", {"calibration": table, "chosen": best, "held_out": held, "kpi": kpi,
                   "cost_us_per_msg": cost, "n_train_drones": len(train), "n_test_drones": len(test)})


# --------------------------------------------------------------------------- model diagnostics
def _collect_err(args):
    """Epoch evidence with the position error kept, so labels can be re-defined."""
    p, seeds, beta = args
    from aerofence.fusion import epoch_features, run_fusion, track
    from aerofence.sim import simulate_scene, GENUINE, GO_SILENT
    te = np.arange(0, p.T_obs, p.dt_eval)
    pm = p if beta is None else replace(p, beta=tuple(beta))
    X, E, D, S = [], [], [], []
    for sd in seeds:
        rx, drones = simulate_scene(p, sd)
        for j, d in enumerate(drones):
            if d.kind == GO_SILENT:
                continue
            tr = track(d, rx, p); f = epoch_features(d, tr, p, te)
            res = run_fusion(d, rx, pm, "aerofence", tr=tr, feats=f)
            rel = (res.zone >= 1) & (res.age <= p.T_silent)
            if not rel.any():
                continue
            gi = np.clip(np.round(te / 0.1).astype(int), 0, len(d.t_grid) - 1)
            err = np.linalg.norm(res.est - d.true_xy[gi], axis=1) if d.kind == GENUINE else np.full(len(te), np.inf)
            X.append(res.S[rel]); E.append(err[rel])
            D.append(np.full(rel.sum(), sd * 100 + j)); S.append(np.full(rel.sum(), sd))
    return np.vstack(X), np.concatenate(E), np.concatenate(D), np.concatenate(S)


def _cat(parts):
    return [np.concatenate([pp[i] for pp in parts]) if i else np.vstack([pp[0] for pp in parts]) for i in range(4)]


def stage_diag():
    from aerofence.analysis import fit_logistic, bootstrap_logistic
    conds = [replace(BASE, Nr=nr, sigma_gnss=s) for nr in (4, 8, 12) for s in (3.0, 10.0)]
    jobs = [(c, range(1000 + 10 * i, 1010 + 10 * i), None) for i, c in enumerate(conds)]
    with Pool(POOL) as pool:
        X, E, D, S = _cat(pool.map(_collect_err, jobs))
    y = (E < 50.0).astype(int)
    b = fit_logistic(X, y)
    out = {"beta4": b.tolist()}
    # 1000-replicate bootstrap, resampling drones and, separately, whole scenes
    for name, g in (("drone", D), ("scene", S)):
        bt = bootstrap_logistic(X, y, g, B=1000, seed=1)
        w = np.maximum(bt[:, 1:], 0); w = w / w.sum(1, keepdims=True)
        out[f"boot_{name}"] = {"beta_lo": np.percentile(bt, 2.5, 0).tolist(), "beta_hi": np.percentile(bt, 97.5, 0).tolist(),
                               "w_lo": np.percentile(w, 2.5, 0).tolist(), "w_hi": np.percentile(w, 97.5, 0).tolist(),
                               "share_nonneg": float((bt[:, 1:] >= 0).all(1).mean())}
    # sensitivity of the fitted weights to the accuracy criterion in the label
    lab = {}
    for thr in (10.0, 20.0, 50.0, 100.0):
        yy = (E < thr).astype(int); bb = fit_logistic(X, yy)
        ww = np.maximum(bb[1:], 0); ww = ww / ww.sum()
        lab[str(int(thr))] = {"beta": bb.tolist(), "w": ww.tolist(), "pos_frac": float(yy.mean())}
    out["label_threshold"] = lab
    # probability calibration on held-out epochs (Nr = 8, sigma = 3 m, ROC test seeds)
    with Pool(POOL) as pool:
        Xt, Et, Dt, St = _cat(pool.map(_collect_err, [(BASE, range(2000, 2020), None), (BASE, range(2020, 2040), None)]))
    yt = (Et < 50.0).astype(int)
    pr = 1 / (1 + np.exp(-(b[0] + Xt @ b[1:])))
    bins = np.linspace(0, 1, 11); k = np.clip(np.digitize(pr, bins) - 1, 0, 9)
    rel = [{"bin": [float(bins[i]), float(bins[i + 1])], "n": int((k == i).sum()),
            "mean_pred": float(pr[k == i].mean()) if (k == i).any() else None,
            "obs_freq": float(yt[k == i].mean()) if (k == i).any() else None} for i in range(10)]
    ece = float(sum(r["n"] * abs(r["mean_pred"] - r["obs_freq"]) for r in rel if r["n"]) / len(pr))
    out["calibration_heldout"] = {"brier": float(np.mean((pr - yt) ** 2)),
                                  "brier_climatology": float(np.mean((yt.mean() - yt) ** 2)),
                                  "ece": ece, "reliability": rel, "n_epochs": int(len(yt)),
                                  "pos_frac": float(yt.mean())}
    save("diag", out)
    print(json.dumps({k: v for k, v in out.items() if k != "calibration_heldout"}, indent=0)[:3000])
    print({k: out["calibration_heldout"][k] for k in ("brier", "brier_climatology", "ece", "n_epochs")})


# --------------------------------------------------------------------------- robustness to RF assumptions
def stage_robust():
    ov, kw = learned()
    L = load("learn"); X = load("combo")["chosen"]
    ov2 = dict(ov); ov2["C_on"] = X["C_on"]; ov2["C_off"] = X["C_on"] - 0.1
    modes = ("aerofence:L", "aeroinnov", "rss")
    ovs = {"aerofence:L": ov, "aeroinnov": ov2}
    kws = {**kw, "aeroinnov": {"innov_thr": X["innov_thr"]}}
    settings = {"nominal": {}, "rxcal2": {"rx_gain_sd": 2.0}, "rxcal4": {"rx_gain_sd": 4.0},
                "rxcal6": {"rx_gain_sd": 6.0}, "pattern3": {"tx_pattern_db": 3.0},
                "pattern6": {"tx_pattern_db": 6.0}, "spoofpwr6": {"spoof_pwr_db": 6.0}}
    out = {}
    for name, chg in settings.items():
        p = replace(BASE, **chg)
        out[name] = merge(par(split(p, RUNS, modes, 5000, ovs, kws)))
        r = out[name]
        print(name, {m: (round(r[m]["escalation"][0], 3), round(r[m]["spoof_detect"][0], 3)) for m in modes})
    save("robust", out)

# --------------------------------------------------------------------------- sensitivity
SENS = {  # parameter: (low, high) around the learned/nominal design
    "H": (5.0, 20.0), "T_ver": (2.0, 8.0), "T_low": (2.0, 8.0), "T_silent": (2.0, 6.0),
    "tau_p": (30.0, 120.0), "tau_v": (4.0, 16.0), "m_h": (5.0, 30.0), "T_dwell": (1.0, 4.0),
    "n_target": (2, 5), "C_on": (-0.1, 0.1),
}


def stage_sens():
    ov, _ = learned()
    p = replace(BASE, sigma_gnss=10.0)
    modes, ovs = ["aerofence:nom"], {"aerofence:nom": ov}
    for k, (lo, hi) in SENS.items():
        for tag, v in (("lo", lo), ("hi", hi)):
            o = dict(ov)
            if k == "C_on":
                o["C_on"] = ov["C_on"] + v; o["C_off"] = max(o["C_on"] - 0.1, 0)
            else:
                o[k] = type(getattr(p, k))(v)
            modes.append(f"aerofence:{k}_{tag}"); ovs[f"aerofence:{k}_{tag}"] = o
    # weights: shrink each learned coefficient by 50 %
    w0 = np.array(ov["w"])
    for j, nm in enumerate(["div", "con", "mot", "geo"]):
        for tag, f in (("lo", 0.5), ("hi", 1.5)):
            ww = w0.copy(); ww[j] *= f; ww /= ww.sum()
            modes.append(f"aerofence:b{nm}_{tag}"); ovs[f"aerofence:b{nm}_{tag}"] = {**ov, "w": tuple(ww)}
    modes.append("aerofence:eqw"); ovs["aerofence:eqw"] = {"w": (0.34, 0.33, 0.33, 0.0)}
    parts = par(split(p, 40, tuple(modes), 7000, ovs))
    save("sens", merge(parts))
    # global: Latin hypercube over the timing/gating parameters, standardised rank regression
    rng = np.random.default_rng(3)
    keys = [k for k in SENS if k != "n_target"]
    N = 60
    u = (rng.permuted(np.tile(np.arange(N), (len(keys), 1)), axis=1).T + rng.random((N, len(keys)))) / N
    jobs, samples = [], []
    for i in range(N):
        o = dict(ov)
        for j, k in enumerate(keys):
            lo, hi = SENS[k]
            v = lo + u[i, j] * (hi - lo)
            if k == "C_on":
                o["C_on"] = ov["C_on"] + v; o["C_off"] = max(o["C_on"] - 0.1, 0)
            else:
                o[k] = v
        samples.append([u[i, j] for j in range(len(keys))])
        jobs.append((p, 8, ("aerofence:x",), 8000, {"aerofence:x": o}, None))
    res = par(jobs)
    Xs = np.array(samples)
    kpis = ["latency", "escalation", "spoof_detect", "spurious_flip", "verified", "silent_delay"]
    src = {}
    for kp in kpis:
        y = np.array([r["aerofence:x"][kp][0] for r in res])
        ok = np.isfinite(y)
        rx_ = np.argsort(np.argsort(Xs[ok], 0), 0).astype(float)
        ry = np.argsort(np.argsort(y[ok])).astype(float)
        rx_ = (rx_ - rx_.mean(0)) / rx_.std(0); ry = (ry - ry.mean()) / ry.std()
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), rx_]), ry, rcond=None)
        r2 = 1 - np.sum((ry - np.column_stack([np.ones(ok.sum()), rx_]) @ coef) ** 2) / np.sum(ry ** 2)
        src[kp] = {"srrc": dict(zip(keys, coef[1:].tolist())), "r2": float(r2)}
    save("sens_global", {"keys": keys, "src": src, "N": N})


if __name__ == "__main__":
    stages = sys.argv[1:] or ["learn", "roc", "kpi", "ttff", "tb", "esc", "con", "flip", "compare", "combo", "diag", "robust", "sens"]
    for st in stages:
        t0 = time.time()
        globals()[f"stage_{st}"]()
        print(f"== {st} done in {time.time() - t0:.0f} s", flush=True)
