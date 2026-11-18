"""Run Monte Carlo experiments for one parameter setting and several methods."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .fusion import epoch_features, run_fusion, track
from .metrics import aggregate, drone_kpis
from .sim import Params, simulate_scene

FEATURE_KEYS = {"H", "n_target", "tau_c", "tau_p", "tau_v", "tau_t", "q_acc", "v_max", "dt_eval"}


def run_setting(p: Params, runs: int, modes=("aerofence",), seed0: int = 0, overrides=None,
                kw=None, keep_rows=False):
    """modes: names like 'aerofence' or 'aerofence:opt'; text before ':' selects the method.
    overrides: mode -> dict of Params fields used for that mode only.
    kw: mode -> extra keyword arguments for run_fusion (e.g. thresholds)."""
    overrides = overrides or {}
    kw = kw or {}
    per_mode = {m: [] for m in modes}
    rows_all = {m: [] for m in modes}
    for r in range(runs):
        rx, drones = simulate_scene(p, seed0 + r)
        te = np.arange(0, p.T_obs, p.dt_eval)
        base_tr = [track(d, rx, p) for d in drones]
        base_ft = [epoch_features(d, t, p, te) for d, t in zip(drones, base_tr)]
        for m in modes:
            ov = overrides.get(m, {})
            pm = replace(p, **ov)
            recompute = bool(FEATURE_KEYS & set(ov))
            rows = []
            for d, t, f in zip(drones, base_tr, base_ft):
                if recompute:
                    t = track(d, rx, pm)
                    f = epoch_features(d, t, pm, te)
                res = run_fusion(d, rx, pm, m.split(":")[0], tr=t, feats=f, **kw.get(m, {}))
                rows.append(drone_kpis(d, res, pm))
            per_mode[m].append(aggregate(rows))
            if keep_rows:
                rows_all[m].extend(rows)
    out = {}
    for m, lst in per_mode.items():
        out[m] = {k: (float(np.nanmean([x[k] for x in lst])), float(np.nanstd([x[k] for x in lst])))
                  for k in lst[0]}
    return (out, rows_all) if keep_rows else out
