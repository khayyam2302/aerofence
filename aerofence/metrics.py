"""KPIs computed per drone and aggregated per run."""
from __future__ import annotations

import numpy as np

from .sim import GENUINE, GO_SILENT, SPOOF_JUMP, SPOOF_SMOOTH, zone_of


def drone_kpis(log, res, p):
    te = res.te
    gi = np.clip(np.round(te / (log.t_grid[1] - log.t_grid[0])).astype(int), 0, len(log.t_grid) - 1)
    tz = zone_of(log.true_xy[gi], p)
    k = {"kind": log.kind}
    k["ttff"] = res.first_arr - log.t_on if np.isfinite(res.first_arr) else np.nan
    entries = [a for a in res.alerts if a[1] == "entry"]
    k["n_alerts"] = len(res.alerts)
    k["cued"] = len(res.cues) > 0
    if log.kind == GENUINE:
        on = te >= log.t_on
        rel = (tz >= 1) & on
        k["continuity"] = np.mean(res.age[rel] <= 2 * p.Tb + 0.5) if rel.any() else np.nan
        k["verified"] = np.mean(res.verified[rel]) if rel.any() else np.nan
        k["in_warning"] = bool(rel.any())
        fine = zone_of(log.true_xy, p)
        hit = np.where((fine == 2) & (log.t_grid >= log.t_on))[0]
        if len(hit):
            t_e = log.t_grid[hit[0]]
            later = [a for a in entries if a[0] >= t_e - 5]
            k["latency"] = later[0][0] - t_e if later else np.nan
            k["entry_label"] = later[0][2] if later else None
        else:
            k["latency"], k["entry_label"] = np.nan, None
        changes = np.sum(np.diff(res.zone[on]) != 0)
        true_changes = np.sum(np.diff(tz[on]) != 0)
        k["spurious_flips_per_min"] = max(changes - true_changes, 0) / (p.T_obs / 60)
        k["flips_per_min"] = changes / (p.T_obs / 60)
    if log.kind in (SPOOF_SMOOTH, SPOOF_JUMP):
        alerted = len(entries) > 0 or any(a[1] == "approach" for a in res.alerts)
        k["spoof_alerted"] = alerted
        flagged = any(a[2] == "investigate" for a in res.alerts) or any(c[1] == "low_conf" for c in res.cues)
        k["spoof_flagged"] = flagged if alerted else np.nan
    if log.kind == GO_SILENT and np.isfinite(log.t_off):
        sil = [c[0] for c in res.cues if c[1] == "silent" and c[0] >= log.t_off]
        k["silent_delay"] = sil[0] - log.t_off if sil else np.nan
    return k


def aggregate(rows):
    def m(key, kinds=None, fn=np.nanmean):
        v = [r[key] for r in rows if key in r and (kinds is None or r["kind"] in kinds)]
        v = np.array([np.nan if x is None else float(x) for x in v], float)
        return fn(v) if len(v) and np.isfinite(v).any() else np.nan
    G = (GENUINE,)
    g_warn = [r for r in rows if r["kind"] == GENUINE and r.get("in_warning")]
    out = {
        "ttff": m("ttff", G),
        "latency": m("latency", G),
        "continuity": m("continuity", G),
        "verified": m("verified", G),
        "escalation": np.mean([r["cued"] for r in g_warn]) if g_warn else np.nan,
        "flip": m("flips_per_min", G),
        "spurious_flip": m("spurious_flips_per_min", G),
        "spoof_detect": m("spoof_flagged", (SPOOF_SMOOTH, SPOOF_JUMP)),
        "spoof_smooth_detect": m("spoof_flagged", (SPOOF_SMOOTH,)),
        "spoof_jump_detect": m("spoof_flagged", (SPOOF_JUMP,)),
        "genuine_investigate": np.mean([r["entry_label"] == "investigate" for r in rows
                                        if r["kind"] == GENUINE and r.get("entry_label")]) if any(
            r["kind"] == GENUINE and r.get("entry_label") for r in rows) else np.nan,
        "silent_delay": m("silent_delay", (GO_SILENT,)),
    }
    return out
