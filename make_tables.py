"""Regenerate the numerical tables of the paper from results/*.json (run experiments.py first).

Writes one Markdown file with Tables 3, 7, 8, 10, 11 and 13 so every number in the paper can be
checked against the stored results.

Usage: python3 make_tables.py [output_folder]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES
OUT.mkdir(parents=True, exist_ok=True)


def J(name):
    return json.loads((RES / f"{name}.json").read_text())


def fmt(v, d=2):
    return "--" if v is None else f"{v:.{d}f}"


def table3(L, D=None):
    """Intervals from the 1000-replicate drone bootstrap of the diag stage when available."""
    if D is not None:
        L = {**L, "boot_lo": D["boot_drone"]["beta_lo"], "boot_hi": D["boot_drone"]["beta_hi"]}
    names = ["Intercept", "S_div", "S_con", "S_mot", "S_geo"]
    rows = ["| Term | beta | 95% bootstrap interval | Weight w |", "|---|---|---|---|"]
    for i, n in enumerate(names):
        w = "--" if i == 0 else fmt(L["w4"][i - 1], 3)
        rows.append(f"| {n} | {L['beta4'][i]:.2f} | [{L['boot_lo'][i]:.2f}, {L['boot_hi'][i]:.2f}] | {w} |")
    b = np.array(L["beta4"]); B = np.maximum(b[1:], 0).sum(); p = L["thr"]["C_on_L"]
    con = (np.log(p / (1 - p)) - b[0]) / B
    rows.append(f"\nP_on = {p:.3f} -> C_on = {con:.3f}, C_off = {con - 0.1:.3f}; "
                f"training epochs = {L['n_epochs']}, positive share = {L['pos_frac']:.3f}; "
                f"equal-weight threshold at 5% = {L['thr']['C_on_eq_5pct']:.3f}")
    return "\n".join(rows)


def table7(K):
    keys = [("ttff", 2), ("latency", 2), ("continuity", 2), ("verified", 2), ("escalation", 3),
            ("spoof_detect", 2), ("genuine_investigate", 3), ("silent_delay", 2)]
    rows = ["| sigma | Nr | " + " | ".join(k for k, _ in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for s in ("3", "10"):
        for nr in (1, 4, 8, 12):
            r = K[f"s{s}_Nr{nr}"]["aerofence:L"]
            rows.append(f"| {s} m | {nr} | " + " | ".join(f"{r[k][0]:.{d}f} ± {r[k][1]:.{d}f}" for k, d in keys) + " |")
    return "\n".join(rows)


def table8(C, R, X):
    names = [("raw", "(a) Raw"), ("kf", "(b) KF"), ("innov", "(c) Innovation"), ("rss", "(d) RSS check"),
             ("aerofence:eq", "(e) Equal w"), ("aerofence:L", "(f) Learned w")]
    rk = {"innov": "innov", "rss": "rss", "aerofence:eq": "aerofence_eq", "aerofence:L": "aerofence_L"}
    roc = R["Nr8_s3"]
    cost = C["cost_us_per_msg"]
    rows = ["| Method | latency | flips s3 | flips s10 | escalation | spoof all | smooth | jump | AUC | TPR@5% | cost us/msg |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in names:
        a, b = C["s3"][k], C["s10"][k]
        auc = roc[rk[k]]["all"]["auc"] if k in rk else None
        tpr = roc[rk[k]]["all"].get("tpr5") if k in rk else None
        c = cost.get(k, cost.get(k.split(":")[0]))
        if k != "raw":
            c = (c or 0) + cost["track_us_per_msg"]
        rows.append(f"| {lab} | {a['latency'][0]:.2f} | {a['spurious_flip'][0]:.3f} | {b['spurious_flip'][0]:.3f} | "
                    f"{a['escalation'][0]:.3f} | {a['spoof_detect'][0]:.2f} | {a['spoof_smooth_detect'][0]:.2f} | "
                    f"{a['spoof_jump_detect'][0]:.2f} | {fmt(auc)} | {fmt(tpr)} | {fmt(c, 1)} |")
    g = X["kpi"]["aeroinnov"]
    rows.append(f"| (g) (f) OR (c) | {g['latency'][0]:.2f} | -- | -- | {g['escalation'][0]:.3f} | "
                f"{g['spoof_detect'][0]:.2f} | {g['spoof_smooth_detect'][0]:.2f} | {g['spoof_jump_detect'][0]:.2f} | -- | -- | "
                f"{X['cost_us_per_msg'] + cost['track_us_per_msg']:.1f} |")
    h = X["held_out"]
    rows.append("\nHeld-out operating points (thresholds fixed on training scenes): " + "; ".join(
        f"{k}: FPR {v['fpr']:.3f}, detection {v['tpr_all']:.3f} (smooth {v['tpr_smooth']:.3f}, jump {v['tpr_jump']:.3f})"
        for k, v in h.items()))
    rows.append(f"Joint calibration chosen: C_on = {X['chosen']['C_on']:.3f}, innovation threshold = {X['chosen']['innov_thr']:.1f}")
    return "\n".join(rows)


def table_robust(Rb):
    names = [("nominal", "Nominal"), ("rxcal2", "Rx gain offsets 2 dB"), ("rxcal4", "Rx gain offsets 4 dB"),
             ("rxcal6", "Rx gain offsets 6 dB"), ("pattern3", "Tx pattern 3 dB"), ("pattern6", "Tx pattern 6 dB"),
             ("spoofpwr6", "Spoofer power +/-6 dB")]
    rows = ["| Condition | (f) esc | (f) det | (g) esc | (g) det | (d) esc | (d) det |", "|---|---|---|---|---|---|---|"]
    for k, lab in names:
        r = Rb[k]
        rows.append(f"| {lab} | " + " | ".join(f"{r[m]['escalation'][0]:.3f} | {r[m]['spoof_detect'][0]:.2f}"
                                             for m in ("aerofence:L", "aeroinnov", "rss")) + " |")
    return "\n".join(rows)


def diag_text(D):
    c = D["calibration_heldout"]
    out = [f"Held-out calibration: Brier {c['brier']:.3f} (climatology {c['brier_climatology']:.3f}), "
           f"ECE {c['ece']:.3f}, {c['n_epochs']} epochs"]
    for k, v in D["label_threshold"].items():
        out.append(f"Label {k} m: w = " + ", ".join(f"{x:.3f}" for x in v["w"]))
    for k in ("boot_drone", "boot_scene"):
        b = D[k]
        out.append(f"{k}: beta 95% = " + ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(b["beta_lo"], b["beta_hi"]))
                   + f"; share with all coefficients >= 0: {b['share_nonneg']:.3f}")
    return "\n".join(out)


def table10(F):
    rows = ["| Nr | raw: flips / spurious, latency | KF no hysteresis | KF hysteresis + dwell |", "|---|---|---|---|"]
    for nr in (1, 4, 8, 12):
        r = F[f"Nr{nr}"]
        cell = lambda m: f"{r[m]['flip'][0]:.2f} / {r[m]['spurious_flip'][0]:.3f}, {r[m]['latency'][0]:.1f} s"
        rows.append(f"| {nr} | {cell('raw')} | {cell('aerofence:nohyst')} | {cell('aerofence:L')} |")
    return "\n".join(rows)


def table12(G):
    kp = list(G["src"].keys())
    rows = ["| parameter | " + " | ".join(kp) + " |", "|---|" + "---|" * len(kp)]
    for p in G["keys"]:
        rows.append(f"| {p} | " + " | ".join(f"{G['src'][k]['srrc'][p]:.2f}" for k in kp) + " |")
    rows.append("| R^2 | " + " | ".join(f"{G['src'][k]['r2']:.2f}" for k in kp) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    parts = ["# AeroFence: tables regenerated from results/*.json\n"]
    def _d():
        try:
            return J("diag")
        except FileNotFoundError:
            return None
    for title, fn in [("Table 3. Learned logistic model", lambda: table3(J("learn"), _d())),
                      ("Model diagnostics (Section VI)", lambda: diag_text(J("diag"))),
                      ("Table 7. Operational indicators (learned weights)", lambda: table7(J("kpi"))),
                      ("Table 8. Comparison of decision methods (Nr = 8)", lambda: table8(J("compare"), J("roc"), J("combo"))),
                      ("Table 10. Robustness to radio assumptions", lambda: table_robust(J("robust"))),
                      ("Table 11. Zone flips for boundary-skimming drones", lambda: table10(J("flip"))),
                      ("Table 13. Global sensitivity (SRRC)", lambda: table12(J("sens_global")))]:
        try:
            parts.append(f"## {title}\n\n{fn()}\n")
            print("ok", title)
        except (FileNotFoundError, KeyError) as e:
            parts.append(f"## {title}\n\nnot available: {e}\n")
            print("skip", title, e)
    (OUT / "tables.md").write_text("\n".join(parts))
    print("wrote", OUT / "tables.md")
