"""Draw the paper figures from results/*.json (run experiments.py first)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
    "xtick.color": "#333333", "ytick.color": "#333333"})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a5cc4"]
NR = (1, 4, 8, 12)
W, Hh = 3.45, 2.25


def J(name):
    return json.loads((RES / f"{name}.json").read_text())


def style(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "both", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_ttff():
    d = J("ttff")
    fig, ax = plt.subplots(figsize=(W, Hh))
    for i, nr in enumerate(NR):
        v = np.sort(np.array(d[f"Nr{nr}"], float))
        v = v[np.isfinite(v)]
        ax.step(v, np.arange(1, len(v) + 1) / len(d[f"Nr{nr}"]), where="post", color=C[i], lw=1.6,
                label=f"$N_r$ = {nr}")
    ax.set_xscale("log"); ax.set_xlim(0.05, 100)
    ax.set_xlabel("Time to first fix after power-on (s)"); ax.set_ylabel("Empirical CDF")
    ax.legend(frameon=False, loc="lower right"); style(ax, False)
    save(fig, "fig_ttff_cdf.pdf")


def fig_tb():
    d = J("tb")
    fig, ax = plt.subplots(figsize=(W, Hh))
    tbs = (0.5, 1.0, 2.0)
    for i, nr in enumerate(NR):
        m = [d[f"Tb{tb}_Nr{nr}"]["aerofence:L"]["latency"][0] for tb in tbs]
        s = [d[f"Tb{tb}_Nr{nr}"]["aerofence:L"]["latency"][1] for tb in tbs]
        ax.errorbar(tbs, m, yerr=s, color=C[i], lw=1.6, marker="o", ms=3.5, capsize=2,
                    label=f"$N_r$ = {nr}")
    ax.set_xticks(tbs); ax.set_xlabel("Broadcast period $T_b$ (s)")
    ax.set_ylabel("Entry-alert latency (s)")
    ax.legend(frameon=False, ncol=2); style(ax)
    save(fig, "fig_latency_tb.pdf")


def fig_con():
    d = J("con")
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, axs = plt.subplots(1, 2, figsize=(2 * W * 0.98, Hh))
    for i, nr in enumerate(NR):
        r = d[f"Nr{nr}"]
        axs[0].plot(grid, [r[f"aerofence:c{c}"]["verified"][0] for c in grid], color=C[i], lw=1.6,
                    marker="o", ms=3, label=f"$N_r$ = {nr}")
        axs[1].plot(grid, [r[f"aerofence:c{c}"]["escalation"][0] for c in grid], color=C[i], lw=1.6,
                    marker="o", ms=3)
    c_on = learned_con()
    for ax in axs:
        ax.axvline(c_on, color="#555555", lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel(r"Verification threshold $C_{\min}^{on}$"); style(ax)
    axs[0].text(c_on + 0.01, 0.05, "design", fontsize=6.5, color="#555555")
    axs[0].set_ylabel("Verified-track rate"); axs[0].set_ylim(0, 1.02)
    axs[1].set_ylabel("Unnecessary escalation rate"); axs[1].set_ylim(0, 1.02)
    axs[0].legend(frameon=False, loc="lower left")
    save(fig, "fig_verified_con.pdf")


def learned_con():
    L = J("learn")
    b = np.array(L["beta4"]); B = np.maximum(b[1:], 0).sum(); p = L["thr"]["C_on_L"]
    return float((np.log(p / (1 - p)) - b[0]) / B)


def fig_esc():
    d = J("esc")
    fig, ax = plt.subplots(figsize=(W, Hh))
    for i, e in enumerate((0.05, 0.10, 0.15)):
        for corr, ls in ((0.0, "-"), (0.5, (0, (3, 2)))):
            v = [d[f"e{e}_Nr{nr}_c{corr}"]["aerofence:L"]["escalation"][0] for nr in NR]
            ax.plot(NR, v, color=C[i], lw=1.6, ls=ls, marker="o", ms=3,
                    label=fr"$\epsilon$ = {e:.2f}" if corr == 0 else None)
    ax.plot([], [], color="#555555", lw=1.2, label="independent loss")
    ax.plot([], [], color="#555555", lw=1.2, ls=(0, (3, 2)), label="50 % common loss")
    ax.set_xticks(NR); ax.set_xlabel("Number of receivers $N_r$")
    ax.set_ylabel("Unnecessary escalation rate"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5); style(ax)
    save(fig, "fig_escalation.pdf")


def fig_roc():
    d = J("roc")["Nr8_s3"]
    names = [("aerofence_L", "AeroFence (learned $w$)"), ("aerofence_eq", "AeroFence (equal $w$)"),
             ("rss", "RSS claim check [GhostBuster]"), ("innov", "Innovation test [Bjorkman]")]
    fig, ax = plt.subplots(figsize=(W, Hh * 1.1))
    for i, (k, lab) in enumerate(names):
        r = d[k]["all"]
        ax.plot(r["fpr"], r["tpr"], color=C[i], lw=1.6, label=f"{lab}, AUC {r['auc']:.2f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=(0, (1, 2)))
    ax.axvline(0.05, color="#555555", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("False-escalation rate (genuine tracks)")
    ax.set_ylabel("Detection rate (spoofed tracks)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=6.3, loc="lower right"); style(ax, False)
    save(fig, "fig_roc.pdf")


def fig_sens():
    d = J("sens")
    nom = d["aerofence:nom"]
    params = [("H", "$H$ (5/20 s)"), ("T_ver", r"$T_{ver}$ (2/8 s)"), ("T_low", r"$T_{low}$ (2/8 s)"),
              ("T_silent", r"$T_{silent}$ (2/6 s)"), ("tau_p", r"$\tau_p$ (30/120 m)"),
              ("tau_v", r"$\tau_v$ (4/16 m/s)"), ("m_h", "$m_h$ (5/30 m)"), ("T_dwell", r"$T_{dwell}$ (1/4 s)"),
              ("n_target", "$n_{target}$ (2/5)"), ("C_on", r"$C^{on}_{\min}$ ($\pm$0.1)"),
              ("bdiv", r"$w_{div}$ ($\times$0.5/1.5)"), ("bcon", r"$w_{con}$ ($\times$0.5/1.5)"),
              ("bmot", r"$w_{mot}$ ($\times$0.5/1.5)"), ("bgeo", r"$w_{geo}$ ($\times$0.5/1.5)")]
    kpis = [("spoof_detect", "Spoof detection rate"), ("escalation", "Unnecessary escalation"),
            ("latency", "Entry-alert latency (s)"), ("spurious_flip", "Spurious flips (1/min)")]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 3.0), sharey=True)
    y = np.arange(len(params))[::-1]
    for a, (kp, lab) in enumerate(kpis):
        ax = axs[a]
        base = nom[kp][0]
        for j, (k, _) in enumerate(params):
            lo = d[f"aerofence:{k}_lo"][kp][0] - base
            hi = d[f"aerofence:{k}_hi"][kp][0] - base
            ax.barh(y[j] + 0.18, lo, 0.34, color=C[0])
            ax.barh(y[j] - 0.18, hi, 0.34, color=C[1])
        ax.axvline(0, color="#333333", lw=0.6)
        ax.set_title(f"{lab}\nnominal {base:.2f}", fontsize=7)
        ax.grid(axis="x", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([p[1] for p in params], fontsize=6.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C[0], label="low value"), Patch(color=C[1], label="high value")],
               frameon=False, fontsize=6.5, loc="lower left", ncol=2)
    fig.supxlabel("Change from nominal design", fontsize=7)
    save(fig, "fig_sensitivity.pdf")


if __name__ == "__main__":
    for f in (fig_ttff, fig_tb, fig_con, fig_esc, fig_roc, fig_sens):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e.filename)
