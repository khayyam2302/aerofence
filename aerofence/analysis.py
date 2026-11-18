"""Threshold-free scores, ROC analysis, and learning of the confidence weights."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .fusion import epoch_features, run_fusion, track
from .sim import GENUINE, GO_SILENT, SPOOF_JUMP, SPOOF_SMOOTH, Params, simulate_scene, zone_of


def _roll(x, n, fn):
    if len(x) < n:
        return np.array([])
    return fn(sliding_window_view(x, n), axis=1)


def drone_scores(log, res, p: Params):
    """Suspicion score per method (higher = more suspicious). A drone is escalated for low
    confidence when its score exceeds the method threshold for T_low continuously, so the
    scores map one-to-one onto the escalation decision and give exact ROC curves."""
    n = max(int(round(p.T_low / p.dt_eval)), 1)
    rel = (res.zone >= 1) & (res.age <= p.T_silent)
    if rel.sum() < n:
        return None
    # windows fully inside relevant epochs
    ok = _roll(rel.astype(float), n, np.min) > 0
    if not ok.any():
        return None
    C = _roll(res.C, n, np.max)[ok]
    nis = _roll(res.nis_mean, n, np.min)[ok]
    rss = _roll(np.nan_to_num(res.rss_mean, nan=-np.inf), n, np.min)[ok]
    return {"aero": 1 - C.min(), "innov": nis.max(), "rss": rss.max()}


def roc(scores_neg, scores_pos):
    s = np.r_[scores_neg, scores_pos]
    y = np.r_[np.zeros(len(scores_neg)), np.ones(len(scores_pos))]
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    keep = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0, tp[keep] / max(y.sum(), 1)]
    fpr = np.r_[0, fp[keep] / max((1 - y).sum(), 1)]
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def tpr_at(fpr, tpr, target=0.05):
    i = np.searchsorted(fpr, target, "right") - 1
    return float(tpr[max(i, 0)])


def threshold_at(neg_scores, target=0.05):
    """Score threshold that flags a fraction `target` of genuine tracks."""
    v = np.sort(np.asarray(neg_scores)[np.isfinite(neg_scores)])
    return float(np.quantile(v, 1 - target)) if len(v) else np.inf


def collect(p: Params, seeds, beta=None):
    """Per-drone scores (all methods) and epoch-level evidence for learning."""
    te = np.arange(0, p.T_obs, p.dt_eval)
    pm = p if beta is None else Params(**{**p.__dict__, "beta": tuple(beta)})
    drones_out, X, Y, G = [], [], [], []
    for sd in seeds:
        rx, drones = simulate_scene(p, sd)
        for d in drones:
            tr = track(d, rx, p)
            f = epoch_features(d, tr, p, te)
            res = run_fusion(d, rx, pm, "aerofence", tr=tr, feats=f)
            sc = drone_scores(d, res, pm)
            if sc is not None and d.kind != GO_SILENT:
                sc["kind"] = d.kind
                drones_out.append(sc)
            # epoch-level learning data: relevant, broadcasting epochs
            rel = (res.zone >= 1) & (res.age <= p.T_silent)
            if not rel.any() or d.kind == GO_SILENT:
                continue
            gi = np.clip(np.round(te / 0.1).astype(int), 0, len(d.t_grid) - 1)
            if d.kind == GENUINE:
                err = np.linalg.norm(res.est - d.true_xy[gi], axis=1)
                y = (err < 50.0).astype(int)
            else:
                y = np.zeros(len(te), int)
            X.append(res.S[rel]); Y.append(y[rel]); G.append(np.full(rel.sum(), sd * 100 + len(G)))
    return drones_out, (np.vstack(X), np.concatenate(Y), np.concatenate(G))


def fit_logistic(X, y, l2=1.0, iters=50):
    """L2-regularised logistic regression by Newton's method (MAP estimate under a
    Gaussian prior on the coefficients). Returns (b0, b) on the raw feature scale."""
    Xa = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xa.shape[1])
    lam = np.r_[0, np.full(X.shape[1], l2)]
    for _ in range(iters):
        z = Xa @ b
        pr = 1 / (1 + np.exp(-z))
        g = Xa.T @ (pr - y) + lam * b
        Hm = (Xa * (pr * (1 - pr))[:, None]).T @ Xa + np.diag(lam)
        step = np.linalg.solve(Hm, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def bootstrap_logistic(X, y, groups, B=200, seed=0, **kw):
    """Cluster bootstrap over tracks (epochs of one track are correlated)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        out.append(fit_logistic(X[ii], y[ii], **kw))
    return np.array(out)
