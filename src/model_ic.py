"""§3.4.2 / §3.4.3 — predictive power of the combined alpha model.

Each day t (T+1 daily):
  Step1  orthogonalise every alpha vs [industry + style] -> X̃_t (standardised);
         residualise the forward 1-day return vs [industry + style] -> realised
         alpha cross-section ω_{t+1}.
  Step2  multivariate (ridge) regression ω on X̃ -> factor-return vector f_t.
  Step3  forecast Ê(ω_{t+1}) = X̃_t · f̄, f̄ = mean of the trailing-T realised f_s.
  Step4  model IC_t = corr_xs(Ê, ω_{t+1}); collect the daily series.

`compute_model_ic_multi_memmap` does the per-date orthogonalisation ONCE and
evaluates several factor subsets (for §3.4.3's factor-count curve) in one pass,
streaming the alpha exposures per-date off the signal memmap (full-A, 16 GB-safe).

Ridge (config.MODEL_IC_RIDGE · n) is a deliberate deviation from the report's OLS:
our out-of-sample 189-alpha set is collinear, so plain OLS blows up as correlated
factors are added; ridge stabilises the multivariate fit.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config
from single_factor_test import _winsor_standardize, _orthogonalize


def _industry_dummies(industry_map, uni):
    ind = pd.Series(industry_map).reindex(uni).fillna("UNKNOWN")
    return pd.get_dummies(ind).astype(float)


def _ridge_solve(X, y, lam):
    """Ridge coefficients (XᵀX + λI)⁻¹ Xᵀy — stabilises the multivariate
    factor-return regression against the collinearity of the alpha set."""
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    try:
        return np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def compute_model_ic_multi_memmap(mmap_dir, style_long, industry_map, close_wide,
                                   subsets, T=None, min_T=None, log_every=400):
    """subsets: dict[name -> list of alpha column names]. Returns dict[name -> IC
    Series indexed by date]. Streams alpha exposures per-date from the memmap."""
    T = T or config.MODEL_IC_WINDOW
    min_T = min_T or config.MODEL_IC_MIN_WINDOW
    meta = np.load(os.path.join(mmap_dir, "meta.npz"), allow_pickle=True)
    dates = meta["dates"].astype(str)
    codes = meta["codes"].astype(str)
    valid = meta["valid"]
    Tn, N, K = int(meta["T"]), int(meta["N"]), int(meta["K"])
    ids = meta["alpha_ids"].tolist()
    alpha_cols = [f"alpha{i}" for i in ids]
    col_idx = {c: i for i, c in enumerate(alpha_cols)}
    sub_idx = {name: np.array([col_idx[c] for c in cols if c in col_idx])
               for name, cols in subsets.items()}
    mm = np.memmap(os.path.join(mmap_dir, "signals.f32"), dtype=np.float32, mode="r",
                   shape=(Tn * N, K))
    uni = list(codes)

    ind_arr = _industry_dummies(industry_map, uni).to_numpy()
    cw = close_wide.reindex(index=dates, columns=uni)
    fwd1 = (cw.shift(-1) / cw - 1.0).to_numpy()
    sl = style_long.set_index(["trade_date", "ts_code"])
    style_arr = {s: sl[s].unstack("ts_code").reindex(index=dates, columns=uni).to_numpy()
                 for s in config.STYLE_FACTORS}

    buffers = {name: [] for name in subsets}
    ic_recs = {name: [] for name in subsets}

    for j in range(Tn):
        sel = valid[j * N:(j + 1) * N]
        if sel.sum() < config.MIN_CROSS_SECTION:
            continue
        Y = np.asarray(mm[j * N:(j + 1) * N], dtype=float)[sel]
        st = np.column_stack([style_arr[s][j][sel] for s in config.STYLE_FACTORS])
        st = np.nan_to_num(_winsor_standardize(st), nan=0.0)
        Zind = ind_arr[sel]
        Zind = Zind[:, Zind.sum(0) > 0]
        Z = np.hstack([Zind, st])
        Xt = np.nan_to_num(_winsor_standardize(_orthogonalize(_winsor_standardize(Y), Z)), nan=0.0)

        r = fwd1[j][sel]
        mask = np.isfinite(r)
        if mask.sum() < config.MIN_CROSS_SECTION:
            continue
        rr = r[mask]
        wcoef, *_ = np.linalg.lstsq(Z[mask], rr, rcond=None)
        omega = rr - Z[mask] @ wcoef
        Xm = Xt[mask]
        lam = config.MODEL_IC_RIDGE * mask.sum()
        # the factor-count subsets are nested column-slices of Xm, so compute the
        # Gram matrix and cross-product ONCE and slice sub-blocks for each ridge.
        G = Xm.T @ Xm
        bvec = Xm.T @ omega

        for name, idx in sub_idx.items():
            if idx.size == 0:
                continue
            f_t = np.linalg.solve(G[np.ix_(idx, idx)] + lam * np.eye(idx.size), bvec[idx])
            buf = buffers[name]
            if len(buf) >= min_T:
                E = Xm[:, idx] @ np.mean(buf, axis=0)
                if np.std(E) > 0 and np.std(omega) > 0:
                    ic = np.corrcoef(E, omega)[0, 1]
                    if np.isfinite(ic):
                        ic_recs[name].append((dates[j], ic))
            buf.append(f_t)
            if len(buf) > T:
                buf.pop(0)
        if log_every and (j + 1) % log_every == 0:
            print(f"  model-IC {j+1}/{Tn} dates", flush=True)

    return {name: pd.Series(dict(recs)).sort_index() for name, recs in ic_recs.items()}


def export_predictions_memmap(mmap_dir, style_long, industry_map, close_wide,
                              factor_cols=None, T=None, min_T=None, log_every=400):
    """Stream the combined-model forecast Ê(ω_{t+1}) = X̃_t · f̄ for EVERY tradable
    stock each day (point-in-time: f̄ = trailing-T mean of realised factor returns
    f_s, s<t). Returns a long DataFrame [trade_date, ts_code, pred] — the §3.6
    optimiser's expected-alpha objective. Uses the ridge fit, all valid factors."""
    T = T or config.MODEL_IC_WINDOW
    min_T = min_T or config.MODEL_IC_MIN_WINDOW
    meta = np.load(os.path.join(mmap_dir, "meta.npz"), allow_pickle=True)
    dates = meta["dates"].astype(str)
    codes = meta["codes"].astype(str)
    valid = meta["valid"]
    Tn, N, K = int(meta["T"]), int(meta["N"]), int(meta["K"])
    ids = meta["alpha_ids"].tolist()
    alpha_cols = [f"alpha{i}" for i in ids]
    col_idx = {c: i for i, c in enumerate(alpha_cols)}
    if factor_cols is None:
        factor_cols = alpha_cols
    fidx = np.array([col_idx[c] for c in factor_cols if c in col_idx])
    mm = np.memmap(os.path.join(mmap_dir, "signals.f32"), dtype=np.float32, mode="r",
                   shape=(Tn * N, K))
    uni = list(codes)
    ind_arr = _industry_dummies(industry_map, uni).to_numpy()
    cw = close_wide.reindex(index=dates, columns=uni)
    fwd1 = (cw.shift(-1) / cw - 1.0).to_numpy()
    sl = style_long.set_index(["trade_date", "ts_code"])
    style_arr = {s: sl[s].unstack("ts_code").reindex(index=dates, columns=uni).to_numpy()
                 for s in config.STYLE_FACTORS}
    codes_arr = np.asarray(uni)

    buf, recs = [], []
    for j in range(Tn):
        sel = valid[j * N:(j + 1) * N]
        if sel.sum() < config.MIN_CROSS_SECTION:
            continue
        Y = np.asarray(mm[j * N:(j + 1) * N], dtype=float)[sel]
        st = np.nan_to_num(_winsor_standardize(
            np.column_stack([style_arr[s][j][sel] for s in config.STYLE_FACTORS])), nan=0.0)
        Zind = ind_arr[sel]
        Zind = Zind[:, Zind.sum(0) > 0]
        Z = np.hstack([Zind, st])
        Xt = np.nan_to_num(_winsor_standardize(_orthogonalize(_winsor_standardize(Y), Z)), nan=0.0)
        Xs = Xt[:, fidx]                                  # exposures, ALL valid stocks
        # point-in-time forecast from PAST factor returns
        if len(buf) >= min_T:
            E = Xs @ np.mean(buf, axis=0)
            sel_codes = codes_arr[sel]
            for cc, e in zip(sel_codes, E):
                recs.append((dates[j], cc, e))
        # update buffer with this date's realised factor returns (ridge on masked)
        r = fwd1[j][sel]
        mask = np.isfinite(r)
        if mask.sum() >= config.MIN_CROSS_SECTION:
            rr = r[mask]
            wcoef, *_ = np.linalg.lstsq(Z[mask], rr, rcond=None)
            omega = rr - Z[mask] @ wcoef
            Xm = Xs[mask]
            f_t = _ridge_solve(Xm, omega, config.MODEL_IC_RIDGE * mask.sum())
            buf.append(f_t)
            if len(buf) > T:
                buf.pop(0)
        if log_every and (j + 1) % log_every == 0:
            print(f"  predict {j+1}/{Tn} dates", flush=True)
    return pd.DataFrame(recs, columns=["trade_date", "ts_code", "pred"])


def ic_stats(ic):
    """Mean IC, model t-stat (mean/std·√N), daily win rate."""
    ic = ic.dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    tstat = mean / std * np.sqrt(n) if std > 0 else np.nan
    return {"n_days": n, "ic_mean": float(mean), "tstat": float(tstat),
            "win_rate": float((ic > 0).mean())}
