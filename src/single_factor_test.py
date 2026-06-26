"""§3.4.1 single-factor significance & effective-period test.

For each date t the cross-section of every alpha is winsorised+standardised, then
orthogonalised against the shared control matrix Z = [industry dummies | 9 BARRA
style factors] (Step-1 residual, identical across horizons). For each horizon d the
per-date factor return f_{k,t,d} is the cross-sectional slope of the NON-overlapping
forward d-day return on the orthogonalised exposure. Annualised (report §3.4.1):

    FR_k(d) = 252 * mean_t(f_{k,t,d}) / d
    IR_k(d) = mean_t(f_{k,t,d}) / std_t(f_{k,t,d}) * sqrt(252 / d)

Two backends share the per-date logic: ``run_test`` (in-RAM long table, for the
topN universe) and ``run_test_memmap`` (reads per-date slices from the disk-backed
signal memmap, for full-A on a 16 GB machine).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config

ALPHA_COLS_ALL = [f"alpha{i}" for i in range(1, 192)]


def _winsor_standardize(M):
    """Column-wise MAD winsorise then z-score. M: (n,k) float array (NaNs ok)."""
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-12, np.nan, mad)
    lo = med - config.WINSOR_SIGMA * mad
    hi = med + config.WINSOR_SIGMA * mad
    M = np.clip(M, lo, hi)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    sd = np.where(sd < config.MIN_FACTOR_STD, np.nan, sd)
    return (M - mu) / sd


def _orthogonalize(Y, Z):
    """Residualise each column of Y (n,k) against Z (n,p)."""
    Yf = np.nan_to_num(Y, nan=0.0)
    coef, *_ = np.linalg.lstsq(Z, Yf, rcond=None)
    return Yf - Z @ coef


def _factor_returns_one_date(Y, st, Zind, fwd_by_d, j, horizons, fr_records):
    """Shared per-date core: orthogonalise exposures, regress each horizon's
    forward return, append f-vectors to ``fr_records``. ``fwd_by_d[d]`` is the
    (n,) forward d-day return for this date's selected stocks."""
    Y = _winsor_standardize(Y)
    st = np.nan_to_num(_winsor_standardize(st), nan=0.0)
    Zind = Zind[:, Zind.sum(0) > 0]                       # drop empty industries
    Z = np.hstack([Zind, st])
    resid = _orthogonalize(Y, Z)
    resid = np.nan_to_num(_winsor_standardize(resid), nan=0.0)
    ss = (resid ** 2).sum(0)
    ss = np.where(ss < config.MIN_FACTOR_STD, np.nan, ss)
    for d in horizons:
        if config.NON_OVERLAPPING and (j % d != 0):
            continue
        r = fwd_by_d[d]
        mask = np.isfinite(r)
        if mask.sum() < config.MIN_CROSS_SECTION:
            continue
        rc = np.where(mask, r, 0.0)
        rc = rc - rc[mask].mean()
        rc[~mask] = 0.0
        f = (resid * rc[:, None]).sum(0) / ss
        fr_records[d].append((j, f))


def _assemble(fr_records, dates, alpha_cols, horizons):
    factor_returns, fr_table, ir_table, counts = {}, {}, {}, {}
    fr_table = pd.DataFrame(index=alpha_cols, columns=horizons, dtype=float)
    ir_table = pd.DataFrame(index=alpha_cols, columns=horizons, dtype=float)
    counts = pd.DataFrame(index=alpha_cols, columns=horizons, dtype=float)
    for d in horizons:
        recs = fr_records[d]
        if not recs:
            continue
        idx = [dates[r[0]] for r in recs]
        mat = np.vstack([r[1] for r in recs])
        fr = pd.DataFrame(mat, index=idx, columns=alpha_cols)
        factor_returns[d] = fr
        n = len(fr)
        count = fr.count(axis=0)
        valid = count >= config.MIN_OBS_FRAC * n
        mean, std = fr.mean(axis=0), fr.std(axis=0)
        fr_table[d] = (config.TRADING_DAYS_PER_YEAR * mean / d).where(valid)
        ir_table[d] = (mean / std * np.sqrt(config.TRADING_DAYS_PER_YEAR / d)).where(valid)
        counts[d] = count
    return factor_returns, fr_table, ir_table, counts


# --------------------------------------------------------------- memmap backend
def run_test_memmap(mmap_dir, style_long, industry_map, close_wide, horizons=None):
    """Stream the full-A single-factor test from the signal memmap written by
    panel_engine.run_panel_to_memmap. Styles/returns/industry are held wide in RAM
    (small); only the 9.4 GB alpha block is read per-date off disk."""
    horizons = horizons or config.HORIZONS
    meta = np.load(os.path.join(mmap_dir, "meta.npz"), allow_pickle=True)
    dates = meta["dates"].astype(str)
    codes = meta["codes"].astype(str)
    valid = meta["valid"]
    T, N, K = int(meta["T"]), int(meta["N"]), int(meta["K"])
    ids = meta["alpha_ids"].tolist()
    alpha_cols = [f"alpha{i}" for i in ids]
    mm = np.memmap(os.path.join(mmap_dir, "signals.f32"), dtype=np.float32, mode="r",
                   shape=(T * N, K))
    uni = list(codes)

    # industry dummies (S x I) aligned to codes order
    ind = pd.Series(industry_map).reindex(uni).fillna("UNKNOWN")
    ind_arr = pd.get_dummies(ind).astype(float).to_numpy()

    # forward d-day returns, wide (T x N) aligned to dates/codes
    cw = close_wide.reindex(index=dates, columns=uni)
    fwd = {d: (cw.shift(-d) / cw - 1.0).to_numpy() for d in horizons}

    # style loadings wide (T x N) per style
    sl = style_long.set_index(["trade_date", "ts_code"])
    style_arr = {s: sl[s].unstack("ts_code").reindex(index=dates, columns=uni).to_numpy()
                 for s in config.STYLE_FACTORS}

    fr_records = {d: [] for d in horizons}
    for j in range(T):
        if dates[j] < config.TEST_START:                  # warmup-only dates (in-sample)
            continue
        sel = valid[j * N:(j + 1) * N]
        if sel.sum() < config.MIN_CROSS_SECTION:
            continue
        Y = np.asarray(mm[j * N:(j + 1) * N], dtype=float)[sel]
        st = np.column_stack([style_arr[s][j][sel] for s in config.STYLE_FACTORS])
        Zind = ind_arr[sel]
        fwd_by_d = {d: fwd[d][j][sel] for d in horizons}
        _factor_returns_one_date(Y, st, Zind, fwd_by_d, j, horizons, fr_records)
    return _assemble(fr_records, dates, alpha_cols, horizons)


# --------------------------------------------------------------- long-table backend
def run_test(alpha_long, style_long, industry_map, close_wide, horizons=None):
    """In-RAM backend (topN universe). Same result as run_test_memmap."""
    horizons = horizons or config.HORIZONS
    alpha_cols = [c for c in ALPHA_COLS_ALL if c in alpha_long.columns]
    dates = close_wide.index.to_numpy()
    date_pos = {d: i for i, d in enumerate(dates)}
    cw = close_wide
    fwd = {d: (cw.shift(-d) / cw - 1.0) for d in horizons}
    sl = style_long.set_index(["trade_date", "ts_code"])
    ind = pd.Series(industry_map).reindex(cw.columns.tolist()).fillna("UNKNOWN")
    ind_dummies = pd.get_dummies(ind).astype(float)

    fr_records = {d: [] for d in horizons}
    for t, g in alpha_long.sort_values("trade_date").groupby("trade_date", sort=True):
        if len(g) < config.MIN_CROSS_SECTION:
            continue
        j = date_pos.get(t)
        if j is None:
            continue
        codes = g["ts_code"].to_numpy()
        Y = g[alpha_cols].to_numpy(dtype=float)
        try:
            st = sl.loc[t].reindex(codes)[config.STYLE_FACTORS].to_numpy(dtype=float)
        except KeyError:
            continue
        Zind = ind_dummies.reindex(codes).fillna(0.0).to_numpy()
        fwd_by_d = {d: fwd[d].loc[t].reindex(codes).to_numpy(dtype=float) for d in horizons}
        _factor_returns_one_date(Y, st, Zind, fwd_by_d, j, horizons, fr_records)
    return _assemble(fr_records, dates, alpha_cols, horizons)


# --------------------------------------------------------------- pruning
def valid_factors(counts, factor_returns, horizon):
    n = len(factor_returns[horizon])
    c = counts[horizon]
    return c.index[c >= config.MIN_OBS_FRAC * n].tolist()


def prune_correlated(factor_returns, ir_table, threshold=None, horizon=None):
    """Drop the lower-|IR| factor of each pair whose factor-return series correlate
    above ``threshold`` (report's de-duplication). Returns (kept, dropped, corr_before,
    corr_after)."""
    threshold = threshold or config.PRUNE_CORR_THRESHOLD
    horizon = horizon or config.PRUNE_HORIZON
    fr = factor_returns[horizon]
    n = len(fr)
    well = fr.count(axis=0) >= config.MIN_OBS_FRAC * n
    fr = fr.loc[:, well[well].index].dropna(axis=1, how="all")
    ir_abs = ir_table[horizon].abs()
    corr = fr.corr()
    cols = list(corr.columns)
    kept, dropped = set(cols), []
    for a in ir_abs.reindex(cols).sort_values(ascending=False).index.tolist():
        if a not in kept:
            continue
        for b in cols:
            if b == a or b not in kept:
                continue
            c = corr.loc[a, b]
            if np.isfinite(c) and abs(c) > threshold:
                kept.discard(b)
                dropped.append((b, a, float(c)))
    kept = [c for c in cols if c in kept]
    return kept, dropped, corr, fr[kept].corr()
