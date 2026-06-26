"""BARRA CNE5S style factors (the 9 styles used in the report's neutralisation),
each composited from its descriptors with the MSCI CNE5 descriptor weights
(config.CNE5S_WEIGHTS, from gtja191-reference/Barra CNE5S Model Descriptor.pdf).

Descriptors (report appendix-1 parameters: vs CSI300, half-lives 60/120/40):
  Beta        BETA                          (250d ewm hl60 vs CSI300)
  Momentum    RSTR                          (mean log ret, ewm hl120, lag 21)
  Size        LNCAP                         (ln total market cap)
  EarnYield   EPIBS*, ETOP, CETOP           (.68/.11/.21)
  Volatility  DASTD, CMRA, HSIGMA           (ewm hl40 / 12-mo range / Beta resid)
  Growth      SGRO, EGRO, EGIBS*, EGIBS_s*  (.47/.24/.18/.11; 5y annual reg slope)
  Value       BTOP                          (book equity / market cap)
  Leverage    MLEV, DTOA, BLEV              (.38/.35/.27; PE≈0 approx)
  Liquidity   STOM, STOQ, STOA              (.35/.35/.30; log turnover sums)

* = sell-side analyst consensus, NOT on this tushare token. Per the locked
decision (DATA_AUDIT.md §6.4) they are PROXIED by company guidance:
  EPIBS    ~ forward expected net profit / market cap
             (forecast net_profit midpoint, express n_income fallback);
  EGIBS,
  EGIBS_s  ~ forward YoY net-profit growth (forecast p_change midpoint,
             express yoy_net_profit fallback) — the long/short horizons are
             not separable from company guidance, so both use the same series.

Descriptors are cap-weighted-mean-0 / unit-std standardised (winsorised ±3) per
date. The composite is a NaN-AWARE per-cell weighted average: when a descriptor
(typically a sparse consensus proxy) is missing for a stock-day, the style is
renormalised over that cell's available descriptors — so e.g. EarnYield collapses
to ETOP 0.344 / CETOP 0.656 exactly where the EPIBS proxy is absent. The composite
is then re-standardised.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config

WEIGHTS = {k: config.CNE5S_WEIGHTS[k] for k in config.STYLE_FACTORS}
STYLES = list(WEIGHTS)
ALL_DESCRIPTORS = sorted({d for w in WEIGHTS.values() for d in w})
_NEED = set(ALL_DESCRIPTORS)                              # descriptors actually requested


# ---------------------------------------------------------------- descriptors
def compute_descriptors(panel, bench, batch_size=600, log=True):
    """Driver: compute descriptors in batches of stocks so peak memory is one
    batch's wide working frame (not all of full-A's 8.7M rows at once, which
    swap-thrashes a 16 GB machine, especially with the 8.4 GB signal memmap also
    in page cache). Returns the slim (keys + total_mv + descriptors) frame."""
    import math
    codes = sorted(panel["ts_code"].unique())
    nb = math.ceil(len(codes) / batch_size)
    parts = []
    for bi in range(nb):
        batch = set(codes[bi * batch_size:(bi + 1) * batch_size])
        sub = panel[panel["ts_code"].isin(batch)]
        parts.append(_descriptors_batch(sub, bench))
        if log:
            print(f"  descriptors batch {bi+1}/{nb} ({len(batch)} stocks)", flush=True)
    return pd.concat(parts, ignore_index=True)


def _descriptors_batch(panel, bench):
    df = panel.sort_values(["ts_code", "trade_date"]).copy()
    # downcast to float32 — float64 working columns blow past RAM on full-A.
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df["ret"] = df.groupby("ts_code", group_keys=False)["close_hfq"].pct_change()
    df["logret"] = np.log1p(df["ret"])
    b = bench.sort_values("trade_date").copy()
    b["mkt"] = b["bench_close"].pct_change()
    df = df.merge(b[["trade_date", "mkt"]], on="trade_date", how="left")
    df = df.sort_values(["ts_code", "trade_date"])

    def _ewm_mean(col, hl, mp):
        return df.groupby("ts_code")[col].transform(lambda s: s.ewm(halflife=hl, min_periods=mp).mean())

    # --- Beta / HSIGMA via half-life-60 ewm moments ---
    bw = config.BETA_HALFLIFE
    mp = config.BETA_WINDOW // 2
    df["_rm"] = df["ret"] * df["mkt"]
    df["_mm"] = df["mkt"] ** 2
    df["_rr"] = df["ret"] ** 2
    er, em = _ewm_mean("ret", bw, mp), _ewm_mean("mkt", bw, mp)
    erm, emm, err = _ewm_mean("_rm", bw, mp), _ewm_mean("_mm", bw, mp), _ewm_mean("_rr", bw, mp)
    cov, var_m, var_r = erm - er * em, emm - em * em, err - er * er
    df["BETA"] = cov / var_m.where(var_m > 0)
    df["HSIGMA"] = np.sqrt((var_r - df["BETA"] * cov).clip(lower=0))
    df.drop(columns=["_rm", "_mm", "_rr"], inplace=True)
    del er, em, erm, emm, err, cov, var_m, var_r

    # --- Volatility DASTD (ewm hl40); Momentum RSTR (ewm hl120 mean log ret, lag 21) ---
    df["DASTD"] = df.groupby("ts_code")["ret"].transform(
        lambda s: s.ewm(halflife=config.DASTD_HALFLIFE, min_periods=config.DASTD_WINDOW // 2).std())
    df["RSTR"] = df.groupby("ts_code")["logret"].transform(
        lambda s: s.ewm(halflife=config.RSTR_HALFLIFE, min_periods=config.BETA_WINDOW).mean().shift(config.RSTR_LAG))

    # --- level descriptors (LNCAP price-based; the rest need financials → guard) ---
    mv = df["total_mv"].where(df["total_mv"] > 0)
    df["LNCAP"] = np.log(mv)
    if _NEED & {"ETOP", "CETOP", "BTOP", "MLEV", "DTOA", "BLEV"}:
        df["ETOP"] = df["n_income_attr_p_ttm"] / df["total_mv"]
        df["CETOP"] = df["n_cashflow_act_ttm"] / df["total_mv"]
        df["BTOP"] = df["total_hldr_eqy_exc_min_int"] / df["total_mv"]
        LD = df[["lt_borr", "bond_payable", "non_cur_liab_due_1y"]].sum(axis=1, min_count=1).fillna(0)
        ME = df["total_mv"]
        BE = df["total_hldr_eqy_exc_min_int"]
        df["MLEV"] = (ME + LD) / ME.where(ME > 0)             # PE (preferred) ≈ 0
        df["DTOA"] = df["total_liab"] / df["total_assets"].where(df["total_assets"] > 0)
        df["BLEV"] = (BE + LD) / BE.where(BE != 0)

    # --- consensus PROXIES (forecast 业绩预告 / express 业绩快报) ---
    if _NEED & {"EPIBS", "EGIBS", "EGIBS_s"}:
        fc_np_mid = df[["fc_net_profit_min", "fc_net_profit_max"]].mean(axis=1)    # 万元
        fwd_np = fc_np_mid.where(fc_np_mid.notna(), df["exp_n_income"] / 1e4)
        df["EPIBS"] = fwd_np / mv                                                  # forward E/P proxy
        fc_g = df[["fc_p_change_min", "fc_p_change_max"]].mean(axis=1) / 100.0
        fwd_g = fc_g.where(fc_g.notna(), df["exp_yoy_net_profit"] / 100.0)
        df["EGIBS"] = fwd_g
        df["EGIBS_s"] = fwd_g

    # --- Liquidity STOM/STOQ/STOA from daily turnover (换手率, %) ---
    def _rsum(w, minp):
        return df.groupby("ts_code")["turnover_rate"].transform(
            lambda s: s.fillna(0).rolling(w, min_periods=minp).sum())

    md = config.STO_MONTH_DAYS
    df["STOM"] = np.log(_rsum(md, md // 2).clip(lower=1e-6))
    df["STOQ"] = np.log((_rsum(md * 3, md * 3 // 2) / 3.0).clip(lower=1e-6))
    df["STOA"] = np.log((_rsum(md * 12, md * 6) / 12.0).clip(lower=1e-6))

    # --- CMRA: 12-month cumulative log-return range, lagged onto the next month ---
    df["ym"] = (df["trade_date"].str[:6] if df["trade_date"].dtype == object
                else (df["trade_date"] // 100))
    monthly = df.groupby(["ts_code", "ym"])["logret"].sum().reset_index()

    def _cmra(s):
        return s.rolling(config.CMRA_MONTHS, min_periods=config.CMRA_MONTHS).apply(
            lambda x: np.expm1(np.cumsum(x)).max() - np.expm1(np.cumsum(x)).min(), raw=True)
    monthly["CMRA"] = monthly.groupby("ts_code", group_keys=False)["logret"].apply(_cmra)
    lag = monthly[["ts_code", "ym", "CMRA"]].copy()
    lag["ym"] = monthly.groupby("ts_code")["ym"].shift(-1)
    df = df.merge(lag.dropna(subset=["ym"]), on=["ts_code", "ym"], how="left")

    # --- SGRO / EGRO: 5y annual-statement growth (as-of by ann_date) — reads income
    # files per stock, so only when Growth descriptors are actually requested ---
    if _NEED & {"SGRO", "EGRO"}:
        growth = _annual_growth(df["ts_code"].unique())
        df = _asof_merge_growth(df, growth)

    # slim to keys + total_mv (cap weights) + descriptors; drop the raw price /
    # financial / forecast inputs so the standardisation stage stays memory-lean.
    keep = ["trade_date", "ts_code", "total_mv"] + [d for d in ALL_DESCRIPTORS if d in df.columns]
    df = df[keep].copy()
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


def _annual_growth(codes):
    """Per stock: 5y trailing growth = OLS slope / |mean| of annual total_revenue
    (SGRO) and n_income_attr_p (EGRO), keyed by ann_date for point-in-time use."""
    rows = []
    inc_dir = os.path.join(config.RAW_DIR, "income")
    for code in codes:
        p = os.path.join(inc_dir, f"{code}.parquet")
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_parquet(p, columns=["end_date", "ann_date", "total_revenue", "n_income_attr_p"])
        except Exception:
            continue
        if d.empty:
            continue
        d = d[d["end_date"].astype(str).str[4:] == "1231"]
        d = d.dropna(subset=["ann_date"]).sort_values("end_date").drop_duplicates("end_date", keep="first")
        if len(d) < 3:
            continue
        for col, name in (("total_revenue", "SGRO"), ("n_income_attr_p", "EGRO")):
            vals = d[["end_date", "ann_date", col]].dropna()
            for i in range(len(vals)):
                window = vals.iloc[max(0, i - (config.GROWTH_YEARS - 1)):i + 1]
                if len(window) < 3:
                    continue
                y = window[col].to_numpy(dtype=float)
                slope = np.polyfit(np.arange(len(y)), y, 1)[0]
                rows.append((code, int(vals.iloc[i]["ann_date"]), name,
                             slope / (np.abs(y.mean()) + 1e-9)))
    gdf = pd.DataFrame(rows, columns=["ts_code", "ann_date", "desc", "val"])
    if gdf.empty:
        return gdf
    return gdf.pivot_table(index=["ts_code", "ann_date"], columns="desc", values="val").reset_index()


def _asof_merge_growth(df, growth):
    if growth is None or growth.empty:
        df["SGRO"] = np.nan; df["EGRO"] = np.nan
        return df
    df["_td"] = df["trade_date"].astype(int)
    growth = growth.copy()
    growth["ann_date"] = growth["ann_date"].astype(int)
    for c in ("SGRO", "EGRO"):
        if c not in growth.columns:
            growth[c] = np.nan
    growth = growth.sort_values("ann_date")
    left = df.sort_values("_td")
    m = pd.merge_asof(left, growth[["ts_code", "ann_date", "SGRO", "EGRO"]],
                      left_on="_td", right_on="ann_date", by="ts_code", direction="backward")
    m.index = left.index
    df["SGRO"] = m["SGRO"]; df["EGRO"] = m["EGRO"]
    return df.drop(columns="_td")


# ---------------------------------------------------------------- standardise + composite
def _standardize_xs(df, x, cap_pos):
    """Per-date: cap-weighted mean 0 (over finite x), equal-weight unit std, winsor ±3."""
    td = df["trade_date"]
    w = cap_pos.where(x.notna())
    mu = (w * x).groupby(td).transform("sum") / w.groupby(td).transform("sum")
    sd = x.groupby(td).transform("std")
    return ((x - mu) / sd.replace(0, np.nan)).clip(-3, 3)


def compute_style_factors(panel, bench, return_descriptors=False):
    """Return long [trade_date, ts_code, <9 styles>] (+ descriptor z-scores if asked)."""
    df = compute_descriptors(panel, bench)
    cap_pos = df["total_mv"].where(df["total_mv"] > 0)
    for d in ALL_DESCRIPTORS:
        df[d + "_z"] = _standardize_xs(df, df[d], cap_pos).astype("float32")

    for style, w in WEIGHTS.items():
        num = den = None
        for d, wt in w.items():
            z = df[d + "_z"]
            avail = z.notna()
            term = (z * wt).where(avail, 0.0)
            wcol = pd.Series(np.where(avail.to_numpy(), wt, 0.0), index=df.index)
            num = term if num is None else num + term
            den = wcol if den is None else den + wcol
        comp = num / den.where(den > 0)
        df[style] = _standardize_xs(df, comp, cap_pos)

    keep = ["trade_date", "ts_code"] + STYLES
    if return_descriptors:
        keep += [d + "_z" for d in ALL_DESCRIPTORS]
    return df[keep].reset_index(drop=True)
