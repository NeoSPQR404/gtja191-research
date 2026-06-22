"""Load the tushare panel and assemble alpha / style-factor inputs.

Provides: the yearly panel concatenated over the configured window; a full-A
(ex-ST) or top-N liquid universe; the CSI300 benchmark open+close; and the
申万一级 (SW2021 L1) industry map fetched into data/raw/ of this repo.

Units: tushare ``vol`` is in 手 (100 shares), ``amount`` in 千元 (thousand yuan),
so per-share vwap = amount*1000 / (vol*100) = amount*10/vol (verified ≈ close).
Prices are made hfq-comparable by × ``adj_factor`` (``close_hfq`` already in panel).
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd

import config

# Columns needed for the alpha engine (§3.3). The BARRA style step loads a wider
# set via STYLE_COLS.
PRICE_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "vol", "amount", "adj_factor", "close_hfq",
]
STYLE_COLS = PRICE_COLS + [
    "total_mv", "circ_mv", "float_share", "total_share", "pb", "turnover_rate",
    "n_income_attr_p_ttm", "n_cashflow_act_ttm", "total_revenue_ttm",
    "total_hldr_eqy_exc_min_int", "total_liab", "total_assets",
    "lt_borr", "bond_payable", "non_cur_liab_due_1y", "total_ncl",
    # consensus proxies (forecast / express):
    "fc_p_change_min", "fc_p_change_max", "fc_net_profit_min", "fc_net_profit_max",
    "fc_last_parent_net", "exp_diluted_eps", "exp_n_income", "exp_yoy_net_profit",
    "basic_eps",
]


def load_panel(columns=PRICE_COLS, universe=None):
    """Concatenate the yearly panel parquets within [START_DATE, END_DATE]."""
    files = sorted(glob.glob(os.path.join(config.PANEL_DIR, "year=*", "panel.parquet")))
    if not files:
        raise FileNotFoundError(f"no panel parquets under {config.PANEL_DIR}")
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=columns)
        df = df[(df["trade_date"] >= config.START_DATE) & (df["trade_date"] <= config.END_DATE)]
        if universe is not None:
            df = df[df["ts_code"].isin(universe)]
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def load_stock_basic():
    return pd.read_parquet(config.STOCK_BASIC, columns=["ts_code", "name", "industry"])


def load_industry():
    """申万一级 (SW2021 L1) industry label per stock, from the fetched membership
    table. Returns a Series ts_code -> l1_name (current-snapshot classification;
    see DATA_AUDIT.md §2a for the look-ahead caveat)."""
    mem = pd.read_parquet(config.SW_MEMBER, columns=["ts_code", "l1_code", "l1_name"])
    mem = mem.drop_duplicates("ts_code").set_index("ts_code")
    return mem["l1_name"]


def select_universe(panel, basic):
    """full_a: all stocks with >= MIN_HISTORY_DAYS history, ex-ST.
    topN: additionally restrict to the top UNIVERSE_SIZE by median daily amount."""
    name = basic.set_index("ts_code")["name"].fillna("")
    is_st = name.apply(lambda s: any(m in s for m in config.ST_MARKERS))
    st_codes = set(name.index[is_st])

    grp = panel.groupby("ts_code")
    nobs = grp.size()
    eligible = set(nobs.index[nobs >= config.MIN_HISTORY_DAYS]) - st_codes

    if config.UNIVERSE_MODE == "topN":
        med_amt = grp["amount"].median()
        med_amt = med_amt[med_amt.index.isin(eligible)]
        return sorted(med_amt.sort_values(ascending=False).head(config.UNIVERSE_SIZE).index)
    return sorted(eligible)


def load_benchmark():
    """CSI300 open+close (bench_open is needed by alpha75/149/181/182 and is not
    carried in the panel, so we read it from the raw index_daily parquet)."""
    idx = pd.read_parquet(config.CSI300_RAW, columns=["trade_date", "open", "close"])
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    return idx.rename(columns={"open": "bench_open", "close": "bench_close"})[
        ["trade_date", "bench_open", "bench_close"]
    ]


def zero_ff(dates):
    """A zero Fama-French series. Only alpha30 consumes mkt/smb/hml, and alpha30
    is skipped (config.SKIP_ALPHAS), so the engine just needs the columns present."""
    d = pd.Series(sorted(pd.unique(dates)))
    return pd.DataFrame({"trade_date": d, "mkt": 0.0, "smb": 0.0, "hml": 0.0})


def build_stock_inputs(g, bench, ff):
    """Build the per-stock ``data`` dict consumed by the reference Alpha_Set with
    the per-stock Factor_Calculator (used by the engine-equivalence validator).
    ``g`` is one stock's panel rows; prices are hfq-adjusted and aligned on the
    stock's own trading dates."""
    g = g.sort_values("trade_date")
    g = g.merge(bench, on="trade_date", how="left").merge(ff, on="trade_date", how="left")
    adj = g["adj_factor"].to_numpy()
    vol = g["vol"].to_numpy()
    amount = g["amount"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol > 0, amount * 10.0 / vol, np.nan) * adj
    close_hfq = g["close_hfq"].to_numpy()
    data = {
        "close": pd.Series(close_hfq),
        "open": pd.Series(g["open"].to_numpy() * adj),
        "high": pd.Series(g["high"].to_numpy() * adj),
        "low": pd.Series(g["low"].to_numpy() * adj),
        "volume": pd.Series(vol),
        "amount": pd.Series(amount),
        "vwap": pd.Series(vwap),
        "ret": pd.Series(pd.Series(close_hfq).pct_change().fillna(0).to_numpy()),
        "benchmarkindexclose": pd.Series(g["bench_close"].to_numpy()),
        "benchmarkindexopen": pd.Series(g["bench_open"].to_numpy()),
        "mkt": pd.Series(g["mkt"].fillna(0).to_numpy()),
        "smb": pd.Series(g["smb"].fillna(0).to_numpy()),
        "hml": pd.Series(g["hml"].fillna(0).to_numpy()),
        "date": pd.Series(g["trade_date"].to_numpy()),
        "stock_code": g["ts_code"].iloc[0],
    }
    return data, g["trade_date"].to_numpy()
