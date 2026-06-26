"""A1: build a fast PRICE-ONLY panel for the report's 2010-2017 window.

The report's pre-2017 financials/forecast would be a huge fetch, and build_panel's
~17.5k one-by-one statement reads hit a severe macOS background-I/O throttle. The
189 alphas are price-only, and the in-sample BARRA uses the 5 price-based styles
(config: GTJA_RUN_TAG=insample → STYLE_FACTORS = Beta/Momentum/Size/Volatility/
Liquidity), so a price-only panel suffices. We bulk-read the per-day price files
(fast) and fill the financial/forecast columns with NaN so the pipeline's column
reads still work. Writes <tushare-data>/data/panel_insample/year=YYYY/panel.parquet.
"""
import glob
import os
import sys
import time
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
import config

RAW = config.RAW_DIR
OUT = os.path.join(config.DATA_ROOT, "panel_insample")
YEARS = range(2010, 2018)
END = "20170430"
DB = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
      "pe", "pe_ttm", "pb", "ps_ttm", "total_share", "float_share", "free_share",
      "total_mv", "circ_mv"]
NAN_FIN = ["n_income_attr_p_ttm", "n_cashflow_act_ttm", "total_revenue_ttm",
           "total_hldr_eqy_exc_min_int", "total_liab", "total_assets", "lt_borr",
           "bond_payable", "non_cur_liab_due_1y", "total_ncl", "fc_p_change_min",
           "fc_p_change_max", "fc_net_profit_min", "fc_net_profit_max",
           "fc_last_parent_net", "exp_diluted_eps", "exp_n_income",
           "exp_yoy_net_profit", "basic_eps"]


def _bulk(sub, year):
    fs = [f for f in glob.glob(os.path.join(RAW, sub, str(year), "*.parquet")) if os.path.isfile(f)]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def main():
    for y in YEARS:
        t = time.time()
        dy = _bulk("daily", y)
        aj = _bulk("adj_factor", y)[["ts_code", "trade_date", "adj_factor"]]
        db = _bulk("daily_basic", y)[DB]
        if y == 2017:
            dy = dy[dy.trade_date <= END]; db = db[db.trade_date <= END]; aj = aj[aj.trade_date <= END]
        m = dy.merge(aj, on=["ts_code", "trade_date"], how="left").merge(db, on=["ts_code", "trade_date"], how="left")
        m["close_hfq"] = m["close"] * m["adj_factor"]
        m["amplitude"] = (m["high"] - m["low"]) / m["pre_close"] * 100
        for c in NAN_FIN:
            m[c] = np.nan
        d = os.path.join(OUT, f"year={y}"); os.makedirs(d, exist_ok=True)
        m.to_parquet(os.path.join(d, "panel.parquet"), index=False)
        print(f"  {y}: {len(m)} rows, {m.ts_code.nunique()} stocks, "
              f"{m.trade_date.min()}..{m.trade_date.max()}, {time.time()-t:.1f}s", flush=True)
    print("price-only panel ->", OUT)


if __name__ == "__main__":
    main()
