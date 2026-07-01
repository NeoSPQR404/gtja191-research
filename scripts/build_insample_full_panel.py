"""A1 (full 9-style): merge backfilled financials onto the price-only in-sample panel.

Reuses tsbarra.build_panel's TTM + point-in-time as-of logic, but reads the CONSOLIDATED
financial parquets (data/raw/_consolidated/{income,balancesheet,cashflow}.parquet — made
by chunked reads of the universe stocks, to dodge the macOS I/O throttle) and merges them
onto the existing panel_insample (price-only) base — reading ~11 files total, no per-stock
reads. forecast/express are skipped (5297 per-day files → throttle); the consensus proxies
(EPIBS/EGIBS) then renormalize to ETOP/CETOP & SGRO/EGRO (same as the OOS fallback).
Writes panel_insample_full/year=YYYY/panel.parquet (full 9-style panel).
"""
import os
import sys
import time
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
import config
_TS = os.path.abspath(os.path.join(config.DATA_ROOT, ".."))   # gtja191-tushare-data
sys.path.insert(0, _TS)
from tsbarra import build_panel as bp, storage        # noqa: E402

CONS = os.path.join(config.RAW_DIR, "_consolidated")
PRICE = os.path.join(config.DATA_ROOT, "panel_insample")
OUT = os.path.join(config.DATA_ROOT, "panel_insample_full")
# base placeholder financial columns that income/cf/balance will supply (drop before merge)
DROP = ["n_income_attr_p_ttm", "n_cashflow_act_ttm", "total_revenue_ttm",
        "total_hldr_eqy_exc_min_int", "total_liab", "total_assets", "lt_borr",
        "bond_payable", "non_cur_liab_due_1y", "total_ncl", "basic_eps"]


def _patched(ds):
    p = os.path.join(CONS, f"{ds}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()  # forecast/express -> empty


def main():
    storage.load_dataset = _patched
    bp.storage.load_dataset = _patched
    t = time.time()
    income = bp._prep_income()
    cashflow = bp._prep_cashflow()
    balance = bp._prep_balance()
    print(f"prepped financials (income={len(income)} cf={len(cashflow)} bs={len(balance)}) "
          f"{time.time()-t:.0f}s", flush=True)
    for y in range(2010, 2018):
        base = pd.read_parquet(os.path.join(PRICE, f"year={y}", "panel.parquet"))
        base = base.drop(columns=[c for c in DROP if c in base.columns])
        base["_trade_dt"] = pd.to_datetime(base["trade_date"], format="%Y%m%d")
        base = bp._asof_merge(base, income, "fin_eff_date")
        base = bp._asof_merge(base, cashflow, "cf_eff_date")
        base = bp._asof_merge(base, balance, "bs_eff_date")
        base = base.drop(columns=["_trade_dt"]).sort_values(["trade_date", "ts_code"])
        d = os.path.join(OUT, f"year={y}"); os.makedirs(d, exist_ok=True)
        base.to_parquet(os.path.join(d, "panel.parquet"), index=False)
        cov = base["n_income_attr_p_ttm"].notna().mean() if "n_income_attr_p_ttm" in base else 0
        print(f"  {y}: {len(base)} rows, {base.shape[1]} cols, TTM-cov {cov*100:.0f}%", flush=True)
    print("full panel ->", OUT, f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
