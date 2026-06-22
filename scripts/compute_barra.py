"""Step 3 — compute the 9 BARRA CNE5S style factors and cache them.

Diagnostics: per-descriptor coverage (esp. the forecast/express consensus
proxies), per-style coverage, and the CNE5 style correlation structure (sanity:
Vol–Liq positive, Size–Liq negative). Cross-section RANK is not involved here.

Usage: python scripts/compute_barra.py [--universe full_a|topN] [--smoke N]
"""
import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import config
import data_loader as dl
import barra_cne5s as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=config.UNIVERSE_MODE, choices=["full_a", "topN"])
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    config.UNIVERSE_MODE = args.universe

    t0 = time.time()
    basic = dl.load_stock_basic()
    panel = dl.load_panel(columns=dl.STYLE_COLS)
    uni = dl.select_universe(panel, basic)
    if args.smoke:
        uni = uni[: args.smoke]
    panel = panel[panel["ts_code"].isin(uni)].reset_index(drop=True)
    bench = dl.load_benchmark()
    print(f"universe={args.universe} stocks={panel['ts_code'].nunique()} "
          f"dates={panel['trade_date'].nunique()} rows={len(panel)} load={time.time()-t0:.0f}s",
          flush=True)

    styles = bc.compute_style_factors(panel, bench, return_descriptors=True)
    print(f"computed styles in {time.time()-t0:.0f}s, rows={len(styles)}")

    # --- descriptor coverage (fraction of rows with a finite z-score) ---
    zcols = [d + "_z" for d in bc.ALL_DESCRIPTORS]
    cov = (styles[zcols].notna().mean().rename(lambda c: c[:-2])
           .sort_values().to_frame("coverage"))
    cov["consensus"] = cov.index.isin(config.CONSENSUS_DESCRIPTORS)
    print("\ndescriptor coverage (finite z fraction):")
    print(cov.round(4).to_string())

    # --- style coverage ---
    print("\nstyle coverage:")
    print(styles[bc.STYLES].notna().mean().round(4).to_string())

    # --- CNE5 correlation structure (per-date mean pairwise corr) ---
    corr = styles.groupby("trade_date").apply(lambda g: g[bc.STYLES].corr())
    mean_corr = corr.groupby(level=1).mean().reindex(bc.STYLES)[bc.STYLES]
    print("\nmean cross-sectional style correlation:")
    print(mean_corr.round(2).to_string())
    print(f"\nsanity: Vol–Liq={mean_corr.loc['Volatility','Liquidity']:+.2f} (expect +), "
          f"Size–Liq={mean_corr.loc['Size','Liquidity']:+.2f} (expect −)")

    if not args.smoke:
        out = os.path.join(config.CACHE_DIR, f"barra_cne5s_factors_{args.universe}.parquet")
        styles[["trade_date", "ts_code"] + bc.STYLES].to_parquet(out, index=False)
        print(f"\ncached -> {out}")


if __name__ == "__main__":
    main()
