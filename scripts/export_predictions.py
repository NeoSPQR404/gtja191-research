"""Export the combined-model per-(date,stock) alpha forecast Ê(α) for §3.6.
Streams the signal memmap; point-in-time (trailing factor returns). Caches to
report/cache/alpha_pred_{universe}.parquet.
"""
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
import config           # noqa: E402
import data_loader as dl   # noqa: E402
import model_ic as mic     # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="full_a", choices=["full_a", "topN"])
    ap.add_argument("--rank", default=config.RANK_CONVENTION)
    args = ap.parse_args()
    config.UNIVERSE_MODE = args.universe
    t0 = time.time()

    mmap_dir = os.path.join(config.CACHE_DIR, f"signals_{args.universe}_{args.rank}")
    barra_cache = os.path.join(config.CACHE_DIR, f"barra_cne5s_factors_{args.universe}.parquet")
    basic = dl.load_stock_basic()
    panel = dl.load_panel(columns=["ts_code", "trade_date", "close_hfq", "amount"])
    uni = dl.select_universe(panel, basic)
    panel = panel[panel["ts_code"].isin(uni)]
    close_wide = panel.pivot(index="trade_date", columns="ts_code", values="close_hfq")
    industry_map = dl.load_industry()
    style_long = pd.read_parquet(barra_cache)
    print(f"loaded {time.time()-t0:.0f}s; streaming predictions ...", flush=True)

    pred = mic.export_predictions_memmap(mmap_dir, style_long, industry_map, close_wide)
    out = os.path.join(config.CACHE_DIR, f"alpha_pred_{args.universe}.parquet")
    pred.to_parquet(out, index=False)
    print(f"cached -> {out}  rows={len(pred)}  {time.time()-t0:.0f}s")
    print(pred.head().to_string())


if __name__ == "__main__":
    main()
