"""§3.4.1 — single-factor significance & effective-period test (full-A, streamed).

Prerequisites (run as separate processes so each frees RAM — see README):
  python scripts/run_alphas.py   --universe full_a --memmap   # -> signal memmap
  python scripts/compute_barra.py --universe full_a            # -> style cache

This script then streams the test off the memmap and writes:
  report/tables/table7_mean_factor_return.csv   (mean annualised FR by horizon)
  report/tables/table8_mean_IR.csv              (mean IR by horizon)
  report/tables/fr_ir_by_alpha.csv              (per-alpha FR/IR/obs at each d)
  report/tables/prune.md                        (correlation pruning result)
  report/cache/factor_returns_d{d}.parquet      (per-date factor returns)
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
import config
import data_loader as dl
import single_factor_test as sft


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
    for p in (os.path.join(mmap_dir, "meta.npz"), barra_cache):
        if not os.path.exists(p):
            sys.exit(f"missing prerequisite: {p}\nRun run_alphas.py --memmap and "
                     f"compute_barra.py for --universe {args.universe} first.")

    # universe + close_wide (forward returns) + SW industry map
    basic = dl.load_stock_basic()
    panel = dl.load_panel(columns=["ts_code", "trade_date", "close_hfq", "amount"])
    uni = dl.select_universe(panel, basic)
    panel = panel[panel["ts_code"].isin(uni)]
    close_wide = panel.pivot(index="trade_date", columns="ts_code", values="close_hfq")
    industry_map = dl.load_industry()
    style_long = pd.read_parquet(barra_cache)
    print(f"loaded: universe={len(uni)} dates={close_wide.shape[0]} "
          f"styles={len(style_long)} {time.time()-t0:.0f}s", flush=True)

    factor_returns, fr_table, ir_table, counts = sft.run_test_memmap(
        mmap_dir, style_long, industry_map, close_wide)
    print(f"test done {time.time()-t0:.0f}s", flush=True)

    # ---- Table 7/8: cross-factor mean of annualised FR / IR by horizon ----
    t7 = fr_table.mean(axis=0).to_frame("mean_annual_factor_return").T
    t8 = ir_table.mean(axis=0).to_frame("mean_IR").T
    t7.to_csv(os.path.join(config.TABLE_DIR, "table7_mean_factor_return.csv"))
    t8.to_csv(os.path.join(config.TABLE_DIR, "table8_mean_IR.csv"))
    per_alpha = pd.concat({"FR": fr_table, "IR": ir_table, "obs": counts}, axis=1)
    per_alpha.to_csv(os.path.join(config.TABLE_DIR, "fr_ir_by_alpha.csv"))

    print("\n=== Table 7: mean annualised factor return by horizon ===")
    print((t7 * 100).round(2).to_string())
    print("\n=== Table 8: mean IR by horizon ===")
    print(t8.round(2).to_string())
    nvalid = {d: int(fr_table[d].notna().sum()) for d in config.HORIZONS}
    print("\nvalid factors per horizon:", nvalid)

    # cache per-date factor returns (for §3.4.2/3 model IC)
    for d, fr in factor_returns.items():
        fr.to_parquet(os.path.join(config.CACHE_DIR, f"factor_returns_d{d}.parquet"))

    # ---- correlation pruning (report's de-dup step) ----
    kept, dropped, corr_before, corr_after = sft.prune_correlated(factor_returns, ir_table)
    cvals = corr_before.where(~np.eye(len(corr_before), dtype=bool)).stack()
    with open(os.path.join(config.TABLE_DIR, "prune.md"), "w") as fh:
        fh.write(f"# Factor-return correlation & pruning (d={config.PRUNE_HORIZON}, "
                 f"threshold={config.PRUNE_CORR_THRESHOLD})\n\n")
        fh.write(f"- factors entering pruning: {corr_before.shape[0]}\n")
        fh.write(f"- kept: {len(kept)}   dropped: {len(dropped)}\n")
        fh.write(f"- mean |corr| (off-diag): {cvals.abs().mean():.4f}; "
                 f"within ±0.1: {(cvals.abs() <= 0.1).mean()*100:.1f}%\n")
    print(f"\npruning: {corr_before.shape[0]} -> kept {len(kept)}, dropped {len(dropped)}; "
          f"mean|corr|={cvals.abs().mean():.4f}, within±0.1={(cvals.abs()<=0.1).mean()*100:.1f}%")
    print(f"\nall done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
