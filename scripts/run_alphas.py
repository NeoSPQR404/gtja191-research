"""§3.3 — compute the 189 price-volume alphas on the full-A (ex-ST) universe and
cache the long signal table.

Usage:
  python scripts/run_alphas.py [--rank cross_section|time_series]
                               [--universe full_a|topN] [--smoke N]

--smoke N runs on only N stocks (quick wiring/finite-fraction check) and does not
overwrite the production cache.
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
import panel_engine as pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default=config.RANK_CONVENTION,
                    choices=["cross_section", "time_series"])
    ap.add_argument("--universe", default=config.UNIVERSE_MODE,
                    choices=["full_a", "topN"])
    ap.add_argument("--smoke", type=int, default=0,
                    help="run on only N stocks (quick check, no cache write)")
    ap.add_argument("--memmap", action="store_true",
                    help="stream signals to a disk memmap (required for full-A on 16GB)")
    args = ap.parse_args()
    config.UNIVERSE_MODE = args.universe

    t0 = time.time()
    basic = dl.load_stock_basic()
    panel = dl.load_panel()
    uni = dl.select_universe(panel, basic)
    if args.smoke:
        uni = uni[: args.smoke]
    panel = panel[panel["ts_code"].isin(uni)].reset_index(drop=True)
    bench = dl.load_benchmark()
    ff = dl.zero_ff(panel["trade_date"])
    T = panel["trade_date"].nunique()
    N = panel["ts_code"].nunique()
    print(f"universe={args.universe} stocks={N} dates={T} rows={len(panel)} "
          f"load={time.time()-t0:.0f}s", flush=True)

    if args.memmap and not args.smoke:
        out_dir = os.path.join(config.CACHE_DIR, f"signals_{args.universe}_{args.rank}")
        pe.run_panel_to_memmap(panel, bench, ff, out_dir, rank_mode=args.rank,
                               skip_ids=config.SKIP_ALPHAS, log_every=25)
        sz = os.path.getsize(os.path.join(out_dir, "signals.f32")) / 1e9
        print(f"\nmemmap cached -> {out_dir}  (signals.f32 {sz:.2f} GB, {time.time()-t0:.0f}s)")
        return

    long = pe.run_panel(panel, bench, ff, rank_mode=args.rank, log_every=25,
                        skip_ids=config.SKIP_ALPHAS)
    acols = [c for c in long.columns if c.startswith("alpha")]
    print(f"\ncomputed {len(acols)} alphas, {len(long)} rows, {time.time()-t0:.0f}s")

    # finite (non-zero-filled) fraction per alpha — a health check
    fin = {a: np.isfinite(long[a]).mean() for a in acols}
    nz = {a: (long[a] != 0).mean() for a in acols}
    health = pd.DataFrame({"finite_frac": fin, "nonzero_frac": nz})
    print("\nnon-zero fraction summary:")
    print(health["nonzero_frac"].describe().round(4).to_string())
    weak = health[health["nonzero_frac"] < 0.2].sort_values("nonzero_frac")
    print(f"\nalphas with <20% non-zero ({len(weak)}):")
    print(weak.round(4).to_string() if len(weak) else "  (none)")

    if not args.smoke:
        out = os.path.join(config.CACHE_DIR, f"alpha_signals_{args.rank}.parquet")
        long.to_parquet(out, index=False)
        print(f"\ncached -> {out}  ({os.path.getsize(out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
