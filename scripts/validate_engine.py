"""Validate the vectorized panel engine against the per-stock reference.

For gap-free stocks (present on every union trade date), the panel engine in
time_series-RANK mode must reproduce the per-stock Factor_Calculator alpha values
(same windows, same RANK over the full series). Reports per-alpha agreement on a
sample of such stocks. Run: python scripts/validate_engine.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import config
import data_loader as dl

basic = dl.load_stock_basic()
# Use the topN liquid universe for a tractable, mostly gap-free sample.
panel = dl.load_panel()
prev_mode = config.UNIVERSE_MODE
config.UNIVERSE_MODE = "topN"
uni = dl.select_universe(panel, basic)
config.UNIVERSE_MODE = prev_mode
panel = panel[panel["ts_code"].isin(uni)].reset_index(drop=True)
bench = dl.load_benchmark()
ff = dl.zero_ff(panel["trade_date"])

# stocks present on every union trade date (gap-free)
all_dates = np.sort(panel["trade_date"].unique())
cnt = panel.groupby("ts_code")["trade_date"].nunique()
full = cnt[cnt == len(all_dates)].index.tolist()[:20]
print(f"union dates={len(all_dates)}  gap-free stocks sampled={len(full)}")
sub = panel[panel["ts_code"].isin(full)].reset_index(drop=True)

# --- per-stock reference (time-series RANK, FC = per-stock Factor_Calculator) ---
import importlib
import Alpha_Set as AM
import Factor_Calculator as RefFC
importlib.reload(AM)
AM.FC = RefFC.Factor_Calculator
import alphas as per_stock
ref_long = per_stock.run_universe(sub, bench, ff, full, log_every=0, n_jobs=1)

# --- panel engine (time-series mode) ---
import panel_engine as pe
pan_long = pe.run_panel(sub, bench, ff, rank_mode="time_series", log_every=0,
                        skip_ids=config.SKIP_ALPHAS)

acols = [c for c in pan_long.columns if c.startswith("alpha")]
ref = ref_long.set_index(["trade_date", "ts_code"])[acols].sort_index()
pan = pan_long.set_index(["trade_date", "ts_code"])[acols].reindex(ref.index)

rows = []
for a in acols:
    x, y = ref[a].to_numpy(), pan[a].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        continue
    maxd = np.max(np.abs(x[m] - y[m]))
    scale = np.median(np.abs(x[m])) + 1e-9
    corr = np.corrcoef(x[m], y[m])[0, 1] if np.std(x[m]) > 0 and np.std(y[m]) > 0 else np.nan
    rows.append((a, corr, maxd, maxd / scale))

r = pd.DataFrame(rows, columns=["alpha", "corr", "max_abs_diff", "rel_diff"]).set_index("alpha")
ok = (r["corr"] > 0.999) | (r["max_abs_diff"] < 1e-4)
print(f"\nalphas compared: {len(r)}")
print(f"matching (corr>0.999 or maxdiff<1e-4): {ok.sum()}/{len(r)}")
print("\nworst 10 by correlation:")
print(r.sort_values("corr").head(10).round(6).to_string())
