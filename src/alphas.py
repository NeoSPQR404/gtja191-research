"""Run the 191-alpha set (reference Alpha_Set, time-series RANK convention) over
the reduced universe and assemble a long signal table cached to parquet.

Each alpha is evaluated per stock on its own date-ordered series; failures or
length mismatches (e.g. alpha30, whose REGRESI gets a 2-D regressor) are recorded
as NaN columns rather than aborting the run.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from Alpha_Set import Alpha_Set

ALPHA_IDS = [i for i in range(1, 192) if hasattr(Alpha_Set, f"alpha{i}")]
ALPHA_FUNCS = {i: getattr(Alpha_Set, f"alpha{i}") for i in ALPHA_IDS}


def run_for_stock(data, n):
    """Return {alpha_name: float32 array of length n} for one stock."""
    out = {}
    for i, fn in ALPHA_FUNCS.items():
        name = f"alpha{i}"
        try:
            v = np.asarray(fn(data), dtype=float).ravel()
            if v.shape[0] != n:
                v = np.full(n, np.nan)
        except Exception:
            v = np.full(n, np.nan)
        out[name] = v.astype(np.float32)
    return out


def _stock_frame(code, g, bench, ff):
    from data_loader import build_stock_inputs
    data, dates = build_stock_inputs(g, bench, ff)
    sig = run_for_stock(data, len(dates))
    df = pd.DataFrame(sig)
    df.insert(0, "ts_code", code)
    df.insert(0, "trade_date", dates)
    return df


# --- multiprocessing plumbing (spawn-safe: bench/ff passed via initializer) ---
_BENCH = None
_FF = None


def _init_worker(bench, ff):
    global _BENCH, _FF
    _BENCH, _FF = bench, ff


def _worker(args):
    code, g = args
    return _stock_frame(code, g, _BENCH, _FF)


def run_universe(panel, bench, ff, codes, log_every=50, n_jobs=1):
    """Compute alphas for every stock in ``codes``; return a long DataFrame
    [trade_date, ts_code, alpha1..alpha191]. Set ``n_jobs>1`` to parallelise
    across stocks."""
    groups = dict(tuple(panel.groupby("ts_code")))
    tasks = [(c, groups[c]) for c in codes if c in groups and len(groups[c])]
    frames = []
    t0 = time.time()

    if n_jobs and n_jobs > 1:
        import multiprocessing as mp
        with mp.Pool(n_jobs, initializer=_init_worker, initargs=(bench, ff)) as pool:
            for k, df in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
                frames.append(df)
                if log_every and (k + 1) % log_every == 0:
                    el = time.time() - t0
                    print(f"  {k+1}/{len(tasks)} stocks  {el:.0f}s  ({el/(k+1):.2f}s/stock)", flush=True)
    else:
        for k, (code, g) in enumerate(tasks):
            frames.append(_stock_frame(code, g, bench, ff))
            if log_every and (k + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(tasks)} stocks  {el:.0f}s  ({el/(k+1):.2f}s/stock)", flush=True)

    out = pd.concat(frames, ignore_index=True)
    return out
