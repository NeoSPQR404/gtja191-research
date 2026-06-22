"""Unified panel engine for the 191 alphas.

A single (date x stock) vectorised engine that evaluates the reference Alpha_Set
formulas with either RANK convention, selected by ``PanelFactorCalculator.RANK_MODE``:

  * "cross_section" — RANK ranks across stocks per date (the report's intent);
  * "time_series"   — RANK ranks each stock over its own full history
                      (the reference Alpha_Set convention: ``Series.rank()``).

Every other operator is a time-series operator applied per column (per stock),
numerically matching the per-stock Factor_Calculator. Running both conventions
through THIS one engine makes the TS-vs-CS comparison apples-to-apples (only the
RANK axis differs). Rows = dates (RangeIndex), columns = stocks (RangeIndex).
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd


def _df(A):
    if isinstance(A, pd.DataFrame):
        return A
    if isinstance(A, pd.Series):
        return A.to_frame()
    return pd.DataFrame(A)


class PanelFactorCalculator:
    RANK_MODE = "cross_section"          # or "time_series"

    # ------------------------------------------------------------ RANK
    @staticmethod
    def RANK(A):
        axis = 1 if PanelFactorCalculator.RANK_MODE == "cross_section" else 0
        return _df(A).rank(axis=axis)

    @staticmethod
    def CS_RANK(A):
        return _df(A).rank(axis=1)

    # ------------------------------------------------------------ elementwise
    @staticmethod
    def MAX(A, B):
        return _df(np.maximum(_df(A), _df(B)))

    @staticmethod
    def MIN(A, B):
        return _df(np.minimum(_df(A), _df(B)))

    @staticmethod
    def ABS(A):
        return np.abs(_df(A))

    @staticmethod
    def LOG(A):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(_df(A))

    @staticmethod
    def SIGN(A):
        return np.sign(_df(A))

    # ------------------------------------------------------------ time-series (per column)
    @staticmethod
    def STD(A, n):
        return _df(A).rolling(n, min_periods=max(1, n // 2)).std().fillna(0)

    @staticmethod
    def SUM(A, n):
        return _df(A).rolling(n, min_periods=1).sum().fillna(0)

    @staticmethod
    def SUMAC(A, n):
        return _df(A).rolling(n, min_periods=1).sum().fillna(0)

    @staticmethod
    def MEAN(A, n):
        return _df(A).rolling(n, min_periods=1).mean().fillna(0)

    @staticmethod
    def DELTA(A, n=1):
        A = _df(A)
        return A - A.shift(n)

    @staticmethod
    def DELAY(A, n):
        return _df(A).shift(n).fillna(0)

    @staticmethod
    def TSMIN(A, n):
        return _df(A).rolling(n, min_periods=1).min().fillna(0)

    @staticmethod
    def TSMAX(A, n):
        return _df(A).rolling(n, min_periods=1).max().fillna(0)

    @staticmethod
    def PROD(A, n):
        return _df(A).rolling(n, min_periods=1).apply(np.prod, raw=True).fillna(0)

    @staticmethod
    def COUNT(condition, n):
        return _df(condition).astype(float).rolling(n, min_periods=1).sum().fillna(0)

    @staticmethod
    def SUMIF(A, n, condition):
        return (_df(A) * _df(condition).astype(float)).rolling(n, min_periods=1).sum().fillna(0)

    @staticmethod
    def CORR(A, B, n):
        A, B = _df(A), _df(B)
        mp = max(1, n // 2)
        ma = A.rolling(n, min_periods=mp).mean()
        mb = B.rolling(n, min_periods=mp).mean()
        mab = (A * B).rolling(n, min_periods=mp).mean()
        va = (A * A).rolling(n, min_periods=mp).mean() - ma * ma
        vb = (B * B).rolling(n, min_periods=mp).mean() - mb * mb
        res = (mab - ma * mb) / np.sqrt(va * vb).replace(0, np.nan)
        return res.fillna(0)

    @staticmethod
    def COVIANCE(A, B, n):
        A, B = _df(A), _df(B)
        mp = max(1, n // 2)
        cnt = A.rolling(n, min_periods=mp).count()
        ma = A.rolling(n, min_periods=mp).mean()
        mb = B.rolling(n, min_periods=mp).mean()
        mab = (A * B).rolling(n, min_periods=mp).mean()
        cov = (mab - ma * mb) * cnt / (cnt - 1)
        return cov.fillna(0)

    @staticmethod
    def REGBETA(A, B, n):
        A, B = _df(A), _df(B)
        mp = max(1, n // 2)
        ma = A.rolling(n, min_periods=mp).mean()
        mb = B.rolling(n, min_periods=mp).mean()
        mab = (A * B).rolling(n, min_periods=mp).mean()
        vb = (B * B).rolling(n, min_periods=mp).mean() - mb * mb
        beta = (mab - ma * mb) / vb.replace(0, np.nan)
        return beta.fillna(0)

    @staticmethod
    def REGRESI(A, B, n):
        A, B = _df(A), _df(B)
        beta = PanelFactorCalculator.REGBETA(A, B, n)
        alpha = PanelFactorCalculator.MEAN(A, n) - beta * PanelFactorCalculator.MEAN(B, n)
        return (A - (alpha + beta * B)).fillna(0)

    @staticmethod
    def _warmup(A, n):
        # True where the trailing n-window has fewer than n valid obs (matches the
        # reference min_periods=n -> NaN -> fillna(0) behaviour for warmup rows).
        return A.notna().rolling(n, min_periods=1).sum() < n

    @staticmethod
    def TSRANK(A, n):
        A = _df(A)
        less = eq = None
        for k in range(n):
            s = A.shift(k)
            lt = (s < A).astype(float)
            e = (s == A).astype(float)
            less = lt if less is None else less + lt
            eq = e if eq is None else eq + e
        res = (less + (eq + 1) / 2.0).mask(PanelFactorCalculator._warmup(A, n))
        return res.fillna(0)

    @staticmethod
    def DECAYLINEAR(A, d):
        A = _df(A)
        S = d * (d + 1) / 2.0
        out = None
        for k in range(d):
            term = (d - k) / S * A.shift(k)
            out = term if out is None else out + term
        return out.mask(PanelFactorCalculator._warmup(A, d)).fillna(0)

    @staticmethod
    def WMA(A, n):
        A = _df(A)
        w = np.array([0.9 ** k for k in range(n)])
        w = w / w.sum()
        out = None
        for k in range(n):
            term = w[k] * A.shift(k)
            out = term if out is None else out + term
        return out.mask(PanelFactorCalculator._warmup(A, n)).fillna(0)

    @staticmethod
    def SMA(A, n, m):
        # recursive EMA; resets to the raw value whenever the running state is NaN
        # (matches the reference per-stock SMA, incl. its post-NaN reset).
        A = _df(A)
        vals = A.to_numpy(dtype=float)
        out = np.empty_like(vals)
        y_prev = np.full(vals.shape[1], np.nan)
        for i in range(vals.shape[0]):
            rec = (vals[i] * m + y_prev * (n - m)) / n
            out[i] = np.where(np.isnan(y_prev), vals[i], rec)
            y_prev = out[i]
        return pd.DataFrame(out, index=A.index, columns=A.columns).fillna(0)

    @staticmethod
    def HIGHDAY(A, n):
        return _df(A).rolling(n, min_periods=n).apply(lambda x: n - 1 - np.argmax(x), raw=True).fillna(0)

    @staticmethod
    def LOWDAY(A, n):
        return _df(A).rolling(n, min_periods=n).apply(lambda x: n - 1 - np.argmin(x), raw=True).fillna(0)

    @staticmethod
    def FILTER(A, condition):
        return _df(A).where(_df(condition).astype(bool), np.nan)

    @staticmethod
    def SEQUENCE(n):
        return np.arange(1, n + 1)


# --------------------------------------------------------------------- runner
def _adjust(panel):
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    adj = p["adj_factor"]
    p["open_hfq"] = p["open"] * adj
    p["high_hfq"] = p["high"] * adj
    p["low_hfq"] = p["low"] * adj
    with np.errstate(divide="ignore", invalid="ignore"):
        p["vwap_hfq"] = np.where(p["vol"] > 0, p["amount"] * 10.0 / p["vol"], np.nan) * adj
    p["ret_"] = p.groupby("ts_code")["close_hfq"].pct_change()
    return p


def build_panels(panel, bench, ff):
    p = _adjust(panel)
    dates = np.sort(p["trade_date"].unique())
    codes = sorted(p["ts_code"].unique())

    def wide(col):
        w = p.pivot(index="trade_date", columns="ts_code", values=col).reindex(index=dates, columns=codes)
        w.index = range(len(dates))
        w.columns = range(len(codes))
        return w

    T, N = len(dates), len(codes)

    def broadcast(series_by_date):
        s = pd.Series(series_by_date).reindex(dates).to_numpy(dtype=float)
        return pd.DataFrame(np.repeat(s[:, None], N, axis=1))

    b = bench.set_index("trade_date")
    ffi = ff.set_index("trade_date")
    data = {
        "close": wide("close_hfq"), "open": wide("open_hfq"),
        "high": wide("high_hfq"), "low": wide("low_hfq"),
        "volume": wide("vol"), "amount": wide("amount"), "vwap": wide("vwap_hfq"),
        "ret": wide("ret_").fillna(0),
        "benchmarkindexclose": broadcast(b["bench_close"]),
        "benchmarkindexopen": broadcast(b["bench_open"]),
        "mkt": broadcast(ffi["mkt"]).fillna(0),
        "smb": broadcast(ffi["smb"]).fillna(0),
        "hml": broadcast(ffi["hml"]).fillna(0),
    }
    return data, dates, codes, T, N


def run_panel(panel, bench, ff, rank_mode="cross_section", log_every=25, skip_ids=()):
    """Evaluate all alphas with the chosen RANK mode; return long
    [trade_date, ts_code, alpha*]. ``skip_ids`` are alpha numbers to omit
    (e.g. 30 needs FF3, 143 is self-referential)."""
    import Alpha_Set as _am
    from Alpha_Set import Alpha_Set
    _am.FC = PanelFactorCalculator
    PanelFactorCalculator.RANK_MODE = rank_mode

    skip = set(skip_ids)
    data, dates, codes, T, N = build_panels(panel, bench, ff)
    ids = [i for i in range(1, 192) if hasattr(Alpha_Set, f"alpha{i}") and i not in skip]
    K = len(ids)
    # fill a preallocated result array, freeing each (T x N) panel immediately,
    # so we never hold all 191 panels and the long table at once (memory-lean).
    res = np.empty((T * N, K), dtype=np.float32)
    t0 = time.time()
    for k, i in enumerate(ids):
        fn = getattr(Alpha_Set, f"alpha{i}")
        try:
            v = np.asarray(fn(data), dtype=float)
            if v.shape != (T, N):
                v = np.full((T, N), np.nan)
        except Exception:
            v = np.full((T, N), np.nan)
        res[:, k] = v.reshape(-1)
        del v
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k+1}/{K} alphas  {time.time()-t0:.0f}s", flush=True)

    long = pd.DataFrame(res, columns=[f"alpha{i}" for i in ids], copy=False)
    long.insert(0, "ts_code", np.tile(np.array(codes), T))
    long.insert(0, "trade_date", np.repeat(dates, N))
    return long


def run_panel_to_memmap(panel, bench, ff, out_dir, rank_mode="cross_section",
                        skip_ids=(), log_every=25):
    """Memory-lean full-A variant: evaluate the alphas and stream each (T×N) result
    into a disk-backed float32 memmap of shape (T*N, K) — column-major, dense over
    the union (date × stock) grid — instead of materialising one giant long table.
    Peak RAM ≈ the wide input panels (~1–2 GB) + one alpha matrix, regardless of K.

    Writes ``out_dir/signals.f32`` (the memmap) and ``out_dir/meta.npz``
    (dates, codes, alpha_ids, and a row-aligned ``valid`` mask = stock traded that
    date). The single-factor test slices rows [j*N:(j+1)*N] per date j.
    """
    import Alpha_Set as _am
    from Alpha_Set import Alpha_Set
    _am.FC = PanelFactorCalculator
    PanelFactorCalculator.RANK_MODE = rank_mode

    skip = set(skip_ids)
    data, dates, codes, T, N = build_panels(panel, bench, ff)
    # validity = the stock actually traded that date (close_hfq finite before fill)
    valid = np.isfinite(np.asarray(data["close"], dtype=float)).reshape(-1)
    ids = [i for i in range(1, 192) if hasattr(Alpha_Set, f"alpha{i}") and i not in skip]
    K = len(ids)

    os.makedirs(out_dir, exist_ok=True)
    mm = np.memmap(os.path.join(out_dir, "signals.f32"), dtype=np.float32,
                   mode="w+", shape=(T * N, K))
    t0 = time.time()
    for k, i in enumerate(ids):
        fn = getattr(Alpha_Set, f"alpha{i}")
        try:
            v = np.asarray(fn(data), dtype=float)
            if v.shape != (T, N):
                v = np.full((T, N), np.nan)
        except Exception:
            v = np.full((T, N), np.nan)
        mm[:, k] = v.reshape(-1).astype(np.float32)
        del v
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k+1}/{K} alphas  {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    del mm
    np.savez(os.path.join(out_dir, "meta.npz"),
             dates=np.asarray(dates), codes=np.asarray(codes),
             alpha_ids=np.asarray(ids), valid=valid,
             T=np.int64(T), N=np.int64(N), K=np.int64(K))
    return out_dir
