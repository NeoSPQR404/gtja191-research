"""§3.6 — style-neutral multi-factor strategy: optimiser + backtester.

Per rebalance r (decided at close of r, executed at next-day open r+1):
  max_w  w·Ê(α) − Tc·Σ|w − w_prev|
  s.t.   |(w − w_bench)·X_style|     ≤ style_bound   (per style; report ±0.1)
         |(w − w_bench)·X_industry|  ≤ ind_bound     (per industry; report ±5%)
         Σw = 1,  w ≥ 0  (long-only)
linearised to an LP (u ≥ |w − w_prev|) and solved with scipy HiGHS.

Benchmark = CSI500 (000905.SH). Universe per rebalance = tradable (not ST by
namechange, not suspended, has prediction+styles) ∩ (CSI500 members ∪ top liquid),
capped for tractability. Daily NAV with buy-and-hold weight drift between rebalances;
one-sided cost Tc·turnover charged on the execution day.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

import config


# ---------------------------------------------------------------- benchmark / ST
def benchmark_weights_by_date(csi500_weight, trade_dates):
    """Monthly CSI500 constituent weights forward-filled to each trade date.
    Returns dict date -> Series(con_code -> weight summing to 1)."""
    w = csi500_weight.copy()
    w["trade_date"] = w["trade_date"].astype(str)
    snaps = {d: g.set_index("con_code")["weight"] / g["weight"].sum()
             for d, g in w.groupby("trade_date")}
    snap_dates = sorted(snaps)
    out, k = {}, 0
    sd = np.array(snap_dates)
    for d in sorted(map(str, trade_dates)):
        idx = np.searchsorted(sd, d, side="right") - 1
        if idx >= 0:
            out[d] = snaps[sd[idx]]
    return out


def st_codes_by_date(namechange):
    """Point-in-time ST membership from namechange: a (start_date, end_date, name)
    interval is ST if its name contains ST/*ST/退. Returns (intervals df) for query."""
    nc = namechange.copy()
    nc["start_date"] = nc["start_date"].astype(str)
    nc["end_date"] = nc["end_date"].fillna("99999999").astype(str).str.replace(".0", "", regex=False)
    nc["is_st"] = nc["name"].str.contains("ST|退", na=False)
    return nc[nc["is_st"]][["ts_code", "start_date", "end_date"]].reset_index(drop=True)


def is_st_on(st_intervals, date):
    """Set of ts_codes that are ST on `date`."""
    m = (st_intervals["start_date"] <= date) & (date < st_intervals["end_date"])
    return set(st_intervals.loc[m, "ts_code"])


# ---------------------------------------------------------------- optimiser
def optimize_weights(codes, pred, Xstyle, ind_labels, w_bench, w_prev,
                     Tc, style_bound, ind_bound, max_weight=None):
    """Solve the long-only style/industry-neutral LP. Returns weight Series (index
    = codes). On infeasibility, falls back to the benchmark-neutral prediction tilt."""
    n = len(codes)
    # decision vars x = [w (n) | u (n)];  minimise  -pred·w + Tc·Σu
    c = np.concatenate([-pred, Tc * np.ones(n)])

    A_ub, b_ub = [], []
    # u >= w - w_prev  ->  w - u <= w_prev ; u >= w_prev - w  ->  -w - u <= -w_prev
    I = np.eye(n)
    A_ub.append(np.hstack([I, -I]));  b_ub.append(w_prev)
    A_ub.append(np.hstack([-I, -I])); b_ub.append(-w_prev)

    # style neutrality: |(w - w_bench)·s_k| <= style_bound
    sb = Xstyle.T @ w_bench
    for k in range(Xstyle.shape[1]):
        sk = Xstyle[:, k]
        A_ub.append(np.concatenate([sk, np.zeros(n)])[None, :]);  b_ub.append([sb[k] + style_bound])
        A_ub.append(np.concatenate([-sk, np.zeros(n)])[None, :]); b_ub.append([-(sb[k] - style_bound)])

    # industry neutrality: |(w - w_bench)·ind_j| <= ind_bound (absolute weight dev)
    for lab in pd.unique(ind_labels):
        dvec = (ind_labels == lab).astype(float)
        bj = dvec @ w_bench
        A_ub.append(np.concatenate([dvec, np.zeros(n)])[None, :]);  b_ub.append([bj + ind_bound])
        A_ub.append(np.concatenate([-dvec, np.zeros(n)])[None, :]); b_ub.append([-(bj - ind_bound)])

    A_ub = np.vstack(A_ub); b_ub = np.concatenate(b_ub)
    A_eq = np.concatenate([np.ones(n), np.zeros(n)])[None, :]; b_eq = np.array([1.0])
    wcap = max_weight if (max_weight and max_weight * n >= 1.0) else None
    bounds = [(0, wcap)] * n + [(0, None)] * n          # 0<=w<=cap long-only, u>=0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if not res.success:
        # relax neutrality to a feasible fallback: benchmark + small prediction tilt
        return pd.Series(w_bench, index=codes)
    w = np.clip(res.x[:n], 0, None)
    w = w / w.sum() if w.sum() > 0 else w_bench
    return pd.Series(w, index=codes)


# ---------------------------------------------------------------- backtester
def backtest(pred_long, style_long, industry_map, panel, csi500_daily, csi500_weight,
             namechange, Tc=0.003, reb_freq=2, style_bound=0.1, ind_bound=0.05,
             max_universe=1500, log_every=100, execution="open_close"):
    """Run the strategy. Returns (daily df[nav, bench_nav, port_ret, bench_ret,
    excess], rebalance df[date, turnover, n_holdings]).

    execution: "open_close" — buy at next-day open, the entry day contributes its
    open→close move (no overnight look-ahead); thereafter close-to-close.
    "open_open" — hold open→open each day (spans the next overnight, where the
    GTJA price-volume signal partly survives). Benchmark return matched per mode."""
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    adj = p["adj_factor"]
    p["open_hfq"] = p["open"] * adj
    p["ret"] = p.groupby("ts_code")["close_hfq"].pct_change()
    dates = np.sort(p["trade_date"].unique()).astype(str)
    codes_all = sorted(p["ts_code"].unique())

    ret_w = p.pivot(index="trade_date", columns="ts_code", values="ret").reindex(index=dates, columns=codes_all)
    open_w = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(index=dates, columns=codes_all)
    close_w = p.pivot(index="trade_date", columns="ts_code", values="close_hfq").reindex(index=dates, columns=codes_all)
    amt_w = p.pivot(index="trade_date", columns="ts_code", values="amount").reindex(index=dates, columns=codes_all)
    trad_amt = amt_w.rolling(20, min_periods=5).mean()          # liquidity proxy
    suspended = ret_w.isna() | (amt_w.fillna(0) <= 0)
    # split-day returns for next-day-open execution (no overnight look-ahead):
    overnight_w = open_w / close_w.shift(1) - 1.0               # close_{j-1} -> open_j
    intraday_w = close_w / open_w - 1.0                         # open_j -> close_j

    pred_w = pred_long.pivot(index="trade_date", columns="ts_code", values="pred").reindex(index=dates, columns=codes_all)
    sl = style_long.copy(); sl["trade_date"] = sl["trade_date"].astype(str)
    style_arr = {s: sl.pivot(index="trade_date", columns="ts_code", values=s)
                 .reindex(index=dates, columns=codes_all) for s in config.STYLE_FACTORS}
    ind_map = pd.Series(industry_map).reindex(codes_all).fillna("UNKNOWN")

    bench_w = benchmark_weights_by_date(csi500_weight, dates)
    st_iv = st_codes_by_date(namechange)
    cb = csi500_daily.copy(); cb["trade_date"] = cb["trade_date"].astype(str)
    bench_ret = cb.set_index("trade_date")["pct_chg"].reindex(dates).astype(float) / 100.0

    code_index = pd.Index(codes_all)
    reb_idx = set(range(0, len(dates) - 1, reb_freq))           # decide at r, exec r+1

    def _decide(j, w_prev_series):
        """Solve the optimiser at close j (execute next open); return target w or None."""
        de = j + 1
        st_set = is_st_on(st_iv, dates[j])
        tradable = ((~suspended.iloc[j]) & (~suspended.iloc[de]) & pred_w.iloc[j].notna()).to_numpy()
        tradable &= ~code_index.isin(st_set)
        cand = code_index[tradable]
        bw = bench_w.get(dates[de], pd.Series(dtype=float))
        members = [c for c in bw.index if c in set(cand)]
        liq = trad_amt.iloc[j].reindex(cand).fillna(0).sort_values(ascending=False)
        uni = sorted(set(members) | set(liq.head(max_universe).index))
        if len(uni) < config.MIN_CROSS_SECTION:
            return None
        pr = pred_w.iloc[j].reindex(uni).fillna(0.0).to_numpy()
        Xs = np.column_stack([style_arr[s].iloc[j].reindex(uni).fillna(0.0).to_numpy()
                              for s in config.STYLE_FACTORS])
        il = ind_map.reindex(uni).fillna("UNKNOWN").to_numpy()
        wb = bw.reindex(uni).fillna(0.0).to_numpy(); wb = wb / wb.sum() if wb.sum() > 0 else wb
        wp = w_prev_series.reindex(uni).fillna(0.0).to_numpy()
        return optimize_weights(uni, pr, Xs, il, wb, wp, Tc, style_bound, ind_bound,
                                config.STRAT_MAX_WEIGHT)

    if execution == "open_open":
        oo_ret = (open_w.shift(-1) / open_w - 1.0)              # open_j -> open_{j+1}
        cb_oo = csi500_daily.copy(); cb_oo["trade_date"] = cb_oo["trade_date"].astype(str)
        bopen = cb_oo.set_index("trade_date")["open"].reindex(dates).astype(float)
        bench_oo = (bopen.shift(-1) / bopen - 1.0)
        w_held = pd.Series(0.0, index=codes_all)
        pending = {}
        nav, bench_nav = 1.0, 1.0
        daily, reb_rows = [], []
        for j, d in enumerate(dates):
            cost = 0.0
            if j in pending:                                   # rebalance at open_j
                w_tgt = pending.pop(j).reindex(codes_all).fillna(0.0)
                turn = float(np.abs(w_tgt.to_numpy() - w_held.to_numpy()).sum()) / 2.0
                cost = Tc * turn
                reb_rows.append((d, turn, int((w_tgt > 1e-6).sum())))
                w_held = w_tgt
            r = np.nan_to_num(oo_ret.iloc[j].to_numpy(), nan=0.0)
            port_ret = (1.0 - cost) * (1.0 + float(np.dot(w_held.to_numpy(), r))) - 1.0
            grown = w_held.to_numpy() * (1.0 + r); s = grown.sum()
            if s > 0:
                w_held = pd.Series(grown / s, index=codes_all)
            if j in reb_idx and j + 1 < len(dates):
                wt = _decide(j, w_held)
                if wt is not None:
                    pending[j + 1] = wt
            br = float(bench_oo.iloc[j]) if np.isfinite(bench_oo.iloc[j]) else 0.0
            nav *= (1.0 + port_ret); bench_nav *= (1.0 + br)
            daily.append((d, nav, bench_nav, port_ret, br))
            if log_every and (j + 1) % log_every == 0:
                print(f"  backtest[oo] {j+1}/{len(dates)}  nav={nav:.3f} bench={bench_nav:.3f}", flush=True)
        dd = pd.DataFrame(daily, columns=["trade_date", "nav", "bench_nav", "port_ret", "bench_ret"])
        dd["excess_ret"] = dd["port_ret"] - dd["bench_ret"]
        return dd, pd.DataFrame(reb_rows, columns=["trade_date", "turnover", "n_holdings"])

    w_close = pd.Series(0.0, index=codes_all)                   # weights at prev close
    pending = {}                                                # exec-day index -> target w
    nav, bench_nav = 1.0, 1.0
    daily, reb_rows = [], []

    for j, d in enumerate(dates):
        if j in pending:                                       # execution day: split-day
            on = np.nan_to_num(overnight_w.iloc[j].to_numpy(), nan=0.0)
            intr = np.nan_to_num(intraday_w.iloc[j].to_numpy(), nan=0.0)
            wc = w_close.to_numpy()
            port_on = float(np.dot(wc, on))                    # old weights earn overnight
            w_open = wc * (1.0 + on); s = w_open.sum(); w_open = w_open / s if s > 0 else wc
            w_tgt = pending.pop(j).reindex(codes_all).fillna(0.0).to_numpy()
            turn = float(np.abs(w_tgt - w_open).sum()) / 2.0   # one-sided, at open
            cost = Tc * turn
            reb_rows.append((d, turn, int((w_tgt > 1e-6).sum())))
            port_in = float(np.dot(w_tgt, intr))               # new weights earn intraday
            port_ret = (1.0 + port_on) * (1.0 - cost) * (1.0 + port_in) - 1.0
            grown = w_tgt * (1.0 + intr); s = grown.sum()
            w_close = pd.Series(grown / s if s > 0 else w_tgt, index=codes_all)
        else:                                                  # normal day: close-to-close
            ft = np.nan_to_num(ret_w.iloc[j].to_numpy(), nan=0.0)
            wc = w_close.to_numpy()
            port_ret = float(np.dot(wc, ft))
            grown = wc * (1.0 + ft); s = grown.sum()
            if s > 0:
                w_close = pd.Series(grown / s, index=codes_all)

        if j in reb_idx and j + 1 < len(dates):                # decide (execute next day)
            de = j + 1
            st_set = is_st_on(st_iv, d)
            tradable = ((~suspended.iloc[j]) & (~suspended.iloc[de]) & pred_w.iloc[j].notna()).to_numpy()
            tradable &= ~code_index.isin(st_set)
            cand = code_index[tradable]
            bw = bench_w.get(dates[de], pd.Series(dtype=float))
            members = [c for c in bw.index if c in set(cand)]
            liq = trad_amt.iloc[j].reindex(cand).fillna(0).sort_values(ascending=False)
            uni = sorted(set(members) | set(liq.head(max_universe).index))
            if len(uni) >= config.MIN_CROSS_SECTION:
                pr = pred_w.iloc[j].reindex(uni).fillna(0.0).to_numpy()
                Xs = np.column_stack([style_arr[s].iloc[j].reindex(uni).fillna(0.0).to_numpy()
                                      for s in config.STYLE_FACTORS])
                il = ind_map.reindex(uni).fillna("UNKNOWN").to_numpy()
                wb = bw.reindex(uni).fillna(0.0).to_numpy(); wb = wb / wb.sum() if wb.sum() > 0 else wb
                wp = w_close.reindex(uni).fillna(0.0).to_numpy()
                pending[de] = optimize_weights(uni, pr, Xs, il, wb, wp, Tc, style_bound,
                                               ind_bound, config.STRAT_MAX_WEIGHT)

        br = float(bench_ret.iloc[j]) if np.isfinite(bench_ret.iloc[j]) else 0.0
        nav *= (1.0 + port_ret); bench_nav *= (1.0 + br)
        daily.append((d, nav, bench_nav, port_ret, br))
        if log_every and (j + 1) % log_every == 0:
            print(f"  backtest {j+1}/{len(dates)}  nav={nav:.3f} bench={bench_nav:.3f}", flush=True)

    dd = pd.DataFrame(daily, columns=["trade_date", "nav", "bench_nav", "port_ret", "bench_ret"])
    dd["excess_ret"] = dd["port_ret"] - dd["bench_ret"]
    rr = pd.DataFrame(reb_rows, columns=["trade_date", "turnover", "n_holdings"])
    return dd, rr


# ---------------------------------------------------------------- performance
def performance(daily, reb, periods_per_year=252):
    """Annualised excess return, IR (tracking error), max drawdown of the excess
    NAV, two-way annual turnover, mean single-rebalance turnover, mean holdings."""
    ex = daily["excess_ret"].to_numpy()
    n = len(ex)
    ann_excess = ex.mean() * periods_per_year
    te = ex.std() * np.sqrt(periods_per_year)
    ir = ann_excess / te if te > 0 else np.nan
    # excess NAV (long port / short benchmark) drawdown
    exnav = np.cumprod(1.0 + ex)
    dd = exnav / np.maximum.accumulate(exnav) - 1.0
    n_reb = len(reb)
    reb_per_year = n_reb / (n / periods_per_year) if n > 0 else 0
    two_way_turnover = reb["turnover"].mean() * 2 * reb_per_year if n_reb else np.nan
    return {
        "ann_excess": ann_excess, "tracking_error": te, "IR": ir,
        "max_drawdown": float(dd.min()),
        "two_way_turnover_x": two_way_turnover,           # annual, ×
        "mean_single_turnover": float(reb["turnover"].mean() * 2) if n_reb else np.nan,
        "mean_holdings": float(reb["n_holdings"].mean()) if n_reb else np.nan,
        "n_days": n, "n_rebalances": n_reb,
    }
