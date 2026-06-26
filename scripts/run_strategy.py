"""§3.6 — style-neutral multi-factor strategy backtest vs CSI500.

Prereqs: export_predictions.py (alpha_pred_full_a.parquet), compute_barra.py,
and the fetched CSI500 daily / weights / namechange in data/raw/.

Produces Table 11 (perf), Table 12 (yearly), Figs 23/24/25; with --sweep also
Table 13 + Fig 26 (Tc sensitivity); §3.6.4 capacity estimate.
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import config            # noqa: E402
import data_loader as dl    # noqa: E402
import strategy as stg       # noqa: E402

for _f in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [_f]
        break
rcParams["axes.unicode_minus"] = False
BLUE, RED = "#1f4e79", "#c00000"
# report Table 11 (2012-2017, vs CSI500)
REPORT_T11 = {"ann_excess": 0.502, "IR": 4.67, "max_drawdown": -0.059,
              "two_way_turnover_x": 48.2, "mean_single_turnover": 0.38, "mean_holdings": 52}


def cache(n):
    return os.path.join(config.CACHE_DIR, n)


def load_inputs():
    config.UNIVERSE_MODE = "full_a"
    basic = dl.load_stock_basic()
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "close_hfq",
                                   "amount", "adj_factor"])
    uni = dl.select_universe(panel, basic)
    panel = panel[panel["ts_code"].isin(uni) & (panel["trade_date"] >= config.TEST_START)].reset_index(drop=True)
    pred = pd.read_parquet(cache("alpha_pred_full_a.parquet"))
    styles = pd.read_parquet(cache("barra_cne5s_factors_full_a.parquet"))
    industry_map = dl.load_industry().to_dict()
    csi = pd.read_parquet(config.CSI500_RAW)
    csiw = pd.read_parquet(config.CSI500_WEIGHT)
    nc = pd.read_parquet(config.NAMECHANGE)
    return panel, pred, styles, industry_map, csi, csiw, nc


def yearly(daily):
    d = daily.copy()
    d["year"] = d["trade_date"].str[:4]
    rows = []
    for y, g in d.groupby("year"):
        ex = g["excess_ret"].to_numpy()
        ann = ex.mean() * config.TRADING_DAYS_PER_YEAR
        te = ex.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        exnav = np.cumprod(1 + ex)
        dd = (exnav / np.maximum.accumulate(exnav) - 1).min()
        rows.append((y, ann, ann / te if te > 0 else np.nan, dd))
    return pd.DataFrame(rows, columns=["年份", "年化超额收益", "信息比率", "最大回撤"]).set_index("年份")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also run the Tc sensitivity sweep")
    args = ap.parse_args()
    t0 = time.time()
    panel, pred, styles, industry_map, csi, csiw, nc = load_inputs()
    print(f"loaded inputs {time.time()-t0:.0f}s; stocks={panel['ts_code'].nunique()} "
          f"dates={panel['trade_date'].nunique()}", flush=True)

    # ---- main backtest (Tc base) ----
    daily, reb = stg.backtest(pred, styles, industry_map, panel, csi, csiw, nc,
                              Tc=config.STRAT_TC, reb_freq=config.STRAT_REB_FREQ,
                              style_bound=config.STRAT_STYLE_BOUND,
                              ind_bound=config.STRAT_IND_BOUND,
                              max_universe=config.STRAT_MAX_UNIVERSE, log_every=200)
    daily.to_parquet(cache("strategy_daily.parquet"))
    reb.to_parquet(cache("strategy_reb.parquet"))
    perf = stg.performance(daily, reb)
    print(f"main backtest done {time.time()-t0:.0f}s", flush=True)

    t11 = pd.DataFrame({
        f"本次复现 ({config.WINDOW_LABEL}, vs中证500)": [
            f"{perf['ann_excess']*100:.2f}%", f"{perf['IR']:.2f}",
            f"{perf['max_drawdown']*100:.2f}%", f"{perf['two_way_turnover_x']:.1f}倍",
            f"{perf['mean_single_turnover']*100:.1f}%", f"{perf['mean_holdings']:.0f}"],
        "研报 (2012-2017)": [
            f"{REPORT_T11['ann_excess']*100:.2f}%", f"{REPORT_T11['IR']:.2f}",
            f"{REPORT_T11['max_drawdown']*100:.2f}%", f"{REPORT_T11['two_way_turnover_x']:.1f}倍",
            f"{REPORT_T11['mean_single_turnover']*100:.1f}%", f"{REPORT_T11['mean_holdings']}"],
    }, index=["年化超额收益", "信息比率", "最大回撤", "年化换手(双边)", "平均单次换仓(双边)", "平均持仓数"])
    t11.to_csv(os.path.join(config.TABLE_DIR, "table11_strategy_perf.csv"))
    t12 = yearly(daily)
    t12_fmt = t12.copy()
    t12_fmt["年化超额收益"] = (t12["年化超额收益"] * 100).round(2).astype(str) + "%"
    t12_fmt["信息比率"] = t12["信息比率"].round(2)
    t12_fmt["最大回撤"] = (t12["最大回撤"] * 100).round(2).astype(str) + "%"
    t12_fmt.to_csv(os.path.join(config.TABLE_DIR, "table12_yearly.csv"))

    # ---- Figs 23 (NAV), 24 (cumulative excess), 25 (turnover) ----
    x = pd.to_datetime([str(d) for d in daily["trade_date"]], format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, daily["nav"], color=BLUE, lw=1.5, label="策略组合净值")
    ax.plot(x, daily["bench_nav"], color="grey", lw=1.2, label="中证500")
    ax.set_title("图23 组合累计净值 [全A · 样本外 2017-2025]"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig23_nav.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    exnav = (1 + daily["excess_ret"]).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, exnav, color=RED, lw=1.5)
    ax.set_title(f"图24 组合累计超额收益 [年化 {perf['ann_excess']*100:.1f}%, IR {perf['IR']:.2f}]")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig24_excess.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    xr = pd.to_datetime([str(d) for d in reb["trade_date"]], format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(xr, reb["turnover"] * 2 * 100, width=3, color=BLUE, alpha=0.8)
    ax.set_title("图25 组合单次换仓比率(双边, %)"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig25_turnover.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- key diagnostic: overnight (untradable) vs intraday (tradable) IC ----
    icd = ic_decomposition(panel, pred)
    cap_note = capacity_estimate(daily, reb, panel)

    print("\n===== Table 11 =====\n", t11.to_string())
    print("\n===== Table 12 (yearly) =====\n", t12_fmt.to_string())
    print("\nIC decomposition (overnight vs intraday):",
          {k.replace(chr(10), " "): round(v, 4) for k, v in icd.items()})
    print("\n", cap_note)

    summary = {"table11": perf, "table11_report": REPORT_T11,
               "yearly": t12.round(4).reset_index().to_dict("records"),
               "ic_decomposition": icd, "capacity": cap_note}

    # ---- §3.6.3 Tc sensitivity sweep ----
    if args.sweep:
        rows = []
        for tc in config.STRAT_TC_SWEEP:
            dd, rb = stg.backtest(pred, styles, industry_map, panel, csi, csiw, nc,
                                  Tc=tc, reb_freq=config.STRAT_REB_FREQ,
                                  style_bound=config.STRAT_STYLE_BOUND,
                                  ind_bound=config.STRAT_IND_BOUND,
                                  max_universe=config.STRAT_MAX_UNIVERSE, log_every=0)
            pf = stg.performance(dd, rb)
            rows.append((tc, pf["ann_excess"], pf["two_way_turnover_x"],
                         pf["mean_single_turnover"], pf["max_drawdown"], pf["IR"]))
            print(f"  Tc={tc*100:.1f}%: excess={pf['ann_excess']*100:.1f}% IR={pf['IR']:.2f} "
                  f"turn={pf['two_way_turnover_x']:.1f}x", flush=True)
        t13 = pd.DataFrame(rows, columns=["Tc", "年化超额收益", "年化换手(双边,x)",
                                          "平均单次换仓", "最大回撤", "信息比率"]).set_index("Tc")
        t13.to_csv(os.path.join(config.TABLE_DIR, "table13_tc_sensitivity.csv"))
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.plot([f"{t*100:.1f}%" for t in t13.index], t13["年化超额收益"] * 100, color=BLUE, marker="o", lw=2)
        ax.set_ylabel("年化超额收益 (%)", color=BLUE); ax.set_xlabel("交易成本 Tc")
        ax.set_title("图26 不同交易成本下组合年化超额收益"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig26_tc.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
        summary["table13"] = t13.round(4).reset_index().to_dict("records")
        print("\n===== Table 13 (Tc sensitivity) =====\n", t13.round(3).to_string())

    json.dump(summary, open(cache("strategy_summary.json"), "w"), ensure_ascii=False, indent=2, default=float)
    with open(os.path.join(config.TABLE_DIR, "tables_11_13.md"), "w") as f:
        f.write("### 表11 组合绩效统计\n\n" + t11.to_markdown() + "\n\n")
        f.write("### 表12 分年度绩效\n\n" + t12_fmt.to_markdown() + "\n\n")
        f.write(f"_{cap_note}_\n")
    print(f"\nDone {time.time()-t0:.0f}s")


def ic_decomposition(panel, pred):
    """Key diagnostic: the combined-alpha IC split into the (untradable) overnight
    gap vs the (tradable) intraday move. Saves a bar figure + returns the means."""
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p["trade_date"].unique()).astype(str)
    codes = sorted(p["ts_code"].unique())
    cl = p.pivot(index="trade_date", columns="ts_code", values="close_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    comps = {"收盘→收盘\n(预测目标)": cl.shift(-1) / cl - 1.0,
             "收盘→次开盘\n(隔夜·不可交易)": op.shift(-1) / cl - 1.0,
             "次开盘→收盘\n(日内·可交易)": cl.shift(-1) / op.shift(-1) - 1.0}
    pw = pred.pivot(index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)
    out = {}
    for name, b in comps.items():
        ics = []
        for j in range(len(dates)):
            x, y = pw.iloc[j].to_numpy(), b.iloc[j].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 200 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                ics.append(np.corrcoef(x[m], y[m])[0, 1])
        out[name] = float(np.mean(ics))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = [BLUE, "#888", RED]
    bars = ax.bar(list(out.keys()), list(out.values()), color=cols, alpha=0.85)
    ax.axhline(0, color="grey", lw=0.6)
    for bar, v in zip(bars, out.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.005 if v >= 0 else -0.012),
                f"{v:+.3f}", ha="center", fontsize=11)
    ax.set_title("图27 组合模型IC分解: 隔夜(不可交易) vs 日内(可交易) [全A, 2017-2025]")
    ax.set_ylabel("日均截面IC")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig27_ic_decomp.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def capacity_estimate(daily, reb, panel):
    """§3.6.4: per holding, capacity = 10% of its daily amount; portfolio capacity
    ≈ 95-pctile-floored min over holdings, scaled by weight. Coarse estimate."""
    amt = panel.groupby("ts_code")["amount"].median() * 1000.0    # 千元 -> 元
    med = float(amt.median())
    mh = reb["n_holdings"].mean() if len(reb) else np.nan
    # rough: 10% of median holding amount × mean holdings, capped by max single weight
    cap_yuan = 0.10 * med * mh
    return (f"§3.6.4 资金容纳(粗估): 持仓中位成交额≈{med/1e8:.2f}亿元/日, 取10%, "
            f"平均持仓{mh:.0f}只 → 组合容量量级≈{cap_yuan/1e8:.1f}亿元 (样本外粗估)")


if __name__ == "__main__":
    main()
