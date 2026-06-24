"""§3.6 — market-neutral long-short backtest at the optimal hold (h=7), open-to-open.

Dollar-neutral decile long-short: long the top-decile of Ê(α), short the bottom-decile,
equal-weight each leg, rebalance every HOLD trading days, execute at next-day open,
mark to market DAILY (open-to-open). Transaction cost Tc on rebalance turnover.
Outputs a daily NAV curve (fig30), long/short leg decomposition vs CSI500, and overall
+ yearly performance (table). This is the tradable form of the signal (vs the failing
long-only book) — requires the ability to short.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import config            # noqa: E402
import data_loader as dl  # noqa: E402

for _f in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [_f]; break
rcParams["axes.unicode_minus"] = False
BLUE, RED, GREEN, GREY = "#1f4e79", "#c00000", "#2e7d32", "#888888"
TC = config.STRAT_TC
HOLD = 7
DECILE = 0.10
MAXUNI = 1500
Y = 252


def perf(daily_ret, nav):
    r = pd.Series(daily_ret).dropna()
    ann = r.mean() * Y
    vol = r.std() * np.sqrt(Y)
    ir = ann / vol if vol > 0 else np.nan
    nav = pd.Series(nav)
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"ann_return": ann, "ann_vol": vol, "IR": ir, "max_drawdown": mdd,
            "win_rate": float((r > 0).mean())}


def main():
    config.UNIVERSE_MODE = "full_a"
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "amount", "adj_factor"])
    basic = dl.load_stock_basic(); uni = dl.select_universe(panel, basic)
    panel = panel[panel.ts_code.isin(uni)]
    nc = pd.read_parquet(config.NAMECHANGE)
    import strategy as stg
    st_iv = stg.st_codes_by_date(nc)

    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p.trade_date.unique()).astype(str); codes = sorted(p.ts_code.unique())
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    amt = p.pivot(index="trade_date", columns="ts_code", values="amount").reindex(dates, axis=0).reindex(codes, axis=1)
    pw = pd.read_parquet(config.CACHE_DIR + "/alpha_pred_full_a.parquet").pivot(
        index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)
    oo_daily = (op.shift(-1) / op - 1.0)                          # open_j -> open_{j+1}
    csi = pd.read_parquet(config.CSI500_RAW); csi["trade_date"] = csi.trade_date.astype(str)
    bo = csi.set_index("trade_date")["open"].reindex(dates).astype(float)
    mkt = (bo.shift(-1) / bo - 1.0)                               # CSI500 open-to-open

    code_index = pd.Index(codes)
    T = len(dates)
    first = pw.notna().any(axis=1).to_numpy().argmax()           # first day with predictions
    exec_days = set(range(first + 1, T - 1, HOLD))               # rebalance executions
    lw = pd.Series(0.0, index=codes); sw = pd.Series(0.0, index=codes)
    pend = {}
    nav_ls, nav_long, nav_short, nav_mkt = 1.0, 1.0, 1.0, 1.0
    rec, turns = [], []

    for j in range(T):
        cost = 0.0
        if j in pend:                                            # rebalance at open_j
            nlw, nsw = pend.pop(j)
            t_l = float(np.abs(nlw.to_numpy() - lw.to_numpy()).sum()) / 2.0
            t_s = float(np.abs(nsw.to_numpy() - sw.to_numpy()).sum()) / 2.0
            cost = TC * (t_l + t_s); turns.append(t_l + t_s)
            lw, sw = nlw, nsw
        r = oo_daily.iloc[j].fillna(0.0).to_numpy()
        lr = float(np.dot(lw.to_numpy(), r)); sr = float(np.dot(sw.to_numpy(), r))
        ls_ret = lr - sr - cost                                  # dollar-neutral spread
        mr = float(mkt.iloc[j]) if np.isfinite(mkt.iloc[j]) else 0.0
        nav_ls *= (1 + ls_ret); nav_long *= (1 + lr - cost / 2); nav_short *= (1 + sr); nav_mkt *= (1 + mr)
        rec.append((dates[j], nav_ls, nav_long, nav_short, nav_mkt, ls_ret, lr, sr, mr))
        for w in (lw, sw):                                       # buy-and-hold drift each leg
            g = w.to_numpy() * (1 + r); s = g.sum()
            if s > 0:
                w.iloc[:] = g / s
        # decide at close j -> execute open_{j+1}
        if (j + 1) in exec_days and j + 1 < T:
            d = dates[j]
            st_set = stg.is_st_on(st_iv, d)
            ok = (oo_daily.iloc[j].notna() & oo_daily.iloc[j + 1].notna() & pw.iloc[j].notna()
                  & (amt.iloc[j].fillna(0) > 0)).to_numpy()
            ok &= ~code_index.isin(st_set)
            cand = code_index[ok]
            liq = amt.iloc[j].reindex(cand).fillna(0).sort_values(ascending=False)
            u = liq.head(MAXUNI).index
            x = pw.iloc[j].reindex(u).dropna().sort_values()
            k = max(int(len(x) * DECILE), 1)
            bot, top = x.index[:k], x.index[-k:]
            nlw = pd.Series(0.0, index=codes); nlw[top] = 1.0 / len(top)
            nsw = pd.Series(0.0, index=codes); nsw[bot] = 1.0 / len(bot)
            pend[j + 1] = (nlw, nsw)

    dd = pd.DataFrame(rec, columns=["trade_date", "nav_ls", "nav_long", "nav_short",
                                    "nav_mkt", "ls_ret", "long_ret", "short_ret", "mkt_ret"])
    dd["year"] = dd["trade_date"].str[:4]
    dd.to_parquet(os.path.join(config.CACHE_DIR, "market_neutral_daily.parquet"), index=False)

    ov = perf(dd["ls_ret"], dd["nav_ls"])
    ann_turn = np.mean(turns) * (Y / HOLD) * 2 if turns else 0   # two-way (both legs)
    # leg decomposition vs market (which side drives the L/S?)
    long_ex = (dd["long_ret"] - dd["mkt_ret"]).mean() * Y        # long leg excess vs CSI500
    short_ex = (dd["mkt_ret"] - dd["short_ret"]).mean() * Y      # short leg contribution

    yr = dd.groupby("year").apply(lambda g: pd.Series({
        "年化收益": g["ls_ret"].mean() * Y,
        "IR": g["ls_ret"].mean() / g["ls_ret"].std() * np.sqrt(Y) if g["ls_ret"].std() > 0 else np.nan,
        "最大回撤": ((1 + g["ls_ret"]).cumprod() / (1 + g["ls_ret"]).cumprod().cummax() - 1).min(),
        "多头超额": (g["long_ret"] - g["mkt_ret"]).mean() * Y,
        "空头贡献": (g["mkt_ret"] - g["short_ret"]).mean() * Y,
    }))

    t = pd.DataFrame({"市场中性 (h=7 多空, 扣费)": [
        f"{ov['ann_return']*100:.1f}%", f"{ov['ann_vol']*100:.1f}%", f"{ov['IR']:.2f}",
        f"{ov['max_drawdown']*100:.1f}%", f"{ov['win_rate']*100:.1f}%", f"{ann_turn:.0f}倍",
        f"{long_ex*100:+.1f}%", f"{short_ex*100:+.1f}%"]},
        index=["年化收益", "年化波动", "信息比率", "最大回撤", "日胜率", "年化双边换手",
               "多头超额(vs500)", "空头贡献(vs500)"])
    t.to_csv(os.path.join(config.TABLE_DIR, "market_neutral_perf.csv"))
    with open(os.path.join(config.TABLE_DIR, "market_neutral.md"), "w") as f:
        f.write("### §3.6 市场中性多空回测 (h=7, 开盘→开盘, 全A, 2017-2025)\n\n")
        f.write(t.to_markdown() + "\n\n### 分年度\n\n")
        f.write(yr.round(3).assign(**{c: (yr[c] * 100).round(1).astype(str) + "%"
                for c in ["年化收益", "最大回撤", "多头超额", "空头贡献"]}).to_markdown() + "\n")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = pd.to_datetime([str(v) for v in dd["trade_date"]], format="%Y%m%d")
    ax.plot(x, dd["nav_ls"], color=GREEN, lw=2, label=f"市场中性多空 (h=7)  IR={ov['IR']:.2f}")
    ax.plot(x, dd["nav_long"], color=BLUE, lw=1, alpha=0.8, label="多头腿")
    ax.plot(x, dd["nav_short"], color=RED, lw=1, alpha=0.8, label="空头腿(净值越低=越该做空)")
    ax.plot(x, dd["nav_mkt"], color=GREY, lw=1, ls="--", label="中证500")
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.5)
    ax.set_title(f"图30 市场中性多空净值 [h=7, 开盘→开盘, 全A]  年化{ov['ann_return']*100:.0f}% "
                 f"回撤{ov['max_drawdown']*100:.0f}%")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig30_market_neutral.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print(t.to_string()); print("\n分年度:\n", yr.round(3).to_string())


if __name__ == "__main__":
    main()
