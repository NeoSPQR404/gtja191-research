"""§3.6 — IC decay curve → optimal holding period.

Decide at close t, execute at open_{t+1}, hold h days, exit at open_{t+1+h}.
For each h: the signal's IC vs the h-day open-to-open return (decay curve), and a
NON-overlapping decile long-short backtest (the tradable signal, fig28) giving gross
& net (after cost) annualised IR. Optimal h = argmax net IR. Saves fig29 + table.
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
BLUE, RED, GREEN = "#1f4e79", "#c00000", "#2e7d32"
TC = config.STRAT_TC
HS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
Y = 252


def main():
    config.UNIVERSE_MODE = "full_a"
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "amount", "adj_factor"])
    basic = dl.load_stock_basic(); uni = dl.select_universe(panel, basic)
    panel = panel[panel.ts_code.isin(uni)]
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p.trade_date.unique()).astype(str); codes = sorted(p.ts_code.unique())
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    pw = pd.read_parquet(config.CACHE_DIR + "/alpha_pred_full_a.parquet").pivot(
        index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)
    T = len(dates)
    opn = op.to_numpy(); pr = pw.to_numpy()

    rows = []
    for h in HS:
        # h-day open-to-open return aligned to decision day t (exec t+1 -> exit t+1+h)
        oo = np.full((T, len(codes)), np.nan)
        if 1 + h < T:
            oo[:T - (1 + h)] = opn[1 + h:] / opn[1:T - h] - 1.0
        # IC decay (overlapping, all t)
        ics = []
        for j in range(T - (1 + h)):
            x, y = pr[j], oo[j]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 200 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                ics.append(np.corrcoef(x[m], y[m])[0, 1])
        ic_h = float(np.mean(ics))
        # NON-overlapping decile long-short backtest, rebalance every h days
        rets, ptop, pbot = [], set(), set()
        for j in range(0, T - (1 + h), h):
            x, y = pr[j], oo[j]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 200:
                continue
            xs = pd.Series(x[m]); ys = pd.Series(y[m]); rk = xs.rank(pct=True)
            top = set(xs.index[rk > 0.9]); bot = set(xs.index[rk < 0.1])
            ls = ys[list(top)].mean() - ys[list(bot)].mean()
            t_top = len(top ^ ptop) / (2 * max(len(top), 1))
            t_bot = len(bot ^ pbot) / (2 * max(len(bot), 1))
            ptop, pbot = top, bot
            rets.append(ls - TC * (t_top + t_bot))         # round-trip cost both legs
        r = pd.Series(rets).dropna()
        ann = r.mean() * (Y / h)
        ir_net = r.mean() / r.std() * np.sqrt(Y / h) if r.std() > 0 else np.nan
        ir_gross = ic_h * np.sqrt(Y / h)                   # IC × √breadth proxy
        rows.append((h, ic_h, ir_gross, ann, ir_net))

    tab = pd.DataFrame(rows, columns=["持有天数", "IC", "毛IR≈IC√(252/h)", "年化净收益", "净IR"]).set_index("持有天数")
    best = tab["净IR"].idxmax()
    tab.round(3).to_csv(os.path.join(config.TABLE_DIR, "holding_period.csv"))
    with open(os.path.join(config.TABLE_DIR, "holding_period.md"), "w") as f:
        f.write("### §3.6 IC衰减与最优持有期 (开盘→开盘, 十分位多空, 2017-2025)\n\n")
        f.write(tab.round(3).to_markdown() + f"\n\n_最优持有期 = **{best} 天**（净IR 最大 "
                f"{tab.loc[best, '净IR']:.2f}, 年化 {tab.loc[best, '年化净收益']*100:.0f}%)_\n")

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(tab.index, tab["IC"], color=BLUE, marker="o", lw=2, label="信号IC (h日开→开)")
    ax1.set_xlabel("持有天数 h"); ax1.set_ylabel("信号 IC", color=BLUE); ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(tab.index, tab["净IR"], color=GREEN, marker="s", lw=2, label="净IR (扣费)")
    ax2.plot(tab.index, tab["毛IR≈IC√(252/h)"], color=RED, ls="--", lw=1.3, label="毛IR≈IC√(252/h)")
    ax2.axvline(best, color="grey", ls=":", lw=1)
    ax2.set_ylabel("年化 IR", color=GREEN)
    ax2.annotate(f"最优 h={best}", (best, tab.loc[best, "净IR"]), textcoords="offset points",
                 xytext=(8, 8), fontsize=10, color=GREEN)
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9)
    ax1.set_title("图29 IC衰减曲线与最优持有期 [开盘→开盘 多空, 全A]")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig29_holding_period.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print(tab.round(3).to_string())
    print(f"\n最优持有期 = {best} 天 (净IR {tab.loc[best,'净IR']:.2f}, 年化净 {tab.loc[best,'年化净收益']*100:.0f}%)")


if __name__ == "__main__":
    main()
