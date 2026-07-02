"""§3.6 tradability diagnostic: is the combined alpha's open-to-open edge tradable?

Decision at close t, execute at open_{t+1}, exit at open_{t+2} (open-to-open hold,
the realistic next-open convention). Compares:
  * decile long-SHORT (gross) — the raw signal strength, and
  * long-only top-N after cost vs CSI500 — what a long-only book (the report's
    construction, A-share short constraint) actually captures.
Saves report/figures/fig28_tradability.png + report/tables/tradability.md.
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
BLUE, RED = "#1f4e79", "#c00000"
TC = config.STRAT_TC
SP = 2          # open-to-open spacing (decide j, hold open_{j+1}->open_{j+2}); ~daily reb


def main():
    config.UNIVERSE_MODE = "full_a"
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "amount", "adj_factor"])
    basic = dl.load_stock_basic(); uni = dl.select_universe(panel, basic)
    panel = panel[panel.ts_code.isin(uni)]
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p.trade_date.unique()).astype(str); codes = sorted(p.ts_code.unique())
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    amt = p.pivot(index="trade_date", columns="ts_code", values="amount").reindex(dates, axis=0).reindex(codes, axis=1)
    pw = pd.read_parquet(config.CACHE_DIR + "/alpha_pred_full_a.parquet").pivot(
        index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)
    # open_{j+1} -> open_{j+SP} (1-day hold); backstop winsorise per-stock at ±22%
    # (all-board daily limit ±20% + buffer). The real fix is .BJ exclusion in
    # select_universe; this only caps impossible glitches that blow up the equal-weight
    # decile / top-N mean (see data_loader.select_universe .BJ note).
    oo = (op.shift(-SP) / op.shift(-1) - 1.0).clip(-0.22, 0.22)
    csi = pd.read_parquet(config.CSI500_RAW); csi["trade_date"] = csi.trade_date.astype(str)
    bo = csi.set_index("trade_date")["open"].reindex(dates).astype(float)
    bench_oo = bo.shift(-SP) / bo.shift(-1) - 1.0

    def annualize(ex):
        ex = pd.Series(ex).dropna()
        return ex.mean() * 252 / SP, ex.mean() / ex.std() * np.sqrt(252 / SP)

    rows, lo_curve = [], {}
    for topn in (50, 100, 200):
        rets, prev = [], set()
        for j in range(len(dates) - SP):
            x, r, liq = pw.iloc[j], oo.iloc[j], amt.iloc[j]
            ok = x.notna() & r.notna() & (liq.fillna(0) > 0)
            if ok.sum() < topn:
                rets.append(np.nan); continue
            sel = x[ok].nlargest(topn).index
            turn = len(set(sel) ^ prev) / (2 * topn); prev = set(sel)
            b = bench_oo.iloc[j]
            rets.append(float(r[sel].mean()) - 2 * TC * turn - (b if np.isfinite(b) else 0))
        a, ir = annualize(rets); lo_curve[topn] = a
        rows.append((f"多头 top{topn} (扣费)", a, ir))

    ls = []
    for j in range(len(dates) - SP):
        x, r = pw.iloc[j], oo.iloc[j]; m = x.notna() & r.notna()
        if m.sum() < 200:
            ls.append(np.nan); continue
        rk = x[m].rank(pct=True)
        ls.append(r[m][rk > 0.9].mean() - r[m][rk < 0.1].mean())
    a_ls, ir_ls = annualize(ls)
    rows.insert(0, ("十分位 多空 (毛, 信号强度)", a_ls, ir_ls))

    tab = pd.DataFrame(rows, columns=["组合", "年化收益", "IR"]).set_index("组合")
    with open(os.path.join(config.TABLE_DIR, "tradability.md"), "w") as f:
        f.write(f"### §3.6 可交易性诊断 (开盘→开盘执行, {config.WINDOW_LABEL})\n\n")
        f.write("_多空毛 IR 衡量信号强度; 多头扣费超额衡量长仓实际可得_\n\n")
        f.write(tab.assign(年化收益=lambda d: (d["年化收益"] * 100).round(1).astype(str) + "%",
                           IR=lambda d: d["IR"].round(2)).to_markdown() + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    names = list(tab.index); vals = [v * 100 for v in tab["年化收益"]]
    cols = [BLUE] + [RED if v < 0 else "#2e7d32" for v in vals[1:]]
    bars = ax.bar(range(len(names)), vals, color=cols, alpha=0.85)
    ax.axhline(0, color="grey", lw=0.6)
    for b, v, ir in zip(bars, vals, tab["IR"]):
        ax.text(b.get_x() + b.get_width() / 2, v + (1 if v >= 0 else -2.5),
                f"{v:+.0f}%\nIR {ir:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("年化 (%)")
    ax.set_title("图28 可交易性: 多空信号强(IR≈研报) 但长仓不可得 [开盘→开盘, 全A]")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig28_tradability.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print(tab.round(3).to_string())
    print(f"\nlong-short gross IR={ir_ls:+.2f} (report strategy IR 4.67); "
          f"long-only top50 after cost={lo_curve[50]*100:+.1f}%")


if __name__ == "__main__":
    main()
