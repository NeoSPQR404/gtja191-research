"""研究延伸：把"隔夜 alpha"量化为学术口径策略——信号的理论上限（冰山的水下部分）。

同一个 Ê(α)（决策于收盘 t），比较三个执行窗口的十分位多空毛收益：
  · 隔夜 close_t→open_{t+1}     —— 信号主场（IC +0.23），但不可交易（收盘即时成交=看穿）
  · 收盘→收盘 close_t→close_{t+1} —— §3.4 的预测目标
  · 开盘→开盘 open_{t+1}→open_{t+2} —— 现实可交易（次开盘成交）
排成一条"理论上限 → 可交易毛 → 计融券净"的递减阶梯，量化"冰山"有多大、水上有多少。
输出 fig36 + overnight_ceiling.{csv,md}。
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
Y = 252
BORROW = 0.0835                                           # 官方融券费率（用于阶梯末端）


def main():
    config.UNIVERSE_MODE = "full_a"
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "close_hfq", "amount", "adj_factor"])
    basic = dl.load_stock_basic(); uni = dl.select_universe(panel, basic)
    panel = panel[panel.ts_code.isin(uni)]
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p.trade_date.unique()).astype(str); codes = sorted(p.ts_code.unique())
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    cl = p.pivot(index="trade_date", columns="ts_code", values="close_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    amt = p.pivot(index="trade_date", columns="ts_code", values="amount").reindex(dates, axis=0).reindex(codes, axis=1)
    pw = pd.read_parquet(config.CACHE_DIR + "/alpha_pred_full_a.parquet").pivot(
        index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)

    windows = {
        "隔夜 close→open\n(信号主场·不可交易)": op.shift(-1) / cl - 1.0,
        "收盘→收盘\n(预测目标)": cl.shift(-1) / cl - 1.0,
        "开盘→开盘\n(现实可交易)": op.shift(-2) / op.shift(-1) - 1.0,
    }
    X = pw.to_numpy(); A = amt.to_numpy()

    def ls_series(ret):                                   # daily decile long-short, gross
        R = ret.to_numpy(); out = np.full(len(dates), np.nan)
        for j in range(len(dates)):
            x, r, a = X[j], R[j], A[j]
            m = np.isfinite(x) & np.isfinite(r) & (np.nan_to_num(a) > 0)
            if m.sum() < 200:
                continue
            # backstop winsorise per-stock at ±22% (all-board daily limit ±20% + buffer;
            # covers 创业板/科创板). The real fix is .BJ exclusion in select_universe;
            # this only caps impossible glitches (e.g. adj discontinuities) that would
            # blow up the equal-weight decile, and is verified not to touch the signal
            # (identical IR to no-clip once .BJ is excluded).
            xs = pd.Series(x[m]); rk = xs.rank(pct=True).to_numpy(); rr = np.clip(r[m], -0.22, 0.22)
            out[j] = rr[rk > 0.9].mean() - rr[rk < 0.1].mean()
        return pd.Series(out, index=dates)

    rows, series = [], {}
    for name, ret in windows.items():
        s = ls_series(ret); series[name] = s
        ann = s.mean() * Y; ir = s.mean() / s.std() * np.sqrt(Y)
        rows.append((name.replace("\n", " "), ann, ir))
    over = rows[0][1]; tradable = rows[2][1]               # overnight ceiling vs open-open gross

    tab = pd.DataFrame([(n, f"{a*100:+.0f}%", f"{i:.2f}") for n, a, i in rows],
                       columns=["执行窗口", "多空年化(毛)", "IR"]).set_index("执行窗口")
    tab.to_csv(os.path.join(config.TABLE_DIR, "overnight_ceiling.csv"))
    with open(os.path.join(config.TABLE_DIR, "overnight_ceiling.md"), "w") as f:
        f.write("### 研究延伸：隔夜 alpha 理论上限（同一 Ê，三执行窗口的十分位多空毛收益）\n\n")
        f.write(tab.to_markdown() + "\n\n")
        f.write(f"_递减阶梯（{config.WINDOW_LABEL}）：隔夜理论上限 ≈{over*100:.0f}%/年（不可交易）→ "
                f"开盘→开盘可交易毛 ≈{tradable*100:.0f}%/年 → 经风格中性+换手上限+融券成本后见市场中性"
                f"回测（图30-35）。研报所谓\"冰山一角\"：水下(隔夜)巨大但摸不到。_\n")

    # fig36: cumulative NAV — overnight ceiling vs tradable (log scale)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = pd.to_datetime([str(v) for v in dates], format="%Y%m%d")
    names = list(windows.keys())
    nav_over = (1 + series[names[0]].fillna(0)).cumprod()
    nav_c2c = (1 + series[names[1]].fillna(0)).cumprod()
    nav_oo = (1 + series[names[2]].fillna(0)).cumprod()
    ax.plot(x, nav_over, color=RED, lw=2, label=f"隔夜(理论上限,不可交易): 年化{rows[0][1]*100:+.0f}% IR{rows[0][2]:.1f}")
    ax.plot(x, nav_c2c, color=BLUE, lw=1.5, label=f"收盘→收盘(预测目标): 年化{rows[1][1]*100:+.0f}% IR{rows[1][2]:.1f}")
    ax.plot(x, nav_oo, color=GREEN, lw=1.8, label=f"开盘→开盘(可交易毛): 年化{rows[2][1]*100:+.0f}% IR{rows[2][2]:.1f}")
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.5)
    ax.set_ylabel("累计净值(对数)")
    ax.set_title("图36 交易型alpha的\"冰山\": 隔夜理论上限(红) >> 可交易(绿) [十分位多空毛, 全A]")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig36_overnight_ceiling.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print(tab.to_string())
    print(f"\n阶梯: 隔夜上限{over*100:+.0f}% → 开→开毛{tradable*100:+.0f}% → 见 market_neutral 的中性+换手+融券净")


if __name__ == "__main__":
    main()
