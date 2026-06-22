"""Render Tables 7-8 and Figures 9-20 from the cached per-date factor returns,
in the layout of GTJA《数量化专题之九十三》§3.4.1.

Self-sufficient: reads report/cache/factor_returns_d{d}.parquet (written by
run_single_factor_test.py) and recomputes the annualised FR/IR tables, the
distributions, the horizon trend, and the factor-return correlation / pruning.
Writes report/tables/*.csv|md and report/figures/*.png.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import config                       # noqa: E402
import single_factor_test as sft    # noqa: E402

# Report's published reference values (period 2010/2012-2017, full A ex-ST).
REPORT_FR = {1: 6.85, 2: 8.08, 3: 6.09, 4: 5.42, 5: 4.78}     # %
REPORT_IR = {1: 2.57, 2: 2.11, 3: 1.73, 4: 1.26, 5: 1.05}

for _f in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [_f]
        break
rcParams["axes.unicode_minus"] = False
BLUE = "#1f4e79"


def load_factor_returns():
    fr_by_date = {}
    for d in config.HORIZONS:
        p = os.path.join(config.CACHE_DIR, f"factor_returns_d{d}.parquet")
        if os.path.exists(p):
            fr_by_date[d] = pd.read_parquet(p)
    if not fr_by_date:
        sys.exit("no factor_returns_d*.parquet in cache — run run_single_factor_test.py first")
    return fr_by_date


def annualise(fr_by_date):
    """Recompute the per-alpha annualised FR/IR tables (valid set only)."""
    alphas = sorted({c for fr in fr_by_date.values() for c in fr.columns},
                    key=lambda a: int(a[5:]))
    fr_table = pd.DataFrame(index=alphas, columns=config.HORIZONS, dtype=float)
    ir_table = pd.DataFrame(index=alphas, columns=config.HORIZONS, dtype=float)
    for d, fr in fr_by_date.items():
        n = len(fr)
        valid = fr.count(axis=0) >= config.MIN_OBS_FRAC * n
        mean, std = fr.mean(axis=0), fr.std(axis=0)
        fr_table[d] = (config.TRADING_DAYS_PER_YEAR * mean / d).where(valid)
        ir_table[d] = (mean / std * np.sqrt(config.TRADING_DAYS_PER_YEAR / d)).where(valid)
    return fr_table, ir_table


def tables(fr, ir):
    h = config.HORIZONS
    t7 = pd.DataFrame({
        f"DAY{d}": [f"{fr[d].mean()*100:.2f}%", f"{REPORT_FR[d]:.2f}%"] for d in h
    }, index=["本次复现 (2017-2025, 全A非ST, 样本外)", "研报 (2012-2017, 全A非ST)"])
    t8 = pd.DataFrame({
        f"DAY{d}": [f"{ir[d].mean():.2f}", f"{ir[d].abs().mean():.2f}", f"{REPORT_IR[d]:.2f}"]
        for d in h
    }, index=["本次复现 IR均值 (signed)", "本次复现 |IR|均值", "研报 IR均值"])
    t7.to_csv(os.path.join(config.TABLE_DIR, "table7_mean_factor_return.csv"))
    t8.to_csv(os.path.join(config.TABLE_DIR, "table8_mean_IR.csv"))
    with open(os.path.join(config.TABLE_DIR, "tables_7_8.md"), "w") as f:
        f.write(f"_有效（非退化）因子数: {int(ir[1].notna().sum())}；预测周期 d=1..5；"
                f"窗口 2017-2025（样本外）_\n\n")
        f.write("### 表7 不同预测周期下所有因子收益率均值\n\n" + t7.to_markdown() + "\n\n")
        f.write("### 表8 不同预测周期下所有因子收益率IR均值\n\n" + t8.to_markdown() + "\n")
    return t7, t8


def _hist(ax, data, title, color=BLUE):
    data = data[np.isfinite(data)]
    ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(data), color="#c00000", lw=1.2, ls="--")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)


def fig_distributions(fr, ir):
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for ax, d in zip(axes, config.HORIZONS):
        _hist(ax, (fr[d] * 100).dropna().to_numpy(), f"图{8+d} FR_DAY{d} (%)")
    fig.suptitle("图9-13 不同预测周期下因子年化收益率分布 [截面RANK]", fontsize=13, y=1.04)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig09-13_FR_dist.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for ax, d in zip(axes, config.HORIZONS):
        _hist(ax, ir[d].dropna().to_numpy(), f"图{13+d} IR_DAY{d}", color="#2e75b6")
    fig.suptitle("图14-18 不同预测周期下因子收益率IR分布 [截面RANK]", fontsize=13, y=1.04)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig14-18_IR_dist.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)


def fig19_trend(fr, ir):
    d = config.HORIZONS
    fr_mean = [fr[h].mean() * 100 for h in d]
    ir_mean = [ir[h].abs().mean() for h in d]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar([f"DAY{h}" for h in d], fr_mean, color=BLUE, alpha=0.85)
    ax1.set_ylabel("年化因子收益率均值 (%)", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot([f"DAY{h}" for h in d], ir_mean, color="#c00000", marker="o", lw=2)
    ax2.set_ylabel("因子收益率|IR|均值", color="#c00000")
    ax1.set_title("图19 不同周期下因子收益率均值及IR均值变化情况 [截面RANK]")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig19_horizon_trend.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)


def fig20_corr(fr_by_date, ir_table):
    factor_returns = {d: fr_by_date[d] for d in fr_by_date}
    kept, dropped, corr_before, corr_after = sft.prune_correlated(factor_returns, ir_table)
    v = corr_after.where(~np.eye(len(corr_after), dtype=bool)).stack().to_numpy()
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(v), color="#c00000", lw=1.3, ls="--", label=f"均值={np.mean(v):.4f}")
    within = np.mean(np.abs(v) < 0.1) * 100
    ax.set_title(f"图20 因子收益率相关系数分布 [截面RANK] (剔除后 |corr|<0.1 占 {within:.1f}%)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig20_corr_dist.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    return kept, dropped, corr_before


def main():
    fr_by_date = load_factor_returns()
    fr_table, ir_table = annualise(fr_by_date)
    t7, t8 = tables(fr_table, ir_table)
    fig_distributions(fr_table, ir_table)
    fig19_trend(fr_table, ir_table)
    kept, dropped, corr_before = fig20_corr(fr_by_date, ir_table)
    print("Table 7 (mean annual factor return):\n", t7, "\n\nTable 8 (mean IR):\n", t8)
    print(f"\nvalid factors: {int(ir_table[1].notna().sum())}; "
          f"pruning {corr_before.shape[0]} -> kept {len(kept)}, dropped {len(dropped)}")
    print(f"figures -> {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
