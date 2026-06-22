"""§3.4.2 (Table 9 + Fig 21) and §3.4.3 (Table 10 + Fig 22): predictive power of
the combined alpha model. Streams the alpha exposures off the signal memmap.

Prerequisites: run_alphas.py --memmap, compute_barra.py, run_single_factor_test.py
(for the factor_returns_d1 ranking) — all for --universe full_a.
"""
import os
import sys
import json
import time
import pickle
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
import config          # noqa: E402
import data_loader as dl  # noqa: E402
import model_ic as mic    # noqa: E402

for _f in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [_f]
        break
rcParams["axes.unicode_minus"] = False
BLUE, RED = "#1f4e79", "#c00000"
# report's §3.4.2/3.4.3 published values (2012-2017, 全A非ST)
REPORT_T9 = {"ic_mean": 0.057, "tstat": 18.23, "win_rate": 0.887}
REPORT_T10 = {20: 0.0338, 40: 0.037, 60: 0.0424, 80: 0.0432, 100: 0.0483,
              120: 0.0518, 140: 0.0534, 160: 0.0549, 189: 0.057}


def cache(n):
    return os.path.join(config.CACHE_DIR, n)


def rank_factors_by_ir1():
    """|IR@d1| ranking of valid (non-degenerate) factors, from cached d1 returns."""
    fr = pd.read_parquet(cache("factor_returns_d1.parquet"))
    n = len(fr)
    valid = fr.count(axis=0) >= config.MIN_OBS_FRAC * n
    ir = (fr.mean(axis=0) / fr.std(axis=0) * np.sqrt(config.TRADING_DAYS_PER_YEAR)).where(valid)
    return ir.dropna().abs().sort_values(ascending=False).index.tolist()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="full_a", choices=["full_a", "topN"])
    ap.add_argument("--rank", default=config.RANK_CONVENTION)
    ap.add_argument("--from-cache", action="store_true",
                    help="reload the cached subset ICs and only (re)build tables/figures")
    args = ap.parse_args()
    config.UNIVERSE_MODE = args.universe
    t0 = time.time()

    mmap_dir = cache(f"signals_{args.universe}_{args.rank}")
    barra_cache = cache(f"barra_cne5s_factors_{args.universe}.parquet")
    for p in (os.path.join(mmap_dir, "meta.npz"), barra_cache, cache("factor_returns_d1.parquet")):
        if not os.path.exists(p):
            sys.exit(f"missing prerequisite: {p}")

    basic = dl.load_stock_basic()
    panel = dl.load_panel(columns=["ts_code", "trade_date", "close_hfq", "amount"])
    uni = dl.select_universe(panel, basic)
    panel = panel[panel["ts_code"].isin(uni)]
    close_wide = panel.pivot(index="trade_date", columns="ts_code", values="close_hfq")
    industry_map = dl.load_industry()
    style_long = pd.read_parquet(barra_cache)

    ranked = rank_factors_by_ir1()
    n_all = len(ranked)
    counts = [c for c in config.FACTOR_COUNTS if c < n_all]
    ics_pkl = cache(f"model_ic_subsets_{args.universe}.pkl")

    if args.from_cache and os.path.exists(ics_pkl):
        ics = pickle.load(open(ics_pkl, "rb"))
        print(f"loaded cached ICs {time.time()-t0:.0f}s", flush=True)
    else:
        subsets = {"all": ranked}
        subsets.update({str(c): ranked[:c] for c in counts})
        print(f"loaded: universe={len(uni)} valid_factors={n_all} subsets={len(subsets)} "
              f"{time.time()-t0:.0f}s", flush=True)
        ics = mic.compute_model_ic_multi_memmap(mmap_dir, style_long, industry_map, close_wide, subsets)
        pickle.dump(ics, open(ics_pkl, "wb"))          # cache BEFORE plotting (crash-safe)
        print(f"model IC done + cached {time.time()-t0:.0f}s", flush=True)

    # ---- Table 9 + Fig 21 (all-factor model) ----
    ic_all = ics["all"]
    ic_all.to_csv(cache("model_ic_series.csv"))
    s9 = mic.ic_stats(ic_all)
    t9 = pd.DataFrame({
        "日度IC均值": [f"{s9['ic_mean']:.4f}", f"{REPORT_T9['ic_mean']:.4f}"],
        "TStas Alpha Model": [f"{s9['tstat']:.2f}", f"{REPORT_T9['tstat']:.2f}"],
        "日度IC胜率": [f"{s9['win_rate']*100:.1f}%", f"{REPORT_T9['win_rate']*100:.1f}%"],
    }, index=["本次复现 (2017-2025, 全A)", "研报 (2012-2017)"])
    t9.to_csv(os.path.join(config.TABLE_DIR, "table9_model_ic.csv"))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = pd.to_datetime([str(d) for d in ic_all.index], format="%Y%m%d")
    ax.plot(x, ic_all.values, color="#9fb6cf", lw=0.6, label="日度模型IC")
    ax.plot(x, ic_all.rolling(30).mean().values, color=BLUE, lw=1.8, label="30日IC均值")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(s9["ic_mean"], color=RED, ls="--", lw=1, label=f"均值={s9['ic_mean']:.4f}")
    ax.set_title("图21 模型预测系数 (IC of Alpha Model) [全A · 截面RANK + CNE5S]")
    ax.legend(loc="upper right", ncol=3, fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig21_model_ic.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- Table 10 + Fig 22 (factor count vs predictive power) ----
    rows = []
    for c in counts + [n_all]:
        key = "all" if c == n_all else str(c)
        s = mic.ic_stats(ics[key])
        rows.append((c, round(s["ic_mean"], 4), round(s["tstat"], 2)))
    t10 = pd.DataFrame(rows, columns=["因子个数", "模型IC", "IC显著性"]).set_index("因子个数")
    t10.to_csv(os.path.join(config.TABLE_DIR, "table10_factor_count.csv"))

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.bar(t10.index.astype(str), t10["模型IC"], color=BLUE, alpha=0.85, width=0.6)
    ax1.set_ylabel("模型IC", color=BLUE); ax1.set_xlabel("因子个数")
    ax2 = ax1.twinx()
    ax2.plot(t10.index.astype(str), t10["IC显著性"], color=RED, marker="o", lw=2)
    ax2.set_ylabel("IC显著性 (t-stat)", color=RED)
    ax1.set_title("图22 因子个数与模型预测能力比较 [全A · 截面RANK + CNE5S]")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig22_factor_count.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    json.dump({"table9": s9, "table9_report": REPORT_T9,
               "table10": {int(c): {"ic_mean": m, "tstat": tt} for c, m, tt in rows}},
              open(cache("model_ic_summary.json"), "w"), ensure_ascii=False, indent=2)
    with open(os.path.join(config.TABLE_DIR, "tables_9_10.md"), "w") as f:
        f.write("### 表9 模型预测系数特征统计\n\n" + t9.to_markdown() + "\n\n")
        f.write("### 表10 因子个数与模型预测能力比较\n\n" + t10.to_markdown() + "\n")
    print("\n===== Table 9 =====\n", t9.to_string())
    print("\n===== Table 10 =====\n", t10.to_string())
    print(f"\nDone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
