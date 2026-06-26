"""§3.6 — market-neutral long-short backtest at the optimal hold (h=7), open-to-open.

Dollar-neutral decile long-short: long the top-decile of Ê(α), short the bottom-decile,
equal-weight each leg, rebalance every HOLD trading days, execute at next-day open,
mark to market DAILY (open-to-open). Transaction cost Tc on rebalance turnover.

A per-leg single-rebalance turnover cap (None / 0.6 / 0.4 / 0.2) is applied by a
partial move toward the fresh decile (w_new = w_old + α·(target − w_old), α = cap/turn)
— the report's §3.5 risk-constraint turnover control. Compares the cap tiers
(fig31/table); also writes the baseline leg decomposition (fig30). Shortable universe
(liquid top-1500, ex-ST — A-share shorting is limited to the securities-lending list).
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
import strategy as stg    # noqa: E402

for _f in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [_f]; break
rcParams["axes.unicode_minus"] = False
BLUE, RED, GREEN, GREY = "#1f4e79", "#c00000", "#2e7d32", "#888888"
TC = config.STRAT_TC
HOLD, DECILE, MAXUNI, Y = 7, 0.10, 1500, 252


def _cap_move(w_old, w_tgt, cap):
    """Partial move toward the target decile so one-side turnover ≤ cap."""
    if cap is None:
        return w_tgt
    t = float(np.abs(w_tgt.to_numpy() - w_old.to_numpy()).sum()) / 2.0
    if t <= cap or t == 0:
        return w_tgt
    return w_old + (cap / t) * (w_tgt - w_old)


def _zstd(M):
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0)
    return np.nan_to_num((M - mu) / np.where(sd < 1e-12, np.nan, sd), nan=0.0)


def run_ls(op, amt, pw, oo_daily, mkt, codes, dates, st_iv, turnover_cap=None,
           style_arr=None, ind_dum=None, code_pos=None, neutralize=False):
    code_index = pd.Index(codes); T = len(dates)
    first = pw.notna().any(axis=1).to_numpy().argmax()
    exec_days = set(range(first + 1, T - 1, HOLD))
    lw = pd.Series(0.0, index=codes); sw = pd.Series(0.0, index=codes)
    pend, rec, turns, expo = {}, [], [], []           # expo = |net style exposure| per reb
    nav_ls = nav_long = nav_short = nav_mkt = 1.0
    for j in range(T):
        cost = 0.0
        if j in pend:
            tl_w, ts_w = pend.pop(j)
            tl_w = _cap_move(lw, tl_w, turnover_cap)
            ts_w = _cap_move(sw, ts_w, turnover_cap)
            t_l = float(np.abs(tl_w.to_numpy() - lw.to_numpy()).sum()) / 2.0
            t_s = float(np.abs(ts_w.to_numpy() - sw.to_numpy()).sum()) / 2.0
            cost = TC * (t_l + t_s); turns.append(t_l + t_s)
            lw, sw = tl_w, ts_w
        r = oo_daily.iloc[j].fillna(0.0).to_numpy()
        lr = float(np.dot(lw.to_numpy(), r)); sr = float(np.dot(sw.to_numpy(), r))
        ls_ret = lr - sr - cost
        mr = float(mkt.iloc[j]) if np.isfinite(mkt.iloc[j]) else 0.0
        nav_ls *= (1 + ls_ret); nav_long *= (1 + lr - cost / 2); nav_short *= (1 + sr); nav_mkt *= (1 + mr)
        rec.append((dates[j], nav_ls, nav_long, nav_short, nav_mkt, ls_ret, lr, sr, mr))
        for w in (lw, sw):
            g = w.to_numpy() * (1 + r); s = g.sum()
            if s > 0:
                w.iloc[:] = g / s
        if (j + 1) in exec_days and j + 1 < T:
            d = dates[j]
            st_set = stg.is_st_on(st_iv, d)
            ok = (oo_daily.iloc[j].notna() & oo_daily.iloc[j + 1].notna() & pw.iloc[j].notna()
                  & (amt.iloc[j].fillna(0) > 0)).to_numpy()
            ok &= ~code_index.isin(st_set)
            liq = amt.iloc[j].reindex(code_index[ok]).fillna(0).sort_values(ascending=False)
            sig = pw.iloc[j].reindex(liq.head(MAXUNI).index).dropna()
            u = sig.index
            pos = np.array([code_pos[c] for c in u]) if code_pos is not None else None
            Zs = None
            if pos is not None:                       # this date's style matrix for u
                Zs = _zstd(np.column_stack([style_arr[s][j][pos] for s in config.STYLE_FACTORS]))
            if neutralize and pos is not None and len(u) > 40:
                Zi = ind_dum[pos]; Zi = Zi[:, Zi.sum(0) > 0]
                Z = np.hstack([Zi, Zs, np.ones((len(u), 1))])
                e = sig.to_numpy()
                coef, *_ = np.linalg.lstsq(Z, e, rcond=None)
                sig = pd.Series(e - Z @ coef, index=u)   # style/industry-neutral signal
            x = sig.sort_values()
            k = max(int(len(x) * DECILE), 1)
            nlw = pd.Series(0.0, index=codes); nlw[x.index[-k:]] = 1.0 / k
            nsw = pd.Series(0.0, index=codes); nsw[x.index[:k]] = 1.0 / k
            pend[j + 1] = (nlw, nsw)
            if Zs is not None:                        # realized net style exposure of the spread
                net = (nlw.reindex(u).to_numpy() - nsw.reindex(u).to_numpy())
                expo.append(float(np.mean(np.abs(net @ Zs))))
    dd = pd.DataFrame(rec, columns=["trade_date", "nav_ls", "nav_long", "nav_short",
                                    "nav_mkt", "ls_ret", "long_ret", "short_ret", "mkt_ret"])
    ann_turn = (np.mean(turns) * (Y / HOLD) * 2) if turns else 0.0
    return dd, ann_turn, (float(np.mean(expo)) if expo else float("nan"))


def stats(dd):
    r = dd["ls_ret"]
    nav = dd["nav_ls"]
    return {"ann": r.mean() * Y, "vol": r.std() * np.sqrt(Y),
            "ir": r.mean() / r.std() * np.sqrt(Y) if r.std() > 0 else np.nan,
            "mdd": (nav / nav.cummax() - 1).min(), "win": (r > 0).mean()}


def sweep(op, amt, pw, oo_daily, mkt, codes, dates, st_iv, caps, style_arr, ind_dum, code_pos, neut):
    results, navs, expos = [], {}, {}
    for name, cap in caps:
        dd, ann_turn, ex = run_ls(op, amt, pw, oo_daily, mkt, codes, dates, st_iv, cap,
                                  style_arr=style_arr, ind_dum=ind_dum, code_pos=code_pos, neutralize=neut)
        s = stats(dd); navs[name] = dd; expos[name] = ex
        long_ex = (dd["long_ret"] - dd["mkt_ret"]).mean() * Y
        short_ex = (dd["mkt_ret"] - dd["short_ret"]).mean() * Y
        results.append((name, s["ann"], s["vol"], s["ir"], s["mdd"], ann_turn, long_ex, short_ex))
    return results, navs, expos


def main():
    config.UNIVERSE_MODE = "full_a"
    panel = dl.load_panel(columns=["ts_code", "trade_date", "open", "amount", "adj_factor"])
    basic = dl.load_stock_basic(); uni = dl.select_universe(panel, basic)
    panel = panel[panel.ts_code.isin(uni)]
    st_iv = stg.st_codes_by_date(pd.read_parquet(config.NAMECHANGE))
    p = panel.sort_values(["ts_code", "trade_date"]).copy()
    p["open_hfq"] = p["open"] * p["adj_factor"]
    dates = np.sort(p.trade_date.unique()).astype(str); codes = sorted(p.ts_code.unique())
    op = p.pivot(index="trade_date", columns="ts_code", values="open_hfq").reindex(dates, axis=0).reindex(codes, axis=1)
    amt = p.pivot(index="trade_date", columns="ts_code", values="amount").reindex(dates, axis=0).reindex(codes, axis=1)
    pw = pd.read_parquet(config.CACHE_DIR + "/alpha_pred_full_a.parquet").pivot(
        index="trade_date", columns="ts_code", values="pred").reindex(dates, axis=0).reindex(codes, axis=1)
    oo_daily = op.shift(-1) / op - 1.0
    csi = pd.read_parquet(config.CSI500_RAW); csi["trade_date"] = csi.trade_date.astype(str)
    bo = csi.set_index("trade_date")["open"].reindex(dates).astype(float)
    mkt = bo.shift(-1) / bo - 1.0
    # styles + industry for the neutralisation
    sl = pd.read_parquet(os.path.join(config.CACHE_DIR, "barra_cne5s_factors_full_a.parquet"))
    sl["trade_date"] = sl["trade_date"].astype(str); sl = sl.set_index(["trade_date", "ts_code"])
    style_arr = {s: sl[s].unstack("ts_code").reindex(index=dates, columns=codes).to_numpy()
                 for s in config.STYLE_FACTORS}
    ind = pd.Series(dl.load_industry()).reindex(codes).fillna("UNKNOWN")
    ind_dum = pd.get_dummies(ind).astype(float).to_numpy()
    code_pos = {c: i for i, c in enumerate(codes)}

    caps = [("无上限(基线)", None), ("≤60%", 0.60), ("≤40%", 0.40), ("≤20%", 0.20),
            ("≤10%", 0.10), ("≤5%", 0.05), ("≤3%", 0.03), ("≤2%", 0.02)]
    args = (op, amt, pw, oo_daily, mkt, codes, dates, st_iv, caps, style_arr, ind_dum, code_pos)
    results, navs, expos = sweep(*args, neut=False)                # raw decile L/S
    nresults, nnavs, nexpos = sweep(*args, neut=True)              # style/industry-neutral L/S

    # baseline leg decomposition (fig30, raw)
    dd = navs["无上限(基线)"]; s = stats(dd)
    dd.to_parquet(os.path.join(config.CACHE_DIR, "market_neutral_daily.parquet"), index=False)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = pd.to_datetime([str(v) for v in dd["trade_date"]], format="%Y%m%d")
    ax.plot(x, dd["nav_ls"], color=GREEN, lw=2, label=f"市场中性多空  IR={s['ir']:.2f}")
    ax.plot(x, dd["nav_long"], color=BLUE, lw=1, alpha=0.8, label="多头腿")
    ax.plot(x, dd["nav_short"], color=RED, lw=1, alpha=0.8, label="空头腿(越低=越该做空)")
    ax.plot(x, dd["nav_mkt"], color=GREY, lw=1, ls="--", label="中证500")
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.5)
    ax.set_title(f"图30 市场中性多空净值 [h=7, 无换手上限]  年化{s['ann']*100:.0f}% 回撤{s['mdd']*100:.0f}%")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig30_market_neutral.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    tab = pd.DataFrame(results, columns=["换手上限", "年化收益", "年化波动", "IR", "最大回撤",
                                         "年化双边换手", "多头超额", "空头贡献"]).set_index("换手上限")
    tab.to_csv(os.path.join(config.TABLE_DIR, "market_neutral_turnover.csv"))
    fmt = tab.copy()
    for c in ["年化收益", "年化波动", "最大回撤", "多头超额", "空头贡献"]:
        fmt[c] = (tab[c] * 100).round(1).astype(str) + "%"
    fmt["IR"] = tab["IR"].round(2); fmt["年化双边换手"] = tab["年化双边换手"].round(0).astype(int).astype(str) + "倍"
    with open(os.path.join(config.TABLE_DIR, "market_neutral.md"), "w") as f:
        f.write("### §3.6 市场中性多空 — 换手率上限对比 (h=7, 开盘→开盘, 可融券口径, 2017-2025)\n\n")
        f.write("_单次单腿换手上限; 通过向新 decile 部分移动实现_\n\n")
        f.write(fmt.to_markdown() + "\n")

    # fig31: NAV overlay (colormap, tight cap = warm)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = pd.to_datetime([str(v) for v in navs["无上限(基线)"]["trade_date"]], format="%Y%m%d")
    cmap = plt.cm.plasma(np.linspace(0.05, 0.9, len(caps)))
    for (name, _), col in zip(caps, cmap):
        s = stats(navs[name]); tn = [r[5] for r in results if r[0] == name][0]
        ax.plot(x, navs[name]["nav_ls"], color=col, lw=1.6,
                label=f"{name}: 年化{s['ann']*100:+.0f}% IR{s['ir']:+.2f} 换手{tn:.0f}x")
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.5)
    ax.set_title("图31 换手率上限对市场中性多空净值的影响 [h=7, 全A可融券]")
    ax.legend(loc="upper left", fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig31_turnover_caps.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # fig32: inflection — return & IR vs turnover cap
    labels = list(tab.index); anns = (tab["年化收益"] * 100).to_numpy(); irs = tab["IR"].to_numpy()
    best_i = int(np.nanargmax(irs))
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    bars = ax1.bar(range(len(labels)), anns, color=[GREEN if v >= 0 else RED for v in anns], alpha=0.8)
    ax1.set_ylabel("年化收益 (%)"); ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=9, rotation=15)
    ax2 = ax1.twinx(); ax2.plot(range(len(labels)), irs, color=BLUE, marker="o", lw=2, label="IR")
    ax2.set_ylabel("IR", color=BLUE)
    ax2.scatter([best_i], [irs[best_i]], s=140, facecolors="none", edgecolors=RED, lw=2, zorder=5)
    ax2.annotate(f"拐点 {labels[best_i]}\nIR={irs[best_i]:.2f}", (best_i, irs[best_i]),
                 textcoords="offset points", xytext=(6, -28), fontsize=10, color=RED)
    ax1.set_title("图32 换手率拐点: 年化收益/IR vs 单次换手上限 [越紧→成本越省, 信号越稀]")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig32_turnover_inflection.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # fig33: HIGHEST-RETURN turnover tier vs benchmark index (CSI500)
    best_name = max(results, key=lambda r: r[1])[0]            # r[1] = annual return
    bdd = navs[best_name]; s = stats(bdd)
    mr = bdd["mkt_ret"]; m_ann = mr.mean() * Y; m_vol = mr.std() * np.sqrt(Y)
    m_nav = bdd["nav_mkt"]; m_mdd = (m_nav / m_nav.cummax() - 1).min()
    corr = bdd["ls_ret"].corr(bdd["mkt_ret"])
    bdd[["trade_date", "nav_ls", "nav_mkt", "ls_ret", "mkt_ret"]].to_parquet(
        os.path.join(config.CACHE_DIR, "market_neutral_best.parquet"), index=False)
    cmp = pd.DataFrame({
        f"市场中性多空 ({best_name})": [f"{s['ann']*100:+.1f}%", f"{s['vol']*100:.1f}%",
                                  f"{s['ir']:.2f}", f"{s['mdd']*100:.1f}%", f"{s['win']*100:.1f}%"],
        "中证500 (基准)": [f"{m_ann*100:+.1f}%", f"{m_vol*100:.1f}%", f"{m_ann/m_vol:.2f}",
                        f"{m_mdd*100:.1f}%", "—"]},
        index=["年化收益", "年化波动", "夏普/IR", "最大回撤", "日胜率"])
    cmp.to_csv(os.path.join(config.TABLE_DIR, "market_neutral_best.csv"))
    with open(os.path.join(config.TABLE_DIR, "market_neutral_best.md"), "w") as f:
        f.write(f"### 收益最高换手档 ({best_name}) 市场中性多空 vs 中证500 (2017-2025)\n\n")
        f.write(cmp.to_markdown() + f"\n\n_与中证500 日收益相关性 = {corr:+.2f}（≈0 = 市场中性）_\n")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = pd.to_datetime([str(v) for v in bdd["trade_date"]], format="%Y%m%d")
    ax.plot(x, bdd["nav_ls"], color=GREEN, lw=2,
            label=f"市场中性多空 {best_name}（最高收益档）: 年化{s['ann']*100:+.1f}%  波动{s['vol']*100:.0f}%  "
                  f"回撤{s['mdd']*100:.0f}%  IR{s['ir']:.2f}")
    ax.plot(x, bdd["nav_mkt"], color=GREY, lw=1.6, ls="--",
            label=f"中证500（基准）: 年化{m_ann*100:+.1f}%  波动{m_vol*100:.0f}%  回撤{m_mdd*100:.0f}%")
    ax.axhline(1, color="grey", lw=0.5)
    ax.set_ylabel("累计净值"); ax.set_title(
        f"图33 收益最高换手档({best_name})市场中性多空 vs 中证500 [日收益相关性≈{corr:+.2f}]")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig33_best_vs_benchmark.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # ===== B3: 裸 decile vs 风格/行业中性化 多空 =====
    ntab = pd.DataFrame(nresults, columns=["换手上限", "年化收益", "年化波动", "IR", "最大回撤",
                                           "年化双边换手", "多头超额", "空头贡献"]).set_index("换手上限")
    L = list(tab.index)
    cmp_rows = [[n, f"{tab.loc[n,'年化收益']*100:+.1f}%", f"{tab.loc[n,'IR']:.2f}", f"{expos[n]:.3f}",
                 f"{ntab.loc[n,'年化收益']*100:+.1f}%", f"{ntab.loc[n,'IR']:.2f}", f"{nexpos[n]:.3f}"] for n in L]
    cdf = pd.DataFrame(cmp_rows, columns=["换手上限", "裸 年化", "裸 IR", "裸 净风格敞口",
                                          "中性 年化", "中性 IR", "中性 净风格敞口"]).set_index("换手上限")
    cdf.to_csv(os.path.join(config.TABLE_DIR, "market_neutral_neutralized.csv"))
    nbest = max(nresults, key=lambda r: r[3])[0]          # neutral, by IR
    nb = nnavs[nbest]; ns = stats(nb); ncorr = nb["ls_ret"].corr(nb["mkt_ret"])
    with open(os.path.join(config.TABLE_DIR, "market_neutral_neutralized.md"), "w") as f:
        f.write("### B3 多空腿风格/行业中性化 vs 裸 decile (h=7, 开盘→开盘, 2017-2025)\n\n")
        f.write(cdf.to_markdown() + "\n\n")
        f.write(f"_中性化把净风格敞口压到≈0（裸 ~{expos['≤10%']:.2f} → 中性 ~{nexpos['≤10%']:.3f}）；"
                f"最优中性档 {nbest}：年化{ns['ann']*100:+.1f}% IR{ns['ir']:.2f} 回撤{ns['mdd']*100:.0f}% "
                f"与中证500相关性{ncorr:+.2f}_\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(range(len(L)), tab["IR"].values, color=GREY, marker="o", lw=1.8, label="裸 decile 多空")
    a1.plot(range(len(L)), ntab["IR"].values, color=GREEN, marker="s", lw=2, label="风格/行业中性化")
    a1.set_xticks(range(len(L))); a1.set_xticklabels(L, rotation=20, fontsize=8)
    a1.set_ylabel("IR"); a1.axhline(0, color="grey", lw=0.5); a1.legend(fontsize=9); a1.grid(alpha=0.25)
    a1.set_title("IR vs 换手上限")
    a2.plot(range(len(L)), [expos[n] for n in L], color=GREY, marker="o", lw=1.8, label="裸 净风格敞口")
    a2.plot(range(len(L)), [nexpos[n] for n in L], color=GREEN, marker="s", lw=2, label="中性化 净风格敞口")
    a2.set_xticks(range(len(L))); a2.set_xticklabels(L, rotation=20, fontsize=8)
    a2.set_ylabel("|净风格敞口|"); a2.legend(fontsize=9); a2.grid(alpha=0.25)
    a2.set_title("净风格敞口(越低越中性)")
    fig.suptitle("图34 多空腿风格/行业中性化效果 [h=7, 全A可融券]", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURE_DIR, "fig34_neutralized.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    print(fmt.to_string()); print(f"\n拐点(IR最大) = {labels[best_i]}  IR={irs[best_i]:.2f}  年化={anns[best_i]:.1f}%")
    print(f"收益最高档 = {best_name}: 年化{s['ann']*100:+.1f}% IR{s['ir']:.2f} 回撤{s['mdd']*100:.0f}% | "
          f"中证500 年化{m_ann*100:+.1f}% 回撤{m_mdd*100:.0f}% | 相关性{corr:+.2f}")
    print("\n=== 裸 vs 中性化 ===\n", cdf.to_string())
    print(f"最优中性档 {nbest}: 年化{ns['ann']*100:+.1f}% IR{ns['ir']:.2f} 回撤{ns['mdd']*100:.0f}% 相关{ncorr:+.2f}")


if __name__ == "__main__":
    main()
