"""Build a single PDF report with all figures + tables (CJK via STSong-Light)."""
import csv
import os
import sys
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "report", "figures")
TAB = os.path.join(ROOT, "report", "tables")
OUT = os.path.join(ROOT, "report", "GTJA191_复现报告.pdf")

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
F = "STSong-Light"
NAVY, RED, GREEN, GREY = colors.HexColor("#1f4e79"), colors.HexColor("#c00000"), colors.HexColor("#2e7d32"), colors.HexColor("#666666")

PW, PH = A4
LM = RM = 16 * mm
CW = PW - LM - RM                     # content width

S = {
    "title": ParagraphStyle("t", fontName=F, fontSize=21, leading=26, alignment=TA_CENTER, textColor=NAVY),
    "sub": ParagraphStyle("s", fontName=F, fontSize=11, leading=15, alignment=TA_CENTER, textColor=GREY),
    "h1": ParagraphStyle("h1", fontName=F, fontSize=15, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=F, fontSize=12.5, leading=17, textColor=colors.black, spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("b", fontName=F, fontSize=9.6, leading=14.5, alignment=TA_LEFT, spaceAfter=4),
    "cap": ParagraphStyle("c", fontName=F, fontSize=8.6, leading=11, alignment=TA_CENTER, textColor=GREY, spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=F, fontSize=8.4, leading=10.5),
    "cellc": ParagraphStyle("cellc", fontName=F, fontSize=8.4, leading=10.5, alignment=TA_CENTER),
}
story = []


def P(t, st="body"):
    story.append(Paragraph(t, S[st]))


def sp(h=4):
    story.append(Spacer(1, h))


def callout(html, color=colors.HexColor("#eef3f8")):
    t = Table([[Paragraph(html, S["body"])]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                           ("BOX", (0, 0), (-1, -1), 0.5, NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t); sp(6)


def fig(name, caption, max_h=None):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return
    iw, ih = PILImage.open(path).size
    w = CW; h = w * ih / iw
    if max_h and h > max_h:
        h = max_h; w = h * iw / ih
    story.append(KeepTogether([Image(path, width=w, height=h), Paragraph(caption, S["cap"])]))


def table_from_rows(rows, header=True, colw=None, align_first_left=True, font=8.4):
    data = [[Paragraph(str(c), ParagraphStyle("x", fontName=F, fontSize=font, leading=font + 2,
            alignment=TA_LEFT if (align_first_left and j == 0) else TA_CENTER)) for j, c in enumerate(r)]
            for r in rows]
    t = Table(data, colWidths=colw, hAlign="LEFT")
    sty = [("FONTNAME", (0, 0), (-1, -1), F), ("FONTSIZE", (0, 0), (-1, -1), font),
           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f9fb"))]
    t.setStyle(TableStyle(sty)); story.append(t); sp(6)


def csv_rows(fname):
    with open(os.path.join(TAB, fname), encoding="utf-8") as f:
        return [r for r in csv.reader(f)]


# ============================================================= COVER
P("国泰君安《数量化专题之九十三》复现报告", "title")
P("基于短周期价量特征的多因子选股体系 · §3.3–§3.6 全流程复现与样本外可交易性深挖", "sub")
P("数据 tushare · 全A非ST 5103只 · 窗口 2017–2025（样本外）· 189/191 因子 · BARRA CNE5S 中性化", "sub")
sp(10)
callout("<b>核心结论：</b>研报 2012–2017 样本内年化超额 50.2%、IR 4.67 的“交易型 alpha”，在 "
        "<b>2017–2025 样本外——信号强度还在</b>（多空毛 IR ≈4.6 ≈ 研报 4.67），"
        "<b>但赚不到钱</b>：(1) 预测力主要落在<b>不可交易的隔夜跳空</b>；(2) 收益<b>偏空头</b>而 A 股<b>难融券</b>；"
        "(3) 信号<b>太快</b>、换手成本(~20%/年)吞噬收益。最优可交易形态"
        "（多空+7日+可融券+风格中性+≤5%换手）毛收益也仅 <b>+1.2%/IR 0.43</b>；而盈亏平衡融券费率仅 ≈1.2%、"
        "<b>计入官方 8.35% 融券成本后净收益 ≈ −7%/年</b>——<b>样本外不可盈利</b>。", colors.HexColor("#fbeeea"))

P("0. 设置与口径", "h1")
table_from_rows([
    ["项", "取值"],
    ["时间窗", "2017–2025 = 样本外（研报 2012.01–2017.04，几乎不重叠）"],
    ["股票池", "全A 非ST、≥250日历史 → 5103 只"],
    ["因子", "191 价量Alpha，跳过 Alpha30(需FF3)/Alpha143(SELF) → 189 个"],
    ["RANK 约定", "截面（每日跨股票排序，研报本意）"],
    ["行业 / 风格", "申万一级 SW2021(31类,覆盖99.95%) / BARRA CNE5S 9大类"],
    ["一致预期描述子", "EPIBS/EGIBS/EGIBS_s 用 forecast业绩预告+express业绩快报 代理(覆盖~90%)"],
    ["工程", "16GB机器：信号 memmap 流式(8.4GB)、BARRA分批、检验按日切片"],
], colw=[CW * 0.24, CW * 0.76])

# ============================================================= §3.3
P("1. §3.3 — 191 价量因子体系（189/191）", "h1")
P("因子按附录2算子实现：<font name='STSong-Light'>Alpha_Set.py</font>（逐股参考）+ panel_engine.py"
  "（向量化 date×stock 引擎），二者在 gap-free 股票上逐位等价——验证脚本复现旧基线 147/188 精确吻合"
  "（剩 41 个是 RANK^RANK/CORR(RANK)/REGBETA(SEQUENCE) 类绝对秩值量级差异，生产用截面RANK模式下"
  "每日重排、下游不受影响）。退化因子(alpha21/116/147/146/152)输出全零，由 §3.4.1 的 MIN_OBS_FRAC=0.6 过滤剔除。")

# ============================================================= §3.4.1
story.append(PageBreak())
P("2. §3.4.1 — 单因子显著性 & 有效周期检验", "h1")
P("方法（研报四步）：每日截面 winsorize(MAD±5σ)+标准化 → 对 Z=[行业哑变量|9风格] 取最小二乘残差 → "
  "对非重叠前瞻 d 日收益截面回归得单期因子收益率 f → 年化 FR=252·mean(f)/d、IR=mean(f)/std(f)·√(252/d) → d=1..5。")
P("表7 / 表8（全A，对照研报）", "h2")
t7 = csv_rows("table7_mean_factor_return.csv"); t8 = csv_rows("table8_mean_IR.csv")
rows = [["指标 \\ 预测周期", "DAY1", "DAY2", "DAY3", "DAY4", "DAY5"],
        ["本次 |IR|均值"] + t8[2][1:], ["研报 IR均值"] + t8[3][1:],
        ["本次 signed IR均值"] + t8[1][1:],
        ["本次 signed FR均值"] + t7[1][1:], ["研报 FR均值"] + t7[2][1:]]
table_from_rows(rows, colw=[CW * 0.34] + [CW * 0.132] * 5)
P("结论：① 预测力量级被复现（|IR| 2.25→1.78 ≈ 研报 2.57→1.05）；② 随周期单调递减被复现（图19，吻合"
  "T+4 为预测极限）；③ 样本外符号发散——signed FR/IR≈0 而 |IR| 高，IR 分布宽且对称于0（图14-18），"
  "约一半因子相对其 2010–2017 原始方向符号翻转：因子仍有效，但原始符号不可靠。有效因子 170/189；相关性剔除 170→66。")
fig("fig09-13_FR_dist.png", "图9-13 不同预测周期下因子年化收益率分布")
fig("fig14-18_IR_dist.png", "图14-18 不同预测周期下因子收益率IR分布（分布宽、对称于0 = 样本外符号发散）")
fig("fig19_horizon_trend.png", "图19 不同周期下因子收益率均值(柱)及|IR|均值(线)变化——单调递减", max_h=210)
fig("fig20_corr_dist.png", "图20 因子收益率相关系数分布（剔除后）", max_h=210)

# ============================================================= §3.4.2/3
story.append(PageBreak())
P("3. §3.4.2 / §3.4.3 — 组合 Alpha 模型预测能力", "h1")
P("方法：每日正交化全部 alpha → 前瞻1日收益对[行业+风格]残差化得真实 alpha 截面 ω；岭回归(λ=0.1·n，"
  "对研报OLS的偏离：样本外189因子共线，OLS随因子增多发散)得因子收益率 f；用过去120日 f 均值×当期暴露"
  "预测 Ê(ω)；模型 IC=corr(Ê,ω)。")
P("表9 模型预测系数（§3.4.2）", "h2")
t9 = csv_rows("table9_model_ic.csv")
table_from_rows([["", "日度IC均值", "t (Alpha Model)", "日度IC胜率"],
                 ["本次复现 (全A)", t9[1][1], t9[1][2], t9[1][3]],
                 ["研报 (2012-2017)", t9[2][1], t9[2][2], t9[2][3]]],
                colw=[CW * 0.34, CW * 0.22, CW * 0.22, CW * 0.22])
P("表10 因子个数 vs 模型 IC（§3.4.3）", "h2")
t10 = csv_rows("table10_factor_count.csv")
hdr = ["因子数"] + [r[0] for r in t10[1:]]; ic = ["模型IC"] + [r[1] for r in t10[1:]]
table_from_rows([hdr, ic], colw=[CW * 0.16] + [CW * 0.105] * 8)
P("胜率 95.1% ≥ 研报 88.7%：组合模型几乎每天预测对 alpha 截面方向；模型 IC 随因子个数单调递增、边际递减"
  "（与研报 §3.4.3 同形）。<b>但注意：这是收盘价对收盘价的 IC，下一节揭示它被隔夜跳空主导、不可落地。</b>")
fig("fig21_model_ic.png", "图21 模型预测系数 (IC of Alpha Model) 时序", max_h=210)
fig("fig22_factor_count.png", "图22 因子个数与模型预测能力（柱=IC, 线=t）——单调↑、边际递减", max_h=210)

# ============================================================= §3.6 main
story.append(PageBreak())
P("4. §3.5 / §3.6 — 风格中性策略 + 可交易性深挖（核心）", "h1")
P("§3.5 方法层：目标函数加单次换仓罚函数 Max w·E(α) − Tc·|w−w_prev|。§3.6 据此组合优化回测（scipy "
  "HiGHS 线性规划）：风格±0.1、行业±5%(相对中证500)、长仓单票≤5%、Σw=1；次日开盘成交(split-day,无隔夜"
  "前视)、剔除停牌/ST(点-in-time改名史)、基准中证500、Tc=0.3%、每2日调仓。")
P("4.1 主回测（表11/12）— 样本外亏损", "h2")
t11 = csv_rows("table11_strategy_perf.csv")
table_from_rows([["指标", "本次复现 (vs 中证500)", "研报 (2012-2017)"]] + [[r[0], r[1], r[2]] for r in t11[1:]],
                colw=[CW * 0.34, CW * 0.36, CW * 0.30])
t12 = csv_rows("table12_yearly.csv")
table_from_rows([t12[0]] + t12[1:], colw=[CW * 0.16] + [CW * 0.28] * 3)
P("分年度除 2024 年 +5.0% 外逐年为负（2022 最差 −57.5%）。", "body")
fig("fig23_nav.png", "图23 组合累计净值", max_h=190)
fig("fig24_excess.png", "图24 组合累计超额收益", max_h=190)
fig("fig25_turnover.png", "图25 组合单次换仓比率", max_h=190)

# ============================================================= §3.6 deep
story.append(PageBreak())
P("4.2 诊断①：alpha 主要在隔夜跳空里（图27）", "h2")
P("把组合模型预测 IC 按日内时段分解：收盘→收盘 +0.107；收盘→次日开盘(隔夜跳空) <b>+0.229</b>(不可交易，"
  "次开盘成交时已错过)；次日开盘→收盘(日内) <b>−0.012</b>(反转)。→ §3.4 的高 IC 被隔夜跳空主导；按"
  "次日开盘成交口径当晚跳空抓不到。<b>这解释了 §3.4 模型 IC 0.115/胜率95% 与策略亏损的反差。</b>")
fig("fig27_ic_decomp.png", "图27 组合模型IC分解：隔夜(不可交易,+0.229) vs 日内(可交易,−0.012)", max_h=220)
P("4.3 诊断②：开盘→开盘下信号仍强（多空），但长仓拿不到（图28）", "h2")
P("跳空预测力衰减地延续到下一晚(pred 对 close_{t+1}→open_{t+2} 的 IC 仍 +0.11)，故持仓开盘→开盘可得可"
  "交易信号：十分位多空(毛) 年化 <b>+28.2% / IR +4.56 ≈ 研报 4.67</b>——信号强度被复现！但 alpha 不对称、"
  "集中在空头，A 股长仓难融券，研报式长仓组合(含中性优化器 −24.5%)拿不到空头收益。多头 top50/100/200 扣费 "
  "−12%/−36%/−42%。<b>信号在(隔夜+空头)，不在(长仓可交易的截面)。</b>")
fig("fig28_tradability.png", "图28 可交易性：多空信号强(IR≈研报) 但长仓不可得", max_h=220)
story.append(PageBreak())
P("4.4 最优持有期：IC 衰减曲线（图29）", "h2")
P("信号 IC 不是单调衰减——开盘→开盘累积窗口使其先升后降、h≈7 天达峰(0.072)。毛 IR∝IC√(252/h) 偏好短持有，"
  "但扣费后 h=1–2 被换手成本摧毁(净IR −7.4/−1.9)。净IR 在 h≈7 达峰(+1.70) → 最优持有期 ≈ 7 个交易日。")
hp = csv_rows("holding_period.csv")
hrow = ["持有天数"] + [r[0] for r in hp[1:]]
icrow = ["信号IC"] + [r[1] for r in hp[1:]]
nirow = ["净IR(扣费)"] + [str(round(float(r[4]), 2)) for r in hp[1:]]
table_from_rows([hrow, icrow, nirow], colw=[CW * 0.16] + [CW * 0.093] * 9, font=8)
fig("fig29_holding_period.png", "图29 IC衰减曲线与最优持有期(7天)：太短被成本摧毁、太长IR下降", max_h=210)
P("4.5 市场中性多空回测（h=7，可融券口径）（图30）", "h2")
P("逐日盯市、美元中性、每7日调仓、开盘→开盘、扣费，universe=可融券口径(流动前1500、非ST)：年化 −1.7%、"
  "IR −0.14、回撤 −47%、年化双边换手 125倍。<b>腿分解决定性</b>：多头超额(vs500) −14.5%(长仓是负担)，"
  "空头贡献 +31.4%(edge全在此)。与 4.4 扫描 +1.70 IR 的差异=universe(那条用全A含不可融券小微盘高估空头)。")
fig("fig30_market_neutral.png", "图30 市场中性多空净值(绿) + 多头腿/空头腿/中证500 分解（对数轴）", max_h=200)

story.append(PageBreak())
P("4.6 换手率拐点（图31/32）— §3.5 不可能三角的实证", "h2")
P("基线换手 125倍/年。对单次单腿换手设上限(向新decile部分移动)：倒 U 型拐点——太松成本主导(IR −0.14)，"
  "太紧信号稀释到≈0。<b>IR 拐点=≤10%(0.24)，收益拐点=≤20%(+1.4%)</b>。最优区间单次单腿 10–20%(≈双边14–29倍)，"
  "成本~20%→~3%/年，回撤 −47%→−14%。但拐点处也只有 +1.3%/IR 0.24，且仍主要靠空头。")
mn = csv_rows("market_neutral_turnover.csv")


def pct(x):
    return f"{float(x)*100:.1f}%"


mrows = [["换手上限", "年化收益", "IR", "最大回撤", "年化双边换手", "多头超额", "空头贡献"]]
for r in mn[1:]:
    mrows.append([r[0], pct(r[1]), f"{float(r[3]):.2f}", pct(r[4]), f"{float(r[5]):.0f}倍", pct(r[6]), pct(r[7])])
table_from_rows(mrows, colw=[CW * 0.16, CW * 0.14, CW * 0.12, CW * 0.14, CW * 0.16, CW * 0.14, CW * 0.14], font=8)
fig("fig31_turnover_caps.png", "图31 换手率上限对市场中性多空净值的影响（越紧越靠上）", max_h=200)
fig("fig32_turnover_inflection.png", "图32 换手率拐点：年化收益(柱)/IR(线) vs 单次换手上限——倒U，IR峰在≤10%", max_h=210)

P("4.7 收益最高换手档（≤20%）vs 中证500（图33）", "h2")
P("单独挑出换手率扫描中<b>年化收益最高的一档（≤20%，+1.4%）</b>与基准中证500 对比：绝对收益低于指数"
  "(+1.4% vs +4.7%)，但<b>波动仅 1/3（6.6% vs 23%）、回撤仅 1/3（−16% vs −42%）、夏普相当（0.21 vs 0.20）、"
  "与指数日收益相关性 ≈ −0.03（真正市场中性）</b>。即：作为<b>低波动、低回撤、零相关</b>的分散化配置仍有价值，"
  "但绝对收益薄、且收益主要靠空头（需融券）。")
mb = csv_rows("market_neutral_best.csv")
table_from_rows([["指标", mb[0][1], mb[0][2]]] + [[r[0], r[1], r[2]] for r in mb[1:]],
                colw=[CW * 0.28, CW * 0.40, CW * 0.32])
fig("fig33_best_vs_benchmark.png", "图33 收益最高换手档(≤20%)市场中性多空 vs 中证500——低波动/低回撤/零相关", max_h=215)

P("4.8 多空腿风格/行业中性化（图34）— IR 翻倍", "h2")
P("裸 decile 多空带风格/行业敞口(净~0.186)。再平衡时把 Ê 对[行业哑变量+9风格]残差化后再分 decile，"
  "净风格敞口压到 ~0.063(−66%)，风险下降、<b>IR 显著提升</b>：≤5% 档 IR 从裸 0.21 提到 <b>0.43</b>(近翻倍)、"
  "峰值左移到更紧的 ≤5%。<b>最优配置(中性化+≤5%换手)：年化 +1.2%、IR 0.43、最大回撤 −6%、与中证500 相关性 "
  "−0.00</b>——收益相近但风险大降(中性化提风险调整后收益、不提原始收益)。这是目前最好的可交易配置，"
  "但绝对收益仍薄、且靠空头(需融券)。")
mn2 = csv_rows("market_neutral_neutralized.csv")
table_from_rows([["换手上限", "裸 年化", "裸 IR", "裸 敞口", "中性 年化", "中性 IR", "中性 敞口"]] +
                [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in mn2[1:]],
                colw=[CW * 0.17, CW * 0.14, CW * 0.12, CW * 0.13, CW * 0.15, CW * 0.13, CW * 0.16], font=7.8)
fig("fig34_neutralized.png", "图34 多空腿风格/行业中性化：左=IR(绿>灰), 右=净风格敞口(0.186→0.063)", max_h=200)

P("4.9 计入融券成本（图35）— 决定性反转", "h2")
P("策略 $1 多/$1 空，做空需付借券利息（中国官方融券费率 ≈8.35%/年，难借小票更高），每日 carry。"
  "对最优中性档(≤5%)按不同费率重算：<b>盈亏平衡融券费率仅 ≈1.2%（= 毛收益），而官方 ≈8.35%</b> → "
  "计入融券后<b>净收益 ≈ −7.1%/年、IR −2.56</b>，策略<b>决定性转负</b>。这给整条链画上句号：即便层层优化"
  "（隔夜→多空→7日→市场中性→换手拐点→风格中性化，毛 IR 0.43），<b>可交易毛 edge 远低于真实做空成本</b>，"
  "样本外不可盈利——根因仍是收益偏空头、A 股做空贵且受限。")
bt = csv_rows("market_neutral_borrow.csv")
table_from_rows([["融券费率(年化)", "年化净收益", "IR", "最大回撤"]] + [[r[0], r[1], r[2], r[3]] for r in bt[1:]],
                colw=[CW * 0.30, CW * 0.26, CW * 0.20, CW * 0.24])
fig("fig35_borrow_cost.png", "图35 融券成本敏感性：盈亏平衡费率≈1.2% << 官方8.35% → 计入后净≈−7%/年", max_h=210)

# ============================================================= research extension
story.append(PageBreak())
P("5. 研究延伸：隔夜 alpha 的理论上限（“冰山”，图36）", "h1")
P("用同一个 Ê 信号，比较三个执行窗口的十分位多空<b>毛</b>收益。<b>递减阶梯</b>：隔夜理论上限 "
  "<b>+209%/年（IR 36）</b> → 开盘→开盘可交易毛 +56% → 风格中性+换手成本后毛 +1.2% → 计入官方融券 "
  "8.35% 后 −7%。图36 对数轴：隔夜/收盘线复利到 ~10⁸、可交易线只到 ~10²，相差 6 个数量级。"
  "这是研报“我们只看到了冰山一角”的定量版：<b>水下（隔夜跳空）巨大但摸不到</b>（信号需收盘数据算、"
  "却要收盘成交），<b>水上（可交易净）为负</b>。整套价量 alpha 的预测力真实存在（IR 36），只是几乎"
  "全部落在不可交易的时段 + 空头，再被做空成本封死。")
oc = csv_rows("overnight_ceiling.csv")
table_from_rows([["执行窗口", "多空年化(毛)", "IR"]] + [[r[0], r[1], r[2]] for r in oc[1:]],
                colw=[CW * 0.50, CW * 0.28, CW * 0.22])
fig("fig36_overnight_ceiling.png", "图36 交易型alpha的“冰山”: 隔夜理论上限(红,+209%) >> 可交易(绿,+56%毛) [对数轴]", max_h=230)

# ============================================================= chain
story.append(PageBreak())
P("6. 贯穿全程的逻辑链与最终判断", "h1")
callout("§3.4 组合模型 IC 0.115/胜率95%（高！）<br/>"
        "&nbsp;&nbsp;└─ 但是收盘价对收盘价的 IC（图27）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 被不可交易的隔夜跳空主导（隔夜+0.23, 日内−0.01）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 次日开盘成交 → 长仓现实回测 −24.5%（表11）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 但信号延续到下一晚 → 开盘→开盘多空 IR +4.56 ≈ 研报（图28）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 偏空头，A股难融券 → 长仓拿不到<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 最优持有≈7天（IC拐点，图29）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 市场中性可融券口径≈持平（图30）<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ 换手拐点 ≤10%：IR 0.24（图31/32）—— 仍很薄")
P("最终判断：研报方法在样本外可完整跑通、单因子/模型 IC 的量级仍在、多空信号强度与研报相当；但可落地的"
  "超额收益基本消失，根因是该价量 alpha 的预测力落在 (隔夜 + 空头) 两个 A 股长仓难以触及的角落，且信号"
  "过快、成本侵蚀严重。这是“交易型 alpha”研究最经典、也最值得记取的样本外教训。")
P("局限与可继续方向：样本外窗口是反号主因(如需样本内对照需回补面板到2010–2017)；一致预期用公司指引代理；"
  "空头依赖融券标的与成本(未计融券利息)；可继续——计入融券成本的净空头收益 / 小微盘活跃年条件配置 / "
  "把隔夜alpha单列为学术口径隔夜策略量化其上限。")

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=14 * mm, bottomMargin=14 * mm,
                        title="GTJA191 复现报告", author="quant research")


def footer(canvas, d):
    canvas.saveState(); canvas.setFont(F, 8); canvas.setFillColor(GREY)
    canvas.drawCentredString(PW / 2, 8 * mm, f"GTJA191 价量多因子体系复现报告（样本外 2017–2025）  ·  第 {d.page} 页")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
