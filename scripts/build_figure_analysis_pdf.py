"""Generate a figure-by-figure analysis report (out-of-sample vs in-sample).

For every figure produced by the reproduction, embed the OOS (2017-2025) and the
in-sample (2012-2017, full 9-style) versions stacked, with a four-part commentary:
含义 / 样本外读数 / 样本内读数 / 对比结论. Output report/GTJA191_图表分析报告.pdf.
"""
import os
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
OOS = os.path.join(ROOT, "report", "figures")
IS = os.path.join(ROOT, "report", "figures_insample_full")
OUT = os.path.join(ROOT, "report", "GTJA191_图表分析报告.pdf")

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
F = "STSong-Light"
NAVY, RED, GREEN, GREY = (colors.HexColor("#1f4e79"), colors.HexColor("#c00000"),
                          colors.HexColor("#2e7d32"), colors.HexColor("#666666"))
PW, PH = A4
LM = RM = 15 * mm
CW = PW - LM - RM
S = {
    "title": ParagraphStyle("t", fontName=F, fontSize=20, leading=25, alignment=TA_CENTER, textColor=NAVY),
    "sub": ParagraphStyle("s", fontName=F, fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=GREY),
    "h1": ParagraphStyle("h1", fontName=F, fontSize=13.5, leading=18, textColor=NAVY, spaceBefore=10, spaceAfter=3),
    "lab": ParagraphStyle("lab", fontName=F, fontSize=9, leading=11, alignment=TA_CENTER, textColor=GREY, spaceBefore=1, spaceAfter=1),
    "body": ParagraphStyle("b", fontName=F, fontSize=9.4, leading=14, alignment=TA_LEFT, spaceAfter=3),
}
story = []


def P(t, st="body"):
    story.append(Paragraph(t, S[st]))


def img(path, max_h):
    if not os.path.exists(path):
        return None
    iw, ih = PILImage.open(path).size
    w = CW; h = w * ih / iw
    if h > max_h:
        h = max_h; w = h * iw / ih
    return Image(path, width=w, height=h)


def callout(html, color):
    t = Table([[Paragraph(html, S["body"])]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.5, NAVY),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(t); story.append(Spacer(1, 5))


def entry(fname, title, meaning, oos, ins, compare, star=False, mh=150):
    story.append(PageBreak())
    P(("★ " if star else "") + title, "h1")
    P(f"<b>【图表含义】</b>{meaning}")
    io = img(os.path.join(OOS, fname), mh); ii = img(os.path.join(IS, fname), mh)
    blk = []
    if io: blk += [io, Paragraph("▲ 样本外 2017-2025（out-of-sample）", S["lab"])]
    if ii: blk += [ii, Paragraph("▲ 样本内 2012-2017（in-sample, 完整9风格）", S["lab"])]
    story.append(KeepTogether(blk))
    story.append(Spacer(1, 4))
    P(f"<b>【样本外读数】</b>{oos}")
    P(f"<b>【样本内读数】</b>{ins}")
    P(f"<b>【对比结论】</b>{compare}")


# ---------------------------------------------------------------- cover
P("GTJA191 价量多因子体系 — 图表分析报告", "title")
P("每张图逐一配详细说明 · 样本外(2017-2025) vs 样本内(2012-2017, 完整9风格) 对照", "sub")
P("国泰君安《数量化专题之九十三》复现 · 全A非ST · 189/191 价量Alpha · BARRA CNE5S 中性化 · tushare 数据", "sub")
story.append(Spacer(1, 8))
callout("<b>阅读指引：</b>本报告把复现产出的 19 张图逐一列出，每张给出 OOS 与样本内两个版本 + 四段说明"
        "（含义 / 样本外读数 / 样本内读数 / 对比结论）。<b>贯穿主线：</b>因子显著性(|IR|、模型IC)在两窗都"
        "与研报量级相当；但<b>可落地的超额收益样本内为正、样本外为负</b>——根因是价量 alpha 的预测力落在"
        "不可交易的隔夜跳空 + 偏空头，且信号在 ~2016 后衰减。★ 标记的 4 张图（27/33/35/36）是核心。",
        colors.HexColor("#eef3f8"))
P("图表分组：§3.4.1 单因子检验(图9-20) · §3.4.2/3 模型IC(图21-22) · §3.6 策略回测(图23-25) · "
  "§3.6 实战深挖(图27-36)。", "body")

# ---------------------------------------------------------------- §3.4.1
P("一、§3.4.1 单因子有效性与有效周期检验", "h1")
entry("fig09-13_FR_dist.png", "图9-13 不同预测周期下因子年化收益率(FR)分布",
      "5 个子图分别是预测周期 d=1..5 下、全部有效因子的<b>年化因子收益率</b>直方图；红虚线为均值。"
      "研报的 191 个因子是预先定向的（多数应为正）。",
      "分布中心落在 0 附近略偏负（signed FR 均值 −0.4~−0.6%/年），远离研报的 +5~8%——因子的原始方向在样本外已失效。",
      "分布中心同样贴近 0（d1 +0.95%、d2-5 略负）——即便样本内，带符号的因子收益也不高。",
      "两窗的<b>带符号</b>因子收益都≈0（不同于研报 6.85%），因为因子方向在两窗都不稳定。但"
      "“带符号≈0”不等于“无信号”——真正的显著性看 |IR|（图14-18/图19）。")
entry("fig14-18_IR_dist.png", "图14-18 不同预测周期下因子收益率 IR 分布",
      "5 个子图是 d=1..5 下全部因子的<b>信息比率(IR)</b>直方图。IR=年化因子收益率/其波动。",
      "分布<b>宽且大致对称于 0</b>，IR 跨越 ±5（d1 甚至到 +15）——约一半因子 IR 为正、一半为负。",
      "分布同样宽、对称于 0。",
      "这是全报告一个关键现象：因子<b>显著（|IR| 大）但符号发散</b>（正负各半），故 signed 均值≈0 而 "
      "|IR| 均值高（~2）。样本内外皆如此——符号不可靠是这套价量因子的固有特征，不是样本外才出现。")
entry("fig19_horizon_trend.png", "图19 因子收益率均值及 |IR| 均值随预测周期变化",
      "柱=年化因子收益率均值(%)，红线=<b>|IR| 均值</b>，横轴 d=1..5。这是研报表7/表8 的可视化。",
      "|IR| 从 d1 的 2.25 单调降到 d5 的 1.78——<b>复现研报“预测力随周期递减、T+4 为极限”的核心结论</b>。",
      "|IR| d1 异常偏低(0.70，或与非重叠采样/中性化口径有关)，d2 起 2.63 递减到 2.26。",
      "两窗 |IR| 量级都与研报（2.57→1.05）相当，且都呈短周期最强、随 d 衰减——<b>因子显著性被稳健复现</b>。")
entry("fig20_corr_dist.png", "图20 因子收益率两两相关系数分布（剔除后）",
      "对高相关因子去重后，剩余因子收益率序列两两相关系数的直方图（研报以此说明因子间独立性）。",
      "分布较宽，|corr|<0.1 占比约 16%（研报 ~94%）。",
      "分布形态类似。",
      "本复现的因子相关性明显高于研报——可能源于因子定义/算子实现口径差异 + 样本外效应；这也是后续"
      "组合模型改用岭回归（而非研报 OLS）的原因（高共线下 OLS 发散）。")

# ---------------------------------------------------------------- §3.4.2/3
P("二、§3.4.2 / §3.4.3 组合 Alpha 模型的预测能力", "h1")
entry("fig21_model_ic.png", "图21 组合模型预测系数 (IC of Alpha Model) 时序",
      "组合 Alpha 模型每日的截面预测系数 IC 时序（浅线）+ 30日均线（深）+ 均值线。IC=corr(模型预测的"
      "alpha截面, 实际alpha截面)。",
      "日度 IC 均值 <b>0.115</b>、t=65.8、<b>胜率 95.1%</b>——组合模型几乎每天都预测对 alpha 截面方向。",
      "日度 IC 均值 <b>0.102</b>、t=38.9、<b>胜率 87.9%</b>（≈研报 88.7%）。",
      "两窗的模型 IC 都高且稳定为正，胜率与研报（88.7%）吻合——组合模型的<b>收盘价对收盘价</b>预测力很强。"
      "但请记住：这个高 IC 被隔夜跳空主导、不可落地（见图27），是全报告最大的“陷阱”。")
entry("fig22_factor_count.png", "图22 因子个数与模型预测能力的关系",
      "柱=模型 IC，红线=IC 显著性(t)，横轴=按 |IR| 排序纳入的因子数(20→170)。研报表10 的可视化。",
      "模型 IC 随因子数 <b>单调递增、边际递减</b>：0.090(20只)→0.115(170只)。",
      "同样单调递增、边际递减：0.073(20只)→0.102(170只)。",
      "两窗都复现研报 §3.4.3 的结论——加因子持续提升预测力但边际递减，说明因子体系尚未触及预测边界。")

# ---------------------------------------------------------------- §3.6 strategy
P("三、§3.6 风格中性多因子选股策略（长仓组合优化回测）", "h1")
entry("fig23_nav.png", "图23 组合累计净值 vs 基准",
      "风格/行业中性、长仓、次日开盘成交的组合优化策略的累计净值，与基准中证500 对比。",
      "组合净值持续<b>跑输</b>基准（年化超额 −24.5%）。",
      "组合净值<b>跑赢</b>基准（年化超额 +9.0%）。",
      "同一套长仓组合优化：样本外亏、样本内赚——这是策略层面 A1 验证的第一眼直观。")
entry("fig24_excess.png", "图24 组合累计超额收益",
      "组合相对基准的累计超额收益曲线（净值化）。",
      "持续下行到 ~0.1（年化超额 −24.5%、IR −1.59、最大回撤 −90%）。",
      "上行到 ~1.5（年化 +9.0%、IR 0.63）；<b>2013-2015 一路强涨、2016-2017 见顶回落</b>。",
      "样本内曲线的“见顶回落”让<b>衰减起点(~2016)一目了然</b>——不是 2017 才突然失效，而是 2016 已开始。")
entry("fig25_turnover.png", "图25 组合单次换仓比率",
      "每次调仓（每2日）的双边换手率时序。",
      "单次双边换手 ~75%、年化 ~94 倍——很高。",
      "单次双边换手 ~88%、年化 ~111 倍——同样很高。",
      "两窗换手都远高于研报（单次 38%、年化 48 倍）；高换手→高成本，是后续要用“换手上限”优化的动因（图31/32）。")

# ---------------------------------------------------------------- §3.6 deep-dive
P("四、§3.6 实战可交易性深挖（核心）", "h1")
entry("fig27_ic_decomp.png", "图27 组合模型 IC 分解：隔夜(不可交易) vs 日内(可交易)",
      "把组合模型 IC 按成交时段拆解：收盘→收盘(预测目标)、收盘→次日开盘(<b>隔夜跳空</b>)、次日开盘→收盘"
      "(<b>当日日内</b>)。按研报“次日开盘成交”，隔夜那段抓不到、只能吃日内。",
      "隔夜 IC <b>+0.229</b>（不可交易）、日内 IC <b>−0.012</b>（可交易但反转）——可落地部分为负。",
      "隔夜 IC <b>+0.178</b>、日内 IC <b>+0.016（正！）</b>——可落地部分为正。",
      "两窗的高 IC 都被隔夜跳空主导；区别在<b>可交易的日内 IC：样本内 +0.016、样本外 −0.012</b>。"
      "策略 P&L 正好随它翻号（+9% → −24%）——<b>这是样本外失效的精确机制</b>。", star=True, mh=175)
entry("fig28_tradability.png", "图28 可交易性：多空信号强 vs 长仓不可得",
      "同一预测信号下，十分位<b>多空</b>组合（毛，衡量信号强度）与<b>长仓</b> top-N（扣费，vs 中证500）的年化对比。",
      "多空毛 IR <b>+4.56 ≈ 研报 4.67</b>（信号强度被复现！）；但长仓 top50/100/200 扣费 −12%/−36%/−42%"
      "——alpha 偏空头、长仓拿不到。",
      "多空信号同样强；长仓侧因 2012-2015 小盘暴涨含较大 compounding artifact（读数偏高、不稳），干净的"
      "可交易结论应看市场中性（图30/33）。",
      "核心信息（样本外尤其清晰）：<b>信号在多空/空头侧，长仓拿不到</b>——A 股长仓为主、难融券，正是长仓"
      "组合失败的根因。", star=False, mh=170)
entry("fig29_holding_period.png", "图29 IC 衰减曲线与最优持有期",
      "信号 IC（蓝，左轴）与扣费净 IR（绿，右轴）随持有天数 h 的变化，用于在“信号强度 vs 换手成本”间找最优持有期。",
      "IC 在 h≈7 达峰(0.072)；净 IR 也在 h≈7 达峰(1.70)——<b>最优持有约 7 个交易日</b>（h=1 因换手成本被摧毁到 IR −7.4）。",
      "曲线更嘈杂，净 IR 在 h≈20 达峰(2.28)；样本内信号更强、对持有期不敏感。",
      "两窗都显示“太短被成本吃掉、需持有数日”；最优持有期样本外≈7天、样本内更长——后续市场中性统一用 h=7。")
entry("fig30_market_neutral.png", "图30 市场中性多空净值 + 多头/空头腿分解",
      "美元中性的十分位多空（h=7、开盘→开盘、可融券口径）累计净值（绿），并分解为多头腿、空头腿、中证500（对数轴）。",
      "市场中性净值≈持平（基线 −1.7%）；<b>多头腿是负担(−14.5%)、空头腿贡献全部 edge(+31.4%)</b>。",
      "市场中性净值稳步上行（+16.0%）；多头/空头都更有效、且回撤远小。",
      "样本外“收益全靠空头”暴露了 A 股难融券的痛点；样本内则是一个稳健上行的市场中性净值——差异巨大。", star=False, mh=180)
entry("fig31_turnover_caps.png", "图31 换手率上限对市场中性净值的影响",
      "对单腿单次换手设不同上限（无上限→≤2%，颜色越暖越紧），叠加各档市场中性净值（对数轴）。",
      "越紧的上限净值越靠上——策略被<b>换手成本主导</b>，砍换手就改善。",
      "各档都盈利，紧档回撤更小、净值更稳。",
      "两窗都印证研报 §3.5“换手率与超额的最优平衡”；但样本外是“砍成本才不亏”，样本内是“本就盈利、砍成本更稳”。")
entry("fig32_turnover_inflection.png", "图32 换手率拐点：收益/IR vs 换手上限",
      "年化收益(柱)与 IR(线) 随单腿换手上限的变化——典型倒 U 型（太松成本主导、太紧信号稀释）。",
      "IR 拐点在 ≤10%（0.24）、收益拐点 ≤20%（+1.4%）。",
      "IR 拐点在 ≤20-40%（<b>2.16-2.19</b>）、收益随上限放松单调升到 +16%。",
      "同样的倒 U 结构，但<b>量级天壤之别</b>：样本外最优 IR 仅 0.24，样本内高达 2.2。")
entry("fig33_best_vs_benchmark.png", "图33 收益最高换手档 市场中性多空 vs 中证500",
      "收益最高一档的市场中性净值（绿）与基准中证500（灰）对比；标注两者年化/波动/回撤/相关性。",
      "市场中性 +1.4%/波动6.6%/回撤−16%，与指数近零相关(−0.03)——低波动零相关但<b>绝对收益薄</b>。",
      "市场中性 +16.0%/波动9.1%/回撤−6%，指数 +17.3%/波动31.5%/回撤−54.6%，相关 +0.11。",
      "样本内是一个<b>优秀的市场中性产品</b>：同等收益、<b>1/3 波动、1/9 回撤、3× 夏普(1.75 vs 0.55)</b>、"
      "近零相关，几乎无视 2015 股灾；样本外则只剩“低波动但没收益”。", star=True, mh=185)
entry("fig34_neutralized.png", "图34 多空腿风格/行业中性化效果",
      "左：IR vs 换手上限（裸 decile vs 风格/行业中性化）；右：实测净风格敞口（越低越中性）。",
      "中性化把净风格敞口从 0.186 压到 0.063，IR 略升（≤10% 档 0.24→0.39）、收益几乎不变。",
      "净敞口 0.121→0.063，IR/收益几乎不变（≤40% 档 2.27→2.19）。",
      "两窗结论一致：<b>风格/行业中性化能把敞口压到≈0 而几乎不损收益</b>——多空价差本就不太依赖风格暴露。")
entry("fig35_borrow_cost.png", "图35 融券借券成本敏感性",
      "最优中性档在不同融券费率下的年化净收益(绿)/IR(灰)；标注盈亏平衡费率与官方 8.35% 融券费率。",
      "盈亏平衡费率仅 ~1.2% <b>≪ 官方 8.35%</b> → 计入官方融券后<b>净 −7.1%/IR −2.56（转负）</b>。",
      "盈亏平衡费率 ~12% <b>&gt; 官方 8.35%</b> → 计入官方融券后<b>净 +3.6%/IR 0.66（仍为正）</b>。",
      "<b>这是决定性分水岭</b>：同一策略、同一 8.35% 融券成本假设——样本内扣完融券仍赚 +3.6%、样本外亏 −7%。"
      "这是 alpha 真实衰减最硬的证据。", star=True, mh=175)
entry("fig36_overnight_ceiling.png", "图36 交易型 alpha 的“冰山”：隔夜理论上限 vs 可交易",
      "同一信号在三个执行窗口（隔夜 close→open、收盘→收盘、开盘→开盘）的十分位多空<b>毛</b>净值（对数轴），"
      "量化“信号有多大、水上能拿到多少”。",
      "隔夜(不可交易) +209%/IR 36.5 ≫ 收盘→收盘 +186% ≫ 可交易开→开 +56%/IR 6.4。",
      "隔夜 +129%/IR 4.03、收盘 +192%、开→开 +208%(毛,波动大 IR 0.45)。",
      "两窗都印证研报“只看到冰山一角”：<b>水下（隔夜跳空）巨大但摸不到</b>。区别在“水上可交易的那层”"
      "经中性+换手+融券后，样本内为正(+3.6%)、样本外为负(−7%)。", star=True, mh=180)

# ---------------------------------------------------------------- summary
story.append(PageBreak())
P("五、总览：19 张图串起的一条主线", "h1")
callout("① 因子显著性(图19 |IR|、图21-22 模型IC/胜率88%)在两窗都与研报量级相当 → 管线正确、方法可复现。<br/>"
        "② 但图27 揭示高 IC 被<b>不可交易的隔夜跳空</b>主导；图28/30 显示信号偏<b>空头</b>、长仓拿不到。<br/>"
        "③ 图29 定最优持有 7 天、图31-32 定换手拐点、图34 做风格中性——把可交易信号榨到最净。<br/>"
        "④ 图33/35 是分水岭：<b>样本内</b>市场中性对指数 3× 夏普、1/9 回撤，扣完 8.35% 融券仍 +3.6%；"
        "<b>样本外</b>同一构造扣融券后 −7%。<br/>"
        "⑤ 图24 显示衰减起于 ~2016。<b>结论：研报方法对、2012-2017 是真实可盈利的市场中性策略，2017 起"
        "因 alpha 衰减而失效。</b>", colors.HexColor("#eef3f8"))
P("配套文档：完整数据/方法/工程见 SUMMARY.md、RESULTS.md、NOTES.md、DATA_AUDIT.md；样本内外详尽对照见 "
  "INSAMPLE_VS_OOS.md；本报告是其图表的逐张详解版。", "body")

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=13 * mm, bottomMargin=12 * mm,
                        title="GTJA191 图表分析报告", author="quant research")


def footer(canvas, d):
    canvas.saveState(); canvas.setFont(F, 8); canvas.setFillColor(GREY)
    canvas.drawCentredString(PW / 2, 7 * mm, f"GTJA191 图表分析报告 · 样本外 vs 样本内 · 第 {d.page} 页")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
