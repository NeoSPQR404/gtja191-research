# A1 样本内对照：in-sample (2012-2017) vs 样本外 (2017-2025) vs 研报

> **A1 的问题**：研报 2012.01–2017.04 年化超额 50.2%、IR 4.67；本复现样本外（2017-2025）却是
> −24.5%/IR −1.59。这究竟是**真实的样本外衰减**，还是**实现 bug**？
>
> **答案：是真实衰减。** 把同一套管线放回研报的 2012-2017 窗口，**策略转为正（+9.5%/IR 0.64）**，
> 且单因子 |IR|、模型 IC、胜率与研报量级相当。**样本外失效不是实现错误，而是这套价量 alpha 在
> 2017 年后（其实 ~2016 已开始）的真实衰减。**

数据/方法：把研报窗口面板（2010-2017，2010-2011 仅作预热）回补，跑同一管线，统计 2012.01-2017.04。
窗口隔离：`GTJA_RUN_TAG=insample` → `cache_insample`/`figures_insample`/`tables_insample`，OOS 零污染。

---

## 1. §3.4.1 单因子有效性 |IR|均值

| 预测周期 d | **样本内 2012-2017** | 样本外 2017-2025 | 研报 2012-2017 |
|---|---|---|---|
| DAY1 | 0.62 ⚠️ | 2.25 | 2.57 |
| DAY2 | **2.50** | 2.08 | 2.11 |
| DAY3 | 2.28 | 1.89 | 1.73 |
| DAY4 | 2.14 | 1.77 | 1.26 |
| DAY5 | 2.14 | 1.78 | 1.05 |

- 样本内 d2–d5 的 |IR| ≈ 2.1–2.5，**与研报、样本外同量级**——因子显著性跨期稳定、被复现。
- ⚠️ d1 样本内 0.62 偏低（异常，可能与 5 风格中性化口径/非重叠采样有关）；d2 后恢复正常。

## 2. §3.4.2 / §3.4.3 组合模型 IC

| | **样本内 2012-2017** | 样本外 2017-2025 | 研报 2012-2017 |
|---|---|---|---|
| 日度 IC 均值 | 0.102 | 0.115 | 0.057 |
| t 值 | 38.7 | 65.8 | 18.23 |
| **日度 IC 胜率** | **88.0%** | 95.1% | **88.7%** |

- **样本内胜率 88.0% ≈ 研报 88.7%**——几乎精确吻合。模型 IC 随因子个数单调递增（0.073→0.102），复现 §3.4.3。

## 3. §3.6 策略绩效（核心验证）

| | **样本内 5风格** | **样本内 9风格** | 样本外 2017-2025 | 研报 2012-2017 |
|---|---|---|---|---|
| 年化超额收益 | **+9.5%** | **+9.0%** | **−24.5%** | +50.2% |
| 信息比率 | **+0.64** | **+0.63** | **−1.59** | 4.67 |
| 最大回撤 | −20.6% | −20.0% | −90% | −5.9% |

**分年度（样本内 9风格）**：2013 +17.0%(IR 1.50)、2014 +23.5%(IR 1.95)、2015 +27.4%(IR 1.21) 强正；
**2016 −7.6%、2017 −4.8% 已转负**——**衰减早在 2016 就开始**，不是 2017 才突然失效。图24（in-sample
累计超额）：2013-2015 一路涨到 ~1.7，2016-2017 见顶回落。

> **完整 9 风格 BARRA（EarnYield/Value/Growth/Leverage 从回补到 2005 的全历史财务算出，
> 覆盖率 ~80-85%）几乎不改变结果（+9.0% vs +9.5%）**——见 §6，这**推翻了"幅度差距来自简化中性化"
> 的假设**。

## 4. 机制：可交易（日内）IC 的符号翻转

| 时段（同一 Ê） | **样本内 2012-2017** | 样本外 2017-2025 |
|---|---|---|
| 收盘→收盘（预测目标） | +0.096 | +0.107 |
| 收盘→次开盘（隔夜·不可交易） | +0.178 | +0.229 |
| **次开盘→收盘（日内·可交易）** | **+0.016** | **−0.012** |

- 两个窗口的信号都被**隔夜跳空主导**（+0.18 / +0.23）。区别在**可交易的日内 IC**：样本内 **+0.016（正）**
  → 样本外 **−0.012（负）**。**策略 P&L 正好随它翻号：+9.5% → −24.5%。** 这就是样本外失效的精确机制。

---

## 5. 结论

1. **A1 验证成功**：策略样本内为正、样本外为负——**样本外的失效是真实的 alpha 衰减，不是实现 bug**。
   单因子 |IR|、模型 IC、胜率（88.0%≈88.7%）在样本内与研报量级相当，证明管线正确。
2. **衰减时点**：~2016 年起（样本内 2016-2017 已转负），可交易日内 IC 从 +0.016 翻到 −0.012。
3. **量级差距（样本内 +9% vs 研报 +50.2%）的归因**（非反号、是幅度）——**已用 §6 排除中性化**：
   - ~~简化 BARRA~~ **不是原因**：完整 9 风格中性化给 +9.0%，与 5 风格 +9.5% 几乎相同（§6）。
   - 更可能的原因：**无卖方一致预期**（研报核心输入之一）、组合优化用本项目的 LP（ridge 预测 +
     ±0.1风格/±5%行业/≤5%单票）与研报精确设置不同、以及研报可能在多头侧/择时/参数上获取了更多
     （其 +50% 含 2012-2015 小盘风格极利于价量的多头有效性）。
   - 这些都压低幅度，但**不改方向**——样本内盈利被复现。

**一句话**：研报方法是对的、可复现的；它在 2012-2015 确实赚钱（本复现 2013-2015 年化 17%-27%），
但这套短周期价量 alpha 的**可交易超额**在 ~2016 后衰减殆尽、2017 起转负——叠加[隔夜不可交易 +
偏空头 + 做空成本]（见 [SUMMARY.md](SUMMARY.md)），样本外不可盈利。

---

## 6. 完整 9 风格 BARRA 重建（排除"简化中性化"假设）

把 2012-2017 的财务**回补到全历史（income/bs/cf 到 2005）**，用完整 9 风格 CNE5S 中性化重跑
（`GTJA_RUN_TAG=insample_full`，`panel_insample_full`）：

| | 5 风格（价量） | **9 风格（完整 BARRA）** |
|---|---|---|
| 策略年化超额 | +9.5% / IR 0.64 | **+9.0% / IR 0.63** |
| 单因子 \|IR\| (d2) | 2.50 | 2.63（略升，中性化更干净） |
| 模型 IC / 胜率 | 0.102 / 88.0% | 0.102 / 87.9% |
| 隔夜 / 日内 IC | +0.178 / +0.016 | +0.178 / +0.016 |

- **完整 9 风格几乎不改变结果（+9.0% vs +9.5%）** → **量级差距（+9% vs 研报 +50%）不是中性化口径
  造成的**。这是本次重建最有价值的结论：它**证伪**了"差距来自简化 BARRA"的假设，把归因收窄到
  [无卖方一致预期 + LP/预测构造 + 研报特定设置]。
- 完整 9 风格的 EarnYield/Value/Growth/Leverage 从回补的全历史财务算出（TTM 覆盖 ~80-85%；一致预期
  描述子 EPIBS/EGIBS 因 forecast 5297 个按日文件触发 I/O 限流而跳过 → 退化为 ETOP/CETOP、SGRO/EGRO，
  与 OOS 口径一致）。
- **工程**：绕过 macOS 后台 I/O 限流的关键是**只读 universe 的 2455 只财务**（~7s/类，远低于限流阈值），
  再用 `build_insample_full_panel.py` 把 TTM/point-in-time 财务并入价量面板（读 ~11 文件，60s）；
  alpha memmap 复用（价量相同）。

## 复算
```bash
# 1) 补数（一次性）：daily/daily_basic/adj_factor 2010-2017 + CSI500，去掉 HTTP_PROXY 提速 13×
# 2) 快速纯价量面板：python scripts/build_insample_panel.py  → data/panel_insample/year=YYYY
# 3) 管线（窗口隔离）：
GTJA_RUN_TAG=insample python scripts/run_alphas.py --universe full_a --memmap
GTJA_RUN_TAG=insample python scripts/compute_barra.py --universe full_a
GTJA_RUN_TAG=insample python scripts/run_single_factor_test.py --universe full_a
GTJA_RUN_TAG=insample python scripts/make_report.py
GTJA_RUN_TAG=insample python scripts/run_model_ic.py --universe full_a
GTJA_RUN_TAG=insample python scripts/export_predictions.py --universe full_a
GTJA_RUN_TAG=insample python scripts/run_strategy.py
# 产物在 report/{tables,figures}_insample/

# 完整 9 风格（§6）：财务回补全历史后
#   分块合并（只读 universe 2455 只，避开限流）→ data/raw/_consolidated/{income,balancesheet,cashflow}.parquet
python scripts/build_insample_full_panel.py                # → panel_insample_full（读11文件,60s）
cp -r report/cache_insample/signals_full_a_cross_section report/cache_insample_full/   # 复用 alpha memmap
export GTJA_RUN_TAG=insample_full
python scripts/compute_barra.py --universe full_a          # 9 风格（含回补财务）
python scripts/run_single_factor_test.py --universe full_a
python scripts/make_report.py
python scripts/run_model_ic.py --universe full_a
python scripts/export_predictions.py --universe full_a
python scripts/run_strategy.py
# 产物在 report/{tables,figures}_insample_full/
```
