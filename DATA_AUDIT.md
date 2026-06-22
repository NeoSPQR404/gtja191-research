# 数据充分性核查 — 复现国泰君安《数量化专题之九十三》

> 目的：在动手复现前，对照研报（`gtja191-reference/国泰君安数量化专题之九十三.pdf`）
> 与 BARRA CNE5S 权重表（`gtja191-reference/Barra CNE5S Model Descriptor.pdf`）的
> 每一步计算需求，核查 `gtja191-tushare-data` 现有数据是否够用，并标出缺口与处置方式。
>
> 核查日期：2026-06-22。数据源：`gtja191-tushare-data/data/panel/year=YYYY/`（2017–2025，
> ~9.55M 行，65 列，关键价量字段 0% 缺失）+ `data/raw/`（income/balancesheet/cashflow/
> forecast/express/index_daily/stock_basic）。

---

## 0. 研报计算管线（按研报步骤）

| 步骤 | 内容 | 主要数据需求 |
|---|---|---|
| §3.3 | 191 个价量 Alpha 因子 | 日频 OHLC、VWAP、VOLUME、AMOUNT、RET、基准指数 OHLC；个别需 FF3 / 自身递归 |
| §2 / §3.4.1 Step1 | 正交标准化：因子对 行业哑变量 + 风格载荷 取残差 | 行业分类；9 大类 BARRA 风格因子载荷 |
| §3.4.1 Step2-4 | 对 d=1..5 日前瞻收益做截面回归，得因子收益率 f_k、年化 FR、IR | 前瞻 d 日个股收益（复权收盘） |
| §3.4.1 末 | 因子收益率两两相关、阈值剔除 | 上一步的 f_k 序列 |
| §3.4.2-3.4.3 | 组合 Alpha 模型 IC、因子个数 vs 预测力 | 同上 |
| §3.6 | 风格中性组合优化回测（2012.01–2017.04，基准中证500，全A非ST，次日开盘成交，行业±5%/风格±0.1） | 中证500 行情+成分权重、ST 历史、停牌/涨跌停、QP 求解器 |

---

## 1. Alpha 因子层（§3.3，191 个） — 基本满足

附录2 变量定义所需输入与可得性：

| 变量 | 研报定义 | 数据来源 | 状态 |
|---|---|---|---|
| OPEN/HIGH/LOW/CLOSE | 当日 OHLC（未复权） | panel `open/high/low/close` | ✅ |
| 复权 OHLC（拆分/分红对齐） | — | `× adj_factor`（panel 有 `adj_factor`、`close_hfq`） | ✅ 可构造 |
| VWAP 均价 | 成交均价 | `amount*10/vol`（已验证 13.54≈close 13.77 等） | ✅ 可构造 |
| VOLUME / AMOUNT | 成交量/成交额 | panel `vol`/`amount` | ✅ |
| RET | close/前收-1 | panel `pct_chg` 或 `close/pre_close-1` | ✅ |
| BANCHMARKINDEXCLOSE/OPEN | 基准指数收/开盘 | raw `index_daily/000300.SH`（含 open）；**panel 仅带 `bench_close`，缺 `bench_open`** | ⚠️ 需把 open 并入 panel（已在 raw，零成本） |
| MKT/SMB/HML（FF3） | 仅 **Alpha30** 用 | **无** | ❌ 需自建 FF3 或跳过 Alpha30 |
| SELF | 仅 **Alpha143** 用（递归引用自身 t-1 值） | 定义自洽性差 | ⚠️ 通常跳过 Alpha143 |

**结论**：189/191 可直接算；Alpha30（需 FF3）、Alpha143（自递归）按惯例排除或单独处理。
基准指数相关的 Alpha75/149/181/182 用沪深300（数据仅有沪深300，可接受）。

---

## 2. 风格中性化层 — 行业待定，2 个风格描述子缺失

### 2a. 行业哑变量 X_industry — ✅ 已补爬申万一级
- 原 `stock_basic.industry` 是 tushare 自有 110 类（口径不符）。
- **已爬申万一级（SW2021 L1，31 个行业）** → `data/raw/sw2021_l1_classify.parquet`
  + `data/raw/sw_member_all.parquet`（`index_classify` + `index_member_all`）。
- 覆盖：面板 2017–2025 共 5700 只股票，**5697 只有 SW L1 映射（99.95%）**，仅 3 只换代码退市股缺失（可忽略）。
- 说明：`index_member_all` 返回**每股当前快照**（带 in_date，无历史重分类）→ 行业为准静态、
  含轻度前视（行业重分类罕见，与旧研究同一注脚）。in_date 可处理新上市。足够 §3.4 中性化。

### 2b. 风格载荷 X_style（9 大类，BARRA CNE5S 合成权重）
逐描述子核查（权重取自 `Barra CNE5S Model Descriptor.pdf`）：

| 大类 | 描述子(权重) | 数据 | 状态 |
|---|---|---|---|
| Beta | BETA 1.00 | close_hfq 收益 vs 沪深300，250d/hl60 | ✅ |
| Momentum | RSTR 1.00 | log 收益 T=500 L=21 hl120 | ✅ |
| Size | LNCAP 1.00 | ln(`total_mv`) | ✅ |
| Earnings Yield | **EPIBS 0.68** / ETOP 0.11 / CETOP 0.21 | ETOP=`n_income_attr_p_ttm`/`total_mv`，CETOP=`n_cashflow_act_ttm`/`total_mv` | ⚠️ **EPIBS（一致预期EP）缺**（token 无分析师一致预期） |
| Residual Vol | DASTD 0.74 / CMRA 0.16 / HSIGMA 0.10 | 收益序列 + Beta 残差 | ✅ |
| Growth | SGRO 0.47 / EGRO 0.24 / **EGIBS 0.18** / **EGIBS_s 0.11** | SGRO/EGRO 来自 raw 年报 `total_revenue`/`n_income_attr_p`（回溯到 2011，5年够） | ⚠️ **EGIBS/EGIBS_s（一致预期增长）缺** |
| Book-to-Price | BTOP 1.00 | `total_hldr_eqy_exc_min_int`/`total_mv` | ✅ |
| Leverage | MLEV 0.38 / DTOA 0.35 / BLEV 0.27 | 长期负债 `lt_borr`(+`bond_payable`/`total_ncl`)，`total_liab`/`total_assets`，账面权益；优先股 PE≈0 近似 | ✅（PE 近似） |
| Liquidity | STOM/STOQ/STOA 0.35/0.35/0.30 | `vol`/`float_share`，21/3M/12M | ✅ |

**缺失描述子均为分析师一致预期类**（EPIBS、EGIBS、EGIBS_s）——标准 tushare token 不提供
（`report_rc` 虽点-in-time 但限频 1/min→1/hour，全市场回补不可行；akshare 一致预期仅当前快照=回测前视）。

按指示「依旧缺少则重新计算单个 style factor 的 weight 并注明」→ **剔除后按比例重归一**：
- **Earnings Yield**：ETOP 0.11/CETOP 0.21 → **ETOP 0.344 / CETOP 0.656**（剔 EPIBS 0.68）
- **Growth**：SGRO 0.47/EGRO 0.24 → **SGRO 0.662 / EGRO 0.338**（剔 EGIBS+EGIBS_s 0.29）

（其余 7 大类权重不变。NLSIZE 非线性市值：研报 §2 的 9 风格列表未含，可选做。）

---

## 3. 策略回测层（§3.6） — 多项需补爬

| 需求 | 现状 | 处置 |
|---|---|---|
| 基准 = **中证500**（000905.SH）行情 | **只有沪深300（000300.SH）** | ❌ 爬 `index_daily(000905.SH)` |
| 基准**成分股权重 + 行业权重向量**（w_bench, w_bench_industry，约束用） | 无 | ❌ 爬 `index_weight(000905.SH)`（月度） |
| 全A 非 ST 股票池 | 有全A 日行情；ST 仅**当前名称快照**（355 只），**无历史改名** | ⚠️ 爬 `namechange` 还原历史 ST，否则前视 |
| 停牌/涨跌停剔除、次日开盘成交 | 停牌可由缺行/0量识别；涨跌停可由 pct_chg 近 ±10/20/5% 识别；有 open | ✅ 可派生 |
| 组合优化（行业±5%、风格±0.1 敞口约束） | 计算问题非数据问题 | 需 QP 求解器（cvxpy/osqp） |

---

## 4. ⚠️ 最大前提：时间窗错配（全样本外）

- **研报回测窗口：2012.01 – 2017.04**。
- **本数据窗口：2017.01 – 2025.12**（仅 2017.01–04 ~3.5 个月重叠）。
- ⇒ 本复现实质是研报方法在 **2017 年后样本外**的重检验，**无法逐数复现研报原始数值**。
  价量 Alpha 在 2017 后普遍衰减，预期 FR/IR 量级会显著低于研报（FR 6.85% / IR 2.57 等）。
- 若要贴近研报窗口，需把 `gtja191-tushare-data` 面板**回补到 2010–2017**（tushare 历史行情可取，
  另一独立工作量）。

---

## 5. 结论与待决策项

**总体：可以开工。** Alpha 因子层与单因子检验层（§3.3 / §3.4.x）数据**基本满足**，
能完整跑通方法论；策略层（§3.6）需补爬中证500 行情+权重、ST 历史。两个一致预期描述子
按重归一处理并注明。唯一不可消除的是**时间窗为样本外**。

## 6. 已决策（2026-06-22，用户确认）

1. **复现范围**：先做 **§3.3 + §3.4**（191 因子 + 单因子检验/相关性/有效周期/模型IC）。§3.6 策略
   回测暂缓（届时再补爬中证500 + ST 改名史）。
2. **时间窗**：**接受 2017–2025 样本外**，用现成面板，文档标注为样本外重检验。
3. **行业口径**：**补爬申万一级**（`index_classify` level=L1 + `index_member_all`，点-in-time 成分）。
4. **缺失一致预期描述子**（EPIBS / EGIBS / EGIBS_s）：**用业绩预告 forecast + 业绩快报 express 做弱代理**
   （panel 已带 `fc_*` / `exp_*` 列），非严格同义，文档注明。
   - EPIBS 代理：用 forecast/express 隐含的预期 EPS 或预期净利润 ÷ 价格（near-term）。
   - EGIBS / EGIBS_s 代理：用 forecast `p_change_min/max`（业绩预告同比增速）等。
   - 代理仍按 CNE5 原始权重位置参与合成（EarnYield 0.68/0.11/0.21、Growth 0.47/0.24/0.18/0.11）；
     若某股某期代理也缺，则该大类内对**可得描述子**重归一。
5. **Alpha30（FF3）/Alpha143（SELF）**：跳过（不影响其余 189 个因子）。
