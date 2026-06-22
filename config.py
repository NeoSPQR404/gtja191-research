"""Central configuration — GTJA《数量化专题之九十三》(191-alpha) reproduction.

Output repo: gtja191-research-tushare-data/. Reads the read-only tushare panel
from the sibling gtja191-tushare-data/data/ (2017–2025). Decisions locked
2026-06-22 (see DATA_AUDIT.md):
  * scope = §3.3 (191 alphas) + §3.4 (single-factor test + model IC);
  * window = 2017–2025, OUT-OF-SAMPLE vs the report's 2012.01–2017.04;
  * universe = full-A ex-ST (faithful to the report);
  * industry = 申万一级 (SW2021 L1, fetched to data/raw/);
  * missing consensus descriptors (EPIBS / EGIBS / EGIBS_s) proxied by
    forecast(业绩预告)+express(业绩快报);
  * alpha30 (FF3) + alpha143 (SELF) skipped → 189/191.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(ROOT, "..", "gtja191-tushare-data", "data"))
PANEL_DIR = os.path.join(DATA_ROOT, "panel")                 # year=YYYY/panel.parquet
RAW_DIR = os.path.join(DATA_ROOT, "raw")
STOCK_BASIC = os.path.join(RAW_DIR, "stock_basic.parquet")
CSI300_RAW = os.path.join(RAW_DIR, "index_daily", "000300.SH.parquet")

# Supplementary data fetched into THIS repo (SW industry classification).
LOCAL_RAW = os.path.join(ROOT, "data", "raw")
SW_MEMBER = os.path.join(LOCAL_RAW, "sw_member_all.parquet")        # ts_code -> l1
SW_CLASSIFY = os.path.join(LOCAL_RAW, "sw2021_l1_classify.parquet") # 31 L1 industries

REPORT_DIR = os.path.join(ROOT, "report")
TABLE_DIR = os.path.join(REPORT_DIR, "tables")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")
CACHE_DIR = os.path.join(REPORT_DIR, "cache")
for _d in (TABLE_DIR, FIGURE_DIR, CACHE_DIR, LOCAL_RAW):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- sample
START_DATE = "20170101"
END_DATE = "20251231"

# Universe. "full_a": every A-share with >= MIN_HISTORY_DAYS history, ex-ST
# (the report's choice). "topN": top UNIVERSE_SIZE by median daily amount
# (a reduced liquid universe for fast iteration / lower memory).
UNIVERSE_MODE = "full_a"            # "full_a" | "topN"
UNIVERSE_SIZE = 1000               # only used when UNIVERSE_MODE == "topN"
MIN_HISTORY_DAYS = 250
# ST exclusion is by current name only (no namechange history on this token —
# mild look-ahead, documented in DATA_AUDIT.md §3).
ST_MARKERS = ("ST", "*ST", "退")

# Alphas to skip (inputs unavailable / definition self-referential).
SKIP_ALPHAS = (30, 143)            # 30 needs FF3 MKT/SMB/HML; 143 uses SELF

# ---------------------------------------------------------------- §3.4 test
HORIZONS = [1, 2, 3, 4, 5]          # prediction horizons d (trading days)
TRADING_DAYS_PER_YEAR = 252
# Non-overlapping forward returns (sample every d days) so IR annualisation
# sqrt(252/d) is valid.
NON_OVERLAPPING = True
MIN_CROSS_SECTION = 100             # skip a date with fewer valid stocks
WINSOR_SIGMA = 5.0                  # MAD winsorisation of factor exposures
MIN_FACTOR_STD = 1e-8              # drop degenerate near-constant cross-sections
MIN_OBS_FRAC = 0.6                 # a factor must yield a finite factor return on
                                    # >= this fraction of dates to be admitted
# Factor-return correlation pruning (report: drop lower-IR of high-corr pairs).
PRUNE_CORR_THRESHOLD = 0.7
PRUNE_HORIZON = 1

# §3.4.2/§3.4.3 alpha-model IC.
MODEL_IC_WINDOW = 120
MODEL_IC_MIN_WINDOW = 20
FACTOR_COUNTS = [20, 40, 60, 80, 100, 120, 140, 160]
MODEL_IC_RIDGE = 0.1               # ridge λ (× n_stocks) on the multivariate fit

# RANK convention: cross-sectional (rank across stocks per date) is the report's
# intent and this project's default. The unified panel engine can also do the
# per-stock time-series RANK of the reference Alpha_Set.
RANK_CONVENTION = "cross_section"   # "cross_section" | "time_series"

# ---------------------------------------------------------------- BARRA CNE5S
# Descriptor weights per the MSCI "Barra CNE5 Descriptor Details" (Sep 2013),
# gtja191-reference/Barra CNE5S Model Descriptor.pdf. Each style is a weighted
# composite of standardized descriptors.
CNE5S_WEIGHTS = {
    "Beta":       {"BETA": 1.00},
    "Momentum":   {"RSTR": 1.00},
    "Size":       {"LNCAP": 1.00},
    "EarnYield":  {"EPIBS": 0.68, "ETOP": 0.11, "CETOP": 0.21},
    "Volatility": {"DASTD": 0.74, "CMRA": 0.16, "HSIGMA": 0.10},
    "Growth":     {"SGRO": 0.47, "EGRO": 0.24, "EGIBS": 0.18, "EGIBS_s": 0.11},
    "Value":      {"BTOP": 1.00},
    "Leverage":   {"MLEV": 0.38, "DTOA": 0.35, "BLEV": 0.27},
    "Liquidity":  {"STOM": 0.35, "STOQ": 0.35, "STOA": 0.30},
}
# Descriptors that strictly require sell-side analyst consensus, which is NOT on
# this tushare token. Per the locked decision they are PROXIED by forecast /
# express (company guidance); barra_cne5s.py builds the proxy and, where even the
# proxy is unavailable for a stock-period, renormalizes the style over the
# remaining descriptors. Renormalised fallback weights (proxy fully dropped):
#   EarnYield -> ETOP 0.344 / CETOP 0.656 ;  Growth -> SGRO 0.662 / EGRO 0.338.
CONSENSUS_DESCRIPTORS = ("EPIBS", "EGIBS", "EGIBS_s")

# The 9 CNE5S styles used as neutralisation controls (report §2 omits Non-linear
# Size from its style list, so we follow the report and use these 9).
STYLE_FACTORS = list(CNE5S_WEIGHTS.keys())

# BARRA descriptor estimation parameters (report appendix 1 values; the report
# uses 沪深300 + half-life 60/120/40, which differ slightly from MSCI's
# 252/126/42 — we follow the report's appendix-1 numbers).
BETA_WINDOW, BETA_HALFLIFE = 250, 60
RSTR_T, RSTR_LAG, RSTR_HALFLIFE = 500, 21, 120
DASTD_WINDOW, DASTD_HALFLIFE = 250, 40
CMRA_MONTHS, MONTH_DAYS = 12, 21
HSIGMA_WINDOW = 250
STO_MONTH_DAYS = 21
GROWTH_YEARS = 5
