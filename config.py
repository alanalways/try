# -*- coding: utf-8 -*-
"""
策略配置文件 (Strategy Configuration)
所有可調參數集中在這裡，方便優化和測試
"""

# ============================================================
# 交易設定 (Trading Settings)
# ============================================================

# 交易對清單 (Trading Pairs)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# 使用槓桿 (Leverage)
LEVERAGE = 10

# 最大同時持倉數 (Maximum Concurrent Positions)
MAX_POSITIONS = 2

# ============================================================
# 風險管理 (Risk Management)
# ============================================================

# 每筆交易風險比例 (Risk Per Trade)
RISK_PER_TRADE = 0.01  # 1%

# 止盈比例設定 (Take Profit Ratios)
TP1_RATIO = 0.382  # 第一止盈: D到C的38.2%
TP2_RATIO = 0.618  # 第二止盈: D到C的61.8%

# 止損倍數 (Stop Loss Multiplier based on pattern range)
STOP_LOSS_MULTIPLIER = 1.272  # 超過X點的1.272倍

# ATR 週期 (ATR Period)
ATR_PERIOD = 14

# ============================================================
# 諧波形態設定 (Harmonic Pattern Settings)
# ============================================================

# Fibonacci 比例容差 (Tolerance for Fibonacci Ratios)
TOLERANCE = 0.03  # 3%

# ZigZag 參數
ZIGZAG_CONFIRMATION_BARS = 3  # 確認K線數量
ZIGZAG_MIN_RETRACE_PERCENT = 1.0  # 最小回撤百分比

# 形態最大跨度 (Maximum bars between X and D)
MAX_PATTERN_BARS = 100

# ============================================================
# SMC 設定 (Smart Money Concepts Settings)
# ============================================================

# Order Block 回溯K線數 (Order Block Lookback)
ORDER_BLOCK_LOOKBACK = 50

# FVG 回溯K線數 (Fair Value Gap Lookback)
FVG_LOOKBACK = 30

# BOS (Break of Structure) 靈敏度
BOS_SENSITIVITY = 1.5  # ATR倍數

# Order Block 最小尺寸 (Minimum OB Size)
MIN_ORDER_BLOCK_SIZE_ATR = 0.5  # ATR的0.5倍

# ============================================================
# 機器學習設定 (Machine Learning Settings)
# ============================================================

# 信心度閾值 (Confidence Threshold)
CONFIDENCE_THRESHOLD = 0.75

# ML模型名稱 (在ObjectStore中的名稱)
ML_MODEL_NAME = "harmonic_smc_rf_model.pkl"

# 是否啟用ML (Enable ML Scoring)
ML_ENABLED = True

# 無ML時的回退分數 (Fallback Score when ML unavailable)
ML_FALLBACK_SCORE = 0.5

# ============================================================
# 入場確認設定 (Entry Confirmation Settings)
# ============================================================

# PRZ 接近閾值 (PRZ Proximity Threshold)
PRZ_PROXIMITY_PERCENT = 2.0  # 價格在PRZ 2%範圍內觸發監控

# 需要入場確認K線 (Require Entry Confirmation Candle)
REQUIRE_ENTRY_CONFIRMATION = True

# 確認K線類型: "hammer", "engulfing", "any"
ENTRY_CONFIRMATION_TYPE = "any"

# ============================================================
# 回測設定 (Backtest Settings)
# ============================================================

# 起始日期
BACKTEST_START_YEAR = 2023
BACKTEST_START_MONTH = 1
BACKTEST_START_DAY = 1

# 結束日期 (None = 到今天)
BACKTEST_END_YEAR = None
BACKTEST_END_MONTH = None
BACKTEST_END_DAY = None

# 起始資金
STARTING_CASH = 10000

# 預熱K線數量 (Warmup Period)
WARMUP_PERIOD = 500

# ============================================================
# 技術指標設定 (Technical Indicator Settings)
# ============================================================

# RSI 設定
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# EMA 設定
EMA_FAST = 50
EMA_SLOW = 200

# Bollinger Bands 設定
BB_PERIOD = 20
BB_STD_DEV = 2.0

# ============================================================
# 日誌與除錯 (Logging & Debug)
# ============================================================

# 是否啟用詳細日誌
DEBUG_MODE = True

# 日誌等級: "INFO", "DEBUG", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# 是否繪製圖表
ENABLE_PLOTTING = True

# ============================================================
# 諧波形態 Fibonacci 比例定義
# (Harmonic Pattern Fibonacci Ratio Definitions)
# ============================================================

HARMONIC_RATIOS = {
    "Gartley": {
        "XAB": {"min": 0.618, "max": 0.618},
        "ABC": {"min": 0.382, "max": 0.886},
        "BCD": {"min": 1.27, "max": 1.618},
        "XAD": {"min": 0.786, "max": 0.786},
        "direction_change_at_B": True
    },
    "Bat": {
        "XAB": {"min": 0.382, "max": 0.5},
        "ABC": {"min": 0.382, "max": 0.886},
        "BCD": {"min": 1.618, "max": 2.618},
        "XAD": {"min": 0.886, "max": 0.886},
        "direction_change_at_B": True
    },
    "Butterfly": {
        "XAB": {"min": 0.786, "max": 0.786},
        "ABC": {"min": 0.382, "max": 0.886},
        "BCD": {"min": 1.618, "max": 2.618},
        "XAD": {"min": 1.27, "max": 1.618},
        "direction_change_at_B": True
    },
    "Crab": {
        "XAB": {"min": 0.382, "max": 0.618},
        "ABC": {"min": 0.382, "max": 0.886},
        "BCD": {"min": 2.618, "max": 3.618},
        "XAD": {"min": 1.618, "max": 1.618},
        "direction_change_at_B": True
    }
}

