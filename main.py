# -*- coding: utf-8 -*-
"""
諧波形態 + SMC 交易系統 - QuantConnect 主程式
Harmonic Pattern + Smart Money Concepts Trading System

🦄 "Unicorn Setup" - 當諧波形態與 SMC 區域完美匯合時入場
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 導入自定義模組
from config import (
    SYMBOLS,
    LEVERAGE,
    MAX_POSITIONS,
    RISK_PER_TRADE,
    CONFIDENCE_THRESHOLD,
    TOLERANCE,
    PRZ_PROXIMITY_PERCENT,
    REQUIRE_ENTRY_CONFIRMATION,
    ENTRY_CONFIRMATION_TYPE,
    WARMUP_PERIOD,
    STARTING_CASH,
    BACKTEST_START_YEAR,
    BACKTEST_START_MONTH,
    BACKTEST_START_DAY,
    DEBUG_MODE,
    ENABLE_PLOTTING
)

from utils import (
    DataAdapter,
    ZigZagCalculator,
    IndicatorCalculator,
    EntryConfirmation
)

from harmonic_patterns import HarmonicPatternEngine, HarmonicPattern
from smc_analysis import SMCAnalyzer, SMCZone
from ml_scoring import MLScorer, FeatureSet
from risk_manager import RiskManager, TradeSetup


class HarmonicSMCAlgorithm(QCAlgorithm):
    """
    諧波形態 + SMC 交易算法
    
    策略邏輯:
    1. 使用 ZigZag 識別擺動點
    2. 掃描發展中的諧波形態 (XABC → 預測 D)
    3. 檢查價格是否接近 PRZ
    4. 驗證 PRZ 與 SMC 區域的匯合
    5. 使用 ML 評分過濾
    6. 等待入場確認信號
    7. 執行交易並管理風險
    """
    
    def Initialize(self):
        """
        初始化算法
        設定回測參數、交易對、模組
        """
        # ========================================
        # 基本設定
        # ========================================
        
        self.SetStartDate(BACKTEST_START_YEAR, BACKTEST_START_MONTH, BACKTEST_START_DAY)
        self.SetCash(STARTING_CASH)
        
        # 設定經紀商模型 (幣安期貨)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        
        # ========================================
        # 添加交易對
        # ========================================
        
        self.symbols = {}
        for symbol_str in SYMBOLS:
            crypto = self.AddCrypto(symbol_str, Resolution.Hour)
            crypto.SetLeverage(LEVERAGE)
            self.symbols[symbol_str] = crypto.Symbol
        
        # 預熱數據
        self.SetWarmUp(WARMUP_PERIOD)
        
        # ========================================
        # 初始化模組
        # ========================================
        
        self.data_adapter = DataAdapter(self)
        self.zigzag = ZigZagCalculator()
        self.harmonic_engine = HarmonicPatternEngine(tolerance=TOLERANCE)
        self.smc_analyzer = SMCAnalyzer()
        self.ml_scorer = MLScorer(self, enabled=True, confidence_threshold=CONFIDENCE_THRESHOLD)
        self.risk_manager = RiskManager(
            self,
            risk_per_trade=RISK_PER_TRADE,
            max_positions=MAX_POSITIONS,
            leverage=LEVERAGE
        )
        
        # ========================================
        # 狀態追蹤
        # ========================================
        
        # 待處理的交易設置 (等待入場確認)
        self.pending_setups = {}  # {symbol: TradeSetup}
        
        # 已識別的形態 (避免重複)
        self.detected_patterns = {}  # {symbol: [pattern_id]}
        
        # 計數器
        self.patterns_scanned = 0
        self.setups_created = 0
        self.trades_executed = 0
        
        # ========================================
        # 圖表設定 (可視化)
        # ========================================
        
        if ENABLE_PLOTTING:
            # 創建自定義圖表
            chart = Chart("Strategy Dashboard")
            chart.AddSeries(Series("PRZ Distance", SeriesType.Line, 0))
            chart.AddSeries(Series("ML Confidence", SeriesType.Line, 1))
            chart.AddSeries(Series("Position Value", SeriesType.Line, 2))
            self.AddChart(chart)
        
        # ========================================
        # 定時任務
        # ========================================
        
        # 每天記錄一次統計
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(0, 0),
            self.DailyReport
        )
        
        self.Debug("🚀 諧波 + SMC 交易系統初始化完成!")
    
    # ========================================
    # 主要交易邏輯
    # ========================================
    
    def OnData(self, data: Slice):
        """
        主要數據處理函數
        每根 K 線觸發一次
        """
        # 預熱期間不交易
        if self.IsWarmingUp:
            return
        
        # 管理現有持倉
        self.risk_manager.manage_positions(data)
        
        # 遍歷所有交易對
        for symbol_str, symbol in self.symbols.items():
            if not data.ContainsKey(symbol):
                continue
            
            try:
                self.ProcessSymbol(symbol, symbol_str, data)
            except Exception as e:
                if DEBUG_MODE:
                    self.Debug(f"❌ {symbol_str} 處理錯誤: {str(e)}")
    
    def ProcessSymbol(self, symbol: Symbol, symbol_str: str, data: Slice):
        """
        處理單個交易對
        
        Args:
            symbol: QuantConnect Symbol
            symbol_str: 交易對字符串
            data: Slice 數據
        """
        # ========================================
        # 步驟 1: 獲取歷史數據
        # ========================================
        
        history = self.History(symbol, 500, Resolution.Hour)
        df = self.data_adapter.adapt_history(history, symbol)
        
        if len(df) < 100:
            return
        
        # 添加技術指標
        df = IndicatorCalculator.add_all_indicators(df)
        
        current_price = self.Securities[symbol].Price
        
        # ========================================
        # 步驟 2: 檢查待處理的設置
        # ========================================
        
        if symbol_str in self.pending_setups:
            self.CheckPendingSetup(symbol, symbol_str, df, current_price)
            return  # 有待處理設置時不掃描新形態
        
        # ========================================
        # 步驟 3: 計算 ZigZag
        # ========================================
        
        swing_points = self.zigzag.calculate(df)
        
        if len(swing_points) < 4:
            return
        
        # ========================================
        # 步驟 4: 掃描發展中形態
        # ========================================
        
        developing_patterns = self.harmonic_engine.scan_developing_patterns(
            swing_points, current_price, max_patterns=3
        )
        
        if not developing_patterns:
            return
        
        self.patterns_scanned += len(developing_patterns)
        
        # ========================================
        # 步驟 5: 處理每個形態
        # ========================================
        
        for pattern in developing_patterns:
            # 檢查價格是否接近 PRZ
            prz_distance = pattern.prz.distance_percent(current_price)
            
            if prz_distance > PRZ_PROXIMITY_PERCENT:
                continue  # 價格距離 PRZ 還太遠
            
            # 檢查是否已經處理過這個形態
            pattern_id = self._get_pattern_id(pattern)
            if symbol_str in self.detected_patterns:
                if pattern_id in self.detected_patterns[symbol_str]:
                    continue
            
            # 驗證形態
            is_valid, setup = self.ValidateAndCreateSetup(
                symbol, pattern, df, current_price
            )
            
            if is_valid and setup:
                # 記錄已處理的形態
                if symbol_str not in self.detected_patterns:
                    self.detected_patterns[symbol_str] = []
                self.detected_patterns[symbol_str].append(pattern_id)
                
                # 保存待處理設置
                self.pending_setups[symbol_str] = setup
                self.setups_created += 1
                
                if DEBUG_MODE:
                    self.Debug(f"""
🎯 發現有效交易設置: {symbol_str}
{pattern.pattern_type} ({pattern.direction})
PRZ: {pattern.prz.price_low:.4f} - {pattern.prz.price_high:.4f}
當前價格: {current_price:.4f}
距離 PRZ: {prz_distance:.2f}%
""")
                
                # 如果不需要入場確認，直接執行
                if not REQUIRE_ENTRY_CONFIRMATION:
                    self.ExecuteSetup(symbol_str)
    
    def ValidateAndCreateSetup(
        self,
        symbol: Symbol,
        pattern: HarmonicPattern,
        df: pd.DataFrame,
        current_price: float
    ) -> tuple:
        """
        驗證形態並創建交易設置
        
        驗證步驟:
        1. SMC 匯合檢查
        2. ML 評分
        3. 風險報酬比檢查
        
        Returns:
            (is_valid, TradeSetup or None)
        """
        # ========================================
        # SMC 匯合驗證
        # ========================================
        
        order_blocks, fvgs = self.smc_analyzer.get_all_zones(df)
        
        has_confluence, confluent_zones = self.smc_analyzer.check_prz_confluence(
            pattern.prz.price_low,
            pattern.prz.price_high,
            pattern.direction,
            order_blocks,
            fvgs
        )
        
        if not has_confluence:
            if DEBUG_MODE:
                self.Debug(f"⚠️ {symbol}: 無 SMC 匯合，跳過")
            return False, None
        
        # ========================================
        # ML 評分
        # ========================================
        
        features = MLScorer.extract_features(df, pattern, confluent_zones, current_price)
        score_result = self.ml_scorer.calculate_score(features, pattern.direction)
        
        if not self.ml_scorer.should_trade(score_result):
            if DEBUG_MODE:
                self.Debug(f"⚠️ {symbol}: ML 分數不足 ({score_result.confidence:.2%})")
            return False, None
        
        # 繪製 ML 分數
        if ENABLE_PLOTTING:
            self.Plot("Strategy Dashboard", "ML Confidence", score_result.confidence * 100)
        
        # ========================================
        # 創建交易設置
        # ========================================
        
        setup = self.risk_manager.create_trade_setup(symbol, pattern, df, current_price)
        
        if setup is None:
            return False, None
        
        # 檢查風險報酬比
        if setup.risk_reward_ratio < 1.5:
            if DEBUG_MODE:
                self.Debug(f"⚠️ {symbol}: R:R 不足 ({setup.risk_reward_ratio:.2f})")
            return False, None
        
        # 記錄 SMC 區域信息
        if DEBUG_MODE:
            for zone in confluent_zones:
                self.Debug(f"  📦 {self.smc_analyzer.get_zone_summary(zone)}")
        
        return True, setup
    
    def CheckPendingSetup(
        self,
        symbol: Symbol,
        symbol_str: str,
        df: pd.DataFrame,
        current_price: float
    ):
        """
        檢查待處理的交易設置
        等待入場確認信號
        """
        setup = self.pending_setups.get(symbol_str)
        
        if setup is None:
            return
        
        # 檢查是否超時 (例如 24 小時)
        time_elapsed = (self.Time - setup.created_at).total_seconds() / 3600
        if time_elapsed > 24:
            if DEBUG_MODE:
                self.Debug(f"⏰ {symbol_str}: 設置超時，移除")
            del self.pending_setups[symbol_str]
            return
        
        # 檢查價格是否仍在 PRZ 附近
        # (這裡使用止損和止盈來估計 PRZ)
        if setup.direction == "long":
            if current_price < setup.stop_loss:
                if DEBUG_MODE:
                    self.Debug(f"❌ {symbol_str}: 價格跌破止損，設置失效")
                del self.pending_setups[symbol_str]
                return
        else:
            if current_price > setup.stop_loss:
                if DEBUG_MODE:
                    self.Debug(f"❌ {symbol_str}: 價格突破止損，設置失效")
                del self.pending_setups[symbol_str]
                return
        
        # 檢查入場確認 K 線
        has_confirmation = EntryConfirmation.check_entry_candle(
            df, setup.pattern_direction, ENTRY_CONFIRMATION_TYPE
        )
        
        if has_confirmation:
            if DEBUG_MODE:
                self.Debug(f"✅ {symbol_str}: 入場確認信號出現!")
            self.ExecuteSetup(symbol_str)
    
    def ExecuteSetup(self, symbol_str: str):
        """
        執行交易設置
        """
        setup = self.pending_setups.get(symbol_str)
        
        if setup is None:
            return
        
        # 檢查是否已有持倉
        if self.risk_manager.has_position(setup.symbol):
            if DEBUG_MODE:
                self.Debug(f"⚠️ {symbol_str}: 已有持倉，跳過")
            del self.pending_setups[symbol_str]
            return
        
        # 執行交易
        success = self.risk_manager.execute_trade(setup)
        
        if success:
            self.trades_executed += 1
            
            # 繪製持倉價值
            if ENABLE_PLOTTING:
                self.Plot("Strategy Dashboard", "Position Value", setup.position_value)
        
        # 移除待處理設置
        del self.pending_setups[symbol_str]
    
    # ========================================
    # 事件處理
    # ========================================
    
    def OnOrderEvent(self, orderEvent: OrderEvent):
        """
        訂單事件處理
        """
        self.risk_manager.on_order_event(orderEvent)
    
    def OnEndOfAlgorithm(self):
        """
        算法結束時調用
        """
        self.Debug("=" * 50)
        self.Debug("📊 最終績效報告")
        self.Debug("=" * 50)
        self.Debug(self.risk_manager.get_statistics_summary())
        self.Debug(f"掃描形態數: {self.patterns_scanned}")
        self.Debug(f"創建設置數: {self.setups_created}")
        self.Debug(f"執行交易數: {self.trades_executed}")
        self.Debug("=" * 50)
    
    # ========================================
    # 定時任務
    # ========================================
    
    def DailyReport(self):
        """
        每日報告
        """
        if not DEBUG_MODE:
            return
        
        stats = self.risk_manager.get_statistics()
        
        self.Debug(f"""
━━━━━━━ 每日報告 ({self.Time.date()}) ━━━━━━━
資金: ${self.Portfolio.TotalPortfolioValue:.2f}
持倉數: {stats['active_positions']}
今日勝率: {stats['win_rate']:.1f}%
總盈虧: ${stats['total_pnl']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    # ========================================
    # 輔助方法
    # ========================================
    
    def _get_pattern_id(self, pattern: HarmonicPattern) -> str:
        """生成形態唯一 ID"""
        return f"{pattern.pattern_type}_{pattern.X.index}_{pattern.C.index}"

