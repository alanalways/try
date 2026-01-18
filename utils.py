# -*- coding: utf-8 -*-
"""
工具函數模組 (Utility Functions Module)
包含數據處理、ZigZag計算、技術指標等輔助功能
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config import (
    ZIGZAG_CONFIRMATION_BARS,
    ZIGZAG_MIN_RETRACE_PERCENT,
    RSI_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD_DEV
)


# ============================================================
# 數據結構定義 (Data Structures)
# ============================================================

@dataclass
class SwingPoint:
    """ZigZag 擺動點結構"""
    index: int          # K線索引
    price: float        # 價格
    swing_type: str     # 'high' 或 'low'
    timestamp: datetime # 時間戳
    confirmed: bool     # 是否已確認


@dataclass
class TechnicalIndicators:
    """技術指標集合"""
    rsi: float
    ema_fast: float
    ema_slow: float
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volume_sma: float


# ============================================================
# QuantConnect 數據適配器
# ============================================================

class DataAdapter:
    """
    將 QuantConnect 的 History 數據轉換為標準 DataFrame 格式
    """
    
    def __init__(self, algorithm: QCAlgorithm):
        """
        初始化數據適配器
        
        Args:
            algorithm: QCAlgorithm 實例
        """
        self.algorithm = algorithm
    
    def adapt_history(self, qc_history: pd.DataFrame, symbol: Symbol) -> pd.DataFrame:
        """
        將 QC History 轉換為標準 OHLCV DataFrame
        
        Args:
            qc_history: QuantConnect History 返回的 DataFrame
            symbol: 交易對符號
            
        Returns:
            標準化的 OHLCV DataFrame
        """
        if qc_history.empty:
            return pd.DataFrame()
        
        try:
            # QC History 可能是 MultiIndex (symbol, time)
            if isinstance(qc_history.index, pd.MultiIndex):
                df = qc_history.loc[symbol].copy()
            else:
                df = qc_history.copy()
            
            # 確保索引是 datetime
            df.index = pd.to_datetime(df.index)
            
            # 標準化列名 (QC 使用小寫)
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 確保必要列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.algorithm.Debug(f"警告: 缺少欄位 {col}")
                    return pd.DataFrame()
            
            return df[required_cols]
            
        except Exception as e:
            self.algorithm.Debug(f"數據適配錯誤: {str(e)}")
            return pd.DataFrame()
    
    def get_multi_timeframe_data(
        self, 
        symbol: Symbol, 
        resolutions: List[Resolution], 
        periods: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        獲取多時間框架數據
        
        Args:
            symbol: 交易對符號
            resolutions: Resolution 列表
            periods: 每個時間框架的K線數量
            
        Returns:
            {resolution_name: DataFrame} 字典
        """
        data = {}
        
        for resolution in resolutions:
            history = self.algorithm.History(symbol, periods, resolution)
            df = self.adapt_history(history, symbol)
            
            res_name = str(resolution).lower()
            data[res_name] = df
        
        return data


# ============================================================
# 即時 ZigZag 計算器 (不使用未來數據)
# ============================================================

class ZigZagCalculator:
    """
    即時 ZigZag 計算器
    使用價格回撤和確認K線邏輯，避免未來函數問題
    """
    
    def __init__(
        self,
        confirmation_bars: int = ZIGZAG_CONFIRMATION_BARS,
        min_retrace_percent: float = ZIGZAG_MIN_RETRACE_PERCENT
    ):
        """
        初始化 ZigZag 計算器
        
        Args:
            confirmation_bars: 確認K線數量
            min_retrace_percent: 最小回撤百分比
        """
        self.confirmation_bars = confirmation_bars
        self.min_retrace_percent = min_retrace_percent
    
    def calculate(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        計算 ZigZag 擺動點
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            SwingPoint 列表
        """
        if len(df) < self.confirmation_bars + 2:
            return []
        
        swing_points = []
        last_swing = None
        last_swing_type = None
        
        for i in range(self.confirmation_bars, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # 檢查是否為潛在高點
            is_potential_high = True
            for j in range(1, self.confirmation_bars + 1):
                if i - j >= 0 and df['high'].iloc[i - j] >= current_high:
                    is_potential_high = False
                    break
            
            # 檢查是否為潛在低點
            is_potential_low = True
            for j in range(1, self.confirmation_bars + 1):
                if i - j >= 0 and df['low'].iloc[i - j] <= current_low:
                    is_potential_low = False
                    break
            
            # 確認擺動點
            if is_potential_high:
                # 檢查是否有足夠的回撤
                if last_swing is None or last_swing_type == 'low':
                    if self._check_retrace(df, i, 'high'):
                        swing_point = SwingPoint(
                            index=i,
                            price=current_high,
                            swing_type='high',
                            timestamp=df.index[i],
                            confirmed=True
                        )
                        swing_points.append(swing_point)
                        last_swing = current_high
                        last_swing_type = 'high'
            
            if is_potential_low:
                if last_swing is None or last_swing_type == 'high':
                    if self._check_retrace(df, i, 'low'):
                        swing_point = SwingPoint(
                            index=i,
                            price=current_low,
                            swing_type='low',
                            timestamp=df.index[i],
                            confirmed=True
                        )
                        swing_points.append(swing_point)
                        last_swing = current_low
                        last_swing_type = 'low'
        
        return swing_points
    
    def _check_retrace(self, df: pd.DataFrame, index: int, swing_type: str) -> bool:
        """
        檢查是否有足夠的價格回撤
        
        Args:
            df: OHLCV DataFrame
            index: 當前索引
            swing_type: 'high' 或 'low'
            
        Returns:
            是否滿足回撤條件
        """
        lookback = min(20, index)
        
        if swing_type == 'high':
            current_price = df['high'].iloc[index]
            recent_low = df['low'].iloc[index - lookback:index].min()
            retrace_pct = (current_price - recent_low) / recent_low * 100
        else:
            current_price = df['low'].iloc[index]
            recent_high = df['high'].iloc[index - lookback:index].max()
            retrace_pct = (recent_high - current_price) / recent_high * 100
        
        return retrace_pct >= self.min_retrace_percent


# ============================================================
# 技術指標計算器
# ============================================================

class IndicatorCalculator:
    """計算各種技術指標"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """
        計算 RSI 指標
        
        Args:
            prices: 價格序列 (通常是收盤價)
            period: RSI 週期
            
        Returns:
            RSI 序列
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        計算 EMA
        
        Args:
            prices: 價格序列
            period: EMA 週期
            
        Returns:
            EMA 序列
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        """
        計算 ATR (Average True Range)
        
        Args:
            df: OHLCV DataFrame
            period: ATR 週期
            
        Returns:
            ATR 序列
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = BB_PERIOD,
        std_dev: float = BB_STD_DEV
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        計算布林帶
        
        Args:
            prices: 價格序列
            period: 週期
            std_dev: 標準差倍數
            
        Returns:
            (上軌, 中軌, 下軌) 元組
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def detect_rsi_divergence(
        prices: pd.Series,
        rsi: pd.Series,
        lookback: int = 20
    ) -> str:
        """
        偵測 RSI 背離
        
        Args:
            prices: 價格序列
            rsi: RSI 序列
            lookback: 回溯週期
            
        Returns:
            "bullish", "bearish", 或 "none"
        """
        if len(prices) < lookback or len(rsi) < lookback:
            return "none"
        
        recent_prices = prices.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        
        # 找出價格高低點
        price_high_idx = recent_prices.idxmax()
        price_low_idx = recent_prices.idxmin()
        
        # 檢查看跌背離 (價格創新高但 RSI 沒有)
        if price_high_idx == recent_prices.index[-1]:
            prev_highs = recent_prices.iloc[:-5].nlargest(1)
            if len(prev_highs) > 0:
                prev_high_idx = prev_highs.index[0]
                if recent_prices.iloc[-1] > prev_highs.iloc[0]:
                    if rsi.loc[recent_prices.index[-1]] < rsi.loc[prev_high_idx]:
                        return "bearish"
        
        # 檢查看漲背離 (價格創新低但 RSI 沒有)
        if price_low_idx == recent_prices.index[-1]:
            prev_lows = recent_prices.iloc[:-5].nsmallest(1)
            if len(prev_lows) > 0:
                prev_low_idx = prev_lows.index[0]
                if recent_prices.iloc[-1] < prev_lows.iloc[0]:
                    if rsi.loc[recent_prices.index[-1]] > rsi.loc[prev_low_idx]:
                        return "bullish"
        
        return "none"
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        為 DataFrame 添加所有技術指標
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            添加了指標的 DataFrame
        """
        df = df.copy()
        
        # RSI
        df['rsi'] = IndicatorCalculator.calculate_rsi(df['close'])
        
        # EMA
        df['ema_fast'] = IndicatorCalculator.calculate_ema(df['close'], EMA_FAST)
        df['ema_slow'] = IndicatorCalculator.calculate_ema(df['close'], EMA_SLOW)
        
        # ATR
        df['atr'] = IndicatorCalculator.calculate_atr(df)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = \
            IndicatorCalculator.calculate_bollinger_bands(df['close'])
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Trend direction
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        return df


# ============================================================
# 入場確認工具
# ============================================================

class EntryConfirmation:
    """入場確認K線形態檢測"""
    
    @staticmethod
    def is_hammer(
        open_price: float,
        high: float,
        low: float,
        close: float,
        atr: float
    ) -> bool:
        """
        檢測錘子線 (Hammer)
        
        Args:
            open_price, high, low, close: OHLC 價格
            atr: ATR 值
            
        Returns:
            是否為錘子線
        """
        body = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        # 錘子線條件:
        # 1. 下影線至少是實體的2倍
        # 2. 上影線很小
        # 3. 實體不能太小
        if body < atr * 0.1:  # 十字星排除
            return False
            
        if lower_shadow >= body * 2 and upper_shadow <= body * 0.5:
            return True
        
        return False
    
    @staticmethod
    def is_bullish_engulfing(
        prev_open: float,
        prev_close: float,
        curr_open: float,
        curr_close: float
    ) -> bool:
        """
        檢測看漲吞沒 (Bullish Engulfing)
        
        Args:
            prev_*: 前一根K線價格
            curr_*: 當前K線價格
            
        Returns:
            是否為看漲吞沒
        """
        # 前一根是陰線
        prev_is_bearish = prev_close < prev_open
        
        # 當前是陽線
        curr_is_bullish = curr_close > curr_open
        
        # 當前實體完全吞沒前一根實體
        curr_engulfs = curr_open <= prev_close and curr_close >= prev_open
        
        return prev_is_bearish and curr_is_bullish and curr_engulfs
    
    @staticmethod
    def is_bearish_engulfing(
        prev_open: float,
        prev_close: float,
        curr_open: float,
        curr_close: float
    ) -> bool:
        """
        檢測看跌吞沒 (Bearish Engulfing)
        """
        prev_is_bullish = prev_close > prev_open
        curr_is_bearish = curr_close < curr_open
        curr_engulfs = curr_open >= prev_close and curr_close <= prev_open
        
        return prev_is_bullish and curr_is_bearish and curr_engulfs
    
    @staticmethod
    def is_shooting_star(
        open_price: float,
        high: float,
        low: float,
        close: float,
        atr: float
    ) -> bool:
        """
        檢測射擊之星 (Shooting Star)
        """
        body = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        if body < atr * 0.1:
            return False
            
        if upper_shadow >= body * 2 and lower_shadow <= body * 0.5:
            return True
        
        return False
    
    @staticmethod
    def check_entry_candle(
        df: pd.DataFrame,
        pattern_direction: str,  # "bullish" or "bearish"
        confirmation_type: str = "any"
    ) -> bool:
        """
        檢查是否有入場確認K線
        
        Args:
            df: OHLCV DataFrame (至少2根K線)
            pattern_direction: 形態方向
            confirmation_type: 確認類型 ("hammer", "engulfing", "any")
            
        Returns:
            是否有確認信號
        """
        if len(df) < 2:
            return False
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 計算 ATR
        atr = IndicatorCalculator.calculate_atr(df).iloc[-1]
        if pd.isna(atr):
            atr = (curr['high'] - curr['low']) * 2  # 備用計算
        
        if pattern_direction == "bullish":
            # 看漲形態需要看漲確認
            is_hammer = EntryConfirmation.is_hammer(
                curr['open'], curr['high'], curr['low'], curr['close'], atr
            )
            is_engulfing = EntryConfirmation.is_bullish_engulfing(
                prev['open'], prev['close'], curr['open'], curr['close']
            )
            
            if confirmation_type == "hammer":
                return is_hammer
            elif confirmation_type == "engulfing":
                return is_engulfing
            else:  # "any"
                return is_hammer or is_engulfing
        
        else:  # bearish
            is_shooting_star = EntryConfirmation.is_shooting_star(
                curr['open'], curr['high'], curr['low'], curr['close'], atr
            )
            is_engulfing = EntryConfirmation.is_bearish_engulfing(
                prev['open'], prev['close'], curr['open'], curr['close']
            )
            
            if confirmation_type == "shooting_star":
                return is_shooting_star
            elif confirmation_type == "engulfing":
                return is_engulfing
            else:  # "any"
                return is_shooting_star or is_engulfing


# ============================================================
# 通用工具函數
# ============================================================

def calculate_fibonacci_level(
    start_price: float,
    end_price: float,
    ratio: float
) -> float:
    """
    計算 Fibonacci 回撤/延伸水平
    
    Args:
        start_price: 起始價格
        end_price: 結束價格
        ratio: Fibonacci 比例
        
    Returns:
        計算出的價格水平
    """
    return start_price + (end_price - start_price) * ratio


def is_within_tolerance(
    actual_ratio: float,
    target_min: float,
    target_max: float,
    tolerance: float
) -> bool:
    """
    檢查比例是否在容差範圍內
    
    Args:
        actual_ratio: 實際比例
        target_min: 目標最小值
        target_max: 目標最大值
        tolerance: 容差
        
    Returns:
        是否在範圍內
    """
    return (target_min - tolerance) <= actual_ratio <= (target_max + tolerance)


def price_in_zone(
    price: float,
    zone_low: float,
    zone_high: float,
    buffer_percent: float = 0.5
) -> bool:
    """
    檢查價格是否在指定區域內 (含緩衝)
    
    Args:
        price: 當前價格
        zone_low: 區域下限
        zone_high: 區域上限
        buffer_percent: 緩衝百分比
        
    Returns:
        是否在區域內
    """
    buffer = (zone_high - zone_low) * buffer_percent / 100
    return (zone_low - buffer) <= price <= (zone_high + buffer)

