# -*- coding: utf-8 -*-
"""
智慧資金概念分析模組 (Smart Money Concepts Analysis Module)
識別 Order Block、Fair Value Gap (FVG)、Break of Structure (BOS)
用於過濾諧波形態，只交易具有機構級匯合的設置
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config import (
    ORDER_BLOCK_LOOKBACK,
    FVG_LOOKBACK,
    BOS_SENSITIVITY,
    MIN_ORDER_BLOCK_SIZE_ATR
)
from utils import IndicatorCalculator


# ============================================================
# 枚舉與數據結構
# ============================================================

class ZoneType(Enum):
    """SMC 區域類型"""
    ORDER_BLOCK = "order_block"
    FVG = "fair_value_gap"
    BREAKER_BLOCK = "breaker_block"


class ZoneDirection(Enum):
    """區域方向"""
    BULLISH = "bullish"   # 看漲 (供應區)
    BEARISH = "bearish"   # 看跌 (需求區)


@dataclass
class SMCZone:
    """SMC 區域結構"""
    zone_type: ZoneType
    direction: ZoneDirection
    
    # 區域價格範圍
    price_low: float
    price_high: float
    
    # 時間資訊
    start_index: int
    start_time: datetime
    
    # 區域狀態
    is_mitigated: bool = False    # 是否已被觸及
    strength: float = 1.0          # 強度 (0-1)
    touch_count: int = 0           # 被觸及次數
    
    # 額外屬性
    displacement_size: float = 0   # 推進力度
    
    def contains_price(self, price: float, buffer_percent: float = 0.5) -> bool:
        """檢查價格是否在區域內"""
        buffer = (self.price_high - self.price_low) * buffer_percent / 100
        return (self.price_low - buffer) <= price <= (self.price_high + buffer)
    
    def get_midpoint(self) -> float:
        """獲取區域中點"""
        return (self.price_high + self.price_low) / 2


@dataclass 
class StructurePoint:
    """市場結構點"""
    index: int
    price: float
    point_type: str   # 'HH', 'HL', 'LH', 'LL'
    timestamp: datetime


@dataclass
class StructureBreak:
    """結構突破事件"""
    break_type: str   # 'BOS' (Break of Structure) 或 'CHoCH' (Change of Character)
    direction: str    # 'bullish' 或 'bearish'
    break_price: float
    break_index: int
    break_time: datetime


# ============================================================
# SMC 分析引擎
# ============================================================

class SMCAnalyzer:
    """
    智慧資金概念分析器
    
    主要功能:
    1. 識別 Order Blocks (訂單塊)
    2. 識別 Fair Value Gaps (公允價值缺口)
    3. 檢測 Break of Structure (結構突破)
    4. 驗證 PRZ 與 SMC 區域的匯合
    """
    
    def __init__(
        self,
        ob_lookback: int = ORDER_BLOCK_LOOKBACK,
        fvg_lookback: int = FVG_LOOKBACK,
        bos_sensitivity: float = BOS_SENSITIVITY
    ):
        """
        初始化 SMC 分析器
        
        Args:
            ob_lookback: Order Block 回溯K線數
            fvg_lookback: FVG 回溯K線數
            bos_sensitivity: BOS 靈敏度 (ATR倍數)
        """
        self.ob_lookback = ob_lookback
        self.fvg_lookback = fvg_lookback
        self.bos_sensitivity = bos_sensitivity
    
    # ========================================
    # Order Block 識別
    # ========================================
    
    def find_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> List[SMCZone]:
        """
        識別 Order Blocks
        
        Order Block 定義:
        - 在強勁推進 (displacement) 之前的最後一根反向K線
        - 推進必須突破之前的結構 (BOS)
        
        Args:
            df: OHLCV DataFrame
            lookback: 回溯K線數 (預設使用配置值)
            
        Returns:
            Order Block 區域列表
        """
        if lookback is None:
            lookback = self.ob_lookback
        
        order_blocks = []
        
        if len(df) < lookback + 3:
            return order_blocks
        
        # 計算 ATR 用於判斷推進強度
        atr = IndicatorCalculator.calculate_atr(df)
        
        # 從最近的K線向前掃描
        start_idx = max(2, len(df) - lookback)
        
        for i in range(start_idx, len(df) - 2):
            # 檢查是否有強勁推進 (看漲)
            bullish_ob = self._check_bullish_order_block(df, i, atr)
            if bullish_ob:
                order_blocks.append(bullish_ob)
            
            # 檢查是否有強勁推進 (看跌)
            bearish_ob = self._check_bearish_order_block(df, i, atr)
            if bearish_ob:
                order_blocks.append(bearish_ob)
        
        # 移除已被完全穿透的 Order Blocks
        current_price = df['close'].iloc[-1]
        order_blocks = self._filter_mitigated_zones(order_blocks, current_price)
        
        return order_blocks
    
    def _check_bullish_order_block(
        self,
        df: pd.DataFrame,
        idx: int,
        atr: pd.Series
    ) -> Optional[SMCZone]:
        """
        檢查看漲 Order Block (需求區)
        
        條件:
        1. 當前K線是陰線 (下跌)
        2. 後續K線是強勁陽線 (推進)
        3. 推進突破了近期高點 (BOS)
        """
        if idx + 2 >= len(df):
            return None
        
        current_atr = atr.iloc[idx] if not pd.isna(atr.iloc[idx]) else 0
        if current_atr == 0:
            return None
        
        # 當前K線 (潛在 OB)
        ob_candle = df.iloc[idx]
        
        # 檢查是否為陰線
        if ob_candle['close'] >= ob_candle['open']:
            return None
        
        # 檢查K線大小是否足夠
        ob_size = ob_candle['high'] - ob_candle['low']
        if ob_size < current_atr * MIN_ORDER_BLOCK_SIZE_ATR:
            return None
        
        # 檢查後續推進
        next_candle = df.iloc[idx + 1]
        displacement = next_candle['close'] - next_candle['open']
        
        # 推進必須是強勁的陽線
        if displacement < current_atr * self.bos_sensitivity:
            return None
        
        # 檢查是否突破近期高點 (BOS)
        recent_high = df['high'].iloc[max(0, idx-20):idx].max()
        if next_candle['high'] <= recent_high:
            return None
        
        # 創建 Order Block
        return SMCZone(
            zone_type=ZoneType.ORDER_BLOCK,
            direction=ZoneDirection.BULLISH,
            price_low=ob_candle['low'],
            price_high=ob_candle['high'],
            start_index=idx,
            start_time=df.index[idx],
            strength=min(1.0, displacement / (current_atr * 3)),
            displacement_size=displacement
        )
    
    def _check_bearish_order_block(
        self,
        df: pd.DataFrame,
        idx: int,
        atr: pd.Series
    ) -> Optional[SMCZone]:
        """
        檢查看跌 Order Block (供應區)
        
        條件:
        1. 當前K線是陽線 (上漲)
        2. 後續K線是強勁陰線 (推進)
        3. 推進突破了近期低點 (BOS)
        """
        if idx + 2 >= len(df):
            return None
        
        current_atr = atr.iloc[idx] if not pd.isna(atr.iloc[idx]) else 0
        if current_atr == 0:
            return None
        
        ob_candle = df.iloc[idx]
        
        # 檢查是否為陽線
        if ob_candle['close'] <= ob_candle['open']:
            return None
        
        ob_size = ob_candle['high'] - ob_candle['low']
        if ob_size < current_atr * MIN_ORDER_BLOCK_SIZE_ATR:
            return None
        
        next_candle = df.iloc[idx + 1]
        displacement = next_candle['open'] - next_candle['close']  # 陰線推進
        
        if displacement < current_atr * self.bos_sensitivity:
            return None
        
        recent_low = df['low'].iloc[max(0, idx-20):idx].min()
        if next_candle['low'] >= recent_low:
            return None
        
        return SMCZone(
            zone_type=ZoneType.ORDER_BLOCK,
            direction=ZoneDirection.BEARISH,
            price_low=ob_candle['low'],
            price_high=ob_candle['high'],
            start_index=idx,
            start_time=df.index[idx],
            strength=min(1.0, displacement / (current_atr * 3)),
            displacement_size=displacement
        )
    
    # ========================================
    # Fair Value Gap 識別
    # ========================================
    
    def find_fair_value_gaps(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> List[SMCZone]:
        """
        識別 Fair Value Gaps (FVG)
        
        FVG 定義:
        - 三根K線序列
        - 第一根和第三根K線的影線不重疊
        - 形成價格「缺口」(效率不足)
        
        Args:
            df: OHLCV DataFrame
            lookback: 回溯K線數
            
        Returns:
            FVG 區域列表
        """
        if lookback is None:
            lookback = self.fvg_lookback
        
        fvgs = []
        
        if len(df) < lookback + 3:
            return fvgs
        
        start_idx = max(1, len(df) - lookback)
        
        for i in range(start_idx, len(df) - 1):
            if i < 1:
                continue
            
            # 三根K線
            candle_1 = df.iloc[i - 1]  # 第一根
            candle_2 = df.iloc[i]      # 中間根
            candle_3 = df.iloc[i + 1]  # 第三根
            
            # 檢查看漲 FVG (Bullish FVG)
            # 第一根的高點 < 第三根的低點 = 向上缺口
            if candle_1['high'] < candle_3['low']:
                fvg = SMCZone(
                    zone_type=ZoneType.FVG,
                    direction=ZoneDirection.BULLISH,
                    price_low=candle_1['high'],
                    price_high=candle_3['low'],
                    start_index=i,
                    start_time=df.index[i],
                    strength=self._calculate_fvg_strength(
                        candle_1['high'], candle_3['low'], candle_2
                    )
                )
                fvgs.append(fvg)
            
            # 檢查看跌 FVG (Bearish FVG)
            # 第一根的低點 > 第三根的高點 = 向下缺口
            if candle_1['low'] > candle_3['high']:
                fvg = SMCZone(
                    zone_type=ZoneType.FVG,
                    direction=ZoneDirection.BEARISH,
                    price_low=candle_3['high'],
                    price_high=candle_1['low'],
                    start_index=i,
                    start_time=df.index[i],
                    strength=self._calculate_fvg_strength(
                        candle_3['high'], candle_1['low'], candle_2
                    )
                )
                fvgs.append(fvg)
        
        # 過濾已被填補的 FVG
        current_price = df['close'].iloc[-1]
        fvgs = self._filter_mitigated_zones(fvgs, current_price)
        
        return fvgs
    
    def _calculate_fvg_strength(
        self,
        gap_low: float,
        gap_high: float,
        middle_candle: pd.Series
    ) -> float:
        """計算 FVG 強度"""
        gap_size = gap_high - gap_low
        candle_size = middle_candle['high'] - middle_candle['low']
        
        if candle_size == 0:
            return 0.5
        
        # FVG 佔中間K線的比例越大，強度越高
        strength = gap_size / candle_size
        return min(1.0, strength)
    
    # ========================================
    # 結構分析
    # ========================================
    
    def detect_structure_breaks(
        self,
        df: pd.DataFrame,
        swing_highs: List[float],
        swing_lows: List[float]
    ) -> List[StructureBreak]:
        """
        檢測結構突破 (BOS/CHoCH)
        
        Args:
            df: OHLCV DataFrame
            swing_highs: 擺動高點列表
            swing_lows: 擺動低點列表
            
        Returns:
            結構突破事件列表
        """
        breaks = []
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return breaks
        
        # 這裡簡化實作：檢查最近的價格是否突破關鍵水平
        recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
        recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
        
        current_close = df['close'].iloc[-1]
        
        # 突破近期高點 = 看漲 BOS
        if current_close > recent_high:
            breaks.append(StructureBreak(
                break_type='BOS',
                direction='bullish',
                break_price=recent_high,
                break_index=len(df) - 1,
                break_time=df.index[-1]
            ))
        
        # 突破近期低點 = 看跌 BOS
        if current_close < recent_low:
            breaks.append(StructureBreak(
                break_type='BOS',
                direction='bearish',
                break_price=recent_low,
                break_index=len(df) - 1,
                break_time=df.index[-1]
            ))
        
        return breaks
    
    # ========================================
    # 匯合驗證
    # ========================================
    
    def check_prz_confluence(
        self,
        prz_low: float,
        prz_high: float,
        pattern_direction: str,
        order_blocks: List[SMCZone],
        fvgs: List[SMCZone],
        buffer_percent: float = 1.0
    ) -> Tuple[bool, List[SMCZone]]:
        """
        檢查 PRZ 是否與 SMC 區域匯合
        
        這是 "Alpha" 過濾層:
        - 只有當 PRZ 落在 Order Block 或 FVG 內時才有效
        - 方向必須一致 (看漲形態需要看漲 OB/FVG)
        
        Args:
            prz_low: PRZ 下限
            prz_high: PRZ 上限
            pattern_direction: 形態方向 ("bullish" 或 "bearish")
            order_blocks: Order Block 列表
            fvgs: FVG 列表
            buffer_percent: 緩衝百分比
            
        Returns:
            (是否匯合, 匯合的區域列表)
        """
        confluent_zones = []
        
        # 確定需要的 SMC 方向
        if pattern_direction == "bullish":
            required_direction = ZoneDirection.BULLISH
        else:
            required_direction = ZoneDirection.BEARISH
        
        # 檢查 Order Blocks
        for ob in order_blocks:
            if ob.direction != required_direction:
                continue
            
            if self._zones_overlap(
                prz_low, prz_high,
                ob.price_low, ob.price_high,
                buffer_percent
            ):
                confluent_zones.append(ob)
        
        # 檢查 FVGs
        for fvg in fvgs:
            if fvg.direction != required_direction:
                continue
            
            if self._zones_overlap(
                prz_low, prz_high,
                fvg.price_low, fvg.price_high,
                buffer_percent
            ):
                confluent_zones.append(fvg)
        
        has_confluence = len(confluent_zones) > 0
        
        return has_confluence, confluent_zones
    
    def _zones_overlap(
        self,
        zone1_low: float,
        zone1_high: float,
        zone2_low: float,
        zone2_high: float,
        buffer_percent: float
    ) -> bool:
        """檢查兩個區域是否重疊"""
        buffer1 = (zone1_high - zone1_low) * buffer_percent / 100
        buffer2 = (zone2_high - zone2_low) * buffer_percent / 100
        
        # 擴展區域
        z1_low = zone1_low - buffer1
        z1_high = zone1_high + buffer1
        z2_low = zone2_low - buffer2
        z2_high = zone2_high + buffer2
        
        # 檢查重疊
        return not (z1_high < z2_low or z2_high < z1_low)
    
    def _filter_mitigated_zones(
        self,
        zones: List[SMCZone],
        current_price: float
    ) -> List[SMCZone]:
        """過濾已被價格穿透的區域"""
        active_zones = []
        
        for zone in zones:
            # 如果價格完全穿透區域，標記為已緩解
            if zone.direction == ZoneDirection.BULLISH:
                # 看漲區域在價格下方，如果價格跌破則無效
                if current_price < zone.price_low:
                    zone.is_mitigated = True
            else:
                # 看跌區域在價格上方，如果價格突破則無效
                if current_price > zone.price_high:
                    zone.is_mitigated = True
            
            if not zone.is_mitigated:
                active_zones.append(zone)
        
        return active_zones
    
    # ========================================
    # 輔助方法
    # ========================================
    
    def get_zone_summary(self, zone: SMCZone) -> str:
        """生成區域摘要"""
        type_emoji = "📦" if zone.zone_type == ZoneType.ORDER_BLOCK else "📊"
        direction_emoji = "🟢" if zone.direction == ZoneDirection.BULLISH else "🔴"
        
        return f"{type_emoji}{direction_emoji} {zone.zone_type.value} @ {zone.price_low:.2f}-{zone.price_high:.2f} (強度: {zone.strength:.2%})"
    
    def get_all_zones(self, df: pd.DataFrame) -> Tuple[List[SMCZone], List[SMCZone]]:
        """
        獲取所有 SMC 區域
        
        Returns:
            (order_blocks, fvgs)
        """
        order_blocks = self.find_order_blocks(df)
        fvgs = self.find_fair_value_gaps(df)
        
        return order_blocks, fvgs

