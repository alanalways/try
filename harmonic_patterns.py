# -*- coding: utf-8 -*-
"""
諧波形態辨識引擎 (Harmonic Pattern Recognition Engine)
識別 Gartley, Bat, Butterfly, Crab 等諧波形態
支援「發展中形態」掃描，預測 D 點位置
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import TOLERANCE, HARMONIC_RATIOS, MAX_PATTERN_BARS
from utils import SwingPoint, is_within_tolerance, calculate_fibonacci_level


# ============================================================
# 數據結構定義
# ============================================================

@dataclass
class PRZone:
    """潛在反轉區域 (Potential Reversal Zone)"""
    price_low: float      # PRZ 下限
    price_high: float     # PRZ 上限
    price_center: float   # PRZ 中心點
    
    def contains(self, price: float, buffer_percent: float = 2.0) -> bool:
        """檢查價格是否在 PRZ 範圍內"""
        buffer = (self.price_high - self.price_low) * buffer_percent / 100
        return (self.price_low - buffer) <= price <= (self.price_high + buffer)
    
    def distance_percent(self, price: float) -> float:
        """計算價格與 PRZ 中心的距離百分比"""
        return abs(price - self.price_center) / self.price_center * 100


@dataclass
class HarmonicPattern:
    """諧波形態結構"""
    pattern_type: str         # 形態類型 (Gartley, Bat, etc.)
    direction: str            # 方向 (bullish, bearish)
    
    # XABCD 五個點
    X: SwingPoint
    A: SwingPoint
    B: SwingPoint
    C: SwingPoint
    D: Optional[SwingPoint]   # 可能是預測點
    
    # Fibonacci 比例
    XAB_ratio: float
    ABC_ratio: float
    BCD_ratio: float
    XAD_ratio: float
    
    # PRZ (潛在反轉區)
    prz: PRZone
    
    # 形態品質
    clarity_score: float      # 清晰度分數 (0-1)
    is_developing: bool       # 是否為發展中形態
    
    # 時間戳
    detected_at: datetime = field(default_factory=datetime.now)
    
    def get_stop_loss(self) -> float:
        """計算止損價位 (超過 X 點)"""
        if self.direction == "bullish":
            return self.X.price * 0.99  # X 點下方 1%
        else:
            return self.X.price * 1.01  # X 點上方 1%
    
    def get_take_profit_levels(self) -> Tuple[float, float, float]:
        """
        計算止盈價位
        TP1: D 到 C 的 38.2%
        TP2: D 到 C 的 61.8%  
        TP3: D 到 A 的 78.6%
        """
        d_price = self.prz.price_center
        c_price = self.C.price
        a_price = self.A.price
        
        if self.direction == "bullish":
            tp1 = d_price + (c_price - d_price) * 0.382
            tp2 = d_price + (c_price - d_price) * 0.618
            tp3 = d_price + (a_price - d_price) * 0.786
        else:
            tp1 = d_price - (d_price - c_price) * 0.382
            tp2 = d_price - (d_price - c_price) * 0.618
            tp3 = d_price - (d_price - a_price) * 0.786
        
        return tp1, tp2, tp3


# ============================================================
# 諧波形態辨識引擎
# ============================================================

class HarmonicPatternEngine:
    """
    諧波形態辨識引擎
    
    主要功能:
    1. 掃描已完成形態 (XABCD 五點都確認)
    2. 掃描發展中形態 (XABC 確認，預測 D 點)
    3. 計算形態品質分數
    """
    
    def __init__(self, tolerance: float = TOLERANCE):
        """
        初始化引擎
        
        Args:
            tolerance: Fibonacci 比例容差
        """
        self.tolerance = tolerance
        self.pattern_definitions = HARMONIC_RATIOS
    
    # ========================================
    # 主要掃描方法
    # ========================================
    
    def scan_developing_patterns(
        self,
        swing_points: List[SwingPoint],
        current_price: float,
        max_patterns: int = 5
    ) -> List[HarmonicPattern]:
        """
        掃描發展中的諧波形態 (XABC → 預測 D)
        
        這是最重要的方法！它識別已確認的 XABC，
        然後計算 D 點的潛在反轉區 (PRZ)
        
        Args:
            swing_points: ZigZag 擺動點列表
            current_price: 當前價格
            max_patterns: 最大返回形態數量
            
        Returns:
            發展中形態列表
        """
        patterns = []
        n = len(swing_points)
        
        if n < 4:
            return patterns
        
        # 從最新點向前搜尋 (優化: 只搜尋最近的形態)
        # 這避免了 O(N^5) 的問題
        for c_idx in range(n - 1, 3, -1):
            C = swing_points[c_idx]
            
            # B 點 (C 前一個反向點)
            for b_idx in range(c_idx - 1, 2, -1):
                B = swing_points[b_idx]
                
                # B 和 C 必須是反向的
                if B.swing_type == C.swing_type:
                    continue
                
                # A 點
                for a_idx in range(b_idx - 1, 1, -1):
                    A = swing_points[a_idx]
                    
                    if A.swing_type == B.swing_type:
                        continue
                    
                    # X 點
                    for x_idx in range(a_idx - 1, -1, -1):
                        X = swing_points[x_idx]
                        
                        if X.swing_type == A.swing_type:
                            continue
                        
                        # 檢查形態跨度
                        if C.index - X.index > MAX_PATTERN_BARS:
                            break
                        
                        # 嘗試識別形態
                        pattern = self._identify_developing_pattern(
                            X, A, B, C, current_price
                        )
                        
                        if pattern is not None:
                            patterns.append(pattern)
                            
                            if len(patterns) >= max_patterns:
                                return patterns
        
        # 按品質分數排序
        patterns.sort(key=lambda p: p.clarity_score, reverse=True)
        
        return patterns[:max_patterns]
    
    def scan_completed_patterns(
        self,
        swing_points: List[SwingPoint],
        max_patterns: int = 10
    ) -> List[HarmonicPattern]:
        """
        掃描已完成的諧波形態 (XABCD 五點都確認)
        
        Args:
            swing_points: ZigZag 擺動點列表
            max_patterns: 最大返回數量
            
        Returns:
            已完成形態列表
        """
        patterns = []
        n = len(swing_points)
        
        if n < 5:
            return patterns
        
        # 從最新點向前搜尋
        for d_idx in range(n - 1, 3, -1):
            D = swing_points[d_idx]
            
            for c_idx in range(d_idx - 1, 2, -1):
                C = swing_points[c_idx]
                
                if C.swing_type == D.swing_type:
                    continue
                
                for b_idx in range(c_idx - 1, 1, -1):
                    B = swing_points[b_idx]
                    
                    if B.swing_type == C.swing_type:
                        continue
                    
                    for a_idx in range(b_idx - 1, 0, -1):
                        A = swing_points[a_idx]
                        
                        if A.swing_type == B.swing_type:
                            continue
                        
                        for x_idx in range(a_idx - 1, -1, -1):
                            X = swing_points[x_idx]
                            
                            if X.swing_type == A.swing_type:
                                continue
                            
                            if D.index - X.index > MAX_PATTERN_BARS:
                                break
                            
                            pattern = self._identify_completed_pattern(
                                X, A, B, C, D
                            )
                            
                            if pattern is not None:
                                patterns.append(pattern)
                                
                                if len(patterns) >= max_patterns:
                                    return patterns
        
        patterns.sort(key=lambda p: p.clarity_score, reverse=True)
        return patterns[:max_patterns]
    
    # ========================================
    # 內部識別方法
    # ========================================
    
    def _identify_developing_pattern(
        self,
        X: SwingPoint,
        A: SwingPoint,
        B: SwingPoint,
        C: SwingPoint,
        current_price: float
    ) -> Optional[HarmonicPattern]:
        """
        識別發展中形態 (XABC → 預測 D)
        
        Args:
            X, A, B, C: 已確認的四個點
            current_price: 當前價格
            
        Returns:
            識別到的形態或 None
        """
        # 計算實際比例
        XA = A.price - X.price
        AB = B.price - A.price
        BC = C.price - B.price
        
        if XA == 0 or AB == 0:
            return None
        
        XAB_ratio = abs(AB / XA)
        ABC_ratio = abs(BC / AB)
        
        # 確定方向
        if X.swing_type == 'low':  # X 是低點 → 看漲形態
            direction = "bullish"
        else:  # X 是高點 → 看跌形態
            direction = "bearish"
        
        # 遍歷所有形態定義，找出匹配的
        for pattern_name, ratios in self.pattern_definitions.items():
            # 檢查 XAB 比例
            if not is_within_tolerance(
                XAB_ratio,
                ratios["XAB"]["min"],
                ratios["XAB"]["max"],
                self.tolerance
            ):
                continue
            
            # 檢查 ABC 比例
            if not is_within_tolerance(
                ABC_ratio,
                ratios["ABC"]["min"],
                ratios["ABC"]["max"],
                self.tolerance
            ):
                continue
            
            # 計算預測的 D 點 (PRZ)
            prz = self._calculate_prz(X, A, B, C, ratios, direction)
            
            if prz is None:
                continue
            
            # 檢查當前價格是否接近 PRZ
            distance = prz.distance_percent(current_price)
            
            # 只有當價格開始接近 PRZ 時才報告
            # (避免報告太早的形態)
            if distance > 10.0:  # 距離 PRZ 超過 10%
                continue
            
            # 計算品質分數
            clarity_score = self._calculate_clarity_score(
                XAB_ratio, ABC_ratio, 0, 0,  # BCD 和 XAD 還不知道
                ratios
            )
            
            # 創建形態對象
            pattern = HarmonicPattern(
                pattern_type=pattern_name,
                direction=direction,
                X=X, A=A, B=B, C=C, D=None,
                XAB_ratio=XAB_ratio,
                ABC_ratio=ABC_ratio,
                BCD_ratio=0,  # 待定
                XAD_ratio=ratios["XAD"]["min"],  # 使用目標值
                prz=prz,
                clarity_score=clarity_score,
                is_developing=True
            )
            
            return pattern
        
        return None
    
    def _identify_completed_pattern(
        self,
        X: SwingPoint,
        A: SwingPoint,
        B: SwingPoint,
        C: SwingPoint,
        D: SwingPoint
    ) -> Optional[HarmonicPattern]:
        """
        識別已完成形態
        
        Args:
            X, A, B, C, D: 五個已確認的點
            
        Returns:
            識別到的形態或 None
        """
        # 計算所有比例
        XA = A.price - X.price
        AB = B.price - A.price
        BC = C.price - B.price
        CD = D.price - C.price
        XD = D.price - X.price
        
        if XA == 0 or AB == 0 or BC == 0:
            return None
        
        XAB_ratio = abs(AB / XA)
        ABC_ratio = abs(BC / AB)
        BCD_ratio = abs(CD / BC)
        XAD_ratio = abs(XD / XA)
        
        # 確定方向
        if X.swing_type == 'low':
            direction = "bullish"
        else:
            direction = "bearish"
        
        # 遍歷形態定義
        for pattern_name, ratios in self.pattern_definitions.items():
            if not self._check_all_ratios(
                XAB_ratio, ABC_ratio, BCD_ratio, XAD_ratio, ratios
            ):
                continue
            
            # 創建 PRZ (D 點附近)
            prz = PRZone(
                price_low=D.price * 0.99,
                price_high=D.price * 1.01,
                price_center=D.price
            )
            
            # 計算品質分數
            clarity_score = self._calculate_clarity_score(
                XAB_ratio, ABC_ratio, BCD_ratio, XAD_ratio, ratios
            )
            
            pattern = HarmonicPattern(
                pattern_type=pattern_name,
                direction=direction,
                X=X, A=A, B=B, C=C, D=D,
                XAB_ratio=XAB_ratio,
                ABC_ratio=ABC_ratio,
                BCD_ratio=BCD_ratio,
                XAD_ratio=XAD_ratio,
                prz=prz,
                clarity_score=clarity_score,
                is_developing=False
            )
            
            return pattern
        
        return None
    
    def _calculate_prz(
        self,
        X: SwingPoint,
        A: SwingPoint,
        B: SwingPoint,
        C: SwingPoint,
        ratios: Dict,
        direction: str
    ) -> Optional[PRZone]:
        """
        計算潛在反轉區 (PRZ)
        
        PRZ 是多個 Fibonacci 水平的匯合區:
        1. XA 延伸的 XAD 比例
        2. BC 延伸的 BCD 比例
        """
        XA = A.price - X.price
        BC = C.price - B.price
        
        # 計算 D 點的兩個潛在位置
        # 基於 XAD 比例
        xad_target = ratios["XAD"]["min"]
        d_from_xad = X.price + XA * xad_target
        
        # 基於 BCD 比例
        bcd_min = ratios["BCD"]["min"]
        bcd_max = ratios["BCD"]["max"]
        
        if direction == "bullish":
            d_from_bcd_low = C.price - abs(BC) * bcd_min
            d_from_bcd_high = C.price - abs(BC) * bcd_max
        else:
            d_from_bcd_low = C.price + abs(BC) * bcd_min
            d_from_bcd_high = C.price + abs(BC) * bcd_max
        
        # PRZ 是這些水平的匯合
        all_levels = [d_from_xad, d_from_bcd_low, d_from_bcd_high]
        
        prz_low = min(all_levels)
        prz_high = max(all_levels)
        prz_center = (prz_low + prz_high) / 2
        
        # 驗證 PRZ 合理性
        # 對於看漲形態，PRZ 應該低於 C
        # 對於看跌形態，PRZ 應該高於 C
        if direction == "bullish" and prz_center >= C.price:
            return None
        if direction == "bearish" and prz_center <= C.price:
            return None
        
        return PRZone(
            price_low=prz_low,
            price_high=prz_high,
            price_center=prz_center
        )
    
    def _check_all_ratios(
        self,
        XAB: float,
        ABC: float,
        BCD: float,
        XAD: float,
        ratios: Dict
    ) -> bool:
        """檢查所有比例是否符合形態定義"""
        checks = [
            is_within_tolerance(XAB, ratios["XAB"]["min"], ratios["XAB"]["max"], self.tolerance),
            is_within_tolerance(ABC, ratios["ABC"]["min"], ratios["ABC"]["max"], self.tolerance),
            is_within_tolerance(BCD, ratios["BCD"]["min"], ratios["BCD"]["max"], self.tolerance),
            is_within_tolerance(XAD, ratios["XAD"]["min"], ratios["XAD"]["max"], self.tolerance)
        ]
        return all(checks)
    
    def _calculate_clarity_score(
        self,
        XAB: float,
        ABC: float,
        BCD: float,
        XAD: float,
        ratios: Dict
    ) -> float:
        """
        計算形態清晰度分數 (0-1)
        
        分數越高表示比例越接近理想值
        """
        scores = []
        
        # XAB 得分
        xab_target = (ratios["XAB"]["min"] + ratios["XAB"]["max"]) / 2
        xab_deviation = abs(XAB - xab_target) / xab_target
        scores.append(max(0, 1 - xab_deviation))
        
        # ABC 得分
        abc_target = (ratios["ABC"]["min"] + ratios["ABC"]["max"]) / 2
        abc_deviation = abs(ABC - abc_target) / abc_target
        scores.append(max(0, 1 - abc_deviation))
        
        # BCD 和 XAD (如果有)
        if BCD > 0:
            bcd_target = (ratios["BCD"]["min"] + ratios["BCD"]["max"]) / 2
            bcd_deviation = abs(BCD - bcd_target) / bcd_target
            scores.append(max(0, 1 - bcd_deviation))
        
        if XAD > 0:
            xad_target = (ratios["XAD"]["min"] + ratios["XAD"]["max"]) / 2
            xad_deviation = abs(XAD - xad_target) / xad_target
            scores.append(max(0, 1 - xad_deviation))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    # ========================================
    # 輔助方法
    # ========================================
    
    def get_pattern_summary(self, pattern: HarmonicPattern) -> str:
        """生成形態摘要文字"""
        direction_emoji = "🟢" if pattern.direction == "bullish" else "🔴"
        status = "發展中" if pattern.is_developing else "已完成"
        
        summary = f"""
{direction_emoji} {pattern.pattern_type} ({status})
━━━━━━━━━━━━━━━━━━━━━━━━
方向: {pattern.direction}
PRZ: {pattern.prz.price_low:.2f} - {pattern.prz.price_high:.2f}
止損: {pattern.get_stop_loss():.2f}
品質: {pattern.clarity_score:.2%}
━━━━━━━━━━━━━━━━━━━━━━━━
XAB: {pattern.XAB_ratio:.3f}
ABC: {pattern.ABC_ratio:.3f}
"""
        if not pattern.is_developing:
            summary += f"BCD: {pattern.BCD_ratio:.3f}\n"
            summary += f"XAD: {pattern.XAD_ratio:.3f}\n"
        
        return summary

