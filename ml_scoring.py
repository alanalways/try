# -*- coding: utf-8 -*-
"""
機器學習評分模組 (Machine Learning Scoring Module)
使用 RandomForest 對交易信號進行評分
支援模型從 ObjectStore 載入，以及規則基礎的回退機制
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from config import (
    CONFIDENCE_THRESHOLD,
    ML_MODEL_NAME,
    ML_ENABLED,
    ML_FALLBACK_SCORE,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT
)


# ============================================================
# 枚舉與數據結構
# ============================================================

class MLMode(Enum):
    """ML 模式"""
    DISABLED = "disabled"       # 不使用 ML
    COLLECTION = "collection"   # 收集數據模式
    PREDICTION = "prediction"   # 預測模式
    RULE_BASED = "rule_based"   # 規則基礎模式 (無ML時的回退)


@dataclass
class FeatureSet:
    """特徵集合"""
    # 諧波形態特徵
    harmonic_clarity: float       # 形態清晰度
    prz_distance_percent: float   # 價格與 PRZ 的距離
    
    # RSI 特徵
    rsi_value: float              # RSI 值
    rsi_divergence: str           # RSI 背離 ("bullish", "bearish", "none")
    
    # 趨勢特徵
    trend_alignment: float        # 趨勢對齊 (-1 到 1)
    ema_distance: float           # 與 EMA 的距離
    
    # SMC 特徵
    ob_distance: float            # 與 Order Block 的距離
    fvg_overlap: bool             # 是否與 FVG 重疊
    smc_confluence_count: int     # SMC 匯合數量
    
    # 波動性特徵
    atr_normalized: float         # 正規化 ATR
    volume_spike: bool            # 成交量異常
    bb_position: float            # 布林帶位置 (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'harmonic_clarity': self.harmonic_clarity,
            'prz_distance_percent': self.prz_distance_percent,
            'rsi_value': self.rsi_value,
            'rsi_divergence_bullish': 1 if self.rsi_divergence == 'bullish' else 0,
            'rsi_divergence_bearish': 1 if self.rsi_divergence == 'bearish' else 0,
            'trend_alignment': self.trend_alignment,
            'ema_distance': self.ema_distance,
            'ob_distance': self.ob_distance,
            'fvg_overlap': 1 if self.fvg_overlap else 0,
            'smc_confluence_count': self.smc_confluence_count,
            'atr_normalized': self.atr_normalized,
            'volume_spike': 1 if self.volume_spike else 0,
            'bb_position': self.bb_position
        }
    
    def to_array(self) -> np.ndarray:
        """轉換為 numpy 陣列 (用於 ML 預測)"""
        d = self.to_dict()
        return np.array(list(d.values())).reshape(1, -1)


@dataclass
class ScoringResult:
    """評分結果"""
    confidence: float           # 信心分數 (0-1)
    recommendation: str         # 建議 ("strong_buy", "buy", "neutral", "avoid")
    reasoning: List[str]        # 理由列表
    features_used: Dict[str, Any]  # 使用的特徵
    mode: MLMode                # 使用的模式


# ============================================================
# ML 評分器
# ============================================================

class MLScorer:
    """
    機器學習評分器
    
    功能:
    1. 從 ObjectStore 載入預訓練模型
    2. 特徵提取與正規化
    3. 信心分數計算
    4. 規則基礎回退機制
    """
    
    def __init__(
        self,
        algorithm: QCAlgorithm,
        enabled: bool = ML_ENABLED,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        初始化 ML 評分器
        
        Args:
            algorithm: QCAlgorithm 實例
            enabled: 是否啟用 ML
            confidence_threshold: 信心閾值
        """
        self.algorithm = algorithm
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.mode = MLMode.DISABLED if not enabled else MLMode.RULE_BASED
        
        # 數據收集緩衝區 (用於未來訓練)
        self.training_data: List[Dict] = []
        
        # 嘗試載入模型
        if enabled:
            self._load_model()
    
    def _load_model(self):
        """從 ObjectStore 載入模型"""
        try:
            if self.algorithm.ObjectStore.ContainsKey(ML_MODEL_NAME):
                # QuantConnect 的 ObjectStore 讀取
                model_bytes = self.algorithm.ObjectStore.ReadBytes(ML_MODEL_NAME)
                
                # 使用 pickle 反序列化
                import pickle
                self.model = pickle.loads(model_bytes)
                self.mode = MLMode.PREDICTION
                
                self.algorithm.Debug(f"✅ ML 模型載入成功: {ML_MODEL_NAME}")
            else:
                self.algorithm.Debug(f"⚠️ 找不到 ML 模型，使用規則基礎模式")
                self.mode = MLMode.RULE_BASED
                
        except Exception as e:
            self.algorithm.Debug(f"❌ ML 模型載入失敗: {str(e)}")
            self.mode = MLMode.RULE_BASED
    
    # ========================================
    # 主要評分方法
    # ========================================
    
    def calculate_score(
        self,
        features: FeatureSet,
        pattern_direction: str
    ) -> ScoringResult:
        """
        計算信心分數
        
        Args:
            features: 特徵集合
            pattern_direction: 形態方向 ("bullish" 或 "bearish")
            
        Returns:
            評分結果
        """
        if self.mode == MLMode.PREDICTION and self.model is not None:
            return self._ml_prediction(features, pattern_direction)
        else:
            return self._rule_based_scoring(features, pattern_direction)
    
    def _ml_prediction(
        self,
        features: FeatureSet,
        pattern_direction: str
    ) -> ScoringResult:
        """使用 ML 模型預測"""
        try:
            X = features.to_array()
            
            # RandomForest 預測概率
            proba = self.model.predict_proba(X)[0]
            
            # 假設類別 1 是「成功交易」
            confidence = proba[1] if len(proba) > 1 else proba[0]
            
            # 生成建議
            recommendation, reasoning = self._generate_recommendation(
                confidence, features, pattern_direction
            )
            
            return ScoringResult(
                confidence=confidence,
                recommendation=recommendation,
                reasoning=reasoning,
                features_used=features.to_dict(),
                mode=MLMode.PREDICTION
            )
            
        except Exception as e:
            self.algorithm.Debug(f"ML 預測錯誤: {str(e)}")
            return self._rule_based_scoring(features, pattern_direction)
    
    def _rule_based_scoring(
        self,
        features: FeatureSet,
        pattern_direction: str
    ) -> ScoringResult:
        """
        規則基礎評分 (ML 不可用時的回退機制)
        
        評分規則:
        1. 形態清晰度 (權重 25%)
        2. SMC 匯合 (權重 25%)
        3. RSI 背離 (權重 20%)
        4. 趨勢對齊 (權重 15%)
        5. 波動性條件 (權重 15%)
        """
        score = 0.0
        reasoning = []
        
        # 1. 形態清晰度 (0-0.25)
        clarity_score = features.harmonic_clarity * 0.25
        score += clarity_score
        if features.harmonic_clarity > 0.7:
            reasoning.append(f"✅ 形態清晰度優秀: {features.harmonic_clarity:.2%}")
        elif features.harmonic_clarity < 0.5:
            reasoning.append(f"⚠️ 形態清晰度偏低: {features.harmonic_clarity:.2%}")
        
        # 2. SMC 匯合 (0-0.25)
        smc_score = 0.0
        if features.smc_confluence_count >= 2:
            smc_score = 0.25
            reasoning.append(f"✅ 強 SMC 匯合: {features.smc_confluence_count} 個區域重疊")
        elif features.smc_confluence_count == 1:
            smc_score = 0.15
            reasoning.append(f"📊 SMC 匯合: 1 個區域")
        else:
            reasoning.append("⚠️ 無 SMC 匯合")
        score += smc_score
        
        # 3. RSI 背離 (0-0.20)
        rsi_score = 0.0
        if pattern_direction == "bullish":
            if features.rsi_divergence == "bullish":
                rsi_score = 0.20
                reasoning.append("✅ RSI 看漲背離確認")
            elif features.rsi_value < RSI_OVERSOLD:
                rsi_score = 0.15
                reasoning.append(f"📊 RSI 超賣: {features.rsi_value:.1f}")
        else:  # bearish
            if features.rsi_divergence == "bearish":
                rsi_score = 0.20
                reasoning.append("✅ RSI 看跌背離確認")
            elif features.rsi_value > RSI_OVERBOUGHT:
                rsi_score = 0.15
                reasoning.append(f"📊 RSI 超買: {features.rsi_value:.1f}")
        score += rsi_score
        
        # 4. 趨勢對齊 (0-0.15)
        trend_score = 0.0
        if pattern_direction == "bullish" and features.trend_alignment > 0:
            trend_score = 0.15
            reasoning.append("✅ 與主趨勢一致 (看漲)")
        elif pattern_direction == "bearish" and features.trend_alignment < 0:
            trend_score = 0.15
            reasoning.append("✅ 與主趨勢一致 (看跌)")
        elif abs(features.trend_alignment) < 0.1:
            trend_score = 0.08
            reasoning.append("📊 趨勢中性")
        else:
            reasoning.append("⚠️ 逆勢交易")
        score += trend_score
        
        # 5. 波動性與成交量 (0-0.15)
        vol_score = 0.0
        if features.volume_spike:
            vol_score += 0.08
            reasoning.append("✅ 成交量異常 (可能有機構活動)")
        
        # PRZ 距離獎勵 (價格在 PRZ 附近)
        if features.prz_distance_percent < 1.0:
            vol_score += 0.07
            reasoning.append(f"✅ 價格接近 PRZ ({features.prz_distance_percent:.2f}%)")
        score += vol_score
        
        # 最終分數
        confidence = min(1.0, max(0.0, score))
        
        # 生成建議
        recommendation, _ = self._generate_recommendation(
            confidence, features, pattern_direction
        )
        
        return ScoringResult(
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            features_used=features.to_dict(),
            mode=MLMode.RULE_BASED
        )
    
    def _generate_recommendation(
        self,
        confidence: float,
        features: FeatureSet,
        pattern_direction: str
    ) -> tuple:
        """生成交易建議"""
        reasoning = []
        
        if confidence >= 0.80:
            recommendation = "strong_buy" if pattern_direction == "bullish" else "strong_sell"
            reasoning.append(f"🚀 強烈建議: 信心分數 {confidence:.2%}")
        elif confidence >= self.confidence_threshold:
            recommendation = "buy" if pattern_direction == "bullish" else "sell"
            reasoning.append(f"✅ 建議交易: 信心分數 {confidence:.2%}")
        elif confidence >= 0.50:
            recommendation = "neutral"
            reasoning.append(f"📊 中性: 信心分數 {confidence:.2%}")
        else:
            recommendation = "avoid"
            reasoning.append(f"⚠️ 避免交易: 信心分數 {confidence:.2%}")
        
        return recommendation, reasoning
    
    # ========================================
    # 特徵提取輔助方法
    # ========================================
    
    @staticmethod
    def extract_features(
        df: pd.DataFrame,
        pattern,  # HarmonicPattern
        smc_zones: List,  # SMCZone list
        current_price: float
    ) -> FeatureSet:
        """
        從市場數據中提取特徵
        
        Args:
            df: OHLCV DataFrame (需要包含技術指標)
            pattern: HarmonicPattern 對象
            smc_zones: SMC 區域列表
            current_price: 當前價格
            
        Returns:
            FeatureSet 對象
        """
        from utils import IndicatorCalculator
        
        # 確保有指標
        if 'rsi' not in df.columns:
            df = IndicatorCalculator.add_all_indicators(df)
        
        # RSI 相關
        rsi_value = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50.0
        rsi_divergence = IndicatorCalculator.detect_rsi_divergence(
            df['close'], df['rsi']
        )
        
        # 趨勢對齊
        if 'trend' in df.columns:
            trend_alignment = df['trend'].iloc[-1]
        elif 'ema_fast' in df.columns and 'ema_slow' in df.columns:
            trend_alignment = 1 if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] else -1
        else:
            trend_alignment = 0
        
        # EMA 距離
        if 'ema_fast' in df.columns:
            ema_distance = (current_price - df['ema_fast'].iloc[-1]) / current_price
        else:
            ema_distance = 0
        
        # SMC 特徵
        smc_confluence_count = len(smc_zones)
        fvg_overlap = any(z.zone_type.value == 'fair_value_gap' for z in smc_zones)
        
        # 計算與最近 OB 的距離
        ob_distances = []
        for zone in smc_zones:
            if zone.zone_type.value == 'order_block':
                mid = (zone.price_high + zone.price_low) / 2
                distance = abs(current_price - mid) / current_price
                ob_distances.append(distance)
        ob_distance = min(ob_distances) if ob_distances else 1.0
        
        # ATR 正規化
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            atr_normalized = atr / current_price if current_price > 0 else 0
        else:
            atr_normalized = 0.02  # 預設 2%
        
        # 成交量異常
        if 'volume_sma' in df.columns:
            volume_spike = df['volume'].iloc[-1] > df['volume_sma'].iloc[-1] * 1.5
        else:
            volume_spike = False
        
        # 布林帶位置
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]
            if bb_range > 0:
                bb_position = (current_price - df['bb_lower'].iloc[-1]) / bb_range
            else:
                bb_position = 0.5
        else:
            bb_position = 0.5
        
        # PRZ 距離
        prz_distance = pattern.prz.distance_percent(current_price)
        
        return FeatureSet(
            harmonic_clarity=pattern.clarity_score,
            prz_distance_percent=prz_distance,
            rsi_value=rsi_value,
            rsi_divergence=rsi_divergence,
            trend_alignment=trend_alignment,
            ema_distance=ema_distance,
            ob_distance=ob_distance,
            fvg_overlap=fvg_overlap,
            smc_confluence_count=smc_confluence_count,
            atr_normalized=atr_normalized,
            volume_spike=volume_spike,
            bb_position=bb_position
        )
    
    # ========================================
    # 數據收集 (用於未來訓練)
    # ========================================
    
    def collect_training_sample(
        self,
        features: FeatureSet,
        outcome: int,  # 1 = 成功, 0 = 失敗
        pnl_percent: float
    ):
        """
        收集訓練樣本
        
        Args:
            features: 特徵集合
            outcome: 交易結果
            pnl_percent: 盈虧百分比
        """
        sample = {
            'features': features.to_dict(),
            'outcome': outcome,
            'pnl_percent': pnl_percent,
            'timestamp': self.algorithm.Time.isoformat()
        }
        
        self.training_data.append(sample)
        
        # 定期保存到 ObjectStore
        if len(self.training_data) % 50 == 0:
            self._save_training_data()
    
    def _save_training_data(self):
        """保存訓練數據到 ObjectStore"""
        try:
            data_json = json.dumps(self.training_data)
            self.algorithm.ObjectStore.Save(
                "harmonic_smc_training_data.json",
                data_json
            )
            self.algorithm.Debug(f"✅ 已保存 {len(self.training_data)} 筆訓練數據")
        except Exception as e:
            self.algorithm.Debug(f"❌ 保存訓練數據失敗: {str(e)}")
    
    # ========================================
    # 輔助方法
    # ========================================
    
    def should_trade(self, result: ScoringResult) -> bool:
        """判斷是否應該交易"""
        return result.confidence >= self.confidence_threshold
    
    def get_summary(self, result: ScoringResult) -> str:
        """生成評分摘要"""
        mode_str = {
            MLMode.PREDICTION: "🤖 ML",
            MLMode.RULE_BASED: "📏 規則",
            MLMode.COLLECTION: "📝 收集",
            MLMode.DISABLED: "❌ 關閉"
        }
        
        summary = f"""
━━━━━━━ ML 評分報告 ━━━━━━━
模式: {mode_str.get(result.mode, '未知')}
信心分數: {result.confidence:.2%}
建議: {result.recommendation}
━━━━━━━━━━━━━━━━━━━━━━━━
評分理由:
"""
        for reason in result.reasoning:
            summary += f"  • {reason}\n"
        
        return summary

