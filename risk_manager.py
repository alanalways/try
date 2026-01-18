# -*- coding: utf-8 -*-
"""
風險管理模組 (Risk Management Module)
處理倉位計算、止損止盈設定、訂單管理
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config import (
    RISK_PER_TRADE,
    TP1_RATIO,
    TP2_RATIO,
    STOP_LOSS_MULTIPLIER,
    ATR_PERIOD,
    MAX_POSITIONS,
    LEVERAGE
)


# ============================================================
# 數據結構
# ============================================================

class PositionStatus(Enum):
    """持倉狀態"""
    PENDING = "pending"           # 等待入場
    ACTIVE = "active"             # 持倉中
    PARTIAL_CLOSED = "partial"    # 部分平倉
    CLOSED = "closed"             # 已平倉


@dataclass
class TradeSetup:
    """交易設置"""
    symbol: Symbol
    direction: str              # "long" 或 "short"
    
    # 價格水平
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # 倉位
    quantity: float
    position_value: float
    
    # 風險指標
    risk_reward_ratio: float
    risk_amount: float
    
    # 相關形態
    pattern_type: str
    pattern_direction: str
    
    # 時間戳
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> str:
        """生成交易設置摘要"""
        direction_emoji = "🟢" if self.direction == "long" else "🔴"
        
        return f"""
{direction_emoji} {self.pattern_type} 交易設置
━━━━━━━━━━━━━━━━━━━━━━━━
方向: {self.direction.upper()}
入場: {self.entry_price:.4f}
止損: {self.stop_loss:.4f}
TP1: {self.take_profit_1:.4f}
TP2: {self.take_profit_2:.4f}
TP3: {self.take_profit_3:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━
數量: {self.quantity:.6f}
風險: ${self.risk_amount:.2f}
R:R: 1:{self.risk_reward_ratio:.2f}
"""


@dataclass
class ActivePosition:
    """活躍持倉"""
    symbol: Symbol
    direction: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profits: List[float]
    
    # 訂單追蹤
    entry_order_id: int = None
    stop_order_id: int = None
    tp_order_ids: List[int] = field(default_factory=list)
    
    # 狀態
    status: PositionStatus = PositionStatus.PENDING
    partial_closes: int = 0
    
    # 績效追蹤
    unrealized_pnl: float = 0
    entry_time: datetime = None
    
    def update_pnl(self, current_price: float):
        """更新未實現盈虧"""
        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


# ============================================================
# 風險管理器
# ============================================================

class RiskManager:
    """
    風險管理器
    
    功能:
    1. ATR 基礎倉位計算
    2. 止損止盈設定
    3. 訂單執行與管理
    4. 持倉追蹤
    """
    
    def __init__(
        self,
        algorithm: QCAlgorithm,
        risk_per_trade: float = RISK_PER_TRADE,
        max_positions: int = MAX_POSITIONS,
        leverage: int = LEVERAGE
    ):
        """
        初始化風險管理器
        
        Args:
            algorithm: QCAlgorithm 實例
            risk_per_trade: 每筆交易風險 (佔資金比例)
            max_positions: 最大持倉數
            leverage: 槓桿倍數
        """
        self.algorithm = algorithm
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.leverage = leverage
        
        # 持倉追蹤
        self.active_positions: Dict[str, ActivePosition] = {}
        
        # 績效統計
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    # ========================================
    # 倉位計算
    # ========================================
    
    def calculate_position_size(
        self,
        symbol: Symbol,
        entry_price: float,
        stop_loss_price: float,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        """
        計算基於風險的倉位大小
        
        使用公式:
        Position Size = (Account Equity × Risk %) / (Entry - Stop Loss)
        
        Args:
            symbol: 交易對符號
            entry_price: 入場價格
            stop_loss_price: 止損價格
            df: OHLCV DataFrame (用於 ATR 計算)
            
        Returns:
            (quantity, risk_amount)
        """
        # 獲取賬戶權益
        portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        # 計算可承受的風險金額
        risk_amount = portfolio_value * self.risk_per_trade
        
        # 計算止損距離
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            self.algorithm.Debug("⚠️ 止損距離為 0，無法計算倉位")
            return 0, 0
        
        # 計算基礎倉位
        quantity = risk_amount / stop_distance
        
        # 應用槓桿限制
        max_position_value = portfolio_value * self.leverage
        max_quantity = max_position_value / entry_price
        
        # 取較小值
        final_quantity = min(quantity, max_quantity)
        
        # 計算實際風險金額
        actual_risk = final_quantity * stop_distance
        
        return final_quantity, actual_risk
    
    def calculate_atr_based_stop(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        multiplier: float = STOP_LOSS_MULTIPLIER
    ) -> float:
        """
        基於 ATR 計算止損價格
        
        Args:
            df: OHLCV DataFrame
            direction: 交易方向 ("long" 或 "short")
            entry_price: 入場價格
            multiplier: ATR 倍數
            
        Returns:
            止損價格
        """
        from utils import IndicatorCalculator
        
        atr = IndicatorCalculator.calculate_atr(df, ATR_PERIOD)
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else entry_price * 0.02
        
        if direction == "long":
            stop_loss = entry_price - (current_atr * multiplier)
        else:
            stop_loss = entry_price + (current_atr * multiplier)
        
        return stop_loss
    
    # ========================================
    # 交易設置生成
    # ========================================
    
    def create_trade_setup(
        self,
        symbol: Symbol,
        pattern,  # HarmonicPattern
        df: pd.DataFrame,
        current_price: float
    ) -> Optional[TradeSetup]:
        """
        基於諧波形態創建交易設置
        
        Args:
            symbol: 交易對符號
            pattern: HarmonicPattern 對象
            df: OHLCV DataFrame
            current_price: 當前價格
            
        Returns:
            TradeSetup 對象或 None
        """
        # 檢查持倉限制
        if len(self.active_positions) >= self.max_positions:
            self.algorithm.Debug(f"⚠️ 已達最大持倉數 ({self.max_positions})")
            return None
        
        # 確定交易方向
        direction = "long" if pattern.direction == "bullish" else "short"
        
        # 入場價格 (使用 PRZ 中心或當前價格)
        entry_price = current_price
        
        # 止損價格 (超過 X 點)
        stop_loss = pattern.get_stop_loss()
        
        # 止盈價格
        tp1, tp2, tp3 = pattern.get_take_profit_levels()
        
        # 計算倉位
        quantity, risk_amount = self.calculate_position_size(
            symbol, entry_price, stop_loss, df
        )
        
        if quantity <= 0:
            return None
        
        # 計算風險報酬比 (使用 TP1)
        risk = abs(entry_price - stop_loss)
        reward = abs(tp1 - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # 創建交易設置
        setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            quantity=quantity,
            position_value=quantity * entry_price,
            risk_reward_ratio=risk_reward,
            risk_amount=risk_amount,
            pattern_type=pattern.pattern_type,
            pattern_direction=pattern.direction
        )
        
        return setup
    
    # ========================================
    # 訂單執行
    # ========================================
    
    def execute_trade(self, setup: TradeSetup) -> bool:
        """
        執行交易
        
        Args:
            setup: TradeSetup 對象
            
        Returns:
            是否執行成功
        """
        try:
            symbol = setup.symbol
            
            # 記錄交易設置
            self.algorithm.Debug(setup.get_summary())
            
            # 下市價單入場
            if setup.direction == "long":
                entry_order = self.algorithm.MarketOrder(symbol, setup.quantity)
            else:
                entry_order = self.algorithm.MarketOrder(symbol, -setup.quantity)
            
            # 設置止損單
            if setup.direction == "long":
                stop_order = self.algorithm.StopMarketOrder(
                    symbol, -setup.quantity, setup.stop_loss
                )
            else:
                stop_order = self.algorithm.StopMarketOrder(
                    symbol, setup.quantity, setup.stop_loss
                )
            
            # 設置止盈單 (使用 TP1 作為主要目標)
            # 注意: 在 QC 中，同時持有止損和止盈單需要手動管理
            if setup.direction == "long":
                tp_order = self.algorithm.LimitOrder(
                    symbol, -setup.quantity, setup.take_profit_1
                )
            else:
                tp_order = self.algorithm.LimitOrder(
                    symbol, setup.quantity, setup.take_profit_1
                )
            
            # 創建持倉記錄
            position = ActivePosition(
                symbol=symbol,
                direction=setup.direction,
                entry_price=setup.entry_price,
                quantity=setup.quantity,
                stop_loss=setup.stop_loss,
                take_profits=[setup.take_profit_1, setup.take_profit_2, setup.take_profit_3],
                entry_order_id=entry_order.OrderId if entry_order else None,
                stop_order_id=stop_order.OrderId if stop_order else None,
                tp_order_ids=[tp_order.OrderId] if tp_order else [],
                status=PositionStatus.ACTIVE,
                entry_time=self.algorithm.Time
            )
            
            # 保存持倉
            self.active_positions[str(symbol)] = position
            self.total_trades += 1
            
            self.algorithm.Debug(f"✅ 交易執行成功: {setup.direction} {symbol}")
            
            return True
            
        except Exception as e:
            self.algorithm.Debug(f"❌ 交易執行失敗: {str(e)}")
            return False
    
    # ========================================
    # 持倉管理
    # ========================================
    
    def manage_positions(self, data: Slice):
        """
        管理活躍持倉
        
        在 OnData 中調用此方法來:
        1. 更新未實現盈虧
        2. 檢查部分止盈
        3. 移動止損 (可選)
        
        Args:
            data: QuantConnect Slice 數據
        """
        for symbol_str, position in list(self.active_positions.items()):
            symbol = position.symbol
            
            if not data.ContainsKey(symbol):
                continue
            
            current_price = self.algorithm.Securities[symbol].Price
            
            # 更新盈虧
            position.update_pnl(current_price)
            
            # 檢查是否需要移動止損 (追蹤止損)
            self._check_trailing_stop(position, current_price)
    
    def _check_trailing_stop(self, position: ActivePosition, current_price: float):
        """
        檢查是否需要移動止損
        
        當價格達到 TP1 後，將止損移動到入場價 (保本)
        """
        if position.partial_closes > 0:
            return  # 已經移動過
        
        tp1 = position.take_profits[0]
        
        if position.direction == "long":
            if current_price >= tp1:
                # 移動止損到入場價
                self._move_stop_loss(position, position.entry_price)
                position.partial_closes += 1
                self.algorithm.Debug(f"📈 {position.symbol}: 止損移動到入場價 (保本)")
        else:
            if current_price <= tp1:
                self._move_stop_loss(position, position.entry_price)
                position.partial_closes += 1
                self.algorithm.Debug(f"📉 {position.symbol}: 止損移動到入場價 (保本)")
    
    def _move_stop_loss(self, position: ActivePosition, new_stop: float):
        """移動止損價格"""
        try:
            # 取消原止損單
            if position.stop_order_id:
                self.algorithm.Transactions.CancelOrder(position.stop_order_id)
            
            # 下新止損單
            if position.direction == "long":
                new_order = self.algorithm.StopMarketOrder(
                    position.symbol, -position.quantity, new_stop
                )
            else:
                new_order = self.algorithm.StopMarketOrder(
                    position.symbol, position.quantity, new_stop
                )
            
            position.stop_loss = new_stop
            position.stop_order_id = new_order.OrderId if new_order else None
            
        except Exception as e:
            self.algorithm.Debug(f"❌ 移動止損失敗: {str(e)}")
    
    # ========================================
    # 訂單事件處理
    # ========================================
    
    def on_order_event(self, order_event: OrderEvent):
        """
        處理訂單事件
        
        在主算法的 OnOrderEvent 中調用
        
        Args:
            order_event: QuantConnect OrderEvent
        """
        if order_event.Status != OrderStatus.Filled:
            return
        
        symbol_str = str(order_event.Symbol)
        
        if symbol_str not in self.active_positions:
            return
        
        position = self.active_positions[symbol_str]
        
        # 檢查是止損還是止盈被觸發
        if order_event.OrderId == position.stop_order_id:
            # 止損被觸發
            self._close_position(position, "stop_loss", order_event.FillPrice)
        elif order_event.OrderId in position.tp_order_ids:
            # 止盈被觸發
            self._close_position(position, "take_profit", order_event.FillPrice)
    
    def _close_position(
        self,
        position: ActivePosition,
        close_reason: str,
        close_price: float
    ):
        """關閉持倉並記錄"""
        symbol_str = str(position.symbol)
        
        # 計算盈虧
        if position.direction == "long":
            pnl = (close_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - close_price) * position.quantity
        
        pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100
        
        # 更新統計
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        # 記錄
        emoji = "✅" if pnl > 0 else "❌"
        self.algorithm.Debug(f"""
{emoji} 持倉已關閉: {position.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━
原因: {close_reason}
入場: {position.entry_price:.4f}
出場: {close_price:.4f}
盈虧: ${pnl:.2f} ({pnl_percent:.2f}%)
━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
        # 取消剩餘訂單
        self._cancel_remaining_orders(position)
        
        # 移除持倉記錄
        del self.active_positions[symbol_str]
    
    def _cancel_remaining_orders(self, position: ActivePosition):
        """取消剩餘的掛單"""
        try:
            if position.stop_order_id:
                self.algorithm.Transactions.CancelOrder(position.stop_order_id)
            
            for tp_id in position.tp_order_ids:
                self.algorithm.Transactions.CancelOrder(tp_id)
                
        except Exception as e:
            self.algorithm.Debug(f"取消訂單時發生錯誤: {str(e)}")
    
    # ========================================
    # 輔助方法
    # ========================================
    
    def can_open_new_position(self) -> bool:
        """檢查是否可以開新倉"""
        return len(self.active_positions) < self.max_positions
    
    def has_position(self, symbol: Symbol) -> bool:
        """檢查是否已有該交易對的持倉"""
        return str(symbol) in self.active_positions
    
    def get_statistics(self) -> Dict:
        """獲取績效統計"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'active_positions': len(self.active_positions)
        }
    
    def get_statistics_summary(self) -> str:
        """生成績效統計摘要"""
        stats = self.get_statistics()
        
        return f"""
━━━━━━━ 績效統計 ━━━━━━━
總交易數: {stats['total_trades']}
獲利交易: {stats['winning_trades']}
虧損交易: {stats['losing_trades']}
勝率: {stats['win_rate']:.1f}%
總盈虧: ${stats['total_pnl']:.2f}
活躍持倉: {stats['active_positions']}
━━━━━━━━━━━━━━━━━━━━━━━━
"""

