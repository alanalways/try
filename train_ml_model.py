# -*- coding: utf-8 -*-
"""
ML 模型訓練腳本 (Research Notebook)
在 QuantConnect Research 環境中運行

使用方式:
1. 在 QuantConnect 打開 Research Notebook
2. 將此檔案內容複製到 Notebook
3. 逐個 Cell 執行
"""

# ============================================================
# Cell 1: 導入與初始化
# ============================================================

from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 初始化 QuantBook (Research 專用)
qb = QuantBook()

# 設定參數
SYMBOL = "BTCUSDT"
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 1, 1)
MODEL_NAME = "harmonic_smc_rf_model.pkl"

print("✅ 初始化完成")


# ============================================================
# Cell 2: 獲取歷史數據
# ============================================================

# 添加交易對
symbol = qb.AddCrypto(SYMBOL, Resolution.Hour).Symbol

# 獲取歷史數據
history = qb.History(symbol, START_DATE, END_DATE, Resolution.Hour)

# 轉換為標準 DataFrame
if isinstance(history.index, pd.MultiIndex):
    df = history.loc[symbol].copy()
else:
    df = history.copy()

df.index = pd.to_datetime(df.index)
print(f"✅ 獲取數據: {len(df)} 行")
print(df.tail())


# ============================================================
# Cell 3: 技術指標計算
# ============================================================

def add_indicators(df):
    """添加技術指標"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Trend
    df['trend'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
    
    return df

df = add_indicators(df)
print("✅ 指標計算完成")


# ============================================================
# Cell 4: ZigZag 計算
# ============================================================

def calculate_zigzag(df, min_retrace_pct=1.0):
    """計算 ZigZag 擺動點"""
    swing_points = []
    last_swing = None
    last_swing_type = None
    
    for i in range(3, len(df)):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # 檢查是否為潛在高點
        is_high = all(
            df['high'].iloc[i-j] <= current_high 
            for j in range(1, 4)
        )
        
        # 檢查是否為潛在低點
        is_low = all(
            df['low'].iloc[i-j] >= current_low 
            for j in range(1, 4)
        )
        
        if is_high and (last_swing_type != 'high'):
            lookback = min(20, i)
            recent_low = df['low'].iloc[i-lookback:i].min()
            retrace = (current_high - recent_low) / recent_low * 100
            
            if retrace >= min_retrace_pct:
                swing_points.append({
                    'index': i,
                    'price': current_high,
                    'type': 'high',
                    'time': df.index[i]
                })
                last_swing = current_high
                last_swing_type = 'high'
        
        if is_low and (last_swing_type != 'low'):
            lookback = min(20, i)
            recent_high = df['high'].iloc[i-lookback:i].max()
            retrace = (recent_high - current_low) / recent_high * 100
            
            if retrace >= min_retrace_pct:
                swing_points.append({
                    'index': i,
                    'price': current_low,
                    'type': 'low',
                    'time': df.index[i]
                })
                last_swing = current_low
                last_swing_type = 'low'
    
    return swing_points

swing_points = calculate_zigzag(df)
print(f"✅ 找到 {len(swing_points)} 個擺動點")


# ============================================================
# Cell 5: 特徵生成函數
# ============================================================

def generate_features_at_point(df, point_index, lookback=50):
    """在指定點生成特徵"""
    if point_index < lookback:
        return None
    
    try:
        current = df.iloc[point_index]
        recent = df.iloc[point_index-lookback:point_index]
        
        # RSI 相關
        rsi = current['rsi'] if not pd.isna(current['rsi']) else 50
        
        # 趨勢對齊
        trend = current['trend'] if 'trend' in df.columns else 0
        
        # 波動性
        atr = current['atr'] if not pd.isna(current['atr']) else 0
        atr_norm = atr / current['close'] if current['close'] > 0 else 0
        
        # 布林帶位置
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            bb_range = current['bb_upper'] - current['bb_lower']
            bb_pos = (current['close'] - current['bb_lower']) / bb_range if bb_range > 0 else 0.5
        else:
            bb_pos = 0.5
        
        # 成交量異常
        vol_sma = current['volume_sma'] if not pd.isna(current['volume_sma']) else current['volume']
        vol_spike = 1 if current['volume'] > vol_sma * 1.5 else 0
        
        # EMA 距離
        ema_dist = (current['close'] - current['ema_50']) / current['close'] if current['close'] > 0 else 0
        
        # 價格動量
        momentum = (current['close'] - recent['close'].iloc[0]) / recent['close'].iloc[0] if recent['close'].iloc[0] > 0 else 0
        
        # 波動性變化
        vol_recent = recent['atr'].iloc[-10:].mean() if len(recent) >= 10 else atr
        vol_older = recent['atr'].iloc[:10].mean() if len(recent) >= 10 else atr
        vol_change = (vol_recent - vol_older) / vol_older if vol_older > 0 else 0
        
        return {
            'rsi': rsi,
            'trend': trend,
            'atr_norm': atr_norm,
            'bb_pos': bb_pos,
            'vol_spike': vol_spike,
            'ema_dist': ema_dist,
            'momentum': momentum,
            'vol_change': vol_change
        }
    
    except Exception as e:
        return None

print("✅ 特徵生成函數定義完成")


# ============================================================
# Cell 6: 生成訓練數據集
# ============================================================

def generate_training_data(df, swing_points, lookahead=20, profit_threshold=0.015):
    """
    生成訓練數據
    
    lookahead: 向前看多少根K線來確定結果
    profit_threshold: 獲利閾值 (1.5%)
    """
    features_list = []
    labels = []
    
    for sp in swing_points:
        idx = sp['index']
        
        # 確保有足夠的前向數據
        if idx + lookahead >= len(df):
            continue
        
        # 生成特徵
        features = generate_features_at_point(df, idx)
        if features is None:
            continue
        
        # 確定標籤 (未來 lookahead 根 K 線的表現)
        entry_price = df['close'].iloc[idx]
        future_prices = df['close'].iloc[idx+1:idx+lookahead+1]
        
        if sp['type'] == 'low':  # 潛在看漲信號
            max_profit = (future_prices.max() - entry_price) / entry_price
            max_loss = (entry_price - future_prices.min()) / entry_price
        else:  # 潛在看跌信號
            max_profit = (entry_price - future_prices.min()) / entry_price
            max_loss = (future_prices.max() - entry_price) / entry_price
        
        # 標籤: 1 = 成功 (獲利 > 閾值 且 風險報酬 > 1.5)
        # 標籤: 0 = 失敗
        risk_reward = max_profit / max_loss if max_loss > 0 else 0
        
        if max_profit >= profit_threshold and risk_reward >= 1.5:
            label = 1
        else:
            label = 0
        
        # 添加方向特徵
        features['is_bullish'] = 1 if sp['type'] == 'low' else 0
        
        features_list.append(features)
        labels.append(label)
    
    return pd.DataFrame(features_list), labels

# 生成訓練數據
X, y = generate_training_data(df, swing_points)
print(f"✅ 生成訓練數據: {len(X)} 樣本")
print(f"正樣本 (成功交易): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"負樣本 (失敗交易): {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")


# ============================================================
# Cell 7: 訓練模型
# ============================================================

# 分割訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 初始化 RandomForest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # 處理類別不平衡
    random_state=42,
    n_jobs=-1
)

# 訓練
model.fit(X_train, y_train)
print("✅ 模型訓練完成")

# 評估
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"訓練集準確率: {train_score:.4f}")
print(f"測試集準確率: {test_score:.4f}")

# 交叉驗證
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")


# ============================================================
# Cell 8: 詳細評估報告
# ============================================================

# 預測
y_pred = model.predict(X_test)

# 分類報告
print("\n📊 分類報告:")
print(classification_report(y_test, y_pred, target_names=['失敗', '成功']))

# 混淆矩陣
print("\n📊 混淆矩陣:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 特徵重要性
print("\n📊 特徵重要性:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)


# ============================================================
# Cell 9: 保存模型到 ObjectStore
# ============================================================

# 序列化模型
model_bytes = pickle.dumps(model)

# 保存到 ObjectStore
qb.ObjectStore.SaveBytes(MODEL_NAME, model_bytes)
print(f"✅ 模型已保存到 ObjectStore: {MODEL_NAME}")

# 驗證保存
if qb.ObjectStore.ContainsKey(MODEL_NAME):
    print("✅ 模型驗證成功，可在策略中使用")
else:
    print("❌ 模型保存驗證失敗")


# ============================================================
# Cell 10: 測試載入模型
# ============================================================

# 讀取模型
loaded_bytes = qb.ObjectStore.ReadBytes(MODEL_NAME)
loaded_model = pickle.loads(loaded_bytes)

# 測試預測
test_prediction = loaded_model.predict_proba(X_test.iloc[:5])
print("✅ 模型載入測試成功")
print("測試預測概率:")
print(test_prediction)


# ============================================================
# Cell 11: 可視化 (可選)
# ============================================================

"""
# 如果在 Jupyter Notebook 中，可以使用以下代碼可視化

import matplotlib.pyplot as plt

# 特徵重要性圖
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 價格和擺動點
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['close'], label='Close')

swing_highs = [sp for sp in swing_points if sp['type'] == 'high']
swing_lows = [sp for sp in swing_points if sp['type'] == 'low']

plt.scatter([sp['time'] for sp in swing_highs], 
            [sp['price'] for sp in swing_highs], 
            color='red', marker='v', s=100, label='Swing High')
plt.scatter([sp['time'] for sp in swing_lows], 
            [sp['price'] for sp in swing_lows], 
            color='green', marker='^', s=100, label='Swing Low')

plt.legend()
plt.title('Price with Swing Points')
plt.tight_layout()
plt.show()
"""

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 ML 訓練完成！

下一步:
1. 在 QuantConnect 策略中，模型會自動從 ObjectStore 載入
2. 確保 config.py 中的 ML_MODEL_NAME 與這裡一致
3. 運行回測驗證策略效果

提示:
- 如果勝率偏低，嘗試調整 profit_threshold
- 如果樣本不平衡，可以使用 SMOTE 過採樣
- 定期重新訓練模型以適應市場變化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

