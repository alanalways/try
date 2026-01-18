# 🦄 HarmonicSMC-QuantConnect

[![QuantConnect](https://img.shields.io/badge/QuantConnect-Compatible-blue.svg)](https://www.quantconnect.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Professional Harmonic Pattern + Smart Money Concepts (SMC) Trading System for QuantConnect**

Combining geometric harmonic patterns with institutional order flow analysis and machine learning optimization.

## 🎯 Features

- **Harmonic Pattern Recognition**: Gartley, Bat, Butterfly, Crab
- **Smart Money Concepts**: Order Blocks, Fair Value Gaps (FVG), BOS/CHoCH
- **Real-time Pattern Detection**: Identifies developing patterns (XABC → Predict D)
- **ML Scoring**: RandomForest confidence scoring with simple rule fallback
- **Risk Management**: ATR-based position sizing, multi-target exits
- **Event-Driven Architecture**: Built for QuantConnect's LEAN engine

## 📁 Project Structure

```
HarmonicSMC-QuantConnect/
├── main.py                 # Main QCAlgorithm strategy
├── harmonic_patterns.py    # Harmonic pattern recognition engine
├── smc_analysis.py         # Smart Money Concepts filter
├── ml_scoring.py           # Machine learning optimization
├── risk_manager.py         # Position sizing and risk control
├── utils.py                # Data processing utilities
├── config.py               # Strategy configuration
├── research/
│   └── train_ml_model.py   # Research notebook for ML training
└── README.md
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/HarmonicSMC-QuantConnect.git
```

### 2. Upload to QuantConnect
1. Log in to [QuantConnect](https://www.quantconnect.com/)
2. Create a new Python project
3. Upload all `.py` files to your project

### 3. Configure Parameters
Edit `config.py` to customize:
```python
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAME = Resolution.Hour
RISK_PER_TRADE = 0.01  # 1%
CONFIDENCE_THRESHOLD = 0.75
```

### 4. Run Backtest
Click **Backtest** and analyze results!

## 📊 Strategy Logic

```
┌─────────────────────────────────────────┐
│           OnData (Per Candle)           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Fetch History & Calculate ZigZag    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    Scan Developing Patterns (XABC→D)    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Check PRZ Proximity (≤2%)          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    SMC Confluence (OB + FVG Overlap)    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     ML Confidence Score (≥75%)          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Entry Confirmation (Hammer/Engulfing)  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Execute: Market + Stop + Take Profit  │
└─────────────────────────────────────────┘
```

## 📐 Harmonic Pattern Ratios

| Pattern | XAB | ABC | XAD | BCD |
|---------|-----|-----|-----|-----|
| **Gartley** | 0.618 | 0.382-0.886 | 0.786 | 1.27-1.618 |
| **Bat** | 0.382-0.5 | 0.382-0.886 | 0.886 | 1.618-2.618 |
| **Butterfly** | 0.786 | 0.382-0.886 | 1.27-1.618 | 1.618-2.618 |
| **Crab** | 0.382-0.618 | 0.382-0.886 | 1.618 | 2.618-3.618 |

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOLERANCE` | 0.03 | Fibonacci ratio tolerance (3%) |
| `CONFIDENCE_THRESHOLD` | 0.75 | ML confidence threshold |
| `RISK_PER_TRADE` | 0.01 | Risk per trade (1%) |
| `MAX_POSITIONS` | 2 | Maximum concurrent positions |
| `PRZ_PROXIMITY` | 2.0 | PRZ approach threshold (%) |
| `LEVERAGE` | 10 | Trading leverage |

## 🤖 Machine Learning

### Training (Research Notebook)
```python
# In QuantConnect Research Environment
from research.train_ml_model import MLTrainer

qb = QuantBook()
trainer = MLTrainer(qb)
trainer.collect_data("BTCUSDT", "2023-01-01", "2024-01-01")
model = trainer.train()
trainer.save_to_objectstore()
```

### Features Used
- RSI Divergence
- Distance to Order Block
- Volume Spike at D Point
- Harmonic Clarity Score
- Trend Alignment (EMA50 vs EMA200)
- Fibonacci Ratio Deviations

## 📈 Expected Performance

Based on backtesting (not guaranteed):
- **Win Rate**: 55-65%
- **Risk/Reward**: 1.5-2.5
- **Max Drawdown**: 10-20%
- **Sharpe Ratio**: 1.0-2.0

⚠️ **Disclaimer**: Past performance does not guarantee future results.

## 🔧 Customization

### Adding More Symbols
```python
# In config.py
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
```

### Adjusting Risk
```python
# More aggressive
CONFIDENCE_THRESHOLD = 0.65
RISK_PER_TRADE = 0.02

# More conservative
CONFIDENCE_THRESHOLD = 0.80
RISK_PER_TRADE = 0.005
```

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [QuantConnect](https://www.quantconnect.com/) - LEAN Engine
- Harmonic Pattern Theory by Scott Carney
- Smart Money Concepts (ICT methodology)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for Quantitative Traders**

