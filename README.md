# 📈 BIST Quantitative Intelligence & ML Selection System

A professional-grade machine learning framework for real-time stock selection, backtesting, and automated analysis of BIST (Borsa Istanbul) equities.

🤖 **Note:** This project utilizes a *"vibecoding"* approach for rapid development, while maintaining rigorous manual control over feature logic, quantitative constraints, and dataset integrity.

---

## 🚀 Key Features

**📡 Live Data Ingestion**  
Real-time market data integration via Yahoo Finance API for the entire BIST100 universe.

**🤖 AI-Powered Inference**  
Short-term upside probability forecasting using XGBoost and LightGBM models.

**🖥 Interactive Dashboard**  
A modern, dark-themed Streamlit terminal for real-time monitoring and strategy tracking.

**🧠 Dual-Strategy Logic**
- **Policy A (Dynamic):** Pure ML ranking of the top 10 high-probability candidates.  
- **Policy B (Sniper):** High-conviction strategy requiring a **>56% confidence threshold** to act.

**🗂 Automated Snapshot Logging**  
Daily predictions are automatically exported as timestamped CSV reports for historical tracking.

**📊 Backtesting Engine**  
Weekly rolling 5-day holding period simulation with performance metrics:
- Sharpe Ratio  
- Maximum Drawdown  
- Win Rate  

---

## 🏗 Project Structure

```
BIST-Weekly-ML-Stock-Selection-System/
│
├── app.py                # Modern Streamlit Dashboard (Live Terminal)
├── run_live_api.py       # CLI-based live market scanner
├── run_train.py          # Model training pipeline
├── run_backtest.py       # Historical simulation engine
│
├── src/                  # Core Quantitative Logic
│   ├── features.py       # Technical & Cross-sectional signals
│   ├── scoring.py        # Strategy Policy (A & B) implementations
│   ├── data_layer.py     # Data ingestion & API management
│   └── ...
│
├── models/               # Serialized ML Models (XGBoost/LGBM)
├── outputs/              # Daily CSV Snapshots & Reports
└── reports/              # Backtest visualizations & PDFs
```

---

## ⚙️ Installation & Setup

### 1️⃣ Environment Setup
```bash
conda create -n bist python=3.13
conda activate bist
```

### 2️⃣ Dependencies
```bash
pip install pandas numpy xgboost lightgbm yfinance streamlit joblib matplotlib
```

### 3️⃣ Launch the Terminal
To start the live interactive dashboard:

```bash
streamlit run app.py
```

---

## 📊 Live Operations

### 🖥 The Dashboard
Accessible via **localhost:8501**, the terminal provides:

- Real-time Analysis of BIST100 universe  
- Automatic feature computation  
- 15-minute refresh during market hours  
- ML-based target price forecasting  
- Daily CSV snapshot logging  

---

## 🤖 Model Architecture

The system currently utilizes an **XGBoost Classifier** optimized for short-term directionality.

### Feature Set
- Volatility Z-scores & ATR%
- Bollinger Band Width & RSI
- Trend Alignment (EMA relationships)
- Volume Confirmation
- Cross-sectional signals

---

## 📌 Quantitative Logic

### Universe Selection
- **BIST 100** coverage  
- Liquidity filtering via volume confirmation  

### Strategies

**Policy A — Dynamic Exposure**  
Selects the top 10 stocks by ML rank.  
Ideal for maintaining continuous market exposure.

**Policy B — Sniper Strategy**  
Only triggers when probability > **56%**.  
If no candidate qualifies → system recommends **100% cash position**.

---

## 🧪 Research & Disclaimer

⚠️ **For research and educational purposes only.**

Quantitative trading involves significant risk.  
This framework does **not** currently account for:

- Transaction costs & slippage *(planned)*  
- Macro-economic regime shifts  
- Market-wide black swan events  

---

## 👨‍💻 Author

**Ali Berbergil**  
Machine Learning & Quantitative Research
