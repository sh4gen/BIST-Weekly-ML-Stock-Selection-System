# 📈 BIST Weekly ML Stock Selection System

A machine learning–based weekly stock selection and backtesting framework for BIST (Borsa Istanbul) equities.

> 🤖 **Note:** The codebase for this project was largely developed using a **"vibecoding"** approach. However, the dataset preparation, feature logic, and critical quantitative decisions were carefully crafted manually.

This project implements a complete quantitative research pipeline:
* Data ingestion (BIST equity CSV files)
* Feature engineering (technical + cross-sectional signals)
* Label generation (forward returns + risk constraints)
* LightGBM model training
* Weekly A/B stock selection strategies
* Automated PDF reporting
* Weekly rolling backtest (5 trading day holding)

---

## 🚀 What This Project Does

The system:
1.  **Predicts** short-term upside probability for BIST stocks.
2.  **Selects** weekly top candidates.
3.  **Compares** two strategies:
    * **Policy A** → Pure ML ranking
    * **Policy B** → ML ranking + confirmation filters
4.  **Runs** a weekly 5-day holding backtest.
5.  **Generates** equity curves and performance metrics.

---

## 🏗 Project Structure

```text
bist100/
│
├── config/
│   └── settings.yaml
│
├── models/
│   └── lgbm.pkl
│
├── reports/
│   ├── weekly_report_YYYY-MM-DD.pdf
│   ├── weekly_picks_A_*.csv
│   ├── weekly_picks_B_*.csv
│   ├── backtest_summary.csv
│   └── backtest_equity_curve.png
│
├── src/
│   ├── data_layer.py
│   ├── features.py
│   ├── labels.py
│   ├── universe.py
│   ├── scoring.py
│   ├── report.py
│   └── backtest.py
│
├── run_train.py
├── run_weekly.py
└── run_backtest.py
```

---

## ⚙️ Installation

**1️⃣ Create Environment**
```bash
conda create -n bist python=3.10
conda activate bist
```

**2️⃣ Install Dependencies**
```bash
pip install pandas numpy lightgbm scikit-learn matplotlib joblib pyyaml
```

---

## 🧠 Model Training

Train the LightGBM model by running:
```bash
python run_train.py
```
This script will build features, generate labels, train the model, and save it to `models/lgbm.pkl`.

---

## 📊 Weekly Stock Selection

Generate weekly picks and the PDF report:
```bash
python run_weekly.py
```
**Outputs:**
* `weekly_picks_A_YYYY-MM-DD.csv`
* `weekly_picks_B_YYYY-MM-DD.csv`
* `weekly_overlap_YYYY-MM-DD.csv`
* `weekly_report_YYYY-MM-DD.pdf`

### Strategies
* **Policy A:** Top-K stocks ranked purely by ML probability.
* **Policy B:** Top-K stocks ranked by ML probability + confirmation filters (Close > EMA20, Positive short-term momentum, Volume confirmation).

---

## 🔁 Weekly Backtest (5-Day Holding)

Run historical simulation:
```bash
python run_backtest.py
```
**Outputs:**
* `backtest_summary.csv`
* `backtest_equity_curve.csv`
* `backtest_trades_A.csv`
* `backtest_trades_B.csv`
* `backtest_equity_curve.png`

**Metrics Included:** Total Return, Average Weekly Return, Win Rate, Maximum Drawdown, Sharpe Ratio.

---

## 📌 Strategy Logic

**Universe Selection**
* Liquidity filter (rolling volume)
* Volatility filter (ATR%)

**Feature Set**
* 5 / 10 / 20 day returns
* EMA relationships
* RSI (14)
* Volume Z-score
* Trend alignment

**Label Definition**
* Binary classification: Positive forward return above threshold.
* Risk constraint using maximum drawdown filter.

---

## 🗄️ Data Sources

The historical equity data and index information used in this project are sourced from:
* [Kaggle: Borsa Istanbul (BIST100) Index 2010-2020](https://www.kaggle.com/datasets/mertopkaya/borsa-istanbul-bist100-index-20102020)
* [Borsa Istanbul Official Index Data](https://www.borsaistanbul.com/en/index/index-data)

---

## 🧪 Research Disclaimer

⚠️ **This project is for research and educational purposes only.**
Backtest results may contain:
* Lookahead bias
* Data leakage
* Survivorship bias
* No transaction cost modeling
* No slippage modeling

**Do not use in live trading without proper validation.**

---

## 📈 Future Improvements

- [ ] Add transaction cost modeling
- [ ] Add walk-forward retraining
- [ ] Add probability calibration
- [ ] Add portfolio optimization layer
- [ ] Add regime detection
- [ ] Add live data integration

---

## 👨‍💻 Author

**Aşkın Ali Berbergil** *Machine Learning & Quantitative Research* Turkey