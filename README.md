# 📈 Stock Price Movement Predictor

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.3.2-150458.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Leakage--Free%20Verified-brightgreen.svg)]()

A production-grade, scientifically rigorous machine learning pipeline designed to predict **next-day price direction** (Up or Down) on liquid equity instruments (**AAPL**) using daily OHLCV market data.

Built with an unwavering commitment to **leakage-free time-series methodology**, **hand-rolled indicator engineering in pandas (no `pandas-ta`)**, **fair baseline comparisons**, and **honest quantitative interpretation**.

---

## 🎯 Project Highlights

1. **Strict Leakage Prevention**:
   - **Explicit Target Construction**: Directional labels constructed via `close.shift(-horizon)` with auditable separation from historical feature matrices.
   - **Chronological Time Splits**: Zero random shuffling. Train data strictly precedes test data in calendar time.
   - **Isolated Preprocessing**: `StandardScaler` fitted **only** on the training partition and transformed on test.
2. **Hand-Rolled Technical Indicators**:
   - 25 engineered features calculated entirely in pandas (Wilder RSI, MACD, Bollinger Bands, ATR, Stochastic %K/%D, Williams %R, OBV Momentum, moving average cross ratios).
   - Zero reliance on outdated packages (`pandas-ta` fails on NumPy 2.x).
3. **Four Exhaustive Labeling Experiments**:
   - **Variant A**: Next-day direction (canonical benchmark)
   - **Variant B**: Next-day direction with 1% dead-zone (abstain on noise)
   - **Variant C**: 5-day horizon with overlapping daily windows (unmasking the autocorrelation artifact)
   - **Variant D**: 5-day horizon with non-overlapping stride=5 sampling (fair multi-day evaluation)
4. **Honest Statistical Reporting**:
   - Transparently contrasts ML models against **Majority Class** and **Persistence** baselines.
   - Demonstrates why single-stock daily direction is close to a random walk, avoiding cherry-picked metrics.

---

## 📁 Repository Structure

```
├── data/
│   └── AAPL.csv                # Historical daily OHLCV data (1,260 trading days)
├── outputs/                    # Visualizations and exported benchmark results
│   ├── predicted_vs_actual_1day_strict.png
│   ├── predicted_vs_actual_1day_deadzone.png
│   ├── predicted_vs_actual_5day_strict.png
│   ├── predicted_vs_actual_5day_nonoverlap.png
│   ├── feature_importance_1day_strict.png
│   ├── roc_curves_1day_strict.png
│   ├── confusion_matrices_1day_strict.png
│   ├── model_comparison_1day_strict.png
│   ├── walk_forward_cv_1day_strict.png
│   └── results_summary_all_variants.csv
├── features.py                 # Hand-rolled feature engineering (25 engineered + 10 raw)
├── labels.py                   # Explicit, leakage-safe target labeling logic
├── plots.py                    # Publication-ready visualization suite
├── run.py                      # Master execution pipeline
├── requirements.txt            # Pinned reproducible dependencies
├── PROJECT_CONTEXT.md          # Technical documentation & reproduction guide
└── README.md                   # Project documentation
```

---

## ⚙️ Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/UtkarshOver9000/GCSRM-Stock-price-predictor.git
cd GCSRM-Stock-price-predictor
pip install -r requirements.txt
```

### 2. Execute Complete Pipeline
```bash
python run.py
```
This executes all 4 labeling experiments, runs walk-forward cross-validation, reports metrics, and exports all plots and tables to `outputs/`.

---

## 🧪 Methodological Compliance Checklist

| Requirement | Implementation Details | File / Function |
| :--- | :--- | :--- |
| **Naive Baselines** | Evaluates Majority Class and Persistence ("tomorrow repeats today") | `run.py` -> Step 6 |
| **Raw vs. Engineered** | Apples-to-apples comparison using identical Logistic Regression classifier | `run.py` -> Step 7 |
| **Model Family Comparison** | Logistic Regression vs. Random Forest vs. Gradient Boosting | `run.py` -> Step 8 |
| **Class Balance** | Quantified and reported on Full, Train, and Test sets | `run.py` -> `report_class_balance()` |
| **Explicit Label Shift** | Clear `.shift(-horizon)` forward looking mechanism | `labels.py` -> `make_label()` |
| **Time-Based Split** | Chronological 80/20 train/test partition without shuffling | `run.py` -> `time_based_split()` |
| **Isolated Scaler** | `StandardScaler.fit_transform` on train, `transform` on test | `run.py` -> `scale_features()` |
| **Prediction Visualizations** | Dual-panel charts with price markers + directional step curves | `plots.py` -> `plot_predicted_vs_actual()` |

---

## 📊 Benchmark Results

### Cross-Variant ML Model Performance

| Variant | Best Model | Test Accuracy | Baseline Accuracy | Lift | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A. Next-Day (Strict)** | Logistic Regression | **53.31%** | 52.89% (Majority) | +0.42% | Efficient market / Random walk noise |
| **B. Next-Day (1% Dead-Zone)** | Logistic Regression | **63.16%** | 57.89% (Majority) | +5.27% | Improved signal when filtering ambiguous moves |
| **C. 5-Day (Overlapping)** | Persistence Baseline | **83.47%** | 83.47% (Persistence) | +0.00% | **Artifact**: Overlapping windows share 4/5 days |
| **D. 5-Day (Non-Overlapping)** | Random Forest | **67.35%** | 57.14% (Majority) | +10.20% | Promising multi-day signal (caveat: N=49 test rows) |

---

## 🧠 Key Quantitative Takeaways & Honest Analysis

### 1. The Reality of Next-Day Direction (Variant A)
In liquid, mega-cap equities like Apple (AAPL), single-day returns closely follow a **martingale difference sequence**. Achieving ~53.3% accuracy against a ~52.9% majority baseline demonstrates that simple technical indicators do not yield an automatic, outsized arbitrage edge. This honesty separates quantitative engineering from curve-fitted hype.

### 2. The Overlapping Window Illusion (Variant C vs. D)
When forecasting 5 days ahead sampled daily, day $t$ and day $t+1$ share 4 identical trading days. This generates massive serial correlation in the target variable by construction. The persistence baseline surges to **83.47%**, not because future market regimes are trivial to forecast, but because the rolling windows overlap. **Variant D** rectifies this by strictly striding every 5 trading days, revealing a genuine non-overlapping baseline of 57.14%.

---

## 🔭 Further Research & Advanced Directions

1. **Sequence Modeling (LSTMs & Time-Series Transformers)**:
   - Capturing non-linear temporal dependencies across lookback windows. Requires massive cross-sectional datasets to prevent overfitting.
2. **Cross-Sectional Statistical Arbitrage (Pairs / Cointegration)**:
   - Modeling spread dynamics between correlated peers (e.g., AAPL vs. MSFT/QQQ) rather than predicting direction in isolation.
3. **Alternative & Sentiment Features**:
   - Incorporating pre-market news sentiment and institutional order flow with strict timestamp auditing.

---

## 📜 License
Released under the [MIT License](LICENSE).
