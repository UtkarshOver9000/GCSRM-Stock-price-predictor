# 📈 GCSRM Stock Price Movement Predictor & Terminal

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.3.2-150458.svg)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Leakage--Free%20Verified-brightgreen.svg)]()

A production-grade, multi-asset quantitative machine learning system designed to predict **next-day price direction** (Up or Down) across multiple diverse, real-world asset classes using real daily OHLCV market data. The pipeline benchmark evaluates six core assets (**AAPL, NVDA, TSLA, SPY, JPM, AMZN**), while the dashboard also includes additional datasets (**AMD, GOOGL, META, MSFT, QQQ, XOM**).

Equipped with an **Interactive Web Dashboard (`dashboard.html`)**, **Deep-Dive Technical Indicator Charts**, **Backtested Strategy Equity Curves vs. Buy & Hold**, and an **Automated Test Suite**.

---

## 🎯 Key Highlights & Capabilities

1. **Multi-Sector Real-World Data**:
   - **Apple (`AAPL`)**: Mega-Cap Tech / Consumer Electronics
   - **NVIDIA (`NVDA`)**: High-Growth AI & Semiconductors
   - **Tesla (`TSLA`)**: High-Beta / EV Growth Stock
   - **SPDR S&P 500 (`SPY`)**: Broad Macro Market Index
   - **JPMorgan Chase (`JPM`)**: Banking & Financial Services
   - **Amazon (`AMZN`)**: Cloud Infrastructure & E-Commerce
   - **Additional dashboard assets**: AMD, GOOGL, META, MSFT, QQQ, and XOM
2. **Interactive Visual Dashboard (`dashboard.html`)**:
   - Built-in responsive financial terminal with live ticker switching.
   - Interactive price chart with green/red prediction signals, toggleable Bollinger Bands, and Moving Averages.
   - Separate interactive subplots for Wilder RSI (14) and MACD (12, 26, 9).
   - Real-time Strategy Wealth Compounder ($10,000 portfolio growth vs. Buy & Hold benchmark).
   - Interactive Confusion Matrix and Top 10 Feature Importances breakdown.
3. **Advanced Visual Tool Suite**:
   - **Technical Indicator Deep-Dive**: 4-panel publication-ready chart detailing Bollinger Bands, Moving Averages with Golden/Death crosses, Volume with On-Balance Volume (OBV), MACD momentum, and RSI overbought/oversold danger zones.
   - **Strategy Equity & Drawdown Curve**: Quantifies whether directional signals translate to alpha over a passive buy-and-hold benchmark.
   - **Cross-Asset Comparison Matrix**: Evaluates directional predictability across 6 distinct market sectors.
4. **Strict Leakage Prevention & Scientific Honesty**:
   - **Explicit Target Construction**: Directional labels constructed via `close.shift(-horizon)`.
   - **Chronological Time Splits**: Zero random shuffling.
   - **Isolated Preprocessing**: `StandardScaler` fitted strictly on the training partition.
   - **Honest Statistical Reporting**: Quantifies why next-day liquid equity moves closely adhere to a random walk, avoids curve-fitted overfitting, and unmasks multi-day autocorrelation artifacts.

---

## 📁 Repository Structure

```
├── data/
│   ├── AAPL.csv                # Apple daily OHLCV (5 years, 1,255 trading days)
│   ├── NVDA.csv                # NVIDIA daily OHLCV (5 years, 1,255 trading days)
│   ├── TSLA.csv                # Tesla daily OHLCV (5 years, 1,255 trading days)
│   ├── SPY.csv                 # S&P 500 ETF daily OHLCV (5 years, 1,255 trading days)
│   ├── JPM.csv                 # JPMorgan Chase daily OHLCV (5 years, 1,255 trading days)
│   ├── AMZN.csv                # Amazon daily OHLCV (5 years, 1,255 trading days)
│   ├── AMD.csv, GOOGL.csv      # Additional semiconductor and search assets
│   ├── META.csv, MSFT.csv      # Additional platform and software assets
│   ├── QQQ.csv, XOM.csv        # Nasdaq-100 ETF and energy asset
├── outputs/                    # Exported charts and benchmark metrics
│   ├── technical_deep_dive_AAPL.png
│   ├── technical_deep_dive_NVDA.png
│   ├── technical_deep_dive_TSLA.png
│   ├── technical_deep_dive_SPY.png
│   ├── strategy_equity_AAPL.png
│   ├── strategy_equity_NVDA.png
│   ├── strategy_equity_TSLA.png
│   ├── strategy_equity_SPY.png
│   ├── multi_asset_comparison_matrix.png
│   ├── predicted_vs_actual_1day_strict.png
│   ├── feature_importance_1day_strict.png
│   ├── roc_curves_1day_strict.png
│   ├── confusion_matrices_1day_strict.png
│   ├── model_comparison_1day_strict.png
│   ├── walk_forward_cv_1day_strict.png
│   └── results_summary_all_variants.csv
├── tests/
│   └── test_predictor.py       # Automated 12-point unit & integration test suite
├── dashboard.html              # Interactive Financial Terminal & Visual Dashboard
├── features.py                 # Hand-rolled feature engineering (25 engineered + 10 raw)
├── labels.py                   # Explicit, leakage-safe target labeling logic
├── plots.py                    # Advanced visualization suite
├── run.py                      # Multi-asset master execution pipeline
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

### 2. Run the Full Multi-Asset Pipeline
```bash
python run.py
```
To run on a specific real-world stock:
```bash
python run.py --ticker NVDA
python run.py --ticker TSLA
python run.py --ticker SPY
python run.py --ticker JPM
python run.py --ticker AMZN
```

### 3. Open the Interactive Visual Dashboard
Simply open `dashboard.html` in any web browser to view the interactive real-time terminal:
```powershell
start dashboard.html
```

For Vercel, import this GitHub repository as a static project. The included
`vercel.json` routes `/` to `index.html` and `/dashboard` to `dashboard.html`.

### 4. Execute the Automated Test Suite
```bash
python -m pytest -v
```

---

## 📊 Cross-Asset Benchmark Results (Real 5-Year Data)

| Ticker | Asset Class / Sector | Test ML Accuracy | Naive Majority Baseline | Lift Over Base |
| :--- | :--- | :---: | :---: | :---: |
| **AAPL** | Mega-Cap Tech / Consumer | **55.19%** | 52.80% | **+2.39%** |
| **NVDA** | AI Hardware / Semiconductors | **50.62%** | 53.32% | -2.70% |
| **TSLA** | High-Beta Growth / EV | **49.79%** | 50.83% | -1.04% |
| **SPY** | S&P 500 Broad Market ETF | **53.11%** | 53.32% | -0.21% |
| **JPM** | Financials & Commercial Banking | **53.53%** | 54.56% | -1.03% |
| **AMZN** | Cloud Computing & Retail | **50.21%** | 50.62% | -0.41% |

---

## 🧠 Quantitative Insights & Market Realities

1. **The Martingale Nature of Single-Day Direction**:
   In highly liquid equities and broad indices (SPY, AAPL, JPM), daily price changes closely approximate a random walk. Achieving 50%–55% directional accuracy with modest lift is consistent with the Efficient Market Hypothesis.
2. **The 5-Day Overlapping Window Trap**:
   When predicting 5 days forward sampled daily, consecutive target labels share 4 out of 5 trading days. This inflates the persistence baseline to ~84%, an artifact of window autocorrelation rather than market predictability. Striding non-overlapping periods restores the fair baseline of ~57%.

---

## 📜 License
Released under the [MIT License](LICENSE).
