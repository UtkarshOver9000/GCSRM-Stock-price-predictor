# PROJECT CONTEXT - Stock Price Movement Predictor
===================================================

## Executive Mission
Predict next-day price direction (Up/Down) for liquid equities (AAPL) using daily OHLCV
data, demonstrating leakage-free time-series modeling, rigorous technical indicator
engineering (without third-party blackbox libraries), apples-to-apples baseline comparisons,
and honest statistical interpretation.

## Methodological Non-Negotiables
1. **No Data Leakage**:
   - Explicit forward shift in labels (`future_close = close.shift(-horizon)`)
   - Strictly chronological time-based train/test split (no random shuffling)
   - Feature scaling fit strictly on training split (`fit_transform` on train, `transform` on test)
2. **Pure Pandas Feature Engineering**:
   - Zero dependency on unmaintained packages like `pandas-ta`
   - All indicators (RSI, MACD, Bollinger Bands, ATR, Stochastics, Williams %R, OBV) computed directly
3. **Apples-to-Apples Baselines**:
   - Majority-class baseline (uninformed directional bet)
   - Persistence baseline ("tomorrow repeats today")
   - Raw-price feature set evaluated with identical classifier (Logistic Regression)
4. **Honest Interpretation**:
   - Near 50-53% accuracy on 1-day liquid equity moves is expected and reflects market efficiency
   - High multi-day persistence in overlapping windows is an autocorrelation artifact, not genuine predictability
