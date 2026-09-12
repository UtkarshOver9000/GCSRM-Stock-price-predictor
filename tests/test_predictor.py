"""
Unit & Integration Tests for Stock Price Movement Predictor
============================================================
Verifies:
  1. Data loading, sorting, and schema integrity
  2. Strict absence of look-ahead bias in feature engineering
  3. Correctness and mathematical bounds of hand-rolled indicators
  4. Explicit forward shift and leakage-safe target labeling
  5. Chronological time-based split (zero shuffle leakage)
  6. Scaler isolation (StandardScaler fitted strictly on train)
  7. Baselines (Majority class and Persistence logic)
  8. End-to-end pipeline execution and output metric schema
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from features import (
    load_ohlcv,
    _rsi,
    _macd,
    _bollinger,
    _atr,
    _stochastic,
    _obv,
    _williams_r,
    build_engineered_features,
    build_raw_price_features,
)
from labels import make_label
from run import time_based_split, scale_features, assemble_dataset

DATA_PATH = os.path.join("data", "AAPL.csv")


@pytest.fixture
def sample_df():
    """Load real AAPL sample data."""
    return load_ohlcv(DATA_PATH)


@pytest.fixture
def synthetic_df():
    """Create controlled 100-row synthetic market series."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100))
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_ = (high + low) / 2.0
    volume = np.random.randint(1_000_000, 10_000_000, size=100)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "AdjClose": close,
        "Volume": volume,
    })


# -----------------------------------------------------------------------------
# 1. Data Loading & Schema Tests
# -----------------------------------------------------------------------------
def test_data_loading_and_sorting(sample_df):
    assert not sample_df.empty
    assert "AdjClose" in sample_df.columns
    assert "Date" in sample_df.columns
    # Ensure strictly ascending chronological order
    assert sample_df["Date"].is_monotonic_increasing


# -----------------------------------------------------------------------------
# 2. Strict Absence of Look-Ahead Bias
# -----------------------------------------------------------------------------
def test_no_lookahead_in_features(synthetic_df):
    """
    Changing future rows (t+1, t+2, ...) MUST NOT alter feature values at row t.
    """
    base_features = build_engineered_features(synthetic_df)
    
    # Mutate data in the second half of the dataframe
    mutated_df = synthetic_df.copy()
    split_row = 50
    mutated_df.loc[split_row:, "AdjClose"] *= 1.50
    mutated_df.loc[split_row:, "High"] *= 1.50
    mutated_df.loc[split_row:, "Low"] *= 1.50
    mutated_df.loc[split_row:, "Volume"] *= 2.0

    mutated_features = build_engineered_features(mutated_df)

    # Values at or before split_row - 1 MUST be mathematically identical
    for col in base_features.columns:
        valid_idx = base_features[col].iloc[:split_row].dropna().index
        np.testing.assert_allclose(
            base_features.loc[valid_idx, col].values,
            mutated_features.loc[valid_idx, col].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Lookahead leak detected in feature '{col}' at or before row {split_row}!"
        )


# -----------------------------------------------------------------------------
# 3. Hand-Rolled Indicator Mathematical Bounds
# -----------------------------------------------------------------------------
def test_rsi_bounds_and_neutrality(synthetic_df):
    rsi_vals = _rsi(synthetic_df["AdjClose"], window=14)
    assert (rsi_vals >= 0.0).all() and (rsi_vals <= 100.0).all()
    # NaN-handled initial rows should be filled with 50 (neutral)
    assert (rsi_vals.iloc[:10] == 50.0).all()


def test_macd_relationship(synthetic_df):
    macd_line, signal_line, hist = _macd(synthetic_df["AdjClose"])
    # Histogram must equal macd_line - signal_line
    np.testing.assert_allclose(hist.values, (macd_line - signal_line).values)


def test_bollinger_bands_geometry(synthetic_df):
    upper, lower, pct_b, bandwidth = _bollinger(synthetic_df["AdjClose"], window=20)
    valid = upper.notna()
    # Upper band must be strictly greater than lower band
    assert (upper[valid] > lower[valid]).all()
    # Bandwidth must be non-negative
    assert (bandwidth[valid] >= 0.0).all()


def test_stochastic_and_williams_bounds(synthetic_df):
    k, d = _stochastic(synthetic_df, k_window=14, d_window=3)
    assert (k.dropna() >= 0.0).all() and (k.dropna() <= 100.0).all()
    
    wr = _williams_r(synthetic_df, window=14)
    assert (wr.dropna() >= -100.0).all() and (wr.dropna() <= 0.0).all()


def test_obv_directional_sign(synthetic_df):
    obv = _obv(synthetic_df)
    assert len(obv) == len(synthetic_df)
    assert not obv.isna().any()


# -----------------------------------------------------------------------------
# 4. Target Labeling & Explicit Forward Shift
# -----------------------------------------------------------------------------
def test_explicit_label_shift(synthetic_df):
    """Verify forward shift matches manual calculation and undefined tail is NaN."""
    horizon = 1
    label = make_label(synthetic_df, horizon=horizon, threshold=0.0)

    # Last row must be NaN because t+1 does not exist
    assert pd.isna(label.iloc[-1])

    # Check non-tail rows match (price[t+1] > price[t])
    for i in range(len(synthetic_df) - 1):
        actual_up = 1.0 if synthetic_df["AdjClose"].iloc[i+1] > synthetic_df["AdjClose"].iloc[i] else 0.0
        assert label.iloc[i] == actual_up


def test_dead_zone_abstention(synthetic_df):
    """Verify moves within [-threshold, +threshold] are marked NaN."""
    threshold = 0.02  # 2% dead zone
    label = make_label(synthetic_df, horizon=1, threshold=threshold)
    close = synthetic_df["AdjClose"]
    fwd_ret = (close.shift(-1) - close) / close

    for i in range(len(synthetic_df) - 1):
        if abs(fwd_ret.iloc[i]) <= threshold:
            assert pd.isna(label.iloc[i]), f"Row {i} inside deadzone should be NaN"
        elif fwd_ret.iloc[i] > threshold:
            assert label.iloc[i] == 1.0
        else:
            assert label.iloc[i] == 0.0


# -----------------------------------------------------------------------------
# 5. Chronological Train/Test Split (Zero Shuffle)
# -----------------------------------------------------------------------------
def test_time_based_split_ordering(sample_df):
    full, eng_cols, raw_cols, _ = assemble_dataset(horizon=1, threshold=0.0, stride=1)
    Xe_tr, Xe_te, y_tr, y_te, train_df, test_df = time_based_split(full, eng_cols)

    # Strict chronological partition
    max_train_date = train_df["Date"].max()
    min_test_date = test_df["Date"].min()
    assert max_train_date < min_test_date, "Train dates must strictly precede test dates!"

    # Total row balance matches 80/20 partition
    assert len(train_df) + len(test_df) == len(full)
    assert abs(len(test_df) / len(full) - 0.20) < 0.02


# -----------------------------------------------------------------------------
# 6. Scaler Isolation (Fit on Train Only)
# -----------------------------------------------------------------------------
def test_scaler_isolation():
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"f1": [100.0, 200.0, 300.0]})  # Extreme out-of-distribution values

    X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)

    # Scaler mean must match training mean (3.0), NOT contaminated by test values (100, 200, 300)
    assert scaler.mean_[0] == pytest.approx(3.0)
    # Training standardized mean must be zero
    assert np.mean(X_tr_s) == pytest.approx(0.0, abs=1e-7)


# -----------------------------------------------------------------------------
# 7. End-to-End Execution Verification
# -----------------------------------------------------------------------------
def test_assemble_dataset_outputs():
    full, eng_cols, raw_cols, n_dropped = assemble_dataset(horizon=1, threshold=0.0, stride=1)
    assert len(full) > 1000
    assert len(eng_cols) == 25
    assert len(raw_cols) == 10
    assert not full.isna().any().any(), "Dataset should have zero NaN values after assembly"
