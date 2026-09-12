"""
labels.py - Target construction for the Stock Price Movement Predictor
=======================================================================
Constructs direction labels with an explicit, auditable forward shift.

The label at day t determines whether the price moved up or down over
the forward horizon [t -> t + horizon].
The key line is: future_close = close.shift(-horizon)

Parameters:
  horizon   : number of trading days forward (1 = next day, 5 = 1 week)
  threshold : dead-zone percentage (moves within [-threshold, +threshold] are dropped as NaN)
"""

from __future__ import annotations
import pandas as pd


def make_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    close = df["AdjClose"]

    # -----------------------------------------------------------------
    # EXPLICIT FORWARD SHIFT: This is the sole forward-looking step.
    # Because features use only data up to day t close, pairing with
    # close.shift(-horizon) creates zero look-ahead data leakage.
    # -----------------------------------------------------------------
    future_close = close.shift(-horizon)
    fwd_return = (future_close - close) / close

    if threshold <= 0.0:
        label = (fwd_return > 0).astype("float")
    else:
        label = pd.Series(index=df.index, dtype="float")
        label[fwd_return > threshold] = 1.0
        label[fwd_return < -threshold] = 0.0
        # |fwd_return| <= threshold stays NaN (abstain)

    # Final horizon rows cannot have a future close; explicitly mark as NaN
    label.iloc[-horizon:] = pd.NA
    return label
