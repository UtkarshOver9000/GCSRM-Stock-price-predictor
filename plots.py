"""
plots.py - Advanced Visualizations for the Stock Price Movement Predictor
==========================================================================
Generates high-resolution, publication-ready figures for:
  1. Technical Indicator Deep-Dive (Price + BB + SMA + Volume + MACD + RSI)
  2. Predicted vs. Actual Direction (Price series + step plot)
  3. ML Strategy Equity Curve vs. Buy & Hold Benchmark (with Drawdown)
  4. Cross-Asset Performance Comparison Matrix
  5. Feature Importance (Mean Decrease in Impurity)
  6. Confusion Matrices (Counts + normalized proportions)
  7. ROC Curves with AUC evaluation
  8. Walk-Forward Cross-Validation fold progression
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "primary":   "#2563eb",  # Blue
    "secondary": "#f59e0b",  # Amber
    "correct":   "#10b981",  # Emerald Green
    "incorrect": "#ef4444",  # Red
    "neutral":   "#475569",  # Slate
    "bg":        "#ffffff",
    "card_bg":   "#f8fafc",
    "grid":      "#e2e8f0",
    "lr":        "#3b82f6",
    "rf":        "#d97706",
    "gb":        "#059669",
    "raw":       "#dc2626",
    "baseline":  "#64748b",
    "accent1":   "#8b5cf6",
    "accent2":   "#06b6d4",
}


def _apply_theme(ax):
    ax.set_facecolor(PALETTE["card_bg"])
    ax.grid(True, color=PALETTE["grid"], linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=9)


def plot_technical_deep_dive(df: pd.DataFrame, ticker: str = "AAPL") -> str:
    """
    4-Panel High-Impact Financial Chart:
      Panel 1: Candlestick/Close + Bollinger Bands + SMA 20 & 50 with Golden Crosses
      Panel 2: Daily Volume bars + On-Balance Volume (OBV) trend
      Panel 3: MACD Line, Signal Line, and Momentum Histogram
      Panel 4: Wilder RSI (14) with Overbought (70) and Oversold (30) zones
    """
    close = df["AdjClose"]
    dates = pd.to_datetime(df["Date"])

    # 1. SMAs & Bollinger Bands
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    std_20 = close.rolling(20).std()
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20

    # 2. RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_14 = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    # 3. MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 4. Volume & OBV
    obv = (np.sign(close.diff()).fillna(0.0) * df["Volume"]).cumsum()

    # Slice the last 250 trading days for clear, readable visual presentation
    n_days = min(250, len(df))
    sl = slice(-n_days, None)
    d = dates.iloc[sl]

    fig = plt.figure(figsize=(15, 11), facecolor=PALETTE["bg"])
    gs = GridSpec(4, 1, height_ratios=[3.0, 1.0, 1.2, 1.1], hspace=0.15)

    # Panel 1: Price & Bands
    ax0 = fig.add_subplot(gs[0])
    _apply_theme(ax0)
    ax0.plot(d, close.iloc[sl], color="#0f172a", linewidth=1.5, label=f"{ticker} Adj. Close", zorder=4)
    ax0.plot(d, sma_20.iloc[sl], color=PALETTE["primary"], linewidth=1.2, linestyle="-", label="SMA (20)", zorder=3)
    ax0.plot(d, sma_50.iloc[sl], color=PALETTE["secondary"], linewidth=1.2, linestyle="-", label="SMA (50)", zorder=3)
    ax0.plot(d, upper_bb.iloc[sl], color="#94a3b8", linewidth=0.9, linestyle="--", label="Bollinger Bands (20, 2)", zorder=2)
    ax0.plot(d, lower_bb.iloc[sl], color="#94a3b8", linewidth=0.9, linestyle="--", zorder=2)
    ax0.fill_between(d, lower_bb.iloc[sl], upper_bb.iloc[sl], color=PALETTE["primary"], alpha=0.07, zorder=1)
    
    # Detect Golden/Death Crosses in slice
    cross = np.sign(sma_20.iloc[sl] - sma_50.iloc[sl]).diff()
    golden = (cross > 0)
    death = (cross < 0)
    if golden.any():
        ax0.scatter(d[golden], sma_20.iloc[sl][golden], marker="^", color="#10b981", s=90, label="Golden Cross (Bullish)", zorder=5)
    if death.any():
        ax0.scatter(d[death], sma_20.iloc[sl][death], marker="v", color="#ef4444", s=90, label="Death Cross (Bearish)", zorder=5)

    ax0.set_ylabel("Price (USD)", fontsize=10, fontweight="bold")
    ax0.set_title(f"Technical Indicator Deep-Dive: {ticker} (Last {n_days} Trading Days)", fontsize=13, fontweight="bold", pad=10)
    ax0.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5, ncol=3)
    ax0.tick_params(labelbottom=False)

    # Panel 2: Volume & OBV
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    _apply_theme(ax1)
    is_up = close.iloc[sl] >= close.shift(1).iloc[sl]
    vol_colors = [PALETTE["correct"] if u else PALETTE["incorrect"] for u in is_up]
    ax1.bar(d, df["Volume"].iloc[sl] / 1e6, color=vol_colors, alpha=0.7, width=0.8, label="Volume (M)")
    ax1.set_ylabel("Vol (M)", fontsize=9, fontweight="bold")
    ax1.tick_params(labelbottom=False)

    # Panel 3: MACD
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    _apply_theme(ax2)
    ax2.plot(d, macd_line.iloc[sl], color=PALETTE["primary"], linewidth=1.3, label="MACD Line (12, 26)")
    ax2.plot(d, signal_line.iloc[sl], color=PALETTE["secondary"], linewidth=1.2, linestyle="--", label="Signal Line (9)")
    hist_colors = [PALETTE["correct"] if h >= 0 else PALETTE["incorrect"] for h in macd_hist.iloc[sl]]
    ax2.bar(d, macd_hist.iloc[sl], color=hist_colors, alpha=0.6, width=0.8, label="Momentum Hist")
    ax2.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("MACD", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
    ax2.tick_params(labelbottom=False)

    # Panel 4: RSI (14)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    _apply_theme(ax3)
    ax3.plot(d, rsi_14.iloc[sl], color=PALETTE["accent1"], linewidth=1.4, label="Wilder RSI (14)")
    ax3.axhline(70.0, color="#ef4444", linewidth=1.0, linestyle="--", label="Overbought (70)")
    ax3.axhline(30.0, color="#10b981", linewidth=1.0, linestyle="--", label="Oversold (30)")
    ax3.fill_between(d, 70.0, 100.0, color="#ef4444", alpha=0.08)
    ax3.fill_between(d, 0.0, 30.0, color="#10b981", alpha=0.08)
    ax3.set_ylim(10, 90)
    ax3.set_ylabel("RSI", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8, ncol=3)

    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()

    out_path = os.path.join("outputs", f"technical_deep_dive_{ticker}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_strategy_equity_curve(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "AAPL",
    model_name: str = "Random Forest",
) -> str:
    """
    Backtested Cumulative Wealth Curve:
      Compares ML Model Driven Strategy (Long when predicting Up, Cash when predicting Down)
      against Buy & Hold Benchmark on the test set.
      Includes Maximum Drawdown plot in the lower panel.
    """
    close = test_df["AdjClose"].values
    dates = pd.to_datetime(test_df["Date"].values)
    n = len(close)

    daily_returns = np.diff(close) / close[:-1]
    # Prediction on day t determines position on day t+1
    positions = y_pred[:-1]

    # Strategy return: when predicted Up (1) -> earn market return; when Down (0) -> earn 0% (cash)
    strategy_returns = positions * daily_returns
    
    # Cumulative compound growth (starting from $10,000)
    capital = 10000.0
    equity_strategy = capital * np.cumprod(1.0 + strategy_returns)
    equity_benchmark = capital * np.cumprod(1.0 + daily_returns)
    equity_strategy = np.insert(equity_strategy, 0, capital)
    equity_benchmark = np.insert(equity_benchmark, 0, capital)

    # Drawdown calculations
    cummax_strat = np.maximum.accumulate(equity_strategy)
    drawdown_strat = (equity_strategy - cummax_strat) / cummax_strat

    cummax_bench = np.maximum.accumulate(equity_benchmark)
    drawdown_bench = (equity_benchmark - cummax_bench) / cummax_bench

    strat_total_return = (equity_strategy[-1] / capital - 1.0) * 100
    bench_total_return = (equity_benchmark[-1] / capital - 1.0) * 100
    acc = accuracy_score(y_true, y_pred)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel 1: Wealth Index
    _apply_theme(ax0)
    ax0.plot(dates, equity_strategy, color=PALETTE["primary"], linewidth=2.0,
             label=f"ML Strategy ({model_name}): {strat_total_return:+.1f}% ($10k -> ${equity_strategy[-1]:,.0f})")
    ax0.plot(dates, equity_benchmark, color="#64748b", linewidth=1.5, linestyle="--",
             label=f"Buy & Hold {ticker}: {bench_total_return:+.1f}% ($10k -> ${equity_benchmark[-1]:,.0f})")
    ax0.set_ylabel("Portfolio Value ($)", fontsize=10, fontweight="bold")
    ax0.set_title(
        f"Backtested ML Strategy vs. Buy & Hold Benchmark: {ticker} Test Partition\n"
        f"Accuracy: {acc:.1%} | Strategy Total Return: {strat_total_return:+.1f}% vs. Buy & Hold: {bench_total_return:+.1f}%",
        fontsize=11, fontweight="bold", pad=10
    )
    ax0.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    # Panel 2: Drawdown
    _apply_theme(ax1)
    ax1.plot(dates, drawdown_strat * 100, color=PALETTE["primary"], linewidth=1.3, label="ML Strategy Drawdown")
    ax1.plot(dates, drawdown_bench * 100, color="#64748b", linewidth=1.1, linestyle=":", label="Benchmark Drawdown")
    ax1.fill_between(dates, drawdown_strat * 100, 0, color=PALETTE["primary"], alpha=0.15)
    ax1.set_ylabel("Drawdown (%)", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)

    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()

    out_path = os.path.join("outputs", f"strategy_equity_{ticker}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multi_asset_matrix(summary_df: pd.DataFrame) -> str:
    """
    Multi-Asset Cross-Sectional Comparison Chart:
      Visualizes Accuracy, Majority Baseline, and Lift across diverse asset classes
      (Tech, Semiconductors, High-Beta, Broad Index, Banking, E-Commerce).
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _apply_theme(ax)

    tickers = summary_df["ticker"].tolist()
    ml_acc = summary_df["ml_accuracy"].values * 100
    base_acc = summary_df["baseline_accuracy"].values * 100
    lift = summary_df["lift"].values * 100

    x = np.arange(len(tickers))
    width = 0.35

    rects1 = ax.bar(x - width/2, ml_acc, width, label="Best ML Model Accuracy", color=PALETTE["primary"], edgecolor="white")
    rects2 = ax.bar(x + width/2, base_acc, width, label="Naive Majority Baseline", color=PALETTE["baseline"], edgecolor="white")

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#64748b")

    # Add lift badge above each pair
    for i, l in enumerate(lift):
        color = "#10b981" if l >= 0 else "#ef4444"
        prefix = "+" if l >= 0 else ""
        ax.text(x[i], max(ml_acc[i], base_acc[i]) + 4.5, f"Lift: {prefix}{l:.1f}%", ha="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor=color, linewidth=1))

    ax.axhline(50.0, color="#94a3b8", linestyle=":", linewidth=1.0, label="Coin-Flip (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["label"].tolist(), fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_ylim([35, 75])
    ax.set_title("Cross-Asset Directional Predictability (5-Year Real Market Data)\nML Accuracy vs. Naive Baseline Across Multiple Sectors",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join("outputs", "multi_asset_comparison_matrix.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_predicted_vs_actual(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    variant_key: str,
    variant_label: str,
) -> str:
    """Dual-panel chart: test price series with correct/incorrect markers and step plot."""
    dates = pd.to_datetime(test_df["Date"].values)
    prices = test_df["AdjClose"].values
    correct = (y_true == y_pred)
    acc = accuracy_score(y_true, y_pred)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.6, 1]})
    fig.patch.set_facecolor(PALETTE["bg"])

    _apply_theme(ax0)
    ax0.plot(dates, prices, color=PALETTE["neutral"], linewidth=1.3, zorder=1, label="Adj. Close")
    ax0.scatter(dates[correct], prices[correct], color=PALETTE["correct"], s=32,
                label=f"Correct Prediction ({correct.sum()})", zorder=3, edgecolors="white", linewidths=0.5)
    ax0.scatter(dates[~correct], prices[~correct], color=PALETTE["incorrect"], s=34, marker="x",
                label=f"Incorrect Prediction ({(~correct).sum()})", zorder=3, linewidths=1.5)
    ax0.set_ylabel("Adjusted Close ($)", fontsize=10, fontweight="bold", color="#1e293b")
    ax0.set_title(
        f"Predicted vs. Actual Direction - Test Set\n{variant_label}\n"
        f"Best Model: {model_name} | Test Accuracy: {acc:.1%} ({correct.sum()}/{len(y_true)} correct)",
        fontsize=12, fontweight="bold", pad=12, color="#0f172a"
    )
    ax0.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    _apply_theme(ax1)
    ax1.step(dates, y_true, where="post", label="Actual Direction", color=PALETTE["primary"], linewidth=1.8, zorder=2)
    ax1.step(dates, y_pred, where="post", label="Predicted Direction", color=PALETTE["secondary"],
             linewidth=1.4, linestyle="--", alpha=0.9, zorder=3)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Down (0)", "Up (1)"], fontsize=9, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10, fontweight="bold", color="#1e293b")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    out_path = os.path.join("outputs", f"predicted_vs_actual_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(model, feature_names: list[str], variant_key: str, top_n: int = 15) -> str:
    if not hasattr(model, "feature_importances_"):
        return ""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    _apply_theme(ax)

    colors = [PALETTE["rf"] if v >= top.median() else PALETTE["lr"] for v in top.values]
    bars = ax.barh(top.index, top.values, color=colors, edgecolor="white", linewidth=0.6, height=0.65)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8, color="#334155")

    ax.set_xlabel("Relative Importance (Mean Decrease in Impurity)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances - Random Forest ({variant_key})", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    out_path = os.path.join("outputs", f"feature_importance_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(y_true: np.ndarray, models_proba: dict[str, np.ndarray], variant_key: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    _apply_theme(ax)

    colors = [PALETTE["lr"], PALETTE["rf"], PALETTE["gb"], PALETTE["raw"]]
    for (name, proba), color in zip(models_proba.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="#94a3b8", linewidth=1.2, label="Chance Baseline (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title(f"ROC Curves - Model Comparison ({variant_key})", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join("outputs", f"roc_curves_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(y_true: np.ndarray, models_preds: dict[str, np.ndarray], variant_key: str) -> str:
    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    fig.patch.set_facecolor(PALETTE["bg"])
    if n == 1: axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum()
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.45 else "#1e293b"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Down", "Pred Up"], fontsize=8, fontweight="bold")
        ax.set_yticklabels(["Act Down", "Act Up"], fontsize=8, fontweight="bold")
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name}\nAcc: {acc:.1%}", fontsize=10, fontweight="bold", pad=8)

    fig.suptitle(f"Confusion Matrices - Test Set ({variant_key})", fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    out_path = os.path.join("outputs", f"confusion_matrices_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(results_df: pd.DataFrame, variant_key: str, variant_label: str) -> str:
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = results_df["model"].tolist()
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _apply_theme(ax)

    for i, model_name in enumerate(models):
        row = results_df[results_df["model"] == model_name].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        
        if "Baseline" in model_name: color = PALETTE["baseline"]
        elif "RAW" in model_name: color = PALETTE["raw"]
        elif "Random Forest" in model_name: color = PALETTE["rf"]
        elif "Gradient" in model_name: color = PALETTE["gb"]
        else: color = PALETTE["lr"]

        label_short = model_name.replace("Baseline: ", "").replace(" - ENGINEERED features", "").replace(" - RAW price features", " (Raw)")
        rects = ax.bar(x + offset, values, width * 0.92, label=label_short, color=color, alpha=0.9, edgecolor="white")
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
    ax.set_ylim([0, 1.18])
    ax.set_title(f"Model Performance Comparison\n{variant_label}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path = os.path.join("outputs", f"model_comparison_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_walk_forward_cv(cv_scores: dict[str, list[float]], variant_key: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _apply_theme(ax)

    colors = [PALETTE["lr"], PALETTE["rf"], PALETTE["gb"]]
    for (name, scores), color in zip(cv_scores.items(), colors):
        folds = list(range(1, len(scores) + 1))
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.plot(folds, scores, marker="o", linewidth=2.2, markersize=7, color=color,
                label=f"{name} (mean={mean_score:.3f} +/- {std_score:.3f})")
        ax.axhline(mean_score, color=color, linestyle="--", alpha=0.35, linewidth=1.0)

    ax.axhline(0.50, color="#94a3b8", linestyle=":", linewidth=1.2, label="Chance (0.50)")
    ax.set_xlabel("Chronological TimeSeriesSplit Fold (1 -> 5)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Accuracy", fontsize=10, fontweight="bold")
    ax.set_title(f"Walk-Forward Cross-Validation Stability across Folds ({variant_key})", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(range(1, 6))
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    ax.set_ylim([0.25, 0.90])

    fig.tight_layout()
    out_path = os.path.join("outputs", f"walk_forward_cv_{variant_key}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
