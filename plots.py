"""
plots.py - Visualizations for the Stock Price Movement Predictor
================================================================
Generates high-resolution, publication-ready figures:
  1. Predicted vs. Actual Direction (price series + directional step plot)
  2. Model Performance Comparison (Accuracy, Precision, Recall, F1)
  3. Feature Importance (Tree-based mean decrease in impurity)
  4. Confusion Matrices (Counts + normalized proportions)
  5. ROC Curves with AUC evaluation
  6. Walk-Forward Cross-Validation fold progression
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "primary":   "#1f77b4",
    "secondary": "#ff7f0e",
    "correct":   "#2ca02c",
    "incorrect": "#d62728",
    "neutral":   "#475569",
    "bg":        "#ffffff",
    "grid":      "#e2e8f0",
    "lr":        "#2563eb",
    "rf":        "#d97706",
    "gb":        "#059669",
    "raw":       "#dc2626",
    "baseline":  "#64748b",
}


def _apply_theme(ax):
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color=PALETTE["grid"], linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=9)


def plot_predicted_vs_actual(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    variant_key: str,
    variant_label: str,
) -> str:
    """Dual-panel chart: AAPL test price series with correct/incorrect markers and step plot."""
    dates = pd.to_datetime(test_df["Date"].values)
    prices = test_df["AdjClose"].values
    correct = (y_true == y_pred)
    acc = accuracy_score(y_true, y_pred)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1]}
    )
    fig.patch.set_facecolor(PALETTE["bg"])

    # Top panel: Price curve + prediction markers
    _apply_theme(ax0)
    ax0.plot(dates, prices, color=PALETTE["neutral"], linewidth=1.3, zorder=1, label="AAPL Adj. Close")
    ax0.scatter(
        dates[correct], prices[correct], color=PALETTE["correct"], s=32,
        label=f"Correct Prediction ({correct.sum()})", zorder=3, edgecolors="white", linewidths=0.5
    )
    ax0.scatter(
        dates[~correct], prices[~correct], color=PALETTE["incorrect"], s=34, marker="x",
        label=f"Incorrect Prediction ({(~correct).sum()})", zorder=3, linewidths=1.5
    )
    ax0.set_ylabel("Adjusted Close ($)", fontsize=10, fontweight="bold", color="#1e293b")
    ax0.set_title(
        f"Predicted vs. Actual Direction - Test Set\n{variant_label}\n"
        f"Best Model: {model_name} | Test Accuracy: {acc:.1%} ({correct.sum()}/{len(y_true)} correct)",
        fontsize=12, fontweight="bold", pad=12, color="#0f172a"
    )
    ax0.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    # Bottom panel: Step direction plot
    _apply_theme(ax1)
    ax1.step(dates, y_true, where="post", label="Actual Direction", color=PALETTE["primary"], linewidth=1.8, zorder=2)
    ax1.step(
        dates, y_pred, where="post", label="Predicted Direction", color=PALETTE["secondary"],
        linewidth=1.4, linestyle="--", alpha=0.9, zorder=3
    )
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


def plot_feature_importance(
    model,
    feature_names: list[str],
    variant_key: str,
    top_n: int = 15,
) -> str:
    """Horizontal bar chart of top feature importances from tree-based classifier."""
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


def plot_roc_curves(
    y_true: np.ndarray,
    models_proba: dict[str, np.ndarray],
    variant_key: str,
) -> str:
    """Overlay ROC curves with computed AUC for evaluated models."""
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


def plot_confusion_matrices(
    y_true: np.ndarray,
    models_preds: dict[str, np.ndarray],
    variant_key: str,
) -> str:
    """Side-by-side confusion matrices for evaluated models."""
    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    fig.patch.set_facecolor(PALETTE["bg"])
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum()
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.45 else "#1e293b"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
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


def plot_model_comparison(
    results_df: pd.DataFrame,
    variant_key: str,
    variant_label: str,
) -> str:
    """Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across all models."""
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
        
        if "Baseline" in model_name:
            color = PALETTE["baseline"]
        elif "RAW" in model_name:
            color = PALETTE["raw"]
        elif "Random Forest" in model_name:
            color = PALETTE["rf"]
        elif "Gradient" in model_name:
            color = PALETTE["gb"]
        else:
            color = PALETTE["lr"]

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


def plot_walk_forward_cv(
    cv_scores: dict[str, list[float]],
    variant_key: str,
) -> str:
    """Line plot of TimeSeriesSplit CV fold performance for each model."""
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
