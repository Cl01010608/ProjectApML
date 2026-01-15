"""
05_results_analysis.py

Results analysis and visualization.
This script provides functions to analyze and visualize the performance of various machine learning models
based on their evaluation metrics. It generates plots and tables to summarize model performance, including
R², RMSE, and overfitting gap.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =============================================================================
# PATHS
# =============================================================================

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# UTIL
# =============================================================================

def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {path}")

def _get_predictions(pred_df, model_name: str):
    col = f"pred_{model_name}"
    if col not in pred_df.columns:
        raise KeyError(f"Prediction column not found: {col}")
    return pred_df[col].values

# =============================================================================
# PREP
# =============================================================================

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["overfit_gap"] = df["Train_R2"] - df["Val_R2"]
    return df.sort_values("Val_R2", ascending=False).reset_index(drop=True)

# =============================================================================
# 1. MODEL COMPARISON
# =============================================================================

def plot_top_models_by_val_r2(df: pd.DataFrame, top_n: int = 15):
    df_top = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_top["Model"], df_top["Val_R2"], edgecolor="black", alpha=0.8)
    ax.set_xlabel("Validation R²")
    ax.set_title(f"Top {top_n} Models by Validation R²")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    for i, v in enumerate(df_top["Val_R2"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save(fig, "01_top_models_val_r2.png")


def plot_top_models_by_test_rmse(df: pd.DataFrame, top_n: int = 15):
    df_top = df.sort_values("Test_RMSE", ascending=True).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_top["Model"], df_top["Test_RMSE"], edgecolor="black", alpha=0.8)
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title(f"Top {top_n} Models by Test RMSE")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    for i, v in enumerate(df_top["Test_RMSE"]):
        ax.text(v + 0.002, i, f"{v:.5f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save(fig, "02_top_models_test_rmse.png")

# =============================================================================
# 2. METRICS TABLE
# =============================================================================

def plot_metrics_table(df: pd.DataFrame, top_n: int = 8):
    df_top = df.head(top_n)

    display = df_top[[
        "Model","Type",
        "Val_R2","Test_R2",
        "Val_RMSE","Test_RMSE",
        "overfit_gap","train_time"
    ]].copy()

    fmt = {
        "Val_R2": "{:.4f}",
        "Test_R2": "{:.4f}",
        "Val_RMSE": "{:.5f}",
        "Test_RMSE": "{:.5f}",
        "overfit_gap": "{:.4f}",
        "train_time": "{:.2f}s",
    }

    for c, f in fmt.items():
        display[c] = display[c].map(lambda x: f.format(x))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    plt.title("Top Models – Performance Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    _save(fig, "03_metrics_table.png")

# =============================================================================
# 3. MULTI-METRIC COMPARISON
# =============================================================================

def plot_multimetric_comparison(df: pd.DataFrame, top_n: int = 10):
    df_top = df.head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Val R2
    ax = axes[0, 0]
    ax.barh(df_top["Model"], df_top["Val_R2"], alpha=0.8)
    ax.set_title("Validation R²")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    # Test RMSE
    ax = axes[0, 1]
    ax.barh(df_top["Model"], df_top["Test_RMSE"], alpha=0.8)
    ax.set_title("Test RMSE (↓)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    # Overfit gap
    ax = axes[1, 0]
    ax.barh(df_top["Model"], df_top["overfit_gap"], alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Overfit Gap (Train − Val)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    # Train time
    ax = axes[1, 1]
    ax.barh(df_top["Model"], df_top["train_time"], alpha=0.8)
    ax.set_xscale("log")
    ax.set_title("Training Time (log s)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Multi-Metric Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    _save(fig, "04_multimetric_comparison.png")

# =============================================================================
# 4. PREDICTIONS & RESIDUALS
# =============================================================================

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    val_r2: float,
    test_rmse: float,
):
    errors = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="black")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(model_name)
    ax.grid(alpha=0.3)

    ax.text(
        0.05, 0.95,
        f"Val R² = {val_r2:.4f}\nTest RMSE = {test_rmse:.5f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Residuals
    ax = axes[1]
    ax.hist(errors, bins=40, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    try:
        kde = gaussian_kde(errors)
        xs = np.linspace(errors.min(), errors.max(), 200)
        ax.plot(xs, kde(xs), "k-", linewidth=2)
    except Exception:
        pass
    ax.set_title("Residual Distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    _save(fig, f"05_pred_vs_actual_{model_name[:30]}.png")

# =============================================================================
# 5. TOP-4 COMPARISON
# =============================================================================

def plot_top4_predictions(df: pd.DataFrame, predictions: Dict[str, np.ndarray], y_test: np.ndarray):
    top4 = df.head(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top4.iterrows()):
        name = row["Model"]
        y_pred = _get_predictions(predictions, name)


        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="black")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--")
        ax.set_title(f"{name}\nVal R²={row['Val_R2']:.4f}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    _save(fig, "06_top4_predictions.png")

# =============================================================================
# 6. BEST MODEL DETAILED
# =============================================================================

def plot_best_model_detailed(df: pd.DataFrame, predictions: Dict[str, np.ndarray], y_test: np.ndarray):
    best = df.iloc[0]
    name = best["Model"]
    y_pred = _get_predictions(predictions, name)
    errors = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pred vs actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors="black")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0, 0].plot(lims, lims, "r--")
    axes[0, 0].set_title("Predictions vs Actual")
    axes[0, 0].grid(alpha=0.3)

    # Error dist
    axes[0, 1].hist(errors, bins=40, density=True, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set_title("Error Distribution")
    axes[0, 1].grid(alpha=0.3)

    # Residuals vs pred
    axes[1, 0].scatter(y_pred, errors, alpha=0.5, edgecolors="black")
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Residuals vs Predicted")
    axes[1, 0].grid(alpha=0.3)

    # Stats
    axes[1, 1].axis("off")
    stats = f"""
BEST MODEL
Model: {name}

Val R²: {best['Val_R2']:.4f}
Test R²: {best['Test_R2']:.4f}
Test RMSE: {best['Test_RMSE']:.6f}

Overfit gap: {best['overfit_gap']:.4f}
Train time: {best['train_time']:.2f}s
"""
    axes[1, 1].text(
        0.05, 0.5, stats, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()
    _save(fig, "07_best_model_detailed.png")

# =============================================================================
# MAIN ENTRY
# =============================================================================

def run_results_analysis(
    df_results: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
):
    df = _prepare_df(df_results)

    print("=" * 80)
    print("[RESULTS] ANALYSIS")
    print("=" * 80)

    plot_top_models_by_val_r2(df)
    plot_top_models_by_test_rmse(df)
    plot_metrics_table(df)
    plot_multimetric_comparison(df)
    plot_top4_predictions(df, predictions, y_test)
    plot_best_model_detailed(df, predictions, y_test)

    print("[DONE] Figures saved to ../figures")
