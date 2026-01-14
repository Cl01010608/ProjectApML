"""
05_results_analysis.py

Analysis and visualization of model results.

Predictions, residuals, comparisons, detailed model analysis.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set(style="whitegrid", context="notebook", palette="husl")

RESULTS_DIR = "../results"
FIG_DIR = "../figures"

os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str):
    """Save figure."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {path}")
    return path


# =============================================================================
# 1. MODEL COMPARISON VISUALIZATIONS
# =============================================================================

def plot_model_comparison_bars(df_results: pd.DataFrame, metric: str = "r2"):
    """Bar plot of models sorted by metric."""
    top_n = 15
    df_top = df_results.head(top_n).sort_values(metric)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_top)))

    ax.barh(
        range(len(df_top)),
        df_top[metric].values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["model"].values, fontsize=9)
    ax.set_xlabel(metric.upper(), fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Models by {metric.upper()}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(row[metric] + 0.005, i, f"{row[metric]:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save(fig, f"10_model_comparison_top15_{metric}.png")


def plot_metrics_comparison_table(df_results: pd.DataFrame):
    """Visual table of main metrics."""
    top_n = 8
    df_top = df_results.head(top_n)[["model", "r2", "rmse", "mae", "train_time"]].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df_top.values,
        colLabels=df_top.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.15, 0.15, 0.15, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header
    for i in range(len(df_top.columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Rows
    for i in range(1, len(df_top) + 1):
        for j in range(len(df_top.columns)):
            table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")

    plt.title(
        f"Top {top_n} Models Performance Summary",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "11_metrics_table_top8.png")


def plot_multimetric_comparison(df_results: pd.DataFrame):
    """Comparison of multiple metrics for top models."""
    top_n = 10
    df_top = df_results.head(top_n).copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # R²
    ax = axes[0, 0]
    df_top_r2 = df_top.sort_values("r2")
    ax.barh(range(len(df_top_r2)), df_top_r2["r2"].values, color="steelblue", alpha=0.7)
    ax.set_yticks(range(len(df_top_r2)))
    ax.set_yticklabels(df_top_r2["model"].values, fontsize=8)
    ax.set_xlabel("R²", fontweight="bold")
    ax.set_title("R² Score", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # RMSE
    ax = axes[0, 1]
    df_top_rmse = df_top.sort_values("rmse")
    ax.barh(range(len(df_top_rmse)), df_top_rmse["rmse"].values, color="coral", alpha=0.7)
    ax.set_yticks(range(len(df_top_rmse)))
    ax.set_yticklabels(df_top_rmse["model"].values, fontsize=8)
    ax.set_xlabel("RMSE", fontweight="bold")
    ax.set_title("Root Mean Squared Error", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # MAE
    ax = axes[1, 0]
    df_top_mae = df_top.sort_values("mae")
    ax.barh(range(len(df_top_mae)), df_top_mae["mae"].values, color="lightgreen", alpha=0.7)
    ax.set_yticks(range(len(df_top_mae)))
    ax.set_yticklabels(df_top_mae["model"].values, fontsize=8)
    ax.set_xlabel("MAE", fontweight="bold")
    ax.set_title("Mean Absolute Error", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # Train Time
    ax = axes[1, 1]
    df_top_time = df_top.sort_values("train_time")
    ax.barh(
        range(len(df_top_time)),
        df_top_time["train_time"].values,
        color="purple",
        alpha=0.7,
    )
    ax.set_yticks(range(len(df_top_time)))
    ax.set_yticklabels(df_top_time["model"].values, fontsize=8)
    ax.set_xlabel("Training Time (s)", fontweight="bold")
    ax.set_title("Computational Efficiency", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xscale("log")

    fig.suptitle(
        "Multi-Metric Model Comparison (Top 10)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "12_multimetric_comparison.png")


# =============================================================================
# 2. PREDICTIONS ANALYSIS
# =============================================================================

def plot_predictions_vs_actual(
    y_true,
    y_pred,
    model_name: str,
    metrics_dict: dict | None = None,
):
    """Scatter plot of predictions vs actual."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=20,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Ideal (y=x)",
    )
    ax.set_xlabel("Actual Values", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Values", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}\nPredictions vs Actual", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if metrics_dict:
        metrics_text = (
            f"R² = {metrics_dict.get('r2', 0):.4f}\n"
            f"RMSE = {metrics_dict.get('rmse', 0):.5f}\n"
            f"MAE = {metrics_dict.get('mae', 0):.5f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Error distribution
    ax = axes[1]
    errors = y_true - y_pred
    ax.hist(errors, bins=40, color="coral", alpha=0.7, edgecolor="black")
    ax.axvline(
        errors.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean={errors.mean():.5f}",
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Prediction Error (y_true - y_pred)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Error Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Model: {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    _save(fig, f"13_pred_vs_actual_{model_name[:30]}.png")


# =============================================================================
# 3. RESIDUAL ANALYSIS
# =============================================================================

def plot_residuals_analysis(y_true, y_pred, model_name: str):
    """Detailed residual analysis."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(
        y_pred,
        residuals,
        alpha=0.4,
        s=20,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Values", fontsize=10, fontweight="bold")
    ax.set_ylabel("Residuals", fontsize=10, fontweight="bold")
    ax.set_title("Residuals vs Predicted", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    # Histogram of residuals
    ax = axes[0, 1]
    ax.hist(
        residuals,
        bins=40,
        color="coral",
        alpha=0.7,
        edgecolor="black",
        density=True,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=2)

    try:
        kde = gaussian_kde(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x_range, kde(x_range), "k-", linewidth=2, label="KDE")
    except Exception:
        pass

    ax.set_xlabel("Residuals", fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax.set_title("Residual Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Q-Q Plot (simple)
    ax = axes[1, 0]
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(0, 1, len(residuals))
    ax.scatter(
        theoretical_quantiles, sorted_residuals, alpha=0.4, s=20, color="green"
    )
    ax.plot(
        [0, 1],
        [sorted_residuals.min(), sorted_residuals.max()],
        "r--",
        linewidth=2,
    )
    ax.set_xlabel("Theoretical Quantiles", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontsize=10, fontweight="bold")
    ax.set_title("Q-Q Plot", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    # Statistics box
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = f"""
RESIDUAL STATISTICS
{'-'*28}
Mean: {residuals.mean():.6f}
Std Dev: {residuals.std():.6f}
Min: {residuals.min():.6f}
Max: {residuals.max():.6f}
Skewness: {pd.Series(residuals).skew():.4f}
Kurtosis: {pd.Series(residuals).kurtosis():.4f}

ASSUMPTIONS CHECK
{'-'*28}
Zero Mean: {'✓' if abs(residuals.mean()) < 0.01 else '✗'}
Normality: Check Q-Q plot
Homoscedasticity: Check scatter
"""
    ax.text(
        0.1,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(f"Residual Analysis - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    _save(fig, f"14_residuals_{model_name[:30]}.png")


# =============================================================================
# 4. TOP MODELS DETAILED ANALYSIS
# =============================================================================

def plot_top_models_predictions(
    df_results: pd.DataFrame,
    predictions: Dict,
    y_test: np.ndarray,
):
    """Compare predictions of top 4 models side by side."""
    top_4 = df_results.head(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (_, model_row) in enumerate(top_4.iterrows()):
        model_name = model_row["model"]
        y_pred = predictions[model_name]

        ax = axes[idx]
        ax.scatter(
            y_test,
            y_pred,
            alpha=0.4,
            s=15,
            color="steelblue",
            edgecolors="black",
            linewidth=0.5,
        )

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        ax.set_xlabel("Actual", fontsize=9, fontweight="bold")
        ax.set_ylabel("Predicted", fontsize=9, fontweight="bold")
        ax.set_title(
            f"#{idx+1}: {model_name[:35]}\n"
            f"R²={model_row['r2']:.4f}, RMSE={model_row['rmse']:.5f}",
            fontsize=9,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Top 4 Models - Predictions vs Actual", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "15_top4_predictions_comparison.png")


# =============================================================================
# MAIN ANALYSIS FUNCTION (original ranking)
# =============================================================================

def run_results_analysis(
    df_results: pd.DataFrame,
    predictions: Dict,
    y_test: np.ndarray,
):
    """Execute complete results analysis."""
    print("\n" + "=" * 80)
    print("[RESULTS] COMPREHENSIVE MODEL EVALUATION AND VISUALIZATION")
    print("=" * 80)

    print("\n[1/7] Model Comparison Bars (R²)...")
    plot_model_comparison_bars(df_results, metric="r2")

    print("\n[2/7] Metrics Table (Top 8)...")
    plot_metrics_comparison_table(df_results)

    print("\n[3/7] Multi-Metric Comparison...")
    plot_multimetric_comparison(df_results)

    print("\n[4/7] Top 4 Models Predictions...")
    plot_top_models_predictions(df_results, predictions, y_test)

    top_3 = df_results.head(3)
    for idx, (_, model_row) in enumerate(top_3.iterrows()):
        model_name = model_row["model"]
        y_pred = predictions[model_name]

        print(f"\n[5.{idx+1}/7] Predictions Analysis - {model_name}...")
        plot_predictions_vs_actual(y_test, y_pred, model_name, model_row.to_dict())

        print(f"\n[6.{idx+1}/7] Residuals Analysis - {model_name}...")
        plot_residuals_analysis(y_test, y_pred, model_name)

    print("\n[7/7] Results Analysis Complete!")
    print("\n" + "=" * 80)
    print("[SUMMARY] All visualizations saved to ../figures/")
    print("=" * 80)


# =============================================================================
# 1. MODEL COMPARISON BY RMSE
# =============================================================================

def plot_model_comparison_by_rmse(df_results: pd.DataFrame):
    """Bar plot of models sorted by RMSE (lower is better)."""
    top_n = 15
    df_top = df_results.head(top_n).sort_values("rmse", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df_top)))

    ax.barh(
        range(len(df_top)),
        df_top["rmse"].values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["model"].values, fontsize=9)
    ax.set_xlabel("RMSE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Models by RMSE", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # Baseline reference line (optional)
    if "baseline_rmse" in df_top.columns:
        baseline_rmse = df_top["baseline_rmse"].iloc[0]
        ax.axvline(
            baseline_rmse,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Baseline={baseline_rmse:.5f}",
        )
        ax.legend()

    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(row["rmse"] + 0.002, i, f"{row['rmse']:.5f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save(fig, "20_model_comparison_top15_rmse.png")


def plot_model_comparison_by_mae(df_results: pd.DataFrame):
    """Bar plot of models sorted by MAE (lower is better)."""
    top_n = 15
    df_top = df_results.head(top_n).sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df_top)))

    ax.barh(
        range(len(df_top)),
        df_top["mae"].values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["model"].values, fontsize=9)
    ax.set_xlabel("MAE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Models by MAE", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(row["mae"] + 0.002, i, f"{row['mae']:.5f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save(fig, "21_model_comparison_top15_mae.png")


# =============================================================================
# 2. METRICS TABLE BY PERFORMANCE
# =============================================================================

def plot_metrics_table_by_performance(df_results: pd.DataFrame):
    """Visual table of top models by RMSE."""
    top_n = 8
    df_top = df_results.head(top_n)[["model", "rmse", "mae", "r2", "train_time"]].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("tight")
    ax.axis("off")

    # Format values
    df_display = df_top.copy()
    df_display["rmse"] = df_display["rmse"].apply(lambda x: f"{x:.5f}")
    df_display["mae"] = df_display["mae"].apply(lambda x: f"{x:.5f}")
    df_display["r2"] = df_display["r2"].apply(lambda x: f"{x:.4f}")
    df_display["train_time"] = df_display["train_time"].apply(lambda x: f"{x:.3f}s")

    table = ax.table(
        cellText=df_display.values,
        colLabels=["Model", "RMSE ↓", "MAE ↓", "R²", "Time"],
        cellLoc="center",
        loc="center",
        colWidths=[0.40, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color rows
    for i in range(1, len(df_top) + 1):
        for j in range(5):
            table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")

    plt.title(
        f"Top {top_n} Models by Performance (RMSE)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "22_metrics_table_top8_by_rmse.png")


# =============================================================================
# 3. PERFORMANCE COMPARISON (RMSE vs MAE)
# =============================================================================

def plot_rmse_vs_mae_scatter(df_results: pd.DataFrame):
    """Scatter plot of RMSE vs MAE."""
    top_n = 20
    df_top = df_results.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df_top["rmse"],
        df_top["mae"],
        s=100,
        alpha=0.6,
        c=range(len(df_top)),
        cmap="viridis",
        edgecolors="black",
        linewidth=1,
    )

    # Add model ranks for top 5
    for idx, (_, row) in enumerate(df_top.head(5).iterrows()):
        ax.annotate(
            f"#{idx+1}",
            (row["rmse"], row["mae"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("RMSE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Model Performance: RMSE vs MAE", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Rank (Best to Worst)", fontsize=10)

    plt.tight_layout()
    plt.show()
    _save(fig, "23_rmse_vs_mae_scatter.png")


# =============================================================================
# 4. MULTI-METRIC COMPARISON (by RMSE)
# =============================================================================

def plot_multimetric_performance(df_results: pd.DataFrame):
    """Compare top models across multiple metrics."""
    top_n = 10
    df_top = df_results.head(top_n).copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE
    ax = axes[0, 0]
    df_sorted = df_top.sort_values("rmse", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    ax.barh(
        range(len(df_sorted)),
        df_sorted["rmse"].values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["model"].values, fontsize=8)
    ax.set_xlabel("RMSE (Lower Better)", fontweight="bold")
    ax.set_title("Root Mean Squared Error", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # MAE
    ax = axes[0, 1]
    df_sorted = df_top.sort_values("mae", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    ax.barh(
        range(len(df_sorted)),
        df_sorted["mae"].values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["model"].values, fontsize=8)
    ax.set_xlabel("MAE (Lower Better)", fontweight="bold")
    ax.set_title("Mean Absolute Error", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # Improvement over Baseline
    ax = axes[1, 0]
    if "baseline_rmse" in df_top.columns:
        df_top["improvement"] = (
            (df_top["baseline_rmse"] - df_top["rmse"])
            / df_top["baseline_rmse"]
            * 100
        )
        df_sorted = df_top.sort_values("improvement", ascending=False)
        colors = [
            "green" if x > 0 else "red"
            for x in df_sorted["improvement"].values
        ]
        ax.barh(
            range(len(df_sorted)),
            df_sorted["improvement"].values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["model"].values, fontsize=8)
        ax.set_xlabel("% Improvement vs Baseline", fontweight="bold")
        ax.set_title("Improvement Over Baseline", fontweight="bold")
        ax.axvline(0, color="black", linestyle="-", linewidth=1)
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.text(
            0.5,
            0.5,
            "Baseline data not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    # Training Time
    ax = axes[1, 1]
    df_sorted = df_top.sort_values("train_time", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    ax.barh(
        range(len(df_sorted)),
        df_sorted["train_time"].values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["model"].values, fontsize=8)
    ax.set_xlabel("Training Time (s)", fontweight="bold")
    ax.set_title("Computational Efficiency", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xscale("log")

    fig.suptitle(
        "Multi-Metric Performance Comparison (Top 10 by RMSE)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "24_multimetric_performance.png")


# =============================================================================
# 5. TOP MODELS PREDICTIONS (by RMSE)
# =============================================================================

def plot_top_models_predictions_by_rmse(
    df_results: pd.DataFrame,
    predictions: Dict,
    y_test: np.ndarray,
):
    """Compare predictions of top 4 models by RMSE."""
    top_4 = df_results.head(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (_, model_row) in enumerate(top_4.iterrows()):
        model_name = model_row["model"]
        y_pred = predictions[model_name]

        ax = axes[idx]
        ax.scatter(
            y_test,
            y_pred,
            alpha=0.4,
            s=15,
            color="steelblue",
            edgecolors="black",
            linewidth=0.5,
        )

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        title_text = (
            f"#{idx+1}: {model_name[:35]}\n"
            f"RMSE={model_row['rmse']:.5f}, MAE={model_row['mae']:.5f}"
        )
        ax.set_xlabel("Actual", fontsize=9, fontweight="bold")
        ax.set_ylabel("Predicted", fontsize=9, fontweight="bold")
        ax.set_title(title_text, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Top 4 Models by RMSE - Predictions vs Actual",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "25_top4_predictions_by_rmse.png")


# =============================================================================
# 6. DETAILED ANALYSIS OF BEST MODEL (by RMSE)
# =============================================================================

def plot_best_model_detailed_analysis(
    df_results: pd.DataFrame,
    predictions: Dict,
    y_test: np.ndarray,
):
    """Detailed analysis of the best model by RMSE."""
    best_row = df_results.iloc[0]
    best_name = best_row["model"]
    y_pred = predictions[best_name]
    errors = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        s=30,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax.set_xlabel("Actual Values", fontweight="bold")
    ax.set_ylabel("Predicted Values", fontweight="bold")
    ax.set_title("Predictions vs Actual", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    metrics_text = (
        f"RMSE = {best_row['rmse']:.5f}\n"
        f"MAE = {best_row['mae']:.5f}\n"
        f"R² = {best_row['r2']:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 2. Error Distribution
    ax = axes[0, 1]
    ax.hist(
        errors,
        bins=40,
        color="coral",
        alpha=0.7,
        edgecolor="black",
        density=True,
    )
    ax.axvline(
        errors.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean={errors.mean():.5f}",
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Prediction Error", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 3. Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(
        y_pred,
        errors,
        alpha=0.5,
        s=30,
        color="green",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Values", fontweight="bold")
    ax.set_ylabel("Residuals (Actual - Predicted)", fontweight="bold")
    ax.set_title("Residual Plot", fontweight="bold")
    ax.grid(alpha=0.3)

    # 4. Statistics
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = f"""
BEST MODEL STATISTICS
{'='*35}
Model: {best_name}

PERFORMANCE METRICS
{'─'*35}
RMSE: {best_row['rmse']:.6f}
MAE: {best_row['mae']:.6f}
R²: {best_row['r2']:.6f}

ERROR STATISTICS
{'─'*35}
Mean Error: {errors.mean():.6f}
Std Error: {errors.std():.6f}
Min Error: {errors.min():.6f}
Max Error: {errors.max():.6f}

TRAINING INFO
{'─'*35}
Train Time: {best_row['train_time']:.3f} seconds
"""
    ax.text(
        0.1,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    fig.suptitle(
        "Detailed Analysis - Best Model by RMSE",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    _save(fig, "26_best_model_detailed_analysis.png")


# =============================================================================
# MAIN ANALYSIS FUNCTION (ranked by RMSE)
# =============================================================================

def run_results_analysis_by_performance(
    df_results: pd.DataFrame,
    predictions: Dict,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Execute complete results analysis ranked by RMSE."""
    # Sort by RMSE (lower is better)
    df_results = df_results.sort_values("rmse", ascending=True).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("[RESULTS] PERFORMANCE-BASED MODEL EVALUATION (Ranked by RMSE)")
    print("=" * 80)

    print("\n[1/8] Model Comparison by RMSE...")
    plot_model_comparison_by_rmse(df_results)

    print("\n[2/8] Model Comparison by MAE...")
    plot_model_comparison_by_mae(df_results)

    print("\n[3/8] Metrics Table (Top 8 by RMSE)...")
    plot_metrics_table_by_performance(df_results)

    print("\n[4/8] RMSE vs MAE Scatter...")
    plot_rmse_vs_mae_scatter(df_results)

    print("\n[5/8] Multi-Metric Performance...")
    plot_multimetric_performance(df_results)

    print("\n[6/8] Top 4 Models Predictions (by RMSE)...")
    plot_top_models_predictions_by_rmse(df_results, predictions, y_test)

    print("\n[7/8] Best Model Detailed Analysis...")
    plot_best_model_detailed_analysis(df_results, predictions, y_test)

    print("\n[8/8] Performance Analysis Complete!")
    print("\n" + "=" * 80)
    print("[SUMMARY] All visualizations saved to ../figures/")
    print(f"[BEST MODEL] {df_results.iloc[0]['model']}")
    print(f" RMSE: {df_results.iloc[0]['rmse']:.6f}")
    print(f" MAE: {df_results.iloc[0]['mae']:.6f}")
    print(f" R²: {df_results.iloc[0]['r2']:.6f}")
    print("=" * 80)

    return df_results
