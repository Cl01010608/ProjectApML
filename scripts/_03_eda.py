"""
03_eda.py

Exploratory Data Analysis with 9 comprehensive visualizations.
Correlations, distributions, outliers, skewness, kurtosis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set(style="whitegrid", context="notebook", palette="husl")

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

def _save(fig, name: str):
    """Save figure with correct path."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    return path

# =============================================================================
# 1. CORRELATION MATRIX (Heatmap)
# =============================================================================

def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap of correlation between ALL variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title("Correlation Matrix - All Variables", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    _save(fig, "01_correlation_matrix.png")
    
    return corr

# =============================================================================
# 2. CORRELATION WITH TARGET (Bar plot)
# =============================================================================

def plot_correlation_with_target(df: pd.DataFrame, target_col: str = "output"):
    """Barplot of correlation between each variable and target."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = numeric_df.corr()[target_col].drop(target_col).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if x < 0 else 'green' for x in corr_with_target.values]
    bars = ax.barh(range(len(corr_with_target)), corr_with_target.values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(corr_with_target)))
    ax.set_yticklabels(corr_with_target.index, fontsize=9)
    ax.set_xlabel("Correlation with Target", fontsize=11, fontweight='bold')
    ax.set_title(f"Feature Correlation with {target_col}", fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    _save(fig, "02_correlation_with_target.png")

# =============================================================================
# 3. HISTOGRAMS (Simple)
# =============================================================================

def plot_histograms_simple(df: pd.DataFrame):
    """Simple histograms of ALL numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.hist(df[col], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{col}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3, axis='y')
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Distribution of All Variables - Histograms", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "03_histograms.png")

# =============================================================================
# 4. HISTOGRAMS WITH KDE
# =============================================================================

def plot_histograms_kde(df: pd.DataFrame):
    """Histograms with KDE overlay."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        ax.hist(df[col], bins=40, color='steelblue', alpha=0.6, density=True, edgecolor='black', label='Histogram')
        
        try:
            kde = gaussian_kde(df[col].dropna())
            x_range = np.linspace(df[col].min(), df[col].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_title(f"{col}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Distribution of All Variables - Histograms + KDE", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "04_histograms_kde.png")

# =============================================================================
# 5. BOXPLOTS
# =============================================================================

def plot_boxplots(df: pd.DataFrame):
    """Boxplots of all variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_title(f"{col}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, axis='y')
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Distribution of All Variables - Boxplots", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "05_boxplots.png")

# =============================================================================
# 6. BOXPLOTS WITH STATISTICS
# =============================================================================

def plot_boxplots_with_stats(df: pd.DataFrame):
    """Boxplots with statistics annotated."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        stats_text = f"μ={data.mean():.3f}\nσ={data.std():.3f}\nmed={median:.3f}\nIQR={iqr:.3f}"
        ax.text(1.15, data.max() * 0.8, stats_text, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(f"{col}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, axis='y')
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Distribution of All Variables - Boxplots with Statistics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "06_boxplots_with_stats.png")

# =============================================================================
# 7. TARGET DISTRIBUTION (Multiple views)
# =============================================================================

def plot_target_distribution(df: pd.DataFrame, target_col: str = "output"):
    """Complete analysis of target distribution."""
    target = df[target_col]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram
    ax = axes[0, 0]
    ax.hist(target, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel(target_col, fontweight='bold')
    ax.set_ylabel("Frequency")
    ax.set_title("Target Distribution - Histogram", fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # KDE
    ax = axes[0, 1]
    try:
        kde = gaussian_kde(target)
        x_range = np.linspace(target.min(), target.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        ax.fill_between(x_range, kde(x_range), alpha=0.3, color='red')
    except:
        pass
    ax.set_xlabel(target_col, fontweight='bold')
    ax.set_ylabel("Density")
    ax.set_title("Target Distribution - KDE", fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Boxplot
    ax = axes[1, 0]
    bp = ax.boxplot(target, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax.set_ylabel(target_col, fontweight='bold')
    ax.set_title("Target Distribution - Boxplot", fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    TARGET STATISTICS
    {'─'*30}
    Count: {len(target)}
    Mean: {target.mean():.6f}
    Std Dev: {target.std():.6f}
    Min: {target.min():.6f}
    Q1 (25%): {target.quantile(0.25):.6f}
    Median: {target.median():.6f}
    Q3 (75%): {target.quantile(0.75):.6f}
    Max: {target.max():.6f}
    IQR: {target.quantile(0.75) - target.quantile(0.25):.6f}
    Skewness: {pd.Series(target).skew():.6f}
    Kurtosis: {pd.Series(target).kurtosis():.6f}
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f"Comprehensive Analysis of {target_col}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "07_target_distribution.png")


# =============================================================================
# 8. SKEWNESS & KURTOSIS
# =============================================================================

def plot_skewness_kurtosis(df: pd.DataFrame):
    """Analysis of skewness and kurtosis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    skewness = [pd.Series(df[col]).skew() for col in numeric_cols]
    kurtosis = [pd.Series(df[col]).kurtosis() for col in numeric_cols]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Skewness
    ax = axes[0]
    colors_skew = ['red' if x < -0.5 else 'orange' if x < 0.5 else 'green' for x in skewness]
    bars = ax.barh(range(len(numeric_cols)), skewness, color=colors_skew, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=9)
    ax.set_xlabel("Skewness", fontsize=11, fontweight='bold')
    ax.set_title("Skewness by Variable", fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(-0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Moderately left skewed')
    ax.axvline(0.5, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Moderately right skewed')
    ax.grid(alpha=0.3, axis='x')
    ax.legend(fontsize=8)
    
    # Kurtosis
    ax = axes[1]
    colors_kurt = ['red' if x < -0.5 else 'orange' if x < 0.5 else 'green' for x in kurtosis]
    bars = ax.barh(range(len(numeric_cols)), kurtosis, color=colors_kurt, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=9)
    ax.set_xlabel("Excess Kurtosis", fontsize=11, fontweight='bold')
    ax.set_title("Kurtosis by Variable", fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    fig.suptitle("Skewness and Kurtosis Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _save(fig, "08_skewness_kurtosis.png")

# =============================================================================
# MAIN EDA FUNCTION
# =============================================================================

def run_eda(df: pd.DataFrame, target_col: str = "output"):
    """Execute complete EDA with all visualizations."""
    
    print("\n" + "="*80)
    print("[EDA] EXPLORATORY DATA ANALYSIS")
    print("="*80)
    print(f"\nDataset shape: {df.shape}")
    print(f"Target column: {target_col}")
    
    print("\n[1/8] Plotting correlation matrix...")
    corr_matrix = plot_correlation_matrix(df)
    
    print("[2/8] Plotting correlation with target...")
    plot_correlation_with_target(df, target_col=target_col)
    
    print("[3/8] Plotting simple histograms...")
    plot_histograms_simple(df)
    
    print("[4/8] Plotting histograms with KDE...")
    plot_histograms_kde(df)
    
    print("[5/8] Plotting boxplots...")
    plot_boxplots(df)
    
    print("[6/8] Plotting boxplots with statistics...")
    plot_boxplots_with_stats(df)
    
    print("[7/8] Plotting target distribution...")
    plot_target_distribution(df, target_col=target_col)
    
    print("[8/8] Plotting skewness and kurtosis...")
    plot_skewness_kurtosis(df)
    
    print("\n" + "="*80)
    print(" EDA COMPLETE - 8 visualizations generated")
    print("="*80)
