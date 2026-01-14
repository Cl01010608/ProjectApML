"""
02_data_preprocessing.py

This module:
- Splits data (train/val/test)
- Selects best features (mutual information)
- Scales features (StandardScaler on train only)
- Prepares residuals for training (y - mean)
- Returns all needed arrays for training

========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


def preprocess_data_improved(df_train, df_test, target_col='output', 
                            test_size=0.2, n_features=20, verbose=True):
    """
    Preprocess data with optimized R² workflow.
    
    CRITICAL ORDER:
    1. Split raw data (train/val/test) FIRST
    2. Select features based on train data ONLY
    3. Scale using train statistics ONLY
    4. Apply to val/test using train scaler
    5. Calculate residuals from train mean ONLY
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Raw training data
    df_test : pd.DataFrame
        Raw test data
    target_col : str
        Name of target column
    test_size : float
        Fraction for validation split (0.2 = 80/20)
    n_features : int
        Number of top features to select (default: 20)
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    X_train : np.array
        Training features (scaled, selected)
    X_val : np.array
        Validation features (scaled, selected)
    X_test : np.array
        Test features (scaled, selected)
    y_train_res : np.array
        Training targets - mean (for residual learning)
    y_val_res : np.array
        Validation targets - train_mean
    y_test_res : np.array
        Test targets - train_mean
    y_train : np.array
        Original training targets
    y_val : np.array
        Original validation targets
    y_test : np.array
        Original test targets
    scaler : StandardScaler
        Fitted scaler (for future use)
    feature_names : list
        Names of selected features
    y_train_mean : float
        Mean of training target (needed for prediction)
    """
    
    if verbose:
        print("\n" + "="*80)
        print("[PREPROCESSING] Data preprocessing with R² optimization")
        print("="*80)
    
    # Extract target and features
    if verbose:
        print(f"\n[1] Extracting target '{target_col}' from data...")
    
    y_full = df_train[target_col].values
    X_full = df_train.drop(columns=[target_col]).values
    X_test_raw = df_test.drop(columns=[target_col]).values if target_col in df_test.columns else df_test.values
    y_test = df_test[target_col].values if target_col in df_test.columns else None
    
    feature_names_all = df_train.drop(columns=[target_col]).columns.tolist()
    
    if verbose:
        print(f"  ✓ Features: {len(feature_names_all)}, Target samples: {len(y_full)}")
    
    # STEP 1: SPLIT FIRST (before any transformation)
    if verbose:
        print(f"\n[2] Splitting data (train/val ratio: {1-test_size:.1%}/{test_size:.1%})...")
    
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42
    )
    
    if verbose:
        print(f"  ✓ Train: {X_train_raw.shape}, Val: {X_val_raw.shape}")
    
    # STEP 2: FEATURE SELECTION on train data ONLY
    if verbose:
        print(f"\n[3] Selecting top {n_features} features (mutual information)...")
    
    selector = SelectKBest(mutual_info_regression, k=min(n_features, X_train_raw.shape[1]))
    X_train_selected = selector.fit_transform(X_train_raw, y_train)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    feature_names = [feature_names_all[i] for i in selected_indices]
    
    if verbose:
        print(f"  ✓ Selected features: {', '.join(feature_names[:5])}...")
    
    # Apply selection to val and test using TRAIN selector
    X_val_selected = selector.transform(X_val_raw)
    X_test_selected = selector.transform(X_test_raw)
    
    # STEP 3: SCALE using train statistics ONLY
    if verbose:
        print(f"\n[4] Scaling features (fit on train only)...")
    
    scaler = StandardScaler()
    scaler.fit(X_train_selected)  # FIT ONLY ON TRAIN
    
    X_train_scaled = scaler.transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)      # Use train scaler
    X_test_scaled = scaler.transform(X_test_selected)    # Use train scaler
    
    if verbose:
        print(f"  ✓ Scaling complete (train mean ≈ 0, std ≈ 1)")
    
    # STEP 4: RESIDUAL LEARNING - subtract train mean ONLY
    if verbose:
        print(f"\n[5] Preparing residuals (for residual learning)...")
    
    y_train_mean = y_train.mean()
    y_train_res = y_train - y_train_mean
    y_val_res = y_val - y_train_mean      # Use TRAIN mean, not val mean
    y_test_res = y_test - y_train_mean if y_test is not None else None
    
    if verbose:
        print(f"  ✓ Train target mean: {y_train_mean:.4f}")
        print(f"  ✓ Train residuals mean: {y_train_res.mean():.6f} (should be ≈0)")
        print(f"  ✓ Train residuals std: {y_train_res.std():.4f}")
    
    # Summary
    if verbose:
        print(f"\n[6] Summary:")
        print(f"  Train shape: {X_train_scaled.shape}")
        print(f"  Val shape: {X_val_scaled.shape}")
        print(f"  Test shape: {X_test_scaled.shape}")
        print(f"  Selected features: {len(feature_names)}/{len(feature_names_all)}")
        print("="*80)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_res, y_val_res, y_test_res,
            y_train, y_val, y_test,
            scaler, feature_names, y_train_mean)