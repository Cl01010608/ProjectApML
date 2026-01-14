"""
========================================================================
01_data_loading.py

STEP 1: LOAD AND VALIDATE DATA

This module:
- Loads raw CSV files
- Validates data structure
- Checks for missing values
- Reports data statistics
- Prepares for preprocessing

========================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path, test_path, target_col='output', verbose=True):
    """
    Load and validate training and test data.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
    target_col : str
        Name of target column (default: 'output')
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Test data
    """
    
    if verbose:
        print("\n" + "="*80)
        print("[DATA LOADING] Loading raw data")
        print("="*80)
    
    # Load files
    if verbose:
        print(f"\n[1] Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    
    if verbose:
        print(f"[2] Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)
    
    # Validate
    if verbose:
        print(f"\n[3] Validating data structure...")
    
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    if target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data")
    
    if verbose:
        print(f"   Target column '{target_col}' found in both datasets")
    
    # Check missing values
    train_missing = df_train.isnull().sum().sum()
    test_missing = df_test.isnull().sum().sum()
    
    if verbose:
        print(f"\n[4] Checking missing values...")
        print(f"  Train: {train_missing} missing values")
        print(f"  Test: {test_missing} missing values")
    
    if train_missing > 0 or test_missing > 0:
        if verbose:
            print("    WARNING: Found missing values!")
            print("    Consider using imputation or removing rows")
    
    # Report statistics
    if verbose:
        print(f"\n[5] Data statistics:")
        print(f"  Train shape: {df_train.shape}")
        print(f"  Test shape: {df_test.shape}")
        print(f"\n  Target statistics (train):")
        print(f"    Mean: {df_train[target_col].mean():.4f}")
        print(f"    Std:  {df_train[target_col].std():.4f}")
        print(f"    Min:  {df_train[target_col].min():.4f}")
        print(f"    Max:  {df_train[target_col].max():.4f}")
        print(f"\n  Feature columns: {df_train.shape[1] - 1}")
        print(f"    {', '.join(df_train.drop(columns=[target_col]).columns[:5].tolist())}...")
        print("="*80)
    
    return df_train, df_test