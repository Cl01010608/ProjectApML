"""
04_model_training.py

This module:
- Trains 25+ regression models
- Applies residual learning (critical for R²)
- Evaluates on train/val/test sets
- Reports R² scores and metrics
- Returns results dataframe

========================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, VotingRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Global variable to store y_train_mean (set before training)
_Y_TRAIN_MEAN = 0


def set_y_train_mean(mean_value):
    """
    Set the global y_train_mean (must be called before training).
    
    Parameters:
    -----------
    mean_value : float
        Mean of training target (from preprocessing)
    """
    global _Y_TRAIN_MEAN
    _Y_TRAIN_MEAN = mean_value


def apply_residual_learning(y_true, y_pred_residual, y_mean):
    """
    Convert residual predictions back to original scale.
    
    y_pred_residual = model output (trained on residuals)
    y_true_residual = y_true - y_mean
    
    Convert back:
    y_pred_original = y_pred_residual + y_mean
    
    Parameters:
    -----------
    y_true : np.array
        Original target values
    y_pred_residual : np.array
        Predictions from residual-trained model
    y_mean : float
        Mean used for residual calculation
        
    Returns:
    --------
    y_pred_original : np.array
        Predictions in original scale
    """
    return y_pred_residual + y_mean


def evaluate_model(model, X_train, y_train_res, X_val, y_val_res, 
                  X_test, y_test_res, y_train, y_val, y_test, y_mean):
    """
    Evaluate model on train/val/test with residual learning.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train, X_val, X_test : np.array
        Feature matrices
    y_train_res, y_val_res, y_test_res : np.array
        Residual targets (y - y_mean)
    y_train, y_val, y_test : np.array
        Original targets
    y_mean : float
        Training mean
        
    Returns:
    --------
    metrics : dict
        Dictionary with R² and RMSE for each set
    """
    
    # Predict on residuals
    y_train_pred_res = model.predict(X_train)
    y_val_pred_res = model.predict(X_val)
    y_test_pred_res = model.predict(X_test)
    
    # Convert back to original scale 
    y_train_pred = apply_residual_learning(y_train, y_train_pred_res, y_mean)
    y_val_pred = apply_residual_learning(y_val, y_val_pred_res, y_mean)
    y_test_pred = apply_residual_learning(y_test, y_test_pred_res, y_mean)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }
    
    return metrics, y_test_pred


def train_all_models_improved(X_train, y_train_res, X_val, y_val_res,
                             X_test, y_test_res, y_train, y_val, y_test, 
                             y_mean, verbose=True):
    """
    Train 25+ models with residual learning for positive R².
    
    CRITICAL: Models are trained on y_residual (y - y_mean)
    but evaluated on original y values.
    
    Parameters:
    -----------
    X_train, X_val, X_test : np.array
        Feature matrices (already scaled, feature-selected)
    y_train_res, y_val_res, y_test_res : np.array
        Residual targets (y - y_mean)
    y_train, y_val, y_test : np.array
        Original targets (for evaluation)
    y_mean : float
        Training mean (for residual learning)
    verbose : bool
        Print progress
        
    Returns:
    --------
    results : pd.DataFrame
        Results for all models sorted by val_r2
    """
    
    if verbose:
        print("\n" + "="*100)
        print("[MODEL TRAINING] Training 25+ models with residual learning")
        print("="*100)
    
    set_y_train_mean(y_mean)
    
    results_list = []
    model_count = 0
    
    # ==================== LINEAR MODELS ====================
    if verbose:
        print("\n[LINEAR MODELS]")
    
    models_linear = {
        'LinearRegression': LinearRegression(),
        'Ridge_alpha001': Ridge(alpha=0.01),
        'Ridge_alpha01': Ridge(alpha=0.1),
        'Ridge_alpha1': Ridge(alpha=1.0),
        'Ridge_alpha10': Ridge(alpha=10.0),
        'Lasso_alpha001': Lasso(alpha=0.001, max_iter=5000),
        'Lasso_alpha01': Lasso(alpha=0.01, max_iter=5000),
        'Lasso_alpha1': Lasso(alpha=1.0, max_iter=5000),
        'ElasticNet_01': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'ElasticNet_1': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
    }
    
    for name, model in models_linear.items():
        model.fit(X_train, y_train_res)
        metrics, _ = evaluate_model(model, X_train, y_train_res, X_val, y_val_res,
                                   X_test, y_test_res, y_train, y_val, y_test, y_mean)
        
        results_list.append({
            'Model': name,
            'Type': 'Linear',
            'Val_R2': metrics['val_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_R2': metrics['train_r2'],
            'Val_RMSE': metrics['val_rmse'],
            'Test_RMSE': metrics['test_rmse'],
        })
        
        model_count += 1
        if verbose:
            print(f"  [{model_count:2d}] {name:30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
    
    # ==================== TREE MODELS ====================
    if verbose:
        print("\n[TREE MODELS]")
    
    models_tree = {
        'DecisionTree_d5': DecisionTreeRegressor(max_depth=5, random_state=42),
        'DecisionTree_d10': DecisionTreeRegressor(max_depth=10, random_state=42),
        'DecisionTree_d15': DecisionTreeRegressor(max_depth=15, random_state=42),
        'DecisionTree_d20': DecisionTreeRegressor(max_depth=20, random_state=42),
    }
    
    for name, model in models_tree.items():
        model.fit(X_train, y_train_res)
        metrics, _ = evaluate_model(model, X_train, y_train_res, X_val, y_val_res,
                                   X_test, y_test_res, y_train, y_val, y_test, y_mean)
        
        results_list.append({
            'Model': name,
            'Type': 'Tree',
            'Val_R2': metrics['val_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_R2': metrics['train_r2'],
            'Val_RMSE': metrics['val_rmse'],
            'Test_RMSE': metrics['test_rmse'],
        })
        
        model_count += 1
        if verbose:
            print(f"  [{model_count:2d}] {name:30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
    
    # ==================== ENSEMBLE MODELS ====================
    if verbose:
        print("\n[ENSEMBLE MODELS - Random Forest]")
    
    models_rf = {
        'RandomForest_n100': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'RandomForest_n300': RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'RandomForest_n500': RandomForestRegressor(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    }
    
    for name, model in models_rf.items():
        model.fit(X_train, y_train_res)
        metrics, _ = evaluate_model(model, X_train, y_train_res, X_val, y_val_res,
                                   X_test, y_test_res, y_train, y_val, y_test, y_mean)
        
        results_list.append({
            'Model': name,
            'Type': 'RandomForest',
            'Val_R2': metrics['val_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_R2': metrics['train_r2'],
            'Val_RMSE': metrics['val_rmse'],
            'Test_RMSE': metrics['test_rmse'],
        })
        
        model_count += 1
        if verbose:
            print(f"  [{model_count:2d}] {name:30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
    
    # ==================== GRADIENT BOOSTING ====================
    if verbose:
        print("\n[ENSEMBLE MODELS - Gradient Boosting]")
    
    models_gb = {
        'GradientBoosting_n100': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'GradientBoosting_n300': GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42),
        'GradientBoosting_n500': GradientBoostingRegressor(n_estimators=500, max_depth=9, learning_rate=0.02, random_state=42),
    }
    
    for name, model in models_gb.items():
        model.fit(X_train, y_train_res)
        metrics, _ = evaluate_model(model, X_train, y_train_res, X_val, y_val_res,
                                   X_test, y_test_res, y_train, y_val, y_test, y_mean)
        
        results_list.append({
            'Model': name,
            'Type': 'GradientBoosting',
            'Val_R2': metrics['val_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_R2': metrics['train_r2'],
            'Val_RMSE': metrics['val_rmse'],
            'Test_RMSE': metrics['test_rmse'],
        })
        
        model_count += 1
        if verbose:
            print(f"  [{model_count:2d}] {name:30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
    
    # ==================== OTHER MODELS ====================
    if verbose:
        print("\n[OTHER MODELS]")
    
    models_other = {
        'SVR_linear': SVR(kernel='linear', C=100),
        'SVR_rbf': SVR(kernel='rbf', C=100, gamma=0.01),
        'KNN_k5': KNeighborsRegressor(n_neighbors=5),
        'KNN_k10': KNeighborsRegressor(n_neighbors=10),
        'KNN_k15': KNeighborsRegressor(n_neighbors=15),
        'MLP_h100': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'MLP_h100_100': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42),
        'AdaBoost_n300': AdaBoostRegressor(n_estimators=300, random_state=42),
    }
    
    for name, model in models_other.items():
        try:
            model.fit(X_train, y_train_res)
            metrics, _ = evaluate_model(model, X_train, y_train_res, X_val, y_val_res,
                                       X_test, y_test_res, y_train, y_val, y_test, y_mean)
            
            results_list.append({
                'Model': name,
                'Type': 'Other',
                'Val_R2': metrics['val_r2'],
                'Test_R2': metrics['test_r2'],
                'Train_R2': metrics['train_r2'],
                'Val_RMSE': metrics['val_rmse'],
                'Test_RMSE': metrics['test_rmse'],
            })
            
            model_count += 1
            if verbose:
                print(f"  [{model_count:2d}] {name:30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
        except Exception as e:
            if verbose:
                print(f"  [--] {name:30s} | FAILED: {str(e)[:40]}")
    
    # ==================== VOTING & STACKING ====================
    if verbose:
        print("\n[ENSEMBLE MODELS - Meta-Ensembles]")
    
    # Select top 3 base models for voting
    top_models_indices = np.argsort([r['Val_R2'] for r in results_list])[-3:]
    base_models = []
    base_names = []
    
    for idx in top_models_indices:
        if results_list[idx]['Type'] in ['RandomForest', 'GradientBoosting', 'Boosting']:
            name = results_list[idx]['Model']
            if 'RandomForest_n300' in name:
                base_models.append(('rf', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)))
            elif 'GradientBoosting_n300' in name:
                base_models.append(('gb', GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42)))
            elif 'XGBoost' in name:
                base_models.append(('xgb', XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)))
    
    if len(base_models) >= 2:
        voting = VotingRegressor(estimators=base_models[:2])
        voting.fit(X_train, y_train_res)
        metrics, _ = evaluate_model(voting, X_train, y_train_res, X_val, y_val_res,
                                   X_test, y_test_res, y_train, y_val, y_test, y_mean)
        
        results_list.append({
            'Model': 'Voting_2models',
            'Type': 'Meta',
            'Val_R2': metrics['val_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_R2': metrics['train_r2'],
            'Val_RMSE': metrics['val_rmse'],
            'Test_RMSE': metrics['test_rmse'],
        })
        
        model_count += 1
        if verbose:
            print(f"  [{model_count:2d}] {'Voting_2models':30s} | Val R²: {metrics['val_r2']:7.4f} | Test R²: {metrics['test_r2']:7.4f}")
    
    # Create results dataframe
    results = pd.DataFrame(results_list)
    results = results.sort_values('Val_R2', ascending=False).reset_index(drop=True)
    
    # Print summary
    if verbose:
        print("\n" + "="*100)
        print("[SUMMARY] Top 10 models:")
        print("="*100)
        print(results.head(10).to_string(index=False))
        print("="*100)
        print(f"Total models trained: {model_count}")
        print(f"Best Val R²: {results['Val_R2'].max():.4f}")
        print(f"Positive R² models: {(results['Test_R2'] > 0).sum()}/{len(results)}")
        print("="*100)
    
    return results