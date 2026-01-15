

# Steel Production Data Analysis

## Machine Learning for Quality Prediction

**Project:** P1 – Steel Production Quality Prediction
**Author:** Claudia Alexandra Páez
**Date:** January 2026

---

## Project Overview

This project implements a complete machine learning pipeline to analyze and predict steel production quality using industrial process data.
The workflow covers all standard stages of a data science project: data ingestion, preprocessing, exploratory data analysis (EDA), model training, and results evaluation.

**Objective:**
Build, evaluate, and compare multiple regression models to predict the steel quality factor (`output`) from production parameters.

---

## Table of Contents

1. Installation & Environment Setup
2. Project Structure
3. Dataset Description
4. Pipeline Workflow
5. Modules Overview
6. Results Summary
7. Key Findings
8. Usage Guide
9. Requirements & Dependencies
10. Troubleshooting
11. Future Improvements
12. References

---

## 1. Installation & Environment Setup

### Prerequisites

* Python 3.8 or higher
* Jupyter Notebook or JupyterLab
* Conda or pip

### Environment Creation (Conda)

```bash
conda create -n steel-analysis python=3.9
conda activate steel-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas>=1.5.0 numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0 \
            scikit-learn>=1.0.0 xgboost>=1.5.0 jupyter>=1.0.0
```

---

## 2. Project Structure

```
steel_production_analysis/
├── data/
│   ├── steel_production_train.csv
│   └── steel_production_test.csv
├── scripts/
│   ├── 01_data_loading.py
│   ├── 02_data_preprocessing.py
│   ├── 03_eda.py
│   ├── 04_model_training.py
│   └── 05_results_analysis.py
├── figures/
├── results/
│   ├── performance_metrics.csv
│   └── model_predictions/
├── P1-report.pdf
└── README.md
```

---

## 3. Dataset Description

### Source

Industrial steel production dataset obtained from a manufacturing process
(Source: internal/cloud server)

### Dataset Characteristics

| Metric           | Value                    |
| ---------------- | ------------------------ |
| Training samples | 7,000~                   |
| Test samples     | 3,000~                   |
| Total features   | 21                       |
| Target variable  | `output` (steel quality) |
| Feature type     | Numerical                |
| Missing values   | None                     |

### Target Variable

* **Name:** `output`
* **Type:** Continuous (regression)
* **Range:** Normalized quality metric
* **Distribution:** Analyzed in EDA (see figures)

---

## 4. Pipeline Workflow

```
Data Loading
  → Preprocessing
      - Train/Validation split (80/20)
      - Feature selection (MI, k=21)
      - Scaling (StandardScaler)
      - Residual learning
  → Exploratory Data Analysis
  → Model Training (25+ models)
  → Results Analysis & Visualization
```

---

## 5. Modules Overview

### 01_data_loading.py

**Purpose:** Load and validate raw data

* Load CSV files
* Validate structure and column consistency
* Check missing values
* Report basic statistics

**Output:**
Training and test DataFrames (21 features each)

---

### 02_data_preprocessing.py

**Purpose:** Prepare data for optimal regression performance

Key steps:

* Train/validation split (80/20)
* Feature selection using mutual information
* StandardScaler fitted on training data only
* Residual learning:

  ```
  y_residual = y - mean(y_train)
  ```

Residual learning is essential to achieve positive and stable R² scores.

**Main function:** `preprocess_data_improved()`

---

### 03_eda.py

**Purpose:** Exploratory data analysis

Generates 8 high-quality visualizations:

* Feature correlation matrix
* Feature–target correlations
* Histograms (with and without KDE)
* Boxplots with statistical annotations
* Target distribution analysis
* Skewness and kurtosis analysis

All outputs are saved at 300 DPI in `figures/`.

---

### 04_model_training.py

**Purpose:** Train and compare 25+ regression models

**Model categories:**

* Linear models (Linear, Ridge, Lasso, ElasticNet)
* Tree-based models (Decision Tree, Random Forest)
* Ensemble models (Gradient Boosting, XGBoost, AdaBoost)
* Other models (SVR, KNN, MLP)

**Key features:**

* Residual learning applied consistently
* Evaluation on train, validation, and test sets
* Training and inference time tracking
* Automatic ranking by validation R²

---

### 05_results_analysis.py

**Purpose:** Analyze and visualize model performance

Includes:

* Model ranking plots (R², RMSE, overfitting gap)
* Prediction vs actual plots
* Residual diagnostics
* Multi-metric comparison figures

Outputs are saved as publication-quality PNG files.

---

## 6. Results Summary

### Top Models (Validation Performance)

**Purpose:** Analyze and visualize model performance for steel production quality prediction

Includes:

* Model ranking plots (R², RMSE, overfitting gap)
* Prediction vs actual plots
* Residual diagnostics
* Multi-metric comparison figures

Outputs are saved as publication-quality PNG files.

---

## 6. Results Summary

### Top Models (Validation Performance)

| Rank | Model                 | Type             | Val R² | Test R²  | Train R² | Val RMSE | Test RMSE | Train Time (s) |
| ---- | -------------------- | ---------------- | ------ | -------- | -------- | -------- | --------- | -------------- |
| 1    | GradientBoosting_n500 | GradientBoosting | 0.434  | -2.667   | 0.938    | 0.0614   | 0.1811    | 41.42          |
| 2    | RandomForest_n500     | RandomForest     | 0.428  | -2.176   | 0.930    | 0.0617   | 0.1686    | 23.85          |
| 3    | GradientBoosting_n300 | GradientBoosting | 0.427  | -3.174   | 0.879    | 0.0617   | 0.1932    | 18.45          |
| 4    | RandomForest_n300     | RandomForest     | 0.425  | -2.171   | 0.924    | 0.0618   | 0.1684    | 12.90          |
| 5    | RandomForest_n100     | RandomForest     | 0.419  | -2.155   | 0.876    | 0.0621   | 0.1680    | 2.82           |
| 6    | GradientBoosting_n100 | GradientBoosting | 0.373  | -1.594   | 0.660    | 0.0646   | 0.1523    | 10.12          |
| 7    | KNN_k15               | Other            | 0.167  | -1.209   | 0.305    | 0.0744   | 0.1406    | 0.85           |
| 8    | KNN_k10               | Other            | 0.166  | -1.276   | 0.356    | 0.0745   | 0.1427    | 0.78           |
| 9    | KNN_k5                | Other            | 0.132  | -1.522   | 0.464    | 0.0760   | 0.1502    | 0.70           |
| 10   | DecisionTree_d5       | Tree             | 0.128  | -2.477   | 0.260    | 0.0761   | 0.1764    | 0.55           |

---

### Model Training Summary

Total models trained: 28  
Best Validation R²: 0.434  
Positive Validation R² models: 10/28  

---

**Notes:**

* Linear models (LinearRegression, Ridge, Lasso, ElasticNet) show low R² on validation and negative R² on test.
* Tree-based and ensemble models outperform linear and KNN/MLP models.
* Residual learning improves model robustness slightly for ensemble models.

Overall, ensemble-based models combined with residual learning provided the most reliable
results for steel quality prediction. However, the remaining gap between validation and test
performance suggests that further improvements may require better feature engineering and
stronger regularization rather than more complex models
---

## 8. Usage Guide

### Run Complete Pipeline (Python)

```python
df_train, df_test = load_data(...)
X_train, X_val, X_test, y_train_res, ... = preprocess_data_improved(...)
run_eda(df_train)
results = train_all_models_improved(...)
run_results_analysis(results, predictions, y_test)
```

### Customize

* Change number of selected features
* Modify train/validation split
* Enable/disable model groups in `04_model_training.py`

---

## 9. Requirements & Dependencies

* Python ≥ 3.8
* pandas, numpy, scikit-learn
* matplotlib, seaborn
* xgboost
* jupyter

---

## 10. Troubleshooting

* **Missing xgboost:** `pip install xgboost`
* **Out of memory:** reduce `n_estimators`
* **Negative test R²:** expected with overfitting; consider cross-validation

---

## 11. Future Improvements

* Hyperparameter optimization (Bayesian search)
* Time-aware cross-validation
* Advanced feature engineering
* Model stacking and blending
* Production deployment (FastAPI, Docker)

---

## 12. References

* Scikit-learn Documentation
* XGBoost Documentation
* Pandas Documentation
* Matplotlib Gallery
