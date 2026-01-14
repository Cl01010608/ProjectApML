

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
├── README.md
└── requirements.txt
```

---

## 3. Dataset Description

### Source

Industrial steel production dataset obtained from a manufacturing process
(Source: internal/cloud server)

### Dataset Characteristics

| Metric           | Value                    |
| ---------------- | ------------------------ |
| Training samples | 6,113                    |
| Test samples     | 1,533                    |
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
      - Feature selection (MI, k=20)
      - Scaling (StandardScaler)
      - Residual learning
  → Exploratory Data Analysis
  → Model Training (30+ models)
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

**Purpose:** Train and compare 30+ regression models

**Model categories:**

* Linear models (Linear, Ridge, Lasso, ElasticNet)
* Tree-based models (Decision Tree, Random Forest)
* Ensemble models (Gradient Boosting, XGBoost, LightGBM, AdaBoost)
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

* Model ranking plots (R², RMSE, MAE)
* Prediction vs actual plots
* Residual diagnostics
* Multi-metric comparison figures

Outputs are saved as publication-quality PNG files.

---

## 6. Results Summary

### Top Models (Validation Performance)

| Rank | Model                 | Val R² | RMSE | Train Time (s) |
| ---- | --------------------- | ------ | ---- | -------------- |
| 1    | GradientBoosting_n500 | 0.4337 | 2.45 | 41.42          |
| 2    | RandomForest_n500     | 0.4284 | 2.51 | 23.85          |
| 3    | GradientBoosting_n300 | 0.4267 | 2.47 | 18.45          |
| 4    | RandomForest_n300     | 0.4252 | 2.52 | 12.90          |
| 5    | RandomForest_n100     | 0.4195 | 2.62 | 2.82           |

---

## 7. Key Findings

* Ensemble models clearly outperform linear and single-tree models
* Residual learning is critical for achieving positive R²
* Increasing ensemble size improves performance with diminishing returns
* Validation R² ≈ 0.43, but test R² is negative, indicating overfitting
* RandomForest_n100 offers the best speed–accuracy trade-off

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
* xgboost (optional: lightgbm)
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
