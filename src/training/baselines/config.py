import os
import torch

BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASELINES_DIR)))

DATA_PATH = os.path.join(REPO_ROOT, 'dataset', '3.0.3_val_folds.csv')
FOLD_DEF_PATH = os.path.join(BASELINES_DIR, 'fold_definition.csv')

# 3-Tier Results Directory Structure
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
FOLD_METRICS_DIR = os.path.join(RESULTS_DIR, 'fold_metrics')
SUMMARY_DIR = os.path.join(RESULTS_DIR, 'summary')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')
TUNING_DIR = os.path.join(RESULTS_DIR, 'tuning')

ALL_FOLD_METRICS_PATH = os.path.join(FOLD_METRICS_DIR, 'all_fold_metrics.csv')
MODEL_HORIZON_SUMMARY_PATH = os.path.join(SUMMARY_DIR, 'model_horizon_summary.csv')
ALL_PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, 'all_oos_predictions.csv')

RQ1_HORIZON_PATH = os.path.join(ANALYSIS_DIR, 'rq1_horizon.csv')
RQ2_REGIME_PATH = os.path.join(ANALYSIS_DIR, 'rq2_regime.csv')
RQ3_TIME_PATH = os.path.join(ANALYSIS_DIR, 'rq3_time.csv')
STATISTICAL_TESTS_PATH = os.path.join(ANALYSIS_DIR, 'statistical_tests.csv')
BEST_HYPERPARAMS_PATH = os.path.join(TUNING_DIR, 'best_hyperparameters.csv')

# Ensure results directories exist
for _d in [RESULTS_DIR, FOLD_METRICS_DIR, SUMMARY_DIR, PREDICTIONS_DIR, ANALYSIS_DIR, TUNING_DIR]:
    os.makedirs(_d, exist_ok=True)

RANDOM_SEED = 42

# Device selection for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Horizons and Target mapping
HORIZONS = ['1D', '5D', '20D', '60D', '120D']

HORIZON_TO_TARGET = {
    '1D': 'Target_1',
    '5D': 'Target_5',
    '20D': 'Target_20',
    '60D': 'Target_60',
    '120D': 'Target_120',
}

# Time Periods Configuration for RQ3 (Temporal Stability Analysis)
TIME_PERIODS = [
    ("2015-2016", "2015-01-01", "2016-12-31"),
    ("2017-2018", "2017-01-01", "2018-12-31"),
    ("2019-2020", "2019-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2024", "2023-01-01", "2024-12-31"),
    ("2025-2026", "2025-01-01", "2026-12-31"),
]

# Formal Research Metric Framework
METRIC_FRAMEWORK = {
    'regression': {
        'primary': 'OOS_R2',
        'secondary': ['MAE', 'RMSE', 'DA'],
        'inference': ['DM_Test', 'Sign_Test']
    },
    'classification': {
        'primary': 'ROC_AUC',
        'secondary': ['Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Accuracy']
    }
}

# 16 Predictor Features
FEATURE_COLS = [
    'Return_lag_1',
    'Return_lag_5',
    'Return_lag_20',
    'Momentum_5D',
    'Momentum_20D',
    'Momentum_60D',
    'Volatility_5D',
    'Volatility_20D',
    'Volatility_60D',
    'VIXCLS',
    'VIX_lag_1',
    'VIX_change_1D',
    'VIX_pct_change_1D',
    'HL_range',
    'OC_return',
    'Volume_change',
]

# Exogenous variables for ARIMAX
ARIMAX_EXOG_COLS = [
    'VIXCLS',
    'VIX_lag_1',
    'VIX_change_1D',
    'VIX_pct_change_1D',
    'Momentum_5D',
    'Momentum_20D',
    'Volatility_20D',
    'HL_range',
    'OC_return',
    'Volume_change',
]

# Baseline model mapping: folder_name -> (folder, filename)
MODEL_REGISTRY = {
    # 1. Benchmark
    'historical_mean': ('historical_mean', 'historical_mean.py'),
    
    # 2. Classical / Statistical
    'ridge': ('ridge', 'ridge.py'),
    'elastic_net': ('elastic_net', 'elastic_net.py'),
    'arima': ('arima', 'arima.py'),
    'arimax': ('arimax', 'arimax.py'),
    'sarima': ('sarima', 'sarima.py'),
    
    # 3. Kernel-based
    'svr': ('svr', 'svr.py'),
    
    # 4. Tree-based / Ensemble
    'random_forest': ('random_forest', 'random_forest.py'),
    'xgboost': ('xgboost', 'xgboost_model.py'),
    'lightgbm': ('lightgbm', 'lightgbm_model.py'),
    
    # 5. Deep Learning
    'rnn': ('rnn', 'rnn_model.py'),
    'lstm': ('lstm', 'lstm_model.py'),
    'transformer': ('transformer', 'transformer_model.py'),
    
    # 6. Classification Side-task
    'logistic_regression': ('logistic_regression', 'logistic_regression.py'),
}

MODELS = list(MODEL_REGISTRY.keys())

# =========================================================================
# Hyperparameter Tuning Framework (Fixed / Grid Search / Optuna)
# =========================================================================

HYPERPARAM_STRATEGY = {
    'historical_mean': 'None',
    'logistic_regression': 'Grid',
    'ridge': 'Grid',
    'elastic_net': 'Grid',
    'arima': 'Grid',
    'arimax': 'Grid',
    'sarima': 'Grid',
    'svr': 'Grid',
    'random_forest': 'Optuna',
    'xgboost': 'Optuna',
    'lightgbm': 'Optuna',
    'rnn': 'Optuna',
    'lstm': 'Optuna',
    'transformer': 'Optuna'
}

OPTUNA_TRIALS = {
    'random_forest': 30,
    'xgboost': 50,
    'lightgbm': 50,
    'rnn': 30,
    'lstm': 30,
    'transformer': 30
}

# 1. Logistic Regression (Grid Search, 12 combinations)
LOGISTIC_PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 2000
}

# 2. Ridge (Grid Search, 6 combinations)
RIDGE_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 3. Elastic Net (Grid Search, 30 combinations)
ELASTIC_NET_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_iter': 10000
}

# 4. SVR (Grid Search, 64 combinations)
SVR_PARAM_GRID = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'epsilon': [0.001, 0.01, 0.05, 0.1]
}

# 5. ARIMA Grid Space
ARIMA_PARAM_GRID = {
    'p': [0, 1, 2, 3, 5],
    'd': [0, 1],
    'q': [0, 1, 2, 3, 5]
}

# 6. SARIMA Grid Space
SARIMA_PARAM_GRID = {
    'p': [0, 1, 2],
    'd': [0, 1],
    'q': [0, 1, 2],
    'P': [0, 1],
    'D': [0, 1],
    'Q': [0, 1],
    's': 5
}

# 7. Tree-based Optuna Parameter Search Bounds
RF_OPTUNA_SPACE = {
    'n_estimators': [200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 0.5]
}

XGB_OPTUNA_BOUNDS = {
    'n_estimators': (100, 1000),
    'learning_rate': (0.01, 0.2),
    'max_depth': (2, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0.0, 5.0),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-3, 100.0)
}

LGBM_OPTUNA_BOUNDS = {
    'n_estimators': (100, 1000),
    'learning_rate': (0.01, 0.2),
    'num_leaves': (7, 127),
    'max_depth': (-1, 12),
    'min_child_samples': (5, 100),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-3, 100.0)
}

# 8. Deep Learning Default Parameters & Optuna Search Spaces
DL_PARAMS = {
    'lookback': 20,
    'hidden_size': 64,
    'num_layers': 1,
    'dropout': 0.1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 40,
    'patience': 7
}

DL_OPTUNA_SPACE = {
    'lookback': [10, 20, 40, 60],
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': (0.0, 0.5),
    'learning_rate': (1e-5, 1e-3),
    'weight_decay': (1e-6, 1e-2),
    'batch_size': [16, 32, 64],
    'epochs': 40,
    'patience': 7
}

TRANSFORMER_PARAMS = {
    'lookback': 20,
    'd_model': 32,
    'nhead': 2,
    'num_encoder_layers': 1,
    'dim_feedforward': 64,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 40,
    'patience': 7
}

TRANSFORMER_OPTUNA_SPACE = {
    'lookback': [20, 40, 60],
    'd_model': [32, 64],
    'nhead': [2, 4],
    'num_encoder_layers': [1, 2],
    'dim_feedforward': [64, 128],
    'dropout': (0.05, 0.3),
    'learning_rate': (1e-5, 5e-4),
    'weight_decay': (1e-6, 1e-2),
    'batch_size': [16, 32],
    'epochs': 40,
    'patience': 7
}

