import os
import ast
import torch
import pandas as pd

TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTING_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(REPO_ROOT, 'dataset', '3.0.3_val_folds.csv')
TUNING_PATH = os.path.join(REPO_ROOT, 'results', 'Lan01', 'tuning', '3.0.3_best_hyperparameters.csv')

# Output directories for Test Set Evaluation
TEST_RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'Lan01', 'test_evaluation')
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

PREFIX = '3.0.3_'
TEST_PREDICTIONS_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_predictions.csv')
TEST_METRICS_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_metrics_summary.csv')
TEST_CLASSIFICATION_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_classification_summary.csv')
TEST_STATISTICAL_TESTS_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_statistical_tests.csv')
TEST_REGIME_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_regime_breakdown.csv')
TEST_REPORT_PATH = os.path.join(TEST_RESULTS_DIR, f'{PREFIX}test_evaluation_report.md')

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS = ['1D', '5D', '20D', '60D', '120D']

HORIZON_TO_STEPS = {
    '1D': 1, '1': 1,
    '5D': 5, '5': 5,
    '20D': 20, '20': 20,
    '60D': 60, '60': 60,
    '120D': 120, '120': 120,
}

HORIZON_TO_TARGET = {
    '1D': 'Target_1', '1': 'Target_1',
    '5D': 'Target_5', '5': 'Target_5',
    '20D': 'Target_20', '20': 'Target_20',
    '60D': 'Target_60', '60': 'Target_60',
    '120D': 'Target_120', '120': 'Target_120',
}

def normalize_horizon(h):
    h = str(h).strip().upper()
    if not h.endswith('D'):
        h = f"{h}D"
    return h

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

MODELS = [
    'historical_mean',
    'ridge',
    'elastic_net',
    'arima',
    'arimax',
    'sarima',
    'svr',
    'random_forest',
    'xgboost',
    'lightgbm',
    'rnn',
    'lstm',
    'transformer',
    'logistic_regression'
]

MODEL_NAME_MAP = {
    'historical_mean': 'Historical_Mean',
    'ridge': 'Ridge',
    'elastic_net': 'Elastic_Net',
    'arima': 'ARIMA',
    'arimax': 'ARIMAX',
    'sarima': 'SARIMA',
    'svr': 'SVR',
    'random_forest': 'Random_Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'rnn': 'RNN',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'logistic_regression': 'Logistic_Regression'
}

def load_best_hyperparameters(tuning_path=TUNING_PATH):
    """
    Loads best hyperparameters found during Walk-Forward Validation (Lan01).
    Selects parameters from Fold 5 (most mature fold trained up to late 2020),
    or best validation score.
    Returns: dict mapping (model_display_name, horizon) -> dict_of_params
    """
    if not os.path.exists(tuning_path):
        return {}
    
    df_tune = pd.read_csv(tuning_path)
    best_params_map = {}
    
    for (m, h), grp in df_tune.groupby(['Model', 'Horizon']):
        # Prefer Fold 5 (trained on largest expanding window prior to Split 3)
        f5 = grp[grp['Fold'] == 5]
        if len(f5) > 0:
            row = f5.iloc[0]
        else:
            row = grp.sort_values('Best_Val_Score').iloc[0]
            
        try:
            params = ast.literal_eval(row['Best_Params'])
        except Exception:
            params = {}
        best_params_map[(m, h)] = params
        
    return best_params_map
