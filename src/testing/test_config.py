import os
import ast
import torch
import pandas as pd

TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TESTING_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(REPO_ROOT, 'dataset', '3.0.3_val_folds.csv')
RUN_NAME = os.environ.get('TEST_RUN_NAME', 'Lan01')

PREFIX = '3.0.3_'

# Explicit top-level module attributes for static analyzers and IDE
TEST_RESULTS_DIR = None
TUNING_PATH = None
TEST_PREDICTIONS_PATH = None
TEST_METRICS_PATH = None
TEST_CLASSIFICATION_PATH = None
TEST_STATISTICAL_TESTS_PATH = None
TEST_REGIME_PATH = None
TEST_REPORT_PATH = None

def get_run_paths(run_name='Lan01'):
    res_dir = os.path.join(REPO_ROOT, 'results', run_name, 'test_evaluation')
    tune_path = os.path.join(REPO_ROOT, 'results', run_name, 'tuning', f'{PREFIX}best_hyperparameters.csv')
    return {
        'results_dir': res_dir,
        'tuning_path': tune_path if os.path.exists(tune_path) else None,
        'predictions_path': os.path.join(res_dir, f'{PREFIX}test_predictions.csv'),
        'metrics_path': os.path.join(res_dir, f'{PREFIX}test_metrics_summary.csv'),
        'classification_path': os.path.join(res_dir, f'{PREFIX}test_classification_summary.csv'),
        'statistical_tests_path': os.path.join(res_dir, f'{PREFIX}test_statistical_tests.csv'),
        'regime_path': os.path.join(res_dir, f'{PREFIX}test_regime_breakdown.csv'),
        'report_path': os.path.join(res_dir, f'{PREFIX}test_evaluation_report.md'),
    }

def set_run(run_name='Lan01'):
    global RUN_NAME, TEST_RESULTS_DIR, TUNING_PATH, TEST_PREDICTIONS_PATH, TEST_METRICS_PATH
    global TEST_CLASSIFICATION_PATH, TEST_STATISTICAL_TESTS_PATH, TEST_REGIME_PATH, TEST_REPORT_PATH
    RUN_NAME = run_name
    paths = get_run_paths(run_name)
    TEST_RESULTS_DIR = paths['results_dir']
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    TUNING_PATH = paths['tuning_path']
    TEST_PREDICTIONS_PATH = paths['predictions_path']
    TEST_METRICS_PATH = paths['metrics_path']
    TEST_CLASSIFICATION_PATH = paths['classification_path']
    TEST_STATISTICAL_TESTS_PATH = paths['statistical_tests_path']
    TEST_REGIME_PATH = paths['regime_path']
    TEST_REPORT_PATH = paths['report_path']

# Initialize paths for default RUN_NAME
set_run(RUN_NAME)

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

LAN01_MODELS = [
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

ADVANCED_DL_MODELS = [
    'historical_mean',
    'cnn',
    'patchtst',
    'nhits',
    'nbeats',
    'itransformer',
    'tsmixer',
    'tcn'
]

LAN02_MODELS = ADVANCED_DL_MODELS
LAN03_MODELS = ADVANCED_DL_MODELS

MODELS = LAN01_MODELS

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
    'logistic_regression': 'Logistic_Regression',
    'tcn': 'TCN',
    'nbeats': 'N-BEATS',
    'nhits': 'N-HiTS',
    'patchtst': 'PatchTST',
    'tsmixer': 'TSMixer',
    'itransformer': 'iTransformer',
    'cnn': 'CNN'
}

def load_best_hyperparameters(tuning_path=None):
    """
    Loads best hyperparameters found during Walk-Forward Validation.
    Selects parameters from Fold 5 (most mature fold trained up to late 2020),
    or best validation score.
    Returns: dict mapping (model_display_name, horizon) -> dict_of_params
    """
    if tuning_path is None:
        tuning_path = TUNING_PATH
        
    if not tuning_path or not os.path.exists(tuning_path):
        return {}
    
    df_tune = pd.read_csv(tuning_path)
    best_params_map = {}
    
    for (m, h), grp in df_tune.groupby(['Model', 'Horizon']):
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
