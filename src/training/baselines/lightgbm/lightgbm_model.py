import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

from lightgbm import LGBMRegressor

MODEL_NAME = "LightGBM"

def run_model(df=None, horizons=config.HORIZONS, logger=None):
    if df is None:
        df = utils.load_dataset()
    fold_metrics = []
    oos_preds = []
    
    if logger:
        logger.info(f"Running {MODEL_NAME} across horizons: {horizons}")
        
    for horizon in horizons:
        for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
            data = utils.prepare_fold_data(train_df, val_df, horizon, scale=False)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            X_tr_in, y_tr_in, X_val_in, y_val_in = utils.get_internal_train_val_split(X_train, y_train, val_ratio=0.15)
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 7, 127),
                    'max_depth': trial.suggest_int('max_depth', -1, 12),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
                    'random_state': config.RANDOM_SEED,
                    'n_jobs': -1,
                    'verbose': -1
                }
                m_inner = LGBMRegressor(**params)
                m_inner.fit(X_tr_in, y_tr_in)
                preds = m_inner.predict(X_val_in)
                return float(np.sqrt(np.mean((y_val_in - preds)**2)))
                
            n_trials = config.OPTUNA_TRIALS.get('lightgbm', 50)
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params.copy()
            best_params.update({'random_state': config.RANDOM_SEED, 'n_jobs': -1, 'verbose': -1})
            utils.save_tuning_record(MODEL_NAME, horizon, fold, study.best_params, study.best_value, score_metric='Internal_Val_RMSE')
            
            lgbm = LGBMRegressor(**best_params)
            lgbm.fit(X_train, y_train)
            y_preds = lgbm.predict(X_val)
            
            m = utils.calc_regression_metrics(y_val, y_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_params': str(study.best_params),
                'best_val_rmse': round(study.best_value, 6),
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
                'Directional_Accuracy': m['Directional_Accuracy']
            })
            
            val_dates = data['val_dates']
            val_regimes = data['val_regimes']
            for dt, act, pr, reg in zip(val_dates, y_val, y_preds, val_regimes):
                oos_preds.append({
                    'Date': dt, 'Model': MODEL_NAME, 'Horizon': horizon,
                    'Fold': fold, 'Actual': act, 'Prediction': pr, 'Regime': reg
                })
                
    m_df = pd.DataFrame(fold_metrics)
    p_df = pd.DataFrame(oos_preds)
    utils.save_model_latest_results(CURRENT_DIR, m_df, p_df)
    if logger:
        logger.info(f"Completed {MODEL_NAME}.")
    return m_df, p_df

if __name__ == "__main__":
    logger = utils.setup_logger(MODEL_NAME)
    run_model(logger=logger)
