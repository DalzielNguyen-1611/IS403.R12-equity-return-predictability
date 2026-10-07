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

from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

MODEL_NAME = "SVR"

def run_model(df=None, horizons=config.HORIZONS, logger=None):
    if df is None:
        df = utils.load_dataset()
    
    # 1. Load checkpoint if interrupted previously
    completed_keys, fold_metrics, oos_preds = utils.load_checkpoint(CURRENT_DIR)
    cached_tuning = utils.load_saved_tuning_params(MODEL_NAME)
    
    if logger:
        logger.info(f"Running {MODEL_NAME} across horizons: {horizons}")
        if completed_keys:
            logger.info(f"🔄 Found checkpoint with {len(completed_keys)} completed folds. Resuming...")
        if cached_tuning:
            logger.info(f"⚡ Found {len(cached_tuning)} cached tuning records in results/tuning/best_hyperparameters.csv.")
        
    for horizon in horizons:
        for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
            if (str(horizon), int(fold)) in completed_keys:
                if logger:
                    logger.info(f"⏩ [CHECKPOINT] Skipping already completed {MODEL_NAME} | Horizon: {horizon} | Fold: {fold}")
                continue
            
            data = utils.prepare_fold_data(train_df, val_df, horizon, scale=True)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            
            cached_key = (str(horizon), int(fold))
            if cached_key in cached_tuning:
                best_params_raw, val_rmse = cached_tuning[cached_key]
                best_params = best_params_raw.copy() if isinstance(best_params_raw, dict) else eval(best_params_raw)
                best_model = SVR(**best_params)
                best_model.fit(X_train, y_train)
                if logger:
                    logger.info(f"⚡ [CACHED] Recovered best params for Horizon {horizon} Fold {fold}. Skipping GridSearch!")
            else:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(SVR(kernel='rbf'), param_grid=config.SVR_PARAM_GRID, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                val_rmse = float(np.sqrt(-grid.best_score_))
                utils.save_tuning_record(MODEL_NAME, horizon, fold, grid.best_params_, val_rmse, score_metric='CV_RMSE')
                
            y_preds = best_model.predict(X_val)
            
            m = utils.calc_regression_metrics(y_val, y_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_params': str(best_params),
                'best_val_rmse': round(val_rmse, 6),
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
                
            # Save checkpoint after each fold
            utils.save_checkpoint_step(CURRENT_DIR, fold_metrics, oos_preds)
            completed_keys.add((str(horizon), int(fold)))
            if logger:
                logger.info(f"✓ Completed & check-pointed {MODEL_NAME} | Horizon: {horizon} | Fold: {fold}")
                
    m_df = pd.DataFrame(fold_metrics)
    p_df = pd.DataFrame(oos_preds)
    utils.save_model_latest_results(CURRENT_DIR, m_df, p_df)
    utils.finalize_checkpoint(CURRENT_DIR)
    if logger:
        logger.info(f"Completed {MODEL_NAME}.")
    return m_df, p_df

if __name__ == "__main__":
    logger = utils.setup_logger(MODEL_NAME)
    run_model(logger=logger)
