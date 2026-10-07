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

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

MODEL_NAME = "Ridge"

def run_model(df=None, horizons=config.HORIZONS, logger=None):
    if df is None:
        df = utils.load_dataset()
    fold_metrics = []
    oos_preds = []
    
    if logger:
        logger.info(f"Running {MODEL_NAME} across horizons: {horizons}")
        
    for horizon in horizons:
        for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
            data = utils.prepare_fold_data(train_df, val_df, horizon, scale=True)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            
            tscv = TimeSeriesSplit(n_splits=3)
            model = RidgeCV(alphas=config.RIDGE_PARAM_GRID['alpha'], cv=tscv, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            utils.save_tuning_record(MODEL_NAME, horizon, fold, {'alpha': float(model.alpha_)}, 0.0, score_metric='CV_RMSE')
            y_preds = model.predict(X_val)
            
            m = utils.calc_regression_metrics(y_val, y_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_alpha': model.alpha_,
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
