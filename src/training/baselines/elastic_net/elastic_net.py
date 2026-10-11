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

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

MODEL_NAME = "Elastic_Net"

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
            
            tscv = TimeSeriesSplit(n_splits=3)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('enet', ElasticNet(random_state=config.RANDOM_SEED, max_iter=config.ELASTIC_NET_PARAM_GRID['max_iter']))
            ])
            param_grid = {
                'enet__alpha': config.ELASTIC_NET_PARAM_GRID['alpha'],
                'enet__l1_ratio': config.ELASTIC_NET_PARAM_GRID['l1_ratio']
            }
            grid = GridSearchCV(pipeline, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            best_alpha = grid.best_params_['enet__alpha']
            best_l1 = grid.best_params_['enet__l1_ratio']
            val_rmse = float(np.sqrt(-grid.best_score_))
            utils.save_tuning_record(MODEL_NAME, horizon, fold, {'alpha': best_alpha, 'l1_ratio': best_l1}, val_rmse, score_metric='CV_RMSE')
            y_preds = best_model.predict(X_val)
            
            m = utils.calc_regression_metrics(y_val, y_preds, y_bench=data['y_bench'], h=config.HORIZON_TO_STEPS.get(horizon, 1))
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_alpha': best_alpha,
                'best_l1_ratio': best_l1,
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
    return m_df, p_df

if __name__ == "__main__":
    run_model()
