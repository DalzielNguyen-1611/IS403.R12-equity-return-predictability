import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

from sklearn.preprocessing import StandardScaler

MODEL_NAME = "ARIMAX"

def run_model(df=None, horizons=config.HORIZONS, logger=None):
    if df is None:
        df = utils.load_dataset()
    fold_metrics = []
    oos_preds = []
    
    if logger:
        logger.info(f"Running {MODEL_NAME} across horizons: {horizons}")
        
    for horizon in horizons:
        target_col = config.HORIZON_TO_TARGET[horizon]
        for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
            train_y = train_df[target_col].values
            val_y = val_df[target_col].values
            full_y = np.concatenate([train_y, val_y])
            
            exog_cols = config.ARIMAX_EXOG_COLS
            train_exog_raw = train_df[exog_cols].values
            val_exog_raw = val_df[exog_cols].values
            
            scaler = StandardScaler()
            train_exog = scaler.fit_transform(train_exog_raw)
            val_exog = scaler.transform(val_exog_raw)
            full_exog = np.concatenate([train_exog, val_exog], axis=0)
            
            order = (1, 0, 1)
            try:
                mod = ARIMA(train_y, exog=train_exog, order=order)
                res = mod.fit()
            except Exception:
                order = (1, 0, 0)
                mod = ARIMA(train_y, exog=train_exog, order=order)
                res = mod.fit()
                
            utils.save_tuning_record(MODEL_NAME, horizon, fold, {'order': order}, float(getattr(res, 'aic', 0.0)), score_metric='AIC')
            applied = res.apply(full_y, exog=full_exog)
            n_train, n_val = len(train_y), len(val_y)
            val_preds = applied.predict(start=n_train, end=n_train + n_val - 1, exog=val_exog)
            
            m = utils.calc_regression_metrics(val_y, val_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'order': str(order),
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
                'Directional_Accuracy': m['Directional_Accuracy']
            })
            
            val_dates = val_df['Date'].dt.strftime('%Y-%m-%d').values
            val_regimes = val_df['Volatility_Regime'].values
            for dt, act, pr, reg in zip(val_dates, val_y, val_preds, val_regimes):
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
