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

MODEL_NAME = "Historical_Mean"

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
            train_target = train_df[target_col].values
            val_target = val_df[target_col].values
            
            cum_sum = np.sum(train_target)
            cum_count = len(train_target)
            val_preds = []
            for i in range(len(val_target)):
                val_preds.append(cum_sum / cum_count)
                cum_sum += val_target[i]
                cum_count += 1
            val_preds = np.array(val_preds)
            
            m = utils.calc_regression_metrics(val_target, val_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
                'Directional_Accuracy': m['Directional_Accuracy']
            })
            
            val_dates = val_df['Date'].dt.strftime('%Y-%m-%d').values
            val_regimes = val_df['Volatility_Regime'].values
            for dt, act, pr, reg in zip(val_dates, val_target, val_preds, val_regimes):
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
