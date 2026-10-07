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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

MODEL_NAME = "Logistic_Regression"

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
            X_train, y_train = data['X_train'], data['y_train_dir']
            X_val, y_val = data['X_val'], data['y_val_dir']
            actual_cont = data['y_val']
            
            tscv = TimeSeriesSplit(n_splits=3)
            param_grid = {
                'C': config.LOGISTIC_PARAM_GRID['C'],
                'class_weight': config.LOGISTIC_PARAM_GRID['class_weight']
            }
            base_clf = LogisticRegression(
                penalty=config.LOGISTIC_PARAM_GRID['penalty'],
                solver=config.LOGISTIC_PARAM_GRID['solver'],
                max_iter=config.LOGISTIC_PARAM_GRID['max_iter'],
                random_state=config.RANDOM_SEED
            )
            grid = GridSearchCV(base_clf, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            clf = grid.best_estimator_
            utils.save_tuning_record(MODEL_NAME, horizon, fold, grid.best_params_, grid.best_score_, score_metric='CV_ROC_AUC')
            
            y_probs = clf.predict_proba(X_val)[:, 1]
            y_preds = (y_probs >= 0.5).astype(int)
            
            m = utils.calc_classification_metrics(y_val, y_probs, y_preds)
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_C': grid.best_params_['C'],
                'best_class_weight': str(grid.best_params_['class_weight']),
                'ROC_AUC': m['ROC_AUC'], 'Accuracy': m['Accuracy'],
                'Precision': m['Precision'], 'Recall': m['Recall'],
                'F1': m['F1'], 'Balanced_Accuracy': m['Balanced_Accuracy']
            })
            
            val_dates = data['val_dates']
            val_regimes = data['val_regimes']
            for dt, act_c, act_cls, pr_prob, pr_cls, reg in zip(val_dates, actual_cont, y_val, y_probs, y_preds, val_regimes):
                oos_preds.append({
                    'Date': dt, 'Model': MODEL_NAME, 'Horizon': horizon,
                    'Fold': fold, 'Actual': act_c, 'Actual_Class': act_cls,
                    'Predicted_Probability': pr_prob, 'Predicted_Class': pr_cls,
                    'Prediction': pr_cls, 'Regime': reg
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
