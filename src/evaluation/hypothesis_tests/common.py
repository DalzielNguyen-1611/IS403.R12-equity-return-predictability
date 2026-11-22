import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

MODELS_LAN01 = [
    'ARIMA', 'ARIMAX', 'Ridge', 'Elastic_Net', 'SVR',
    'Random_Forest', 'XGBoost', 'LSTM', 'Transformer'
]
MODELS_LAN02 = ['TCN', 'PatchTST', 'TSMixer']
ALL_MODELS = MODELS_LAN01 + MODELS_LAN02

BENCHMARK_MODEL = 'Historical_Mean'
HORIZONS = ['1D', '5D', '20D', '60D', '120D']

def load_unified_predictions(split='test'):
    """
    Loads and merges out-of-sample predictions from Lan01 and Lan02.
    split: 'test' (Hold-out test 2020-11-02 to 2026-04-07) or 'val' (Walk-forward validation)
    Returns: pd.DataFrame with columns ['Date', 'Model', 'Horizon', 'Actual', 'Prediction', 'Regime']
    """
    if split == 'test':
        p1 = os.path.join(REPO_ROOT, 'results', 'Lan01', 'test_evaluation', '3.0.3_test_predictions.csv')
        p2 = os.path.join(REPO_ROOT, 'results', 'Lan02', 'test_evaluation', '3.0.3_test_predictions.csv')
    elif split == 'val':
        p1 = os.path.join(REPO_ROOT, 'results', 'Lan01', 'predictions', '3.0.3_all_oos_predictions.csv')
        p2 = os.path.join(REPO_ROOT, 'results', 'Lan02', 'predictions', '3.0.3_all_oos_predictions.csv')
    else:
        raise ValueError(f"Unknown split: {split}. Choose 'test' or 'val'.")
        
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    
    # Ensure date format
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    # Filter Lan01 models + HM
    sub1 = df1[df1['Model'].isin(MODELS_LAN01 + [BENCHMARK_MODEL])].copy()
    
    # Filter Lan02 models
    sub2 = df2[df2['Model'].isin(MODELS_LAN02)].copy()
    
    df_merged = pd.concat([sub1, sub2], ignore_index=True)
    df_merged = df_merged.sort_values(['Horizon', 'Model', 'Date']).reset_index(drop=True)
    return df_merged

def compute_loss_differentials(df_preds):
    """
    Computes normalized loss differential:
        d_{t,h,m} = (e_{HM,t,h}^2 - e_{m,t,h}^2) / MSE_{HM,h}
    Returns:
        df_diff: DataFrame with columns ['Date', 'Model', 'Horizon', 'Regime', 'd_t', 'e_sq_m', 'e_sq_hm']
        hm_mse: dict mapping horizon -> MSE_{HM,h}
    """
    # Extract Benchmark
    hm_df = df_preds[df_preds['Model'] == BENCHMARK_MODEL].copy()
    hm_df['e_sq_hm'] = (hm_df['Actual'] - hm_df['Prediction']) ** 2
    
    hm_mse = {}
    for h in HORIZONS:
        sub_h = hm_df[hm_df['Horizon'] == h]
        hm_mse[h] = float(np.mean(sub_h['e_sq_hm'].values))
        
    # Merge HM predictions with model predictions
    candidate_df = df_preds[df_preds['Model'].isin(ALL_MODELS)].copy()
    candidate_df['e_sq_m'] = (candidate_df['Actual'] - candidate_df['Prediction']) ** 2
    
    # Merge e_sq_hm by Date and Horizon
    merged = pd.merge(
        candidate_df,
        hm_df[['Date', 'Horizon', 'e_sq_hm']],
        on=['Date', 'Horizon'],
        how='inner'
    )
    
    # Normalized loss differential
    merged['d_t'] = merged.apply(lambda r: (r['e_sq_hm'] - r['e_sq_m']) / hm_mse[r['Horizon']], axis=1)
    
    return merged, hm_mse
