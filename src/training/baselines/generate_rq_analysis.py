import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
import utils

logger = utils.setup_logger("rq_analysis")

def map_date_to_period(dt):
    """Map a timestamp to a configured research time period."""
    dt_str = dt.strftime('%Y-%m-%d')
    for label, start, end in config.TIME_PERIODS:
        if start <= dt_str <= end:
            return label
    return f"{dt.year}"

def run_analysis():
    logger.info("Starting Post-Run Analysis for RQ1, RQ2, RQ3, and Statistical Tests...")
    
    preds_path = config.ALL_PREDICTIONS_PATH
    if not os.path.exists(preds_path):
        logger.warning(f"Prediction file not found at {preds_path}. Run baseline models first!")
        return
        
    df = pd.read_csv(preds_path)
    if df.empty:
        logger.warning("Predictions file is empty. Nothing to analyze.")
        return
        
    logger.info(f"Loaded {len(df):,} OOS predictions across {df['Model'].nunique()} models and {df['Horizon'].nunique()} horizons.")
    
    # Standardize column types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Period'] = df['Date'].apply(map_date_to_period)
    
    # Extract benchmark (historical_mean) predictions
    bench_df = df[df['Model'] == 'historical_mean']
    
    models = sorted(df['Model'].unique())
    # Place historical_mean at the front of models list if present
    if 'historical_mean' in models:
        models.remove('historical_mean')
        models = ['historical_mean'] + models
        
    horizons = [h for h in config.HORIZONS if h in df['Horizon'].unique()]
    
    # -------------------------------------------------------------
    # 1. RQ1 — Horizon Decay Analysis
    # Does equity return predictability decay as forecast horizon increases?
    # -------------------------------------------------------------
    logger.info("Computing RQ1: Horizon Decay Matrix...")
    rq1_records = []
    
    for mod in models:
        row = {'Model': mod}
        for hrz in horizons:
            sub = df[(df['Model'] == mod) & (df['Horizon'] == hrz)]
            if sub.empty:
                continue
            y_true = sub['Actual'].values
            y_pred = sub['Prediction'].values
            
            # Match benchmark if available
            b_sub = bench_df[bench_df['Horizon'] == hrz]
            if not b_sub.empty and mod != 'historical_mean':
                merged = pd.merge(sub[['Date', 'Actual', 'Prediction']], 
                                  b_sub[['Date', 'Prediction']], 
                                  on='Date', suffixes=('', '_bench'))
                if not merged.empty:
                    sse_m = np.sum((merged['Actual'] - merged['Prediction'])**2)
                    sse_b = np.sum((merged['Actual'] - merged['Prediction_bench'])**2)
                    oos_r2 = 1.0 - (sse_m / (sse_b + 1e-12))
                else:
                    oos_r2 = 1.0 - (np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12))
            elif mod == 'historical_mean':
                oos_r2 = 0.0000  # Benchmark baseline by definition
            else:
                oos_r2 = 1.0 - (np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12))
                
            row[f'{hrz}_OOS_R2'] = round(float(oos_r2), 4)
            
            # Directional accuracy
            s_true = np.where(y_true > 0, 1, -1)
            s_pred = np.where(y_pred > 0, 1, -1)
            row[f'{hrz}_DA'] = round(float(np.mean(s_true == s_pred)), 4)
            
        rq1_records.append(row)
        
    rq1_df = pd.DataFrame(rq1_records)
    rq1_df.to_csv(config.RQ1_HORIZON_PATH, index=False)
    logger.info(f"Saved RQ1 Horizon matrix to: {config.RQ1_HORIZON_PATH}")
    
    # -------------------------------------------------------------
    # 2. RQ2 — Regime Dependence
    # How does predictability vary across Volatility Regimes (Low, Normal, High)?
    # -------------------------------------------------------------
    logger.info("Computing RQ2: Volatility Regime Breakdown...")
    rq2_records = []
    regimes = ['Low', 'Normal', 'High']
    
    for mod in models:
        for hrz in horizons:
            sub = df[(df['Model'] == mod) & (df['Horizon'] == hrz)]
            if sub.empty:
                continue
            rec = {'Model': mod, 'Horizon': hrz}
            
            for reg in regimes:
                reg_sub = sub[sub['Regime'] == reg]
                if reg_sub.empty:
                    rec[f'{reg}_OOS_R2'] = np.nan
                    rec[f'{reg}_DA'] = np.nan
                    continue
                    
                y_t = reg_sub['Actual'].values
                y_p = reg_sub['Prediction'].values
                
                # Bench in same regime
                b_sub = bench_df[(bench_df['Horizon'] == hrz) & (bench_df['Regime'] == reg)]
                if not b_sub.empty and mod != 'historical_mean':
                    merged = pd.merge(reg_sub[['Date', 'Actual', 'Prediction']], 
                                      b_sub[['Date', 'Prediction']], 
                                      on='Date', suffixes=('', '_bench'))
                    if not merged.empty:
                        sse_m = np.sum((merged['Actual'] - merged['Prediction'])**2)
                        sse_b = np.sum((merged['Actual'] - merged['Prediction_bench'])**2)
                        r2_reg = 1.0 - (sse_m / (sse_b + 1e-12))
                    else:
                        r2_reg = 1.0 - (np.sum((y_t - y_p)**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-12))
                elif mod == 'historical_mean':
                    r2_reg = 0.0000
                else:
                    r2_reg = 1.0 - (np.sum((y_t - y_p)**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-12))
                    
                s_t = np.where(y_t > 0, 1, -1)
                s_p = np.where(y_p > 0, 1, -1)
                rec[f'{reg}_OOS_R2'] = round(float(r2_reg), 4)
                rec[f'{reg}_DA'] = round(float(np.mean(s_t == s_p)), 4)
                
            rq2_records.append(rec)
            
    rq2_df = pd.DataFrame(rq2_records)
    rq2_df.to_csv(config.RQ2_REGIME_PATH, index=False)
    logger.info(f"Saved RQ2 Regime analysis to: {config.RQ2_REGIME_PATH}")
    
    # -------------------------------------------------------------
    # 3. RQ3 — Time Stability
    # Pattern predictability có ổn định theo thời gian (Time Periods) hay không?
    # -------------------------------------------------------------
    logger.info("Computing RQ3: Time Stability Breakdown across Time Periods...")
    rq3_records = []
    periods_present = [p for p in df['Period'].unique() if p]
    periods_sorted = sorted(periods_present)
    
    for mod in models:
        for prd in periods_sorted:
            rec = {'Model': mod, 'Period': prd}
            prd_sub = df[(df['Model'] == mod) & (df['Period'] == prd)]
            if prd_sub.empty:
                continue
                
            for hrz in horizons:
                h_sub = prd_sub[prd_sub['Horizon'] == hrz]
                if h_sub.empty:
                    rec[f'{hrz}_OOS_R2'] = np.nan
                    rec[f'{hrz}_DA'] = np.nan
                    continue
                    
                y_t = h_sub['Actual'].values
                y_p = h_sub['Prediction'].values
                
                # Bench in same period & horizon
                b_sub = bench_df[(bench_df['Horizon'] == hrz) & (bench_df['Period'] == prd)]
                if not b_sub.empty and mod != 'historical_mean':
                    merged = pd.merge(h_sub[['Date', 'Actual', 'Prediction']], 
                                      b_sub[['Date', 'Prediction']], 
                                      on='Date', suffixes=('', '_bench'))
                    if not merged.empty:
                        sse_m = np.sum((merged['Actual'] - merged['Prediction'])**2)
                        sse_b = np.sum((merged['Actual'] - merged['Prediction_bench'])**2)
                        oos_r2_prd = 1.0 - (sse_m / (sse_b + 1e-12))
                    else:
                        oos_r2_prd = 1.0 - (np.sum((y_t - y_p)**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-12))
                elif mod == 'historical_mean':
                    oos_r2_prd = 0.0000
                else:
                    oos_r2_prd = 1.0 - (np.sum((y_t - y_p)**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-12))
                    
                s_t = np.where(y_t > 0, 1, -1)
                s_p = np.where(y_p > 0, 1, -1)
                
                rec[f'{hrz}_OOS_R2'] = round(float(oos_r2_prd), 4)
                rec[f'{hrz}_DA'] = round(float(np.mean(s_t == s_p)), 4)
                rec[f'{hrz}_RMSE'] = round(float(np.sqrt(np.mean((y_t - y_p)**2))), 4)
                
            rq3_records.append(rec)
            
    rq3_df = pd.DataFrame(rq3_records)
    rq3_df.to_csv(config.RQ3_TIME_PATH, index=False)
    logger.info(f"Saved RQ3 Time stability analysis to: {config.RQ3_TIME_PATH}")
    
    # -------------------------------------------------------------
    # 4. Statistical Inference Tests
    # Columns: Model | Horizon | Test | Benchmark | Statistic | p-value | Significant
    # -------------------------------------------------------------
    logger.info("Computing Statistical Inference Tests (DM Test & Binomial Test)...")
    stat_records = []
    horizon_steps = {'1D': 1, '5D': 5, '20D': 20, '60D': 60, '120D': 120}
    
    # [A] Diebold-Mariano Test: So sánh các candidate models với Historical Mean
    for mod in models:
        if mod == 'historical_mean':
            continue  # Historical Mean là benchmark nên không cần DM test với chính nó
        for hrz in horizons:
            sub = df[(df['Model'] == mod) & (df['Horizon'] == hrz)]
            b_sub = bench_df[bench_df['Horizon'] == hrz]
            if sub.empty or b_sub.empty:
                continue
                
            merged = pd.merge(sub[['Date', 'Actual', 'Prediction']], 
                              b_sub[['Date', 'Prediction']], 
                              on='Date', suffixes=('', '_bench'))
            if merged.empty:
                continue
                
            y_t = merged['Actual'].values
            y_m = merged['Prediction'].values
            y_b = merged['Prediction_bench'].values
            h_step = horizon_steps.get(hrz, 1)
            
            # Diebold Mariano with HLN finite-sample correction
            dm_stat, dm_p = utils.diebold_mariano_test(y_t, y_m, y_b, h=h_step, loss='squared', alternative='greater')
            
            stat_records.append({
                'Model': mod,
                'Horizon': hrz,
                'Test': 'DM',
                'Benchmark': 'Historical Mean',
                'Statistic': round(float(dm_stat), 4),
                'p-value': round(float(dm_p), 4),
                'Significant': 'Yes' if dm_p < 0.05 else 'No'
            })
            
    # [B] Binomial / Sign Test: Kiểm định Directional Accuracy cho tất cả các models
    for mod in models:
        for hrz in horizons:
            sub = df[(df['Model'] == mod) & (df['Horizon'] == hrz)]
            if sub.empty:
                continue
                
            y_t = sub['Actual'].values
            y_p = sub['Prediction'].values
            
            da, binom_p = utils.directional_binomial_test(y_t, y_p, alternative='greater')
            
            stat_records.append({
                'Model': mod,
                'Horizon': hrz,
                'Test': 'Binomial',
                'Benchmark': '50% DA',
                'Statistic': round(float(da), 4),
                'p-value': round(float(binom_p), 4),
                'Significant': 'Yes' if binom_p < 0.05 else 'No'
            })
            
    stat_df = pd.DataFrame(stat_records)
    stat_df.to_csv(config.STATISTICAL_TESTS_PATH, index=False)
    logger.info(f"Saved Statistical tests to: {config.STATISTICAL_TESTS_PATH}")
    logger.info("All post-run research analyses completed successfully!")

if __name__ == "__main__":
    run_analysis()
