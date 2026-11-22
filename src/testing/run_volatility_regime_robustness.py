"""
run_volatility_regime_robustness.py
===================================
Volatility Regime Robustness Evaluation:
Compares Out-of-Sample (OOS) performance, Giacomini-White Conditional Predictive Ability (CPA) tests,
and Horizon-Regime Interaction tests across alternative volatility rolling windows: 10D, 20D, and 60D.

Key Methodological Design:
1. Regime Classification:
   - For each window W in {10D, 20D, 60D}, volatility is computed as the rolling standard deviation
     of daily log returns: Volatility_WD = std(Return_1D, window=W).
   - Expanding Terciles: Thresholds q33(t) and q67(t) are computed via expanding window (min_periods=252)
     without .bfill() to strictly eliminate forward-looking data leakage.
   - Regimes: Low (vol <= q33), Normal (q33 < vol <= q67), High (vol > q67).
2. Model Predictions:
   - Reuses existing holdout predictions from Lan01 and Lan02 (12 models x 5 horizons x 1,362 days).
   - Zero retraining required: model forecasts y_hat_{t,h} remain identical.
3. Statistical Testing:
   - Re-evaluates Giacomini-White (2006) CPA tests with Newey-West HAC (bandwidth=20) and Holm adjustment.
   - Re-evaluates joint Horizon-Regime Interaction tests with Newey-West HAC and Holm adjustment.
4. Output Isolation:
   - Saves all results under results/volatility_robustness/ without overwriting baseline H2 outputs.

Usage:
    python src/testing/run_volatility_regime_robustness.py
    python src/testing/run_volatility_regime_robustness.py --windows 10 20 60
    python src/testing/run_volatility_regime_robustness.py --check-only
"""

import os
import sys
import argparse
import logging
from typing import Any
import numpy as np
import pandas as pd

# Setup repository paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.hypothesis_tests import common, hac, multiple_testing, h2_regime_cpa

# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "volatility_robustness_execution.log")
    
    logger = logging.getLogger("VolatilityRobustness")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear existing
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

# -------------------------------------------------------------------------
# Step 1: Compute Volatility Series & Expanding Tercile Regimes
# -------------------------------------------------------------------------
def generate_regime_labels(feature_path=None, return_path=None, min_periods=252, logger=None):
    """
    Constructs 10D, 20D, and 60D historical volatility and corresponding expanding-tercile
    regime classifications without .bfill() to eliminate look-ahead leakage.
    """
    if feature_path is None:
        feature_path = os.path.join(REPO_ROOT, 'dataset', '2.2.0_feature.csv')
    if return_path is None:
        return_path = os.path.join(REPO_ROOT, 'dataset', '2.0.0_return.csv')
        
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature dataset not found: {feature_path}")
        
    df = pd.read_csv(feature_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure Volatility_10D exists; compute from Return_1D if missing
    if 'Volatility_10D' not in df.columns:
        if 'Return_1D' in df.columns:
            df['Volatility_10D'] = df['Return_1D'].rolling(window=10).std()
        elif os.path.exists(return_path):
            ret_df = pd.read_csv(return_path)
            ret_df['Date'] = pd.to_datetime(ret_df['Date'])
            ret_df['Volatility_10D'] = ret_df['Return_1D'].rolling(window=10).std()
            df = pd.merge(df, ret_df[['Date', 'Volatility_10D']], on='Date', how='left')
        else:
            raise ValueError("Return_1D not found to compute Volatility_10D.")
            
    # Dictionary of volatility columns
    vol_cols = {
        10: 'Volatility_10D',
        20: 'Volatility_20D',
        60: 'Volatility_60D'
    }
    
    regime_results = pd.DataFrame({'Date': df['Date']})
    
    for w, col in vol_cols.items():
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in feature dataset.")
            
        vol_series = df[col]
        regime_results[col] = vol_series
        
        # Expanding terciles with min_periods = 252 (1 trading year)
        # CRITICAL: NO .bfill() used. Initial observations before min_periods are NaN.
        q33 = vol_series.expanding(min_periods=min_periods).quantile(0.33)
        q67 = vol_series.expanding(min_periods=min_periods).quantile(0.67)
        
        regime_results[f'Q33_{w}D'] = q33
        regime_results[f'Q67_{w}D'] = q67
        
        reg_series = pd.Series(index=df.index, dtype=object)
        reg_series[vol_series <= q33] = 'Low'
        reg_series[(vol_series > q33) & (vol_series <= q67)] = 'Normal'
        reg_series[vol_series > q67] = 'High'
        regime_results[f'Regime_{w}D'] = reg_series
        
    if logger:
        logger.info(f"Generated expanding tercile regime labels for {len(regime_results)} trading days.")
        for w in [10, 20, 60]:
            valid_cnt = regime_results[f'Regime_{w}D'].notnull().sum()
            logger.info(f"  Window {w:2d}D: Valid classified days = {valid_cnt} (Initial {min_periods} days strictly unclassified)")
            
    return regime_results

# -------------------------------------------------------------------------
# Step 2: Compute Within-Regime Forecast Performance
# -------------------------------------------------------------------------
def compute_within_regime_metrics(df_preds, regime_col='Regime'):
    """
    Computes within-regime evaluation metrics for each (Model, Horizon, Regime):
    - OOS R^2 relative to Historical Mean within the same regime
    - Directional Accuracy (DA)
    - MAE and RMSE
    - Mean Skill and Std Skill of loss differential d_{t,h,m}
    """
    models = [m for m in common.ALL_MODELS if m in df_preds['Model'].unique()]
    bench_model = common.BENCHMARK_MODEL
    horizons = common.HORIZONS
    regimes = ['Low', 'Normal', 'High']
    
    bench_df = df_preds[df_preds['Model'] == bench_model].copy()
    
    # Compute benchmark SSE per (Horizon, Regime)
    bench_stats = {}
    for h in horizons:
        b_h = bench_df[bench_df['Horizon'] == h]
        for r in regimes:
            b_hr = b_h[b_h[regime_col] == r]
            if len(b_hr) > 0:
                sse_hm = float(np.sum((b_hr['Actual'].values - b_hr['Prediction'].values) ** 2))
                bench_stats[(h, r)] = {
                    'sse_hm': sse_hm,
                    'n_obs': len(b_hr)
                }
            else:
                bench_stats[(h, r)] = {'sse_hm': np.nan, 'n_obs': 0}
                
    # Also overall benchmark MSE per horizon (for loss differential d_t)
    bench_mse_h = {}
    for h in horizons:
        b_h = bench_df[bench_df['Horizon'] == h]
        bench_mse_h[h] = float(np.mean((b_h['Actual'].values - b_h['Prediction'].values) ** 2))
        
    records = []
    matrix_rows = []
    
    for m in models:
        m_df = df_preds[df_preds['Model'] == m]
        for h in horizons:
            sub_mh = m_df[m_df['Horizon'] == h]
            sub_bh = bench_df[bench_df['Horizon'] == h]
            
            piv_entry: dict[str, Any] = {'Model': m, 'Horizon': h}
            
            for r in regimes:
                sub_r = sub_mh[sub_mh[regime_col] == r]
                sub_br = sub_bh[sub_bh[regime_col] == r]
                
                n_obs = len(sub_r)
                if n_obs > 0:
                    y_true = sub_r['Actual'].values
                    y_pred = sub_r['Prediction'].values
                    y_bench = sub_br['Prediction'].values
                    
                    sse_m = float(np.sum((y_true - y_pred) ** 2))
                    sse_hm = bench_stats[(h, r)]['sse_hm']
                    
                    oos_r2 = (1.0 - (sse_m / (sse_hm + 1e-12))) * 100.0
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                    da = float(np.mean(np.where(y_true > 0, 1, -1) == np.where(y_pred > 0, 1, -1))) * 100.0
                    
                    # Normalized loss differential d_t = (e_sq_hm - e_sq_m) / MSE_HM_h
                    e_sq_hm = (y_true - y_bench) ** 2
                    e_sq_m = (y_true - y_pred) ** 2
                    d_t = (e_sq_hm - e_sq_m) / bench_mse_h[h]
                    mean_skill = float(np.mean(d_t))
                    std_skill = float(np.std(d_t, ddof=1)) if n_obs > 1 else 0.0
                else:
                    oos_r2 = np.nan
                    mae = np.nan
                    rmse = np.nan
                    da = np.nan
                    mean_skill = np.nan
                    std_skill = np.nan
                    
                records.append({
                    'Model': m,
                    'Horizon': h,
                    'Regime': r,
                    'N_obs': n_obs,
                    'OOS_R2_pct': round(oos_r2, 4),
                    'MAE': round(mae, 6),
                    'RMSE': round(rmse, 6),
                    'DA_pct': round(da, 2),
                    'Mean_Skill': round(mean_skill, 6),
                    'Std_Skill': round(std_skill, 6)
                })
                
                piv_entry[f'{r}_OOS_R2'] = round(oos_r2, 2)
                piv_entry[f'{r}_DA'] = round(da, 2)
                
            matrix_rows.append(piv_entry)
            
    df_details = pd.DataFrame(records)
    df_matrix = pd.DataFrame(matrix_rows)
    return df_details, df_matrix

# -------------------------------------------------------------------------
# Step 3: Run Full Robustness Evaluation Across Windows
# -------------------------------------------------------------------------
def run_volatility_regime_robustness(windows=[10, 20, 60], output_dir=None, check_only=False):
    """
    Main driver executing the Volatility Regime Robustness experiment.
    """
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'results', 'volatility_robustness')
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info("=====================================================================")
    logger.info("  STARTING VOLATILITY REGIME ROBUSTNESS EVALUATION (10D vs 20D vs 60D)")
    logger.info("=====================================================================")
    logger.info(f"Target Rolling Windows: {windows}")
    logger.info(f"Output Directory: {output_dir}")
    
    # 1. Load predictions
    logger.info("Loading unified holdout predictions (2020-11-02 to 2026-04-07)...")
    preds_df = common.load_unified_predictions(split='test')
    logger.info(f"Loaded {len(preds_df)} prediction records covering {preds_df['Date'].nunique()} holdout dates.")
    
    # 2. Generate regime labels for all windows
    logger.info("Constructing expanding tercile regime classifications without .bfill()...")
    regimes_df = generate_regime_labels(min_periods=252, logger=logger)
    
    # Save regime master file
    regimes_master_path = os.path.join(output_dir, 'regime_labels_10d_20d_60d.csv')
    regimes_df.to_csv(regimes_master_path, index=False)
    logger.info(f"Saved master regime dataset -> {regimes_master_path}")
    
    # Merge regime labels with test predictions by Date
    merged_preds = pd.merge(
        preds_df.drop(columns=['Regime'], errors='ignore'),
        regimes_df[['Date'] + [f'Regime_{w}D' for w in windows] + [f'Volatility_{w}D' for w in windows]],
        on='Date',
        how='left'
    )
    
    if check_only:
        logger.info("Syntax and data check completed successfully (--check-only flag active).")
        return
        
    # Containers for comparative synthesis
    regime_counts_list = []
    oos_r2_comparison_dict = {}
    gw_cpa_results_dict = {}
    interaction_results_dict = {}
    
    # Extract distinct test dates with all regime columns for distributions & cross-tabulations
    all_regime_cols = [f'Regime_{w}D' for w in windows]
    test_dates_all_df = merged_preds[['Date'] + all_regime_cols].drop_duplicates()
    total_test_dates = len(test_dates_all_df)
    
    for w in windows:
        window_dir = os.path.join(output_dir, f'window_{w}d')
        os.makedirs(window_dir, exist_ok=True)
        regime_col = f'Regime_{w}D'
        
        logger.info("---------------------------------------------------------------------")
        logger.info(f"Evaluating Configuration: {w}-Day Volatility Window ({regime_col})")
        logger.info("---------------------------------------------------------------------")
        
        # Check regime distribution on the holdout test set
        counts = test_dates_all_df[regime_col].value_counts()
        
        cnt_record = {
            'Window': f'{w}D',
            'Low_N': int(counts.get('Low', 0)),
            'Low_Pct': round(float(counts.get('Low', 0) / total_test_dates * 100), 2),
            'Normal_N': int(counts.get('Normal', 0)),
            'Normal_Pct': round(float(counts.get('Normal', 0) / total_test_dates * 100), 2),
            'High_N': int(counts.get('High', 0)),
            'High_Pct': round(float(counts.get('High', 0) / total_test_dates * 100), 2),
            'Total_N': total_test_dates
        }
        regime_counts_list.append(cnt_record)
        logger.info(f"  Regime Breakdown ({w}D): Low={cnt_record['Low_N']} ({cnt_record['Low_Pct']}%), "
                    f"Normal={cnt_record['Normal_N']} ({cnt_record['Normal_Pct']}%), "
                    f"High={cnt_record['High_N']} ({cnt_record['High_Pct']}%)")
                    
        # Assign current regime column as standard 'Regime' for compatibility with testing functions
        sub_preds = merged_preds.copy()
        sub_preds['Regime'] = sub_preds[regime_col]
        
        # 1. Within-Regime Performance Metrics
        df_details, df_matrix = compute_within_regime_metrics(sub_preds, regime_col=regime_col)
        details_path = os.path.join(window_dir, f'h2_regime_performance_{w}d.csv')
        matrix_path = os.path.join(window_dir, f'h2_regime_matrix_{w}d.csv')
        df_details.to_csv(details_path, index=False)
        df_matrix.to_csv(matrix_path, index=False)
        logger.info(f"  Saved within-regime performance -> {details_path}")
        
        # Store for comparison
        oos_r2_comparison_dict[f'{w}D'] = df_matrix
        
        # 2. Loss Differentials
        diff_df, _ = common.compute_loss_differentials(sub_preds)
        
        # 3. Giacomini-White CPA Test
        logger.info(f"  Running Giacomini-White CPA tests for {w}D (bandwidth=20, Holm correction)...")
        gw_res = h2_regime_cpa.run_h2_gw_cpa(diff_df, bandwidth=20)
        gw_path = os.path.join(window_dir, f'h2_gw_cpa_results_{w}d.csv')
        gw_res.to_csv(gw_path, index=False)
        gw_cpa_results_dict[f'{w}D'] = gw_res
        sig_gw = gw_res['Significant_05'].sum()
        logger.info(f"  Saved GW CPA test results -> {gw_path} (Significant at 5% after Holm: {sig_gw}/60)")
        
        # 4. Joint Horizon-Regime Interaction Test
        logger.info(f"  Running Horizon-Regime Interaction tests for {w}D (8 df restrictions, Holm correction)...")
        int_res = h2_regime_cpa.run_h2_horizon_regime_interaction(diff_df, bandwidth=20)
        int_path = os.path.join(window_dir, f'h2_horizon_regime_interaction_{w}d.csv')
        int_res.to_csv(int_path, index=False)
        interaction_results_dict[f'{w}D'] = int_res
        sig_int = int_res['Significant_05'].sum()
        logger.info(f"  Saved Interaction test results -> {int_path} (Significant at 5% after Holm: {sig_int}/12)")
        
    # ---------------------------------------------------------------------
    # Step 4: Cross-Window Comparative Synthesis
    # ---------------------------------------------------------------------
    logger.info("=====================================================================")
    logger.info("  BUILDING CROSS-WINDOW COMPARATIVE TABLES AND SYNTHESIS REPORT")
    logger.info("=====================================================================")
    comp_dir = os.path.join(output_dir, 'comparison_summary')
    os.makedirs(comp_dir, exist_ok=True)
    
    # 1. Regime counts comparison
    counts_df = pd.DataFrame(regime_counts_list)
    counts_path = os.path.join(comp_dir, 'regime_counts_comparison.csv')
    counts_df.to_csv(counts_path, index=False)
    logger.info(f"Saved regime counts comparison -> {counts_path}")
    
    # 2. Cross-tabulation Agreement Matrices
    cross_tabs = []
    for w1, w2 in [(10, 20), (20, 60), (10, 60)]:
        ct = pd.crosstab(
            test_dates_all_df[f'Regime_{w1}D'],
            test_dates_all_df[f'Regime_{w2}D'],
            rownames=[f'Window_{w1}D'],
            colnames=[f'Window_{w2}D']
        )
        ct_path = os.path.join(comp_dir, f'agreement_matrix_{w1}d_vs_{w2}d.csv')
        ct.to_csv(ct_path)
        logger.info(f"Saved agreement cross-tabulation -> {ct_path}")
        
    # 3. Side-by-side OOS R^2 Comparison
    comp_r2_rows = []
    base_10 = oos_r2_comparison_dict['10D'].set_index(['Model', 'Horizon'])
    base_20 = oos_r2_comparison_dict['20D'].set_index(['Model', 'Horizon'])
    base_60 = oos_r2_comparison_dict['60D'].set_index(['Model', 'Horizon'])
    
    for idx in base_20.index:
        m, h = idx
        r10 = base_10.loc[idx]
        r20 = base_20.loc[idx]
        r60 = base_60.loc[idx]
        
        comp_r2_rows.append({
            'Model': m,
            'Horizon': h,
            'Low_10D': r10['Low_OOS_R2'],
            'Low_20D': r20['Low_OOS_R2'],
            'Low_60D': r60['Low_OOS_R2'],
            'Normal_10D': r10['Normal_OOS_R2'],
            'Normal_20D': r20['Normal_OOS_R2'],
            'Normal_60D': r60['Normal_OOS_R2'],
            'High_10D': r10['High_OOS_R2'],
            'High_20D': r20['High_OOS_R2'],
            'High_60D': r60['High_OOS_R2'],
        })
    comp_r2_df = pd.DataFrame(comp_r2_rows)
    comp_r2_path = os.path.join(comp_dir, 'regime_oos_r2_comparison.csv')
    comp_r2_df.to_csv(comp_r2_path, index=False)
    logger.info(f"Saved side-by-side OOS R^2 comparison -> {comp_r2_path}")
    
    # 4. Side-by-side GW CPA Test Comparison
    comp_gw_rows = []
    gw_10 = gw_cpa_results_dict['10D'].set_index(['Model', 'Horizon'])
    gw_20 = gw_cpa_results_dict['20D'].set_index(['Model', 'Horizon'])
    gw_60 = gw_cpa_results_dict['60D'].set_index(['Model', 'Horizon'])
    
    for idx in gw_20.index:
        m, h = idx
        g10 = gw_10.loc[idx]
        g20 = gw_20.loc[idx]
        g60 = gw_60.loc[idx]
        
        comp_gw_rows.append({
            'Model': m,
            'Horizon': h,
            'Stat_10D': round(g10['CPA_Stat'], 4),
            'pHolm_10D': round(g10['pvalue_holm'], 4),
            'Sig_10D': bool(g10['Significant_05']),
            'Stat_20D': round(g20['CPA_Stat'], 4),
            'pHolm_20D': round(g20['pvalue_holm'], 4),
            'Sig_20D': bool(g20['Significant_05']),
            'Stat_60D': round(g60['CPA_Stat'], 4),
            'pHolm_60D': round(g60['pvalue_holm'], 4),
            'Sig_60D': bool(g60['Significant_05']),
        })
    comp_gw_df = pd.DataFrame(comp_gw_rows)
    comp_gw_path = os.path.join(comp_dir, 'gw_cpa_significance_comparison.csv')
    comp_gw_df.to_csv(comp_gw_path, index=False)
    logger.info(f"Saved GW CPA test comparison -> {comp_gw_path}")
    
    # 5. Side-by-side Horizon-Regime Interaction Test Comparison
    comp_int_rows = []
    int_10 = interaction_results_dict['10D'].set_index('Model')
    int_20 = interaction_results_dict['20D'].set_index('Model')
    int_60 = interaction_results_dict['60D'].set_index('Model')
    
    for m in int_20.index:
        i10 = int_10.loc[m]
        i20 = int_20.loc[m]
        i60 = int_60.loc[m]
        
        comp_int_rows.append({
            'Model': m,
            'Wald_10D': round(i10['Interaction_Wald_Stat'], 4),
            'pHolm_10D': round(i10['pvalue_holm'], 4),
            'Sig_10D': bool(i10['Significant_05']),
            'Wald_20D': round(i20['Interaction_Wald_Stat'], 4),
            'pHolm_20D': round(i20['pvalue_holm'], 4),
            'Sig_20D': bool(i20['Significant_05']),
            'Wald_60D': round(i60['Interaction_Wald_Stat'], 4),
            'pHolm_60D': round(i60['pvalue_holm'], 4),
            'Sig_60D': bool(i60['Significant_05']),
        })
    comp_int_df = pd.DataFrame(comp_int_rows)
    comp_int_path = os.path.join(comp_dir, 'interaction_test_comparison.csv')
    comp_int_df.to_csv(comp_int_path, index=False)
    logger.info(f"Saved Interaction test comparison -> {comp_int_path}")
    
    # 6. Generate Comprehensive Markdown Report
    report_path = os.path.join(output_dir, 'volatility_regime_robustness_report.md')
    generate_markdown_report(
        report_path, counts_df, comp_r2_df, comp_gw_df, comp_int_df, windows
    )
    logger.info(f"Saved Comprehensive Synthesis Report -> {report_path}")
    logger.info("=====================================================================")
    logger.info("  VOLATILITY REGIME ROBUSTNESS EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=====================================================================")

# -------------------------------------------------------------------------
# Step 5: Report Generator
# -------------------------------------------------------------------------
def generate_markdown_report(report_path, counts_df, r2_df, gw_df, int_df, windows):
    """Generates an academic markdown synthesis of the robustness findings."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Volatility Regime Robustness Report: 10D, 20D, and 60D Window Comparison\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This report examines whether the empirical findings for **Research Question 2 (RQ2)** ")
        f.write("are robust to the choice of the historical volatility window used to define market volatility regimes. ")
        f.write("The baseline specification employs a **20-day window (20D)**. This evaluation compares the baseline ")
        f.write("against two alternative configurations: a shorter **10-day window (10D)** and a longer **60-day window (60D)**.\n\n")
        f.write("- **Methodology:** Expanding terciles (q33, q67) with `min_periods = 252` trading days, strictly avoiding forward-looking leakage (`.bfill()` omitted).\n")
        f.write("- **Data Scope:** 1,362 out-of-sample holdout trading days (2020-11-02 to 2026-04-07).\n")
        f.write("- **Model Scope:** 12 models across 5 forecasting horizons (1D, 5D, 20D, 60D, 120D).\n\n")
        
        f.write("## 2. Regime Distribution on the Holdout Sample\n\n")
        f.write(counts_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Horizon-Regime Interaction Test Robustness (RQ2 Joint Test)\n\n")
        f.write("The table below reports the Wald statistic and Holm-adjusted p-values for the 8 linear interaction restrictions across windows:\n\n")
        f.write(int_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. Giacomini-White CPA Test Significance Comparison\n\n")
        sig_summary = []
        for w in windows:
            sig_cnt = gw_df[f'Sig_{w}D'].sum()
            sig_summary.append({'Window': f'{w}D', 'Significant_Models_at_5pct': f"{sig_cnt} / {len(gw_df)}"})
        f.write(pd.DataFrame(sig_summary).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 5. Detailed OOS R² Comparison Across Regimes\n\n")
        f.write(r2_df.head(25).to_markdown(index=False))
        f.write("\n\n*(Table truncated to first 25 rows for readability; see `comparison_summary/regime_oos_r2_comparison.csv` for the full table)*\n\n")

# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Volatility Regime Robustness Evaluation across 10D, 20D, and 60D.")
    parser.add_argument('--windows', nargs='+', type=int, default=[10, 20, 60],
                        help="List of volatility rolling window sizes in days (default: 10 20 60)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Target output directory (default: results/volatility_robustness)")
    parser.add_argument('--check-only', action='store_true',
                        help="Performs syntax and data sanity check without running the full evaluation.")
                        
    args = parser.parse_args()
    run_volatility_regime_robustness(windows=args.windows, output_dir=args.output_dir, check_only=args.check_only)
