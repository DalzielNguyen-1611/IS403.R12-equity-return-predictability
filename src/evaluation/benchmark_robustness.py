"""
benchmark_robustness.py
=======================
Benchmark Robustness Evaluation Module:
Implements Out-of-Sample (OOS) evaluation against alternative naive benchmarks:
1. Baseline: Expanding Historical Mean Benchmark (Campbell & Thompson, 2008)
2. Alternative: Zero-Return Benchmark (Welch & Goyal, 2008)

Mathematical Framework:
-----------------------
For a given forecasting model m and horizon h:
- Model Predictions: y_hat_{t,h}^m
- Historical Mean Benchmark: y_hat_{t,h}^{HM} = bar{y}_{train,h}
- Zero-Return Benchmark: y_hat_{t,h}^{Zero} = 0

Sum of Squared Errors (SSE):
- SSE_Model = sum_{t=1}^T (y_{t,h} - y_hat_{t,h}^m)^2
- SSE_HM    = sum_{t=1}^T (y_{t,h} - y_hat_{t,h}^{HM})^2
- SSE_Zero  = sum_{t=1}^T (y_{t,h} - 0)^2 = sum_{t=1}^T y_{t,h}^2

Out-of-Sample R-squared (OOS R2):
- OOS_R2_HM   = 1 - (SSE_Model / SSE_HM)
- OOS_R2_Zero = 1 - (SSE_Model / SSE_Zero)
- Delta_OOS_R2 = OOS_R2_Zero - OOS_R2_HM

Theoretical Relationship:
-------------------------
When the underlying asset exhibits a positive unconditional drift (bar{y} > 0),
SSE_Zero = sum y^2 > SSE_HM = sum (y - bar{y})^2.
Consequently, SSE_Model / SSE_Zero < SSE_Model / SSE_HM, which implies:
    OOS_R2_Zero > OOS_R2_HM
Thus, Historical Mean serves as a strictly more conservative benchmark than Zero Return.

Methodological Constraints:
---------------------------
- Zero retraining: Reuses exact pre-computed predictions from Lan01 and Lan02.
- Strict evaluation sample parity: Evaluates all benchmarks on identical dates and horizons.
- Hypothesis testing: Re-uses Diebold-Mariano (1995) test with HLN (1997) finite-sample correction.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup repository paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.evaluation.hypothesis_tests import common
except (ImportError, ValueError):
    import common

# Forecasting horizon step mapping for DM autocorrelation bandwidth
HORIZON_TO_STEPS = {
    '1D': 1,
    '5D': 5,
    '20D': 20,
    '60D': 60,
    '120D': 120
}

HORIZONS_ORDER = ['1D', '5D', '20D', '60D', '120D']
REGIMES_ORDER = ['All', 'Low', 'Normal', 'High']


# =============================================================================
# Statistical Testing: Diebold-Mariano with HLN (1997) Correction
# =============================================================================
def diebold_mariano_test(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_bench: np.ndarray,
    h: int = 1,
    loss: str = 'squared',
    alternative: str = 'greater'
) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Harvey, Leybourne, and Newbold (HLN, 1997) finite-sample correction.
    
    H0: Model and Benchmark have equal predictive accuracy.
    H1: Model is strictly more accurate than Benchmark (alternative='greater').
    
    Loss differential:
        d_t = Loss(bench)_t - Loss(model)_t
    Positive d_t indicates the candidate model incurs lower forecast loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)
    
    if loss == 'squared':
        d = (y_true - y_bench) ** 2 - (y_true - y_model) ** 2
    elif loss == 'absolute':
        d = np.abs(y_true - y_bench) - np.abs(y_true - y_model)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
        
    T = len(d)
    if T <= 1:
        return 0.0, 1.0
        
    mean_d = float(np.mean(d))
    gamma = [float(np.var(d, ddof=0))]
    
    # Autocovariance up to lag h - 1 for multi-step forecasts
    for lag in range(1, int(h)):
        if lag >= T:
            break
        cov_lag = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma.append(float(cov_lag))
        
    var_d = (gamma[0] + 2.0 * sum(gamma[1:])) / T
    if var_d <= 1e-14 or np.isnan(var_d):
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d)
    
    # HLN (1997) finite-sample adjustment factor
    h_int = int(h)
    adj_inner = (T + 1 - 2 * h_int + (h_int * (h_int - 1)) / T) / T
    hln_factor = np.sqrt(max(1e-6, adj_inner))
    dm_stat_adj = float(dm_stat * hln_factor)
    
    df = max(1, T - 1)
    if alternative == 'greater':
        p_val = float(1.0 - stats.t.cdf(dm_stat_adj, df=df))
    elif alternative == 'less':
        p_val = float(stats.t.cdf(dm_stat_adj, df=df))
    else:
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat_adj), df=df)))
        
    return dm_stat_adj, p_val


# =============================================================================
# Core Metric Computation Functions
# =============================================================================
def calc_benchmark_metrics_slice(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_hm: np.ndarray,
    horizon: str,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Computes comparative performance metrics for a single slice:
    - Model vs Historical Mean Benchmark
    - Model vs Zero-Return Benchmark (y_pred_zero = 0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)
    y_pred_hm = np.asarray(y_pred_hm, dtype=float)
    y_pred_zero = np.zeros_like(y_true)
    
    N = len(y_true)
    if N == 0:
        return {}
        
    # Errors & SSE
    err_model = y_true - y_pred_model
    err_hm = y_true - y_pred_hm
    err_zero = y_true - y_pred_zero # equal to y_true
    
    sse_model = float(np.sum(err_model ** 2))
    sse_hm = float(np.sum(err_hm ** 2))
    sse_zero = float(np.sum(err_zero ** 2))
    
    # MAE & RMSE
    mae_model = float(mean_absolute_error(y_true, y_pred_model))
    mae_hm = float(mean_absolute_error(y_true, y_pred_hm))
    mae_zero = float(mean_absolute_error(y_true, y_pred_zero))
    
    rmse_model = float(np.sqrt(mean_squared_error(y_true, y_pred_model)))
    rmse_hm = float(np.sqrt(mean_squared_error(y_true, y_pred_hm)))
    rmse_zero = float(np.sqrt(mean_squared_error(y_true, y_pred_zero)))
    
    # Out-of-Sample R2 (Campbell & Thompson 2008 definition)
    oos_r2_hm = float(1.0 - (sse_model / (sse_hm + eps)))
    oos_r2_zero = float(1.0 - (sse_model / (sse_zero + eps)))
    delta_oos_r2 = float(oos_r2_zero - oos_r2_hm)
    
    # Diebold-Mariano tests (HLN-corrected)
    h_step = HORIZON_TO_STEPS.get(horizon, 1)
    dm_stat_hm, dm_p_hm = diebold_mariano_test(y_true, y_pred_model, y_pred_hm, h=h_step)
    dm_stat_zero, dm_p_zero = diebold_mariano_test(y_true, y_pred_model, y_pred_zero, h=h_step)
    
    return {
        'N_Obs': N,
        'SSE_Model': sse_model,
        'SSE_HM': sse_hm,
        'SSE_Zero': sse_zero,
        'MAE_Model': mae_model,
        'MAE_HM': mae_hm,
        'MAE_Zero': mae_zero,
        'RMSE_Model': rmse_model,
        'RMSE_HM': rmse_hm,
        'RMSE_Zero': rmse_zero,
        'OOS_R2_HM': oos_r2_hm,
        'OOS_R2_Zero': oos_r2_zero,
        'Delta_OOS_R2': delta_oos_r2,
        'DM_Stat_HM': dm_stat_hm,
        'DM_pvalue_HM': dm_p_hm,
        'DM_Sig_HM_05': 'Yes' if dm_p_hm < 0.05 else 'No',
        'DM_Stat_Zero': dm_stat_zero,
        'DM_pvalue_Zero': dm_p_zero,
        'DM_Sig_Zero_05': 'Yes' if dm_p_zero < 0.05 else 'No',
    }


# =============================================================================
# Comprehensive Evaluation Across Models, Horizons, and Regimes
# =============================================================================
def evaluate_benchmark_robustness(
    df_preds: pd.DataFrame,
    models: Optional[List[str]] = None,
    horizons: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates Benchmark Robustness across all candidate models and horizons.
    
    Returns:
    - df_detailed: Performance across All sample + Volatility Regimes (Low, Normal, High).
    - df_summary: Performance across All sample (clean unsegmented comparison).
    """
    if models is None:
        models = [m for m in common.ALL_MODELS if m in df_preds['Model'].unique()]
    if horizons is None:
        horizons = [h for h in HORIZONS_ORDER if h in df_preds['Horizon'].unique()]
        
    # Separate Historical Mean benchmark
    hm_df = df_preds[df_preds['Model'] == common.BENCHMARK_MODEL].copy()
    if len(hm_df) == 0:
        raise ValueError(f"Historical Mean benchmark ('{common.BENCHMARK_MODEL}') not found in predictions DataFrame.")
        
    detailed_rows = []
    
    for h in horizons:
        hm_h = hm_df[hm_df['Horizon'] == h].copy()
        
        for m in models:
            m_df = df_preds[(df_preds['Model'] == m) & (df_preds['Horizon'] == h)].copy()
            if len(m_df) == 0:
                continue
                
            # Align by Date to enforce strict sample parity
            aligned = pd.merge(
                m_df[['Date', 'Actual', 'Prediction', 'Regime']].rename(columns={'Prediction': 'Pred_Model'}),
                hm_h[['Date', 'Prediction']].rename(columns={'Prediction': 'Pred_HM'}),
                on='Date',
                how='inner'
            ).sort_values('Date').reset_index(drop=True)
            
            if len(aligned) == 0:
                if logger:
                    logger.warning(f"No aligned dates found for Model={m}, Horizon={h}")
                continue
                
            # 1. Full Sample Evaluation ('All')
            res_all = calc_benchmark_metrics_slice(
                y_true=aligned['Actual'].values,
                y_pred_model=aligned['Pred_Model'].values,
                y_pred_hm=aligned['Pred_HM'].values,
                horizon=h
            )
            detailed_rows.append({
                'Model': m,
                'Horizon': h,
                'Regime': 'All',
                **res_all
            })
            
            # 2. Regime-Conditional Evaluations ('Low', 'Normal', 'High')
            if 'Regime' in aligned.columns:
                for reg in ['Low', 'Normal', 'High']:
                    sub_reg = aligned[aligned['Regime'] == reg]
                    if len(sub_reg) > 0:
                        res_reg = calc_benchmark_metrics_slice(
                            y_true=sub_reg['Actual'].values,
                            y_pred_model=sub_reg['Pred_Model'].values,
                            y_pred_hm=sub_reg['Pred_HM'].values,
                            horizon=h
                        )
                        detailed_rows.append({
                            'Model': m,
                            'Horizon': h,
                            'Regime': reg,
                            **res_reg
                        })
                        
    df_detailed = pd.DataFrame(detailed_rows)
    
    # Summary dataframe: All sample only
    df_summary = df_detailed[df_detailed['Regime'] == 'All'].copy().reset_index(drop=True)
    
    return df_detailed, df_summary


# =============================================================================
# Pivot Matrix Generation for Research Paper Tables
# =============================================================================
def generate_comparison_matrices(
    df_summary: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Constructs 12x5 pivot matrices formatted for academic tables:
    - OOS R2 under Historical Mean Benchmark
    - OOS R2 under Zero-Return Benchmark
    - Delta OOS R2 (Zero - HM)
    """
    matrices = {}
    
    # 1. Historical Mean OOS R2 Matrix
    piv_hm = df_summary.pivot(index='Model', columns='Horizon', values='OOS_R2_HM')
    piv_hm = piv_hm.reindex(index=common.ALL_MODELS, columns=HORIZONS_ORDER)
    matrices['oos_r2_historical_mean'] = piv_hm
    
    # 2. Zero-Return OOS R2 Matrix
    piv_zero = df_summary.pivot(index='Model', columns='Horizon', values='OOS_R2_Zero')
    piv_zero = piv_zero.reindex(index=common.ALL_MODELS, columns=HORIZONS_ORDER)
    matrices['oos_r2_zero_return'] = piv_zero
    
    # 3. Delta OOS R2 Matrix
    piv_delta = df_summary.pivot(index='Model', columns='Horizon', values='Delta_OOS_R2')
    piv_delta = piv_delta.reindex(index=common.ALL_MODELS, columns=HORIZONS_ORDER)
    matrices['delta_oos_r2'] = piv_delta
    
    # 4. DM p-value comparison matrices
    piv_p_hm = df_summary.pivot(index='Model', columns='Horizon', values='DM_pvalue_HM')
    piv_p_hm = piv_p_hm.reindex(index=common.ALL_MODELS, columns=HORIZONS_ORDER)
    matrices['dm_pvalue_historical_mean'] = piv_p_hm
    
    piv_p_zero = df_summary.pivot(index='Model', columns='Horizon', values='DM_pvalue_Zero')
    piv_p_zero = piv_p_zero.reindex(index=common.ALL_MODELS, columns=HORIZONS_ORDER)
    matrices['dm_pvalue_zero_return'] = piv_p_zero
    
    return matrices


def generate_regime_breakdown_matrix(
    df_detailed: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes cross-model average OOS R2 across Volatility Regimes under both benchmarks.
    """
    sub = df_detailed[df_detailed['Regime'].isin(['Low', 'Normal', 'High'])].copy()
    
    grouped = sub.groupby(['Horizon', 'Regime'])[['OOS_R2_HM', 'OOS_R2_Zero', 'Delta_OOS_R2']].mean().reset_index()
    grouped['Horizon'] = pd.Categorical(grouped['Horizon'], categories=HORIZONS_ORDER, ordered=True)
    grouped['Regime'] = pd.Categorical(grouped['Regime'], categories=['Low', 'Normal', 'High'], ordered=True)
    grouped = grouped.sort_values(['Horizon', 'Regime']).reset_index(drop=True)
    return grouped


# =============================================================================
# Validation Folds Evaluation (Optional)
# =============================================================================
def evaluate_validation_folds(
    df_val_preds: pd.DataFrame,
    models: Optional[List[str]] = None,
    horizons: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Evaluates Benchmark Robustness across 5 Walk-Forward Validation folds.
    """
    if 'Fold' not in df_val_preds.columns:
        raise ValueError("Column 'Fold' required for validation folds evaluation.")
        
    if models is None:
        models = [m for m in common.ALL_MODELS if m in df_val_preds['Model'].unique()]
    if horizons is None:
        horizons = [h for h in HORIZONS_ORDER if h in df_val_preds['Horizon'].unique()]
        
    hm_df = df_val_preds[df_val_preds['Model'] == common.BENCHMARK_MODEL].copy()
    
    val_rows = []
    folds = sorted(df_val_preds['Fold'].unique())
    
    for f in folds:
        hm_f = hm_df[hm_df['Fold'] == f]
        for h in horizons:
            hm_fh = hm_f[hm_f['Horizon'] == h]
            for m in models:
                sub_m = df_val_preds[(df_val_preds['Fold'] == f) & (df_val_preds['Model'] == m) & (df_val_preds['Horizon'] == h)]
                if len(sub_m) == 0:
                    continue
                    
                aligned = pd.merge(
                    sub_m[['Date', 'Actual', 'Prediction']].rename(columns={'Prediction': 'Pred_Model'}),
                    hm_fh[['Date', 'Prediction']].rename(columns={'Prediction': 'Pred_HM'}),
                    on='Date',
                    how='inner'
                )
                if len(aligned) == 0:
                    continue
                    
                res = calc_benchmark_metrics_slice(
                    y_true=aligned['Actual'].values,
                    y_pred_model=aligned['Pred_Model'].values,
                    y_pred_hm=aligned['Pred_HM'].values,
                    horizon=h
                )
                val_rows.append({
                    'Fold': f,
                    'Model': m,
                    'Horizon': h,
                    **res
                })
                
    return pd.DataFrame(val_rows)

