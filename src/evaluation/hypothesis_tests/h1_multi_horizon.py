import os
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import common, hac, bootstrap, multiple_testing
except (ImportError, ValueError):
    import common, hac, bootstrap, multiple_testing


# 4 independent contrast restrictions across 5 horizons
# H0: mu_1D = mu_5D = mu_20D = mu_60D = mu_120D
R_CONTRAST = np.array([
    [1.0, -1.0,  0.0,  0.0,  0.0],
    [0.0,  1.0, -1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0, -1.0,  0.0],
    [0.0,  0.0,  0.0,  1.0, -1.0]
], dtype=float)

def get_model_loss_matrix(df_diff, model_name):
    """
    Pivots loss differentials for model_name into shape (T, 5) aligned by Date.
    """
    sub = df_diff[df_diff['Model'] == model_name].copy()
    piv = sub.pivot(index='Date', columns='Horizon', values='d_t')
    piv = piv[common.HORIZONS].dropna()
    return piv

def run_h1_hac_wald(df_diff, bandwidth=20):
    """
    Runs multi-horizon HAC Wald test for all 12 models.
    """
    rows = []
    for model in common.ALL_MODELS:
        piv = get_model_loss_matrix(df_diff, model)
        X = piv.values # (T, 5)
        T, K = X.shape
        
        theta_hat = np.mean(X, axis=0)
        omega = hac.newey_west_covariance(X, bandwidth=bandwidth, kernel='bartlett')
        V_hat = omega / T
        
        res = hac.wald_test(theta_hat, V_hat, R=R_CONTRAST)
        
        rows.append({
            'Model': model,
            'T': T,
            'Bandwidth': bandwidth,
            'df': res['df'],
            'Wald_Stat': res['stat'],
            'pvalue': res['pvalue'],
            'mean_1D': theta_hat[0],
            'mean_5D': theta_hat[1],
            'mean_20D': theta_hat[2],
            'mean_60D': theta_hat[3],
            'mean_120D': theta_hat[4]
        })
        
    df_res = pd.DataFrame(rows)
    df_res = multiple_testing.adjust_dataframe_column(df_res, pval_col='pvalue', adj_col='pvalue_holm')
    df_res['Significant_05'] = df_res['pvalue_holm'] < 0.05
    return df_res

def run_h1_bandwidth_sensitivity(df_diff, bandwidths=(5, 20, 60, 119)):
    """
    Runs HAC Wald test across multiple lag bandwidths.
    """
    all_rows = []
    for bw in bandwidths:
        df_bw = run_h1_hac_wald(df_diff, bandwidth=bw)
        all_rows.append(df_bw)
    return pd.concat(all_rows, ignore_index=True)

def run_h1_bootstrap(df_diff, block_length=60, n_boot=2000, bandwidth=20, seed=42):
    """
    Paired moving-block bootstrap test for H1.
    Under H0: all 5 horizons have equal mean.
    We center each column by its own mean and add the grand mean.
    """
    rows = []
    for model in common.ALL_MODELS:
        piv = get_model_loss_matrix(df_diff, model)
        X = piv.values # (T, 5)
        T, K = X.shape
        
        # Actual statistic
        theta_hat = np.mean(X, axis=0)
        omega_act = hac.newey_west_covariance(X, bandwidth=bandwidth)
        act_res = hac.wald_test(theta_hat, omega_act / T, R=R_CONTRAST)
        W_act = act_res['stat']
        
        # Enforce H0: subtract column means, add grand mean
        grand_mean = np.mean(X)
        X_h0 = X - theta_hat + grand_mean
        
        # Bootstrap
        b_stats = []
        for X_star in bootstrap.paired_block_bootstrap_matrix(X_h0, block_length=block_length, n_boot=n_boot, seed=seed):
            th_star = np.mean(X_star, axis=0)
            om_star = hac.newey_west_covariance(X_star, bandwidth=bandwidth)
            w_star = hac.wald_test(th_star, om_star / T, R=R_CONTRAST)['stat']
            b_stats.append(w_star)
            
        b_stats = np.array(b_stats)
        boot_pval = float(np.mean(b_stats >= W_act))
        
        rows.append({
            'Model': model,
            'T': T,
            'Block_Length': block_length,
            'N_Boot': n_boot,
            'Wald_Stat': W_act,
            'Asymptotic_pvalue': act_res['pvalue'],
            'Bootstrap_pvalue': boot_pval
        })
        
    df_res = pd.DataFrame(rows)
    df_res = multiple_testing.adjust_dataframe_column(df_res, pval_col='Bootstrap_pvalue', adj_col='Bootstrap_pvalue_holm')
    df_res['Significant_05'] = df_res['Bootstrap_pvalue_holm'] < 0.05
    return df_res
