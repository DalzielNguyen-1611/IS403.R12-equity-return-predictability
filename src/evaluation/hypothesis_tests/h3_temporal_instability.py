import os
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import common, hac, bootstrap, multiple_testing
except (ImportError, ValueError):
    import common, hac, bootstrap, multiple_testing


HOLDOUT_BLOCKS = {
    '2021-2022': ('2021-01-01', '2022-12-31'),
    '2023-2024': ('2023-01-01', '2024-12-31'),
    '2025-2026': ('2025-01-01', '2026-04-07')
}

VAL_BLOCKS = {
    '2015-2016': ('2015-06-10', '2016-12-31'),
    '2017-2018': ('2017-01-01', '2018-12-31'),
    '2019-2020': ('2019-01-01', '2020-10-30')
}

def run_h3_fluctuation_test(df_diff, window_size=252, bandwidth=20, n_boot=1000, seed=42):
    """
    Giacomini-Rossi (2010) style Fluctuation Test for Out-of-Sample Instability.
    Rolling standardized loss differential over window m:
        t_tau = sqrt(m) * mean_{s=tau-m+1}^tau (d_s) / sigma_HAC
    Supremum test statistic:
        S_T = max_{tau} |t_tau|
    Bootstrap null distribution obtained by resampling demeaned series (H0: constant skill).
    """
    rows = []
    fluctuation_series = []
    
    for h in common.HORIZONS:
        sub_h = df_diff[df_diff['Horizon'] == h]
        for model in common.ALL_MODELS:
            sub = sub_h[sub_h['Model'] == model].sort_values('Date').copy()
            d = sub['d_t'].values
            dates = sub['Date'].values
            T = len(d)
            
            if T < window_size:
                continue
                
            # Estimate full-sample HAC standard deviation of d
            omega = hac.newey_west_covariance(d[:, None], bandwidth=bandwidth)
            sigma_hac = np.sqrt(max(1e-8, float(omega[0, 0])))
            
            # Vectorized Rolling statistics using np.convolve
            kernel = np.ones(window_size) / window_size
            rolling_means = np.convolve(d, kernel, mode='valid')
            rolling_stats = np.sqrt(window_size) * (rolling_means / sigma_hac)
            valid_dates = dates[window_size - 1:]
            
            for dt, m_val, stat_val in zip(valid_dates, rolling_means, rolling_stats):
                fluctuation_series.append({
                    'Model': model,
                    'Horizon': h,
                    'Date': str(dt)[:10],
                    'Rolling_Mean': float(m_val),
                    't_stat': float(stat_val)
                })
                
            sup_stat = float(np.max(np.abs(rolling_stats)))
            
            # Bootstrap null p-value (demeaned d under H0: mean skill is constant)
            d_demeaned = d - np.mean(d)
            boot_sup = []
            for d_star in bootstrap.paired_block_bootstrap_matrix(d_demeaned[:, None], block_length=60, n_boot=n_boot, seed=seed):
                b_roll = np.convolve(d_star[:, 0], kernel, mode='valid')
                b_stats = np.sqrt(window_size) * (b_roll / sigma_hac)
                boot_sup.append(np.max(np.abs(b_stats)))
                
            boot_sup = np.array(boot_sup)
            boot_pval = float(np.mean(boot_sup >= sup_stat))
            
            rows.append({
                'Model': model,
                'Horizon': h,
                'T': T,
                'Window_Size': window_size,
                'Bandwidth': bandwidth,
                'Sup_Stat': sup_stat,
                'pvalue': boot_pval,
                'Mean_Skill': float(np.mean(d))
            })
            
    df_res = pd.DataFrame(rows)
    # Holm adjustment within each horizon family across models
    adj_chunks = []
    for h, chunk in df_res.groupby('Horizon'):
        adj = multiple_testing.adjust_dataframe_column(chunk, pval_col='pvalue', adj_col='pvalue_holm')
        adj['Significant_05'] = adj['pvalue_holm'] < 0.05
        adj_chunks.append(adj)
    return pd.concat(adj_chunks, ignore_index=True), pd.DataFrame(fluctuation_series)

def run_h3_block_wald_test(df_diff, block_dict, sample_name='Hold-out', bandwidth=20):
    """
    Supplementary Predefined Block-Level HAC Wald Test:
    H0: mu_{Block 1} = mu_{Block 2} = mu_{Block 3}
    2 independent restrictions on 3 blocks.
    """
    R_BLOCK = np.array([
        [1.0, -1.0,  0.0],
        [0.0,  1.0, -1.0]
    ], dtype=float)
    
    rows = []
    for h in common.HORIZONS:
        sub_h = df_diff[df_diff['Horizon'] == h]
        for model in common.ALL_MODELS:
            sub_m = sub_h[sub_h['Model'] == model].sort_values('Date').copy()
            sub_m['Date'] = pd.to_datetime(sub_m['Date'])
            
            # Assign block indicators
            b_means = []
            b_counts = []
            moments_list = []
            
            # Subset only data falling into predefined blocks
            valid_rows = []
            for b_name, (b_start, b_end) in block_dict.items():
                mask = (sub_m['Date'] >= b_start) & (sub_m['Date'] <= b_end)
                block_data = sub_m[mask]
                b_means.append(float(block_data['d_t'].mean()) if len(block_data) > 0 else np.nan)
                b_counts.append(len(block_data))
                
            # Construct indicator regressors on the concatenated block rows
            block_data_all = []
            for b_idx, (b_name, (b_start, b_end)) in enumerate(block_dict.items()):
                mask = (sub_m['Date'] >= b_start) & (sub_m['Date'] <= b_end)
                b_df = sub_m[mask].copy()
                b_df['Block_Idx'] = b_idx
                block_data_all.append(b_df)
                
            df_blocks = pd.concat(block_data_all, ignore_index=True)
            T_total = len(df_blocks)
            
            # Regression of d_t on 3 block dummy variables without intercept
            D = np.zeros((T_total, 3), dtype=float)
            for b_idx in range(3):
                D[:, b_idx] = (df_blocks['Block_Idx'].values == b_idx).astype(float)
                
            y = df_blocks['d_t'].values
            D_inv = np.linalg.inv(D.T @ D)
            theta_hat = D_inv @ (D.T @ y) # (3,) contains sample means of the 3 blocks
            residuals = y - (D @ theta_hat)
            
            M = D * residuals[:, None]
            omega_M = hac.newey_west_covariance(M, bandwidth=bandwidth)
            bread = D_inv * T_total
            V_theta = (bread @ omega_M @ bread) / T_total
            
            res = hac.wald_test(theta_hat, V_theta, R=R_BLOCK)
            
            row = {
                'Model': model,
                'Horizon': h,
                'Sample': sample_name,
                'T_Total': T_total,
                'df': res['df'],
                'Wald_Stat': res['stat'],
                'pvalue': res['pvalue'],
            }
            for b_idx, b_name in enumerate(block_dict.keys()):
                row[f'Mean_{b_name}'] = b_means[b_idx]
                row[f'N_{b_name}'] = b_counts[b_idx]
            rows.append(row)
            
    df_res = pd.DataFrame(rows)
    # Apply Holm adjustment within each Horizon family
    adj_chunks = []
    for h, chunk in df_res.groupby('Horizon'):
        adj = multiple_testing.adjust_dataframe_column(chunk, pval_col='pvalue', adj_col='pvalue_holm')
        adj['Significant_05'] = adj['pvalue_holm'] < 0.05
        adj_chunks.append(adj)
    return pd.concat(adj_chunks, ignore_index=True)
