import os
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import common, hac, bootstrap, multiple_testing
except (ImportError, ValueError):
    import common, hac, bootstrap, multiple_testing


REGIMES = ['Low', 'Normal', 'High']

def compute_regime_breakdown(df_diff):
    """
    Computes mean predictive skill S_{h,r,m} = E[d_{t,h,m} | R_t = r] and counts.
    """
    rows = []
    for model in common.ALL_MODELS:
        sub_m = df_diff[df_diff['Model'] == model]
        for h in common.HORIZONS:
            sub_h = sub_m[sub_m['Horizon'] == h]
            for r in REGIMES:
                sub_r = sub_h[sub_h['Regime'] == r]
                n_obs = len(sub_r)
                mean_skill = float(sub_r['d_t'].mean()) if n_obs > 0 else np.nan
                std_skill = float(sub_r['d_t'].std()) if n_obs > 1 else np.nan
                rows.append({
                    'Model': model,
                    'Horizon': h,
                    'Regime': r,
                    'N_obs': n_obs,
                    'Mean_Skill': mean_skill,
                    'Std_Skill': std_skill
                })
    return pd.DataFrame(rows)

def run_h2_gw_cpa(df_diff, bandwidth=20):
    """
    Giacomini-White (2006) Conditional Predictive Ability (CPA) Test.
    Instrument vector z_t = [1, I(Low), I(High)]' (Normal is reference).
    Moment vector: m_t = z_t * d_{t,h,m} in R^3.
    H0: E[m_t] = 0.
    """
    rows = []
    for h in common.HORIZONS:
        sub_h = df_diff[df_diff['Horizon'] == h]
        for model in common.ALL_MODELS:
            sub = sub_h[sub_h['Model'] == model].sort_values('Date').copy()
            T = len(sub)
            d = sub['d_t'].values
            
            z_low = (sub['Regime'] == 'Low').astype(float).values
            z_high = (sub['Regime'] == 'High').astype(float).values
            z_const = np.ones(T, dtype=float)
            
            # Instruments matrix Z (T, 3)
            Z = np.column_stack([z_const, z_low, z_high])
            
            # Moment functions (T, 3)
            M = Z * d[:, None]
            
            m_bar = np.mean(M, axis=0) # (3,)
            omega = hac.newey_west_covariance(M, bandwidth=bandwidth)
            V_hat = omega / T
            
            res = hac.wald_test(m_bar, V_hat) # H0: m_bar = 0 (3 df)
            
            rows.append({
                'Model': model,
                'Horizon': h,
                'T': T,
                'Bandwidth': bandwidth,
                'df': res['df'],
                'CPA_Stat': res['stat'],
                'pvalue': res['pvalue'],
                'Instruments': 'Constant, I(Low), I(High)'
            })
            
    df_res = pd.DataFrame(rows)
    # Apply Holm adjustment within each Horizon family across the 12 models
    adjusted_chunks = []
    for h, chunk in df_res.groupby('Horizon'):
        adj_chunk = multiple_testing.adjust_dataframe_column(chunk, pval_col='pvalue', adj_col='pvalue_holm')
        adj_chunk['Significant_05'] = adj_chunk['pvalue_holm'] < 0.05
        adjusted_chunks.append(adj_chunk)
    return pd.concat(adjusted_chunks, ignore_index=True)

def run_h2_horizon_regime_interaction(df_diff, bandwidth=20):
    """
    Joint Horizon-Regime Interaction Test:
    Tests whether the relative performance difference across regimes differs by horizon.
    For each model:
      d_{t,h} = alpha_h + beta_{h,Low} * I(Low_t) + beta_{h,High} * I(High_t) + u_{t,h}
    Null hypothesis:
      beta_{h,Low} = beta_{1D,Low}  and  beta_{h,High} = beta_{1D,High} for all h in {5D, 20D, 60D, 120D}
    Total 8 linear restrictions across the 5 horizons.
    Estimated using stacked moment equations with Newey-West HAC across horizons and regimes.
    """
    rows = []
    for model in common.ALL_MODELS:
        sub_m = df_diff[df_diff['Model'] == model].copy()
        
        # Merge pivoted horizons and regime
        piv_d = sub_m.pivot(index='Date', columns='Horizon', values='d_t')[common.HORIZONS].dropna()
        dates = piv_d.index
        
        # Get regime indicators on same dates
        sub_reg = sub_m[sub_m['Horizon'] == '1D'].set_index('Date').reindex(dates)
        regimes = sub_reg['Regime'].values
        
        T = len(dates)
        z_low = (regimes == 'Low').astype(float)
        z_high = (regimes == 'High').astype(float)
        z_const = np.ones(T, dtype=float)
        
        Z = np.column_stack([z_const, z_low, z_high]) # (T, 3)
        Z_inv = np.linalg.inv(Z.T @ Z) # OLS projection
        
        # Estimate coefficients for each horizon h: beta_h in R^3
        # Stacked theta in R^15: [alpha_1, b_low_1, b_high_1, alpha_5, b_low_5, b_high_5, ...]
        theta_list = []
        residuals_list = []
        for h in common.HORIZONS:
            y_h = piv_d[h].values
            beta_h = Z_inv @ (Z.T @ y_h)
            u_h = y_h - (Z @ beta_h)
            theta_list.append(beta_h)
            residuals_list.append(u_h)
            
        theta_vec = np.concatenate(theta_list) # shape (15,)
        
        # Score/moment matrix for GMM/OLS: S_t = [Z_t * u_{t, 1D}, Z_t * u_{t, 5D}, ...] in R^(T, 15)
        M_list = []
        for i in range(len(common.HORIZONS)):
            M_list.append(Z * residuals_list[i][:, None])
        M_stacked = np.column_stack(M_list) # (T, 15)
        
        omega_M = hac.newey_west_covariance(M_stacked, bandwidth=bandwidth) # (15, 15)
        
        # Bread matrix: Block diagonal with (Z'Z / T)^{-1} on each block of size 3
        Z_bread = Z_inv * T # (3, 3)
        bread = np.zeros((15, 15), dtype=float)
        for i in range(5):
            bread[i*3:(i+1)*3, i*3:(i+1)*3] = Z_bread
            
        # Asymptotic covariance of theta_vec: V_theta = (Bread @ Omega_M @ Bread) / T
        V_theta = (bread @ omega_M @ bread) / T
        
        # Construct Contrast Matrix R for 8 interaction restrictions:
        # For each h in {5D, 20D, 60D, 120D} (indices k = 1, 2, 3, 4):
        # b_{h,Low} - b_{1D,Low} = 0   (index k*3 + 1 minus index 1)
        # b_{h,High} - b_{1D,High} = 0 (index k*3 + 2 minus index 2)
        R_int = np.zeros((8, 15), dtype=float)
        row_idx = 0
        for k in range(1, 5):
            # Low contrast
            R_int[row_idx, k*3 + 1] = 1.0
            R_int[row_idx, 1] = -1.0
            row_idx += 1
            # High contrast
            R_int[row_idx, k*3 + 2] = 1.0
            R_int[row_idx, 2] = -1.0
            row_idx += 1
            
        res = hac.wald_test(theta_vec, V_theta, R=R_int)
        
        rows.append({
            'Model': model,
            'T': T,
            'Bandwidth': bandwidth,
            'df': res['df'],
            'Interaction_Wald_Stat': res['stat'],
            'pvalue': res['pvalue']
        })
        
    df_res = pd.DataFrame(rows)
    df_res = multiple_testing.adjust_dataframe_column(df_res, pval_col='pvalue', adj_col='pvalue_holm')
    df_res['Significant_05'] = df_res['pvalue_holm'] < 0.05
    return df_res
