import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import binomtest

# Load predictions and existing statistical tests
preds = pd.read_csv('results/predictions/all_oos_predictions.csv')
raw_stat = pd.read_csv('results/analysis/statistical_tests.csv')

print("=== 1. RAW STATISTICAL_TESTS.CSV AUDIT ===")
print("Shape:", raw_stat.shape)
print("Columns:", raw_stat.columns.tolist())
print("Unique Tests:", raw_stat['Test'].unique())
print("Unique Benchmarks:", raw_stat['Benchmark'].unique())
print("Models:", len(raw_stat['Model'].unique()), raw_stat['Model'].unique().tolist())
print("Horizons:", raw_stat['Horizon'].unique().tolist())
print("Duplicates (Model, Horizon, Test):", raw_stat.duplicated(subset=['Model', 'Horizon', 'Test']).sum())
print("Missing values:", raw_stat.isna().sum().to_dict())

# Check if all rows are Binomial
print("Are all rows Binomial?", (raw_stat['Test'] == 'Binomial').all())

print("\n=== 2. REPRODUCE BINOMIAL TESTS FROM ALL_OOS_PREDICTIONS.CSV ===")
binom_records = []
max_da_diff = 0.0
max_p_diff = 0.0

horizons = ['1D', '5D', '20D', '60D', '120D']
models = sorted(preds['Model'].unique())

for mod in models:
    for hrz in horizons:
        sub = preds[(preds['Model'] == mod) & (preds['Horizon'] == hrz)]
        y_t = sub['Actual'].values
        y_p = sub['Prediction'].values
        
        s_t = np.where(y_t > 0, 1, -1)
        s_p = np.where(y_p > 0, 1, -1)
        k = int(np.sum(s_t == s_p))
        n = int(len(y_t))
        da = k / n
        res = binomtest(k=k, n=n, p=0.5, alternative='greater')
        p_val = res.pvalue
        
        # Compare with raw_stat
        row_raw = raw_stat[(raw_stat['Model'] == mod) & (raw_stat['Horizon'] == hrz)]
        if not row_raw.empty:
            da_raw = row_raw['Statistic'].values[0]
            p_raw = row_raw['p-value'].values[0]
            da_diff = abs(round(da, 4) - da_raw)
            p_diff = abs(round(p_val, 4) - p_raw)
            if da_diff > max_da_diff: max_da_diff = da_diff
            if p_diff > max_p_diff: max_p_diff = p_diff
            
        binom_records.append({
            'Model': mod, 'Horizon': hrz,
            'Test': 'Binomial',
            'Benchmark': '50% DA',
            'N': n, 'Successes': k,
            'Statistic': round(float(da), 4),
            'DA_raw': da,
            'p_value_exact': p_val,
            'p_value': round(float(p_val), 4),
            'Significant_Raw': 'Yes' if p_val < 0.05 else 'No',
            'P_Actual_Positive': float(np.mean(y_t > 0)),
            'P_Pred_Positive': float(np.mean(y_p > 0))
        })

df_binom = pd.DataFrame(binom_records)
print(f"Max DA difference vs raw CSV: {max_da_diff:.6f}")
print(f"Max p-value difference vs raw CSV: {max_p_diff:.6f}")

print("\n=== 3. DIEBOLD-MARIANO TEST RECALCULATION ===")
def calc_dm(y_true, y_model, y_bench, h_step):
    e_m = y_true - y_model
    e_b = y_true - y_bench
    d = (e_b**2 - e_m**2)
    T = len(d)
    if T <= 1: return 0.0, 1.0, 1.0, 0.0
    mean_d = np.mean(d)
    gamma = [np.var(d, ddof=0)]
    for lag in range(1, int(h_step)):
        gamma.append(np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d)))
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_d <= 1e-12: return 0.0, 1.0, 1.0, mean_d
    dm_stat = mean_d / np.sqrt(var_d)
    hln_factor = np.sqrt(max(1e-6, (T + 1 - 2*h_step + h_step*(h_step-1)/T) / T))
    dm_stat_adj = dm_stat * hln_factor
    df = max(1, T - 1)
    p_greater = 1.0 - stats.t.cdf(dm_stat_adj, df=df)
    p_two_sided = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat_adj), df=df))
    return float(dm_stat_adj), float(p_greater), float(p_two_sided), float(mean_d)

horizon_steps = {'1D': 1, '5D': 5, '20D': 20, '60D': 60, '120D': 120}
dm_records = []

bench_model_name = 'Historical_Mean'
bench_df = preds[preds['Model'] == bench_model_name]

# Regression models (excluding Historical_Mean and Logistic_Regression)
candidate_models = [m for m in models if m not in [bench_model_name, 'Logistic_Regression']]

for mod in candidate_models:
    for hrz in horizons:
        sub = preds[(preds['Model'] == mod) & (preds['Horizon'] == hrz)]
        b_sub = bench_df[bench_df['Horizon'] == hrz]
        
        merged = pd.merge(sub[['Date', 'Actual', 'Prediction']],
                          b_sub[['Date', 'Prediction']],
                          on='Date', suffixes=('', '_bench'))
        
        y_t = merged['Actual'].values
        y_m = merged['Prediction'].values
        y_b = merged['Prediction_bench'].values
        h_s = horizon_steps[hrz]
        
        stat, p_gt, p_2s, mean_loss_diff = calc_dm(y_t, y_m, y_b, h_s)
        
        sse_m = np.sum((y_t - y_m)**2)
        sse_b = np.sum((y_t - y_b)**2)
        oos_r2 = 1.0 - sse_m / sse_b
        rmse_m = np.sqrt(np.mean((y_t - y_m)**2))
        rmse_b = np.sqrt(np.mean((y_t - y_b)**2))
        mae_m = np.mean(np.abs(y_t - y_m))
        mae_b = np.mean(np.abs(y_t - y_b))
        
        dm_records.append({
            'Model': mod, 'Horizon': hrz,
            'Test': 'DM',
            'Benchmark': 'Historical Mean',
            'Statistic': round(float(stat), 4),
            'DM_Stat_Raw': stat,
            'p_value_exact': p_gt,
            'p_value': round(float(p_gt), 4),
            'p_val_two_sided': round(float(p_2s), 4),
            'Mean_Loss_Diff': mean_loss_diff,
            'Candidate_Better': 'Yes' if mean_loss_diff > 0 else 'No',
            'OOS_R2': round(float(oos_r2), 4),
            'RMSE_Model': round(float(rmse_m), 4),
            'RMSE_Bench': round(float(rmse_b), 4),
            'MAE_Model': round(float(mae_m), 4),
            'MAE_Bench': round(float(mae_b), 4),
            'Significant_Raw': 'Yes' if p_gt < 0.05 else 'No'
        })

df_dm = pd.DataFrame(dm_records)
print(f"Total DM tests computed: {len(df_dm)} (12 models * 5 horizons = 60 tests)")

print("\n=== 4. COMPARE WITH REPORT 08 SECTION 4.1 TABLE ===")
pivot_dm = df_dm.pivot(index='Model', columns='Horizon', values='Statistic')
pivot_p = df_dm.pivot(index='Model', columns='Horizon', values='p_value')
print("DM Statistics (Recalculated):")
print(pivot_dm[horizons].round(2))
print("\nDM p-values (greater, Recalculated):")
print(pivot_p[horizons].round(4))

print("\n=== 5. MULTIPLE TESTING CORRECTION (BONFERRONI & BENJAMINI-HOCHBERG FDR) ===")
# Family 1: 60 DM tests
N_tests = len(df_dm) # 60
alpha = 0.05
bonf_thresh = alpha / N_tests
print(f"Bonferroni threshold (alpha / 60): {bonf_thresh:.6f}")

# Compute adjusted p-values
# Bonferroni adjusted p-val = min(1.0, p * N)
df_dm['p_bonferroni'] = np.minimum(1.0, df_dm['p_value_exact'] * N_tests)
df_dm['Significant_Bonferroni'] = np.where(df_dm['p_value_exact'] < bonf_thresh, 'Yes', 'No')

# Benjamini-Hochberg FDR
# Sort by p_value_exact
sorted_indices = np.argsort(df_dm['p_value_exact'].values)
raw_p_sorted = df_dm['p_value_exact'].values[sorted_indices]

# FDR adjusted p-value (cumulative min of p * N / rank from the back)
fdr_p = np.zeros(N_tests)
ranks = np.arange(1, N_tests + 1)
fdr_p = raw_p_sorted * N_tests / ranks
# Monotonize
for i in range(N_tests - 2, -1, -1):
    fdr_p[i] = min(fdr_p[i], fdr_p[i + 1])
fdr_p = np.minimum(1.0, fdr_p)

# Put back in order
fdr_adjusted = np.zeros(N_tests)
fdr_adjusted[sorted_indices] = fdr_p
df_dm['p_fdr_bh'] = fdr_adjusted
df_dm['Significant_FDR_05'] = np.where(df_dm['p_fdr_bh'] < 0.05, 'Yes', 'No')

print("\nSurviving DM tests under Bonferroni (alpha = 0.05, thresh = 0.000833):")
surv_bonf = df_dm[df_dm['Significant_Bonferroni'] == 'Yes'][['Model', 'Horizon', 'Statistic', 'p_value_exact', 'p_bonferroni']]
print(surv_bonf.to_string(index=False))

print("\nSurviving DM tests under Benjamini-Hochberg FDR (q = 0.05):")
surv_fdr = df_dm[df_dm['Significant_FDR_05'] == 'Yes'][['Model', 'Horizon', 'Statistic', 'p_value_exact', 'p_fdr_bh']]
print(surv_fdr.to_string(index=False))

# Multiple testing for Binomial tests (70 tests)
N_binom = len(df_binom) # 70
bonf_binom_thresh = alpha / N_binom
df_binom['p_bonferroni'] = np.minimum(1.0, df_binom['p_value_exact'] * N_binom)
df_binom['Significant_Bonferroni'] = np.where(df_binom['p_value_exact'] < bonf_binom_thresh, 'Yes', 'No')

# Benjamini-Hochberg FDR for Binomial
sorted_b_idx = np.argsort(df_binom['p_value_exact'].values)
raw_b_sorted = df_binom['p_value_exact'].values[sorted_b_idx]
fdr_b = raw_b_sorted * N_binom / np.arange(1, N_binom + 1)
for i in range(N_binom - 2, -1, -1):
    fdr_b[i] = min(fdr_b[i], fdr_b[i + 1])
fdr_b = np.minimum(1.0, fdr_b)
fdr_b_adj = np.zeros(N_binom)
fdr_b_adj[sorted_b_idx] = fdr_b
df_binom['p_fdr_bh'] = fdr_b_adj
df_binom['Significant_FDR_05'] = np.where(df_binom['p_fdr_bh'] < 0.05, 'Yes', 'No')

print(f"\nBinomial tests surviving Bonferroni (p < {bonf_binom_thresh:.6f}): {np.sum(df_binom['Significant_Bonferroni'] == 'Yes')}/{N_binom}")
print(f"Binomial tests surviving FDR (q < 0.05): {np.sum(df_binom['Significant_FDR_05'] == 'Yes')}/{N_binom}")

# Save combined corrected statistical tests CSV
# Standard schema: Model, Horizon, Test, Benchmark, Statistic, p_value_raw, Significant_Raw, p_bonferroni, Significant_Bonferroni, p_fdr_bh, Significant_FDR_05, Notes
records_out = []

# Binomial records
for idx, r in df_binom.iterrows():
    records_out.append({
        'Model': r['Model'],
        'Horizon': r['Horizon'],
        'Test': 'Binomial (DA > 0.5)',
        'Benchmark': '50% Coin Flip',
        'Statistic': r['Statistic'],
        'p_value_raw': r['p_value_exact'],
        'Significant_Raw_05': r['Significant_Raw'],
        'p_bonferroni': r['p_bonferroni'],
        'Significant_Bonferroni': r['Significant_Bonferroni'],
        'p_fdr_bh': r['p_fdr_bh'],
        'Significant_FDR_05': r['Significant_FDR_05'],
        'Positive_Base_Rate': r['P_Actual_Positive'],
        'Predicted_Positive_Rate': r['P_Pred_Positive'],
        'Methodological_Status': 'Descriptive DA only; Overlapping dependence unadjusted'
    })

# DM records
for idx, r in df_dm.iterrows():
    status = 'Methodologically Unvalidated (1-step filtering leakage)' if r['Model'] in ['ARIMA', 'ARIMAX', 'SARIMA'] else 'Methodologically Valid'
    records_out.append({
        'Model': r['Model'],
        'Horizon': r['Horizon'],
        'Test': 'Diebold-Mariano (HLN)',
        'Benchmark': 'Historical Mean',
        'Statistic': r['Statistic'],
        'p_value_raw': r['p_value_exact'],
        'Significant_Raw_05': r['Significant_Raw'],
        'p_bonferroni': r['p_bonferroni'],
        'Significant_Bonferroni': r['Significant_Bonferroni'],
        'p_fdr_bh': r['p_fdr_bh'],
        'Significant_FDR_05': r['Significant_FDR_05'],
        'Positive_Base_Rate': np.nan,
        'Predicted_Positive_Rate': np.nan,
        'Methodological_Status': status
    })

df_corrected = pd.DataFrame(records_out)
out_csv_path = 'results/analysis/statistical_tests_corrected.csv'
df_corrected.to_csv(out_csv_path, index=False)
print(f"\nSaved corrected statistical tests to: {out_csv_path} (Total {len(df_corrected)} rows)")
