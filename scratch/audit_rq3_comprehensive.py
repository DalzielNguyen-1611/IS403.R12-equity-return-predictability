import pandas as pd
import numpy as np

# Load files
preds = pd.read_csv('results/predictions/all_oos_predictions.csv')
rq3 = pd.read_csv('results/analysis/rq3_time.csv')

print("=== 1. BASIC INTEGRITY ===")
print("RQ3 shape:", rq3.shape)
print("RQ3 columns:", rq3.columns.tolist())
print("Models in RQ3:", len(rq3['Model'].unique()), rq3['Model'].unique().tolist())
print("Periods in RQ3:", rq3['Period'].unique().tolist())
print("Duplicates in RQ3:", rq3.duplicated(subset=['Model', 'Period']).sum())
print("Missing values in RQ3:", rq3.isna().sum().sum())

print("\n=== 2. DATE RANGES AND OBSERVATIONS IN PREDICTIONS ===")
preds['Date'] = pd.to_datetime(preds['Date'])
print("Min Date:", preds['Date'].min(), "Max Date:", preds['Date'].max())

def map_date(dt):
    dt_str = dt.strftime('%Y-%m-%d')
    if "2015-01-01" <= dt_str <= "2016-12-31": return "2015-2016"
    if "2017-01-01" <= dt_str <= "2018-12-31": return "2017-2018"
    if "2019-01-01" <= dt_str <= "2020-12-31": return "2019-2020"
    return "Other"

preds['Period'] = preds['Date'].apply(map_date)
single_sub = preds[(preds['Model'] == 'Ridge') & (preds['Horizon'] == '1D')]
print("Counts per Period for 1 model / 1 horizon:")
print(single_sub['Period'].value_counts().sort_index())
print("Total single model/horizon observations:", len(single_sub))

print("\n=== 3. REPRODUCIBILITY OF RQ3 FROM PREDICTIONS ===")
max_diff = {}
for col in rq3.columns:
    if col not in ['Model', 'Period']:
        max_diff[col] = 0.0

for idx, row in rq3.iterrows():
    mod = row['Model']
    prd = row['Period']
    sub_mp = preds[(preds['Model'] == mod) & (preds['Period'] == prd)]
    for hrz in ['1D', '5D', '20D', '60D', '120D']:
        sub_h = sub_mp[sub_mp['Horizon'] == hrz]
        y_t = sub_h['Actual'].values
        y_p = sub_h['Prediction'].values
        
        # In-period sample mean denominator
        sse_m = np.sum((y_t - y_p)**2)
        tss_p = np.sum((y_t - np.mean(y_t))**2)
        r2_calc = round(float(1.0 - sse_m / (tss_p + 1e-12)), 4)
        
        # DA
        s_t = np.where(y_t > 0, 1, -1)
        s_p = np.where(y_p > 0, 1, -1)
        da_calc = round(float(np.mean(s_t == s_p)), 4)
        
        # RMSE
        rmse_calc = round(float(np.sqrt(np.mean((y_t - y_p)**2))), 4)
        
        diff_r2 = abs(r2_calc - row[f'{hrz}_OOS_R2'])
        diff_da = abs(da_calc - row[f'{hrz}_DA'])
        diff_rmse = abs(rmse_calc - row[f'{hrz}_RMSE'])
        
        if diff_r2 > max_diff[f'{hrz}_OOS_R2']: max_diff[f'{hrz}_OOS_R2'] = diff_r2
        if diff_da > max_diff[f'{hrz}_DA']: max_diff[f'{hrz}_DA'] = diff_da
        if diff_rmse > max_diff[f'{hrz}_RMSE']: max_diff[f'{hrz}_RMSE'] = diff_rmse

print("Max discrepancies between recalculation and rq3_time.csv:")
for k, v in max_diff.items():
    print(f"  {k}: {v:.6f}")

print("\n=== 4. AUDIT C: MODEL-BY-MODEL ENDPOINT DIFFERENCE Delta OOS_R2 ===")
delta_rows = []
for mod in rq3['Model'].unique():
    for prd in ['2015-2016', '2017-2018', '2019-2020']:
        sub = rq3[(rq3['Model'] == mod) & (rq3['Period'] == prd)]
        r1 = sub['1D_OOS_R2'].values[0]
        r5 = sub['5D_OOS_R2'].values[0]
        r20 = sub['20D_OOS_R2'].values[0]
        r60 = sub['60D_OOS_R2'].values[0]
        r120 = sub['120D_OOS_R2'].values[0]
        delta = round(r120 - r1, 4)
        is_mono_decreasing = (r1 >= r5 >= r20 >= r60 >= r120)
        delta_rows.append({
            'Model': mod, 'Period': prd,
            '1D': r1, '5D': r5, '20D': r20, '60D': r60, '120D': r120,
            'Delta_120_1': delta,
            'Mono_Dec': is_mono_decreasing
        })
df_delta = pd.DataFrame(delta_rows)
print(df_delta.to_string(index=False))

print("\n=== 5. AUDIT SECTION 4: FAMILY-LEVEL AVERAGES ===")
# Families
families = {
    'Tree Ensembles': ['Random_Forest', 'XGBoost', 'LightGBM'],
    'Linear Regularized': ['Ridge', 'Elastic_Net'],
    'Autoregressive': ['ARIMA', 'ARIMAX', 'SARIMA'],
    'Benchmark (Mean)': ['Historical_Mean'],
    'Deep Learning': ['RNN', 'LSTM', 'Transformer'],
    'Support Vector': ['SVR']
}

for fam_name, mod_list in families.items():
    print(f"\nFamily: {fam_name}")
    for prd in ['2015-2016', '2017-2018', '2019-2020']:
        sub = rq3[(rq3['Model'].isin(mod_list)) & (rq3['Period'] == prd)]
        r1 = sub['1D_OOS_R2'].mean()
        r5 = sub['5D_OOS_R2'].mean()
        r20 = sub['20D_OOS_R2'].mean()
        r60 = sub['60D_OOS_R2'].mean()
        r120 = sub['120D_OOS_R2'].mean()
        d = r120 - r1
        print(f"  {prd}: 1D={r1:.4f}, 5D={r5:.4f}, 20D={r20:.4f}, 60D={r60:.4f}, 120D={r120:.4f}, Delta={d:.4f}")

print("\n=== 6. AUDIT F: TARGET VARIANCE, SD, AND RMSE ACROSS PERIODS ===")
t_rows = []
for prd in ['2015-2016', '2017-2018', '2019-2020']:
    for hrz in ['1D', '5D', '20D', '60D', '120D']:
        sub = preds[(preds['Model'] == 'Ridge') & (preds['Period'] == prd) & (preds['Horizon'] == hrz)]
        y = sub['Actual'].values
        t_rows.append({
            'Period': prd, 'Horizon': hrz, 'N': len(y),
            'Mean': np.mean(y), 'Std': np.std(y, ddof=1), 'Variance': np.var(y, ddof=1)
        })
df_target = pd.DataFrame(t_rows)
print(df_target.to_string(index=False))

print("\n=== 7. VERIFY REPORT SECTION 5.3 SPECIFIC RMSE VALUES ===")
# Check Historical Mean and ARIMA RMSE
for mod in ['Historical_Mean', 'ARIMA', 'Ridge', 'LightGBM', 'LSTM']:
    for prd in ['2015-2016', '2017-2018', '2019-2020']:
        sub = rq3[(rq3['Model'] == mod) & (rq3['Period'] == prd)]
        r_120 = sub['120D_RMSE'].values[0]
        r_1 = sub['1D_RMSE'].values[0]
        print(f"Model {mod:16s} | Period {prd} | 1D RMSE={r_1:.4f} | 120D RMSE={r_120:.4f}")
