import pandas as pd
import numpy as np

preds = pd.read_csv('results/predictions/all_oos_predictions.csv')
rq2 = pd.read_csv('results/analysis/rq2_regime.csv')

print("=== 1. VERIFY RQ2 SHAPE & INTEGRITY ===")
print(f"RQ2 shape: {rq2.shape}")
print(f"Columns: {rq2.columns.tolist()}")
dups = rq2.duplicated(subset=['Model', 'Horizon']).sum()
print(f"Duplicates (Model, Horizon): {dups}")
models = rq2['Model'].unique().tolist()
horizons = rq2['Horizon'].unique().tolist()
print(f"Models ({len(models)}): {models}")
print(f"Horizons ({len(horizons)}): {horizons}")

print("\n=== 2. MAXIMUM ABSOLUTE DISCREPANCIES ===")
max_diff = {
    'Low_OOS_R2': 0.0, 'Normal_OOS_R2': 0.0, 'High_OOS_R2': 0.0,
    'Low_DA': 0.0, 'Normal_DA': 0.0, 'High_DA': 0.0
}
for idx, row in rq2.iterrows():
    mod = row['Model']
    hrz = row['Horizon']
    for reg in ['Low', 'Normal', 'High']:
        sub = preds[(preds['Model'] == mod) & (preds['Horizon'] == hrz) & (preds['Regime'] == reg)]
        y_t = sub['Actual'].values
        y_p = sub['Prediction'].values
        sse = np.sum((y_t - y_p)**2)
        tss = np.sum((y_t - np.mean(y_t))**2)
        calc_r2 = round(float(1.0 - sse / (tss + 1e-12)), 4)
        calc_da = round(float(np.mean(np.where(y_t > 0, 1, -1) == np.where(y_p > 0, 1, -1))), 4)
        
        diff_r2 = abs(calc_r2 - row[f'{reg}_OOS_R2'])
        diff_da = abs(calc_da - row[f'{reg}_DA'])
        if diff_r2 > max_diff[f'{reg}_OOS_R2']: max_diff[f'{reg}_OOS_R2'] = diff_r2
        if diff_da > max_diff[f'{reg}_DA']: max_diff[f'{reg}_DA'] = diff_da

for k, v in max_diff.items():
    print(f"  Max Diff {k}: {v:.6f}")

print("\n=== 3. REGIME SAMPLE SIZES ===")
sub1 = preds[(preds['Model'] == 'Ridge') & (preds['Horizon'] == '1D')]
for reg in ['Low', 'Normal', 'High']:
    print(f"  Count {reg}: {len(sub1[sub1['Regime'] == reg])}")

print("\n=== 4. TARGET VARIANCE AUDIT ===")
var_rows = []
for h in ['1D', '5D', '20D', '60D', '120D']:
    sub_h = preds[(preds['Model'] == 'Ridge') & (preds['Horizon'] == h)]
    var_dict = {}
    for reg in ['Low', 'Normal', 'High']:
        y = sub_h[sub_h['Regime'] == reg]['Actual'].values
        v = np.var(y, ddof=1)
        var_dict[reg] = v
        var_rows.append({
            'Horizon': h,
            'Regime': reg,
            'N': len(y),
            'Mean': np.mean(y),
            'Std': np.std(y, ddof=1),
            'Variance': v,
            'Min': np.min(y),
            'Max': np.max(y),
            'Median': np.median(y),
            'Q25': np.percentile(y, 25),
            'Q75': np.percentile(y, 75)
        })
    ratio = var_dict['High'] / var_dict['Low']
    print(f"Horizon {h:4s}: Var(High)={var_dict['High']:.6f}, Var(Low)={var_dict['Low']:.6f}, Ratio Var(High)/Var(Low)={ratio:.4f}")

df_var = pd.DataFrame(var_rows)
print("\n" + df_var.to_string(index=False))

print("\n=== 5. POSITIVE BASE-RATE & DIRECTIONAL ACCURACY ===")
base_rate_rows = []
for h in ['1D', '5D', '20D', '60D', '120D']:
    for reg in ['Low', 'Normal', 'High']:
        sub_h = preds[(preds['Model'] == 'Ridge') & (preds['Horizon'] == h) & (preds['Regime'] == reg)]
        y = sub_h['Actual'].values
        p_pos = np.mean(y > 0)
        base_rate_rows.append({
            'Horizon': h,
            'Regime': reg,
            'N': len(y),
            'P(Actual > 0)': p_pos,
            'Always_Positive_DA': p_pos
        })
df_base = pd.DataFrame(base_rate_rows)
print(df_base.to_string(index=False))

print("\n=== P(Pred > 0) AND DA FOR SELECTED MODELS ===")
selected_models = ['Historical_Mean', 'ARIMA', 'ARIMAX', 'Ridge', 'LightGBM', 'LSTM']
da_rows = []
for h in ['1D', '5D', '20D', '60D', '120D']:
    for reg in ['Low', 'Normal', 'High']:
        for mod in selected_models:
            sub = preds[(preds['Model'] == mod) & (preds['Horizon'] == h) & (preds['Regime'] == reg)]
            y = sub['Actual'].values
            p = sub['Prediction'].values
            p_pred_pos = np.mean(p > 0)
            da = np.mean(np.where(y > 0, 1, -1) == np.where(p > 0, 1, -1))
            da_rows.append({
                'Horizon': h, 'Regime': reg, 'Model': mod,
                'P(Pred > 0)': p_pred_pos, 'DA': da
            })
df_da = pd.DataFrame(da_rows)
print(df_da.to_string(index=False))

print("\n=== 6. RMSE & MAE COMPARISON ACROSS REGIMES (CATASTROPHIC COLLAPSE AUDIT) ===")
err_rows = []
for h in ['1D', '20D', '120D']:
    for mod in ['Historical_Mean', 'Ridge', 'LightGBM', 'LSTM', 'ARIMAX']:
        for reg in ['Low', 'Normal', 'High']:
            sub = preds[(preds['Model'] == mod) & (preds['Horizon'] == h) & (preds['Regime'] == reg)]
            y = sub['Actual'].values
            p = sub['Prediction'].values
            rmse = np.sqrt(np.mean((y - p)**2))
            mae = np.mean(np.abs(y - p))
            r2_val = rq2[(rq2['Model'] == mod) & (rq2['Horizon'] == h)][f'{reg}_OOS_R2'].values[0]
            err_rows.append({
                'Horizon': h, 'Model': mod, 'Regime': reg,
                'RMSE': rmse, 'MAE': mae, 'OOS_R2': r2_val
            })
df_err = pd.DataFrame(err_rows)
print(df_err.to_string(index=False))
