import pandas as pd
import numpy as np

df = pd.read_csv('results/analysis/rq3_time.csv')
print("Total rows:", len(df))

reg_df = df[df['Model'] != 'Logistic_Regression'].copy()

# Add Model Family
family_map = {
    'Historical_Mean': 'Benchmark',
    'Ridge': 'Linear Regularized',
    'Elastic_Net': 'Linear Regularized',
    'ARIMA': 'Autoregressive',
    'ARIMAX': 'Autoregressive',
    'SARIMA': 'Autoregressive',
    'SVR': 'Support Vector',
    'Random_Forest': 'Tree Ensembles',
    'XGBoost': 'Tree Ensembles',
    'LightGBM': 'Tree Ensembles',
    'RNN': 'Deep Learning',
    'LSTM': 'Deep Learning',
    'Transformer': 'Deep Learning'
}
reg_df['Family'] = reg_df['Model'].map(family_map)

# Let's inspect OOS_R2 across periods for each horizon
r2_cols = ['1D_OOS_R2', '5D_OOS_R2', '20D_OOS_R2', '60D_OOS_R2', '120D_OOS_R2']

for period in ['2015-2016', '2017-2018', '2019-2020']:
    print(f"\n=== Period: {period} (OOS_R2 across Horizons) ===")
    sub = reg_df[reg_df['Period'] == period]
    p_mat = sub.set_index('Model')[r2_cols]
    p_mat.columns = ['1D', '5D', '20D', '60D', '120D']
    print(p_mat.round(4))

# Compute Decay Delta (120D - 1D) for each period
decay_by_period = pd.DataFrame(index=reg_df['Model'].unique())
for period in ['2015-2016', '2017-2018', '2019-2020']:
    sub = reg_df[reg_df['Period'] == period].set_index('Model')
    decay_by_period[period + '_Delta'] = sub['120D_OOS_R2'] - sub['1D_OOS_R2']
    decay_by_period[period + '_120D_R2'] = sub['120D_OOS_R2']

print("\n=== Decay Slope (120D - 1D) across Periods ===")
print(decay_by_period.round(4))

# Family Averages across periods
fam_summary = reg_df.groupby(['Family', 'Period'])[r2_cols].mean()
fam_summary.columns = ['1D', '5D', '20D', '60D', '120D']
print("\n=== Family Summary by Period (Mean OOS_R2 across Horizons) ===")
print(fam_summary.round(4))

# Check RMSE phình to tại 2019-2020
print("\n=== 120D RMSE across Periods by Model ===")
rmse_p = reg_df.pivot(index='Model', columns='Period', values='120D_RMSE')[['2015-2016', '2017-2018', '2019-2020']]
print(rmse_p.round(4))
