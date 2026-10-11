import pandas as pd
import numpy as np

df = pd.read_csv('results/analysis/rq2_regime.csv')
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

# Pivot OOS_R2 by Model x Horizon for each regime
print("\n=== LOW REGIME OOS_R2 ===")
low_pivot = reg_df.pivot(index='Model', columns='Horizon', values='Low_OOS_R2')[['1D', '5D', '20D', '60D', '120D']]
print(low_pivot.round(4))

print("\n=== NORMAL REGIME OOS_R2 ===")
norm_pivot = reg_df.pivot(index='Model', columns='Horizon', values='Normal_OOS_R2')[['1D', '5D', '20D', '60D', '120D']]
print(norm_pivot.round(4))

print("\n=== HIGH REGIME OOS_R2 ===")
high_pivot = reg_df.pivot(index='Model', columns='Horizon', values='High_OOS_R2')[['1D', '5D', '20D', '60D', '120D']]
print(high_pivot.round(4))

# Compute Decay Slope (120D - 1D) for each model across regimes
decay_analysis = pd.DataFrame({
    'Model': low_pivot.index,
    'Low_Delta (120D-1D)': low_pivot['120D'] - low_pivot['1D'],
    'Normal_Delta (120D-1D)': norm_pivot['120D'] - norm_pivot['1D'],
    'High_Delta (120D-1D)': high_pivot['120D'] - high_pivot['1D']
})
print("\n=== Decay Slope (120D - 1D) by Regime ===")
print(decay_analysis.round(4))

# Mean performance by Family x Horizon x Regime
family_summary = reg_df.groupby(['Family', 'Horizon'])[['Low_OOS_R2', 'Normal_OOS_R2', 'High_OOS_R2', 'Low_DA', 'Normal_DA', 'High_DA']].mean()
print("\n=== Family Summary by Horizon (Mean OOS_R2 across Regimes) ===")
for fam in reg_df['Family'].unique():
    print(f"\n--- Family: {fam} ---")
    sub = reg_df[reg_df['Family'] == fam]
    p_low = sub.groupby('Horizon')['Low_OOS_R2'].mean()[['1D', '5D', '20D', '60D', '120D']]
    p_norm = sub.groupby('Horizon')['Normal_OOS_R2'].mean()[['1D', '5D', '20D', '60D', '120D']]
    p_high = sub.groupby('Horizon')['High_OOS_R2'].mean()[['1D', '5D', '20D', '60D', '120D']]
    fam_table = pd.DataFrame({'Low': p_low, 'Normal': p_norm, 'High': p_high})
    print(fam_table.round(4))

# Check overall average OOS_R2 by Regime across all regression models
print("\n=== Overall Mean OOS_R2 by Regime and Horizon (Excluding AR to see ML decay) ===")
ml_models = reg_df[~reg_df['Family'].isin(['Autoregressive'])]
ml_regime_mean = ml_models.groupby('Horizon')[['Low_OOS_R2', 'Normal_OOS_R2', 'High_OOS_R2']].mean().loc[['1D', '5D', '20D', '60D', '120D']]
print(ml_regime_mean.round(4))
