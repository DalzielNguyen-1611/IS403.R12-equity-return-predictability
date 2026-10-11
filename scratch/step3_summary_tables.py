import pandas as pd
import numpy as np

df = pd.read_csv('results/predictions/all_oos_predictions.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

def period_map(y):
    if y in [2015, 2016]: return '2015-2016'
    elif y in [2017, 2018]: return '2017-2018'
    else: return '2019-2020'
df['Period'] = df['Year'].apply(period_map)

# Actual distribution across Horizons
act_dist = df[df['Model'] == 'Historical_Mean'].groupby('Horizon')['Actual'].describe()
print("=== Actual Return Distribution by Horizon ===")
print(act_dist[['count', 'mean', 'std', 'min', '50%', 'max']])

# Overall pooled metrics by Model and Horizon
def pool_metrics(g):
    act = g['Actual'].values
    pred = g['Prediction'].values
    mae = np.mean(np.abs(act - pred))
    rmse = np.sqrt(np.mean((act - pred)**2))
    denom = np.mean((act - act.mean())**2)
    r2 = 1.0 - (rmse**2 / denom) if denom > 0 else np.nan
    da = np.mean((act > 0) == (pred > 0))
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'Pooled_OOS_R2': r2, 'Pooled_DA': da})

reg_df = df[df['Model'] != 'Logistic_Regression']
pooled = reg_df.groupby(['Model', 'Horizon']).apply(pool_metrics, include_groups=False).reset_index()

# Pivot Pooled_OOS_R2
p_r2 = pooled.pivot(index='Model', columns='Horizon', values='Pooled_OOS_R2')[['1D', '5D', '20D', '60D', '120D']]
print("\n=== Pooled OOS R2 by Model x Horizon ===")
print(p_r2.round(4))

# Regime comparison summary: Average OOS_R2 by Model Family x Regime x Horizon
model_families = {
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
reg_df['Family'] = reg_df['Model'].map(model_families)

family_regime = reg_df.groupby(['Family', 'Regime', 'Horizon']).apply(pool_metrics, include_groups=False).reset_index()
print("\n=== Sample Family x Regime x Horizon (Autoregressive vs Tree Ensembles vs Deep Learning) ===")
print(family_regime[family_regime['Family'].isin(['Tree Ensembles', 'Deep Learning', 'Autoregressive']) & (family_regime['Horizon'] == '1D')][['Family', 'Regime', 'Pooled_OOS_R2', 'Pooled_DA']])
