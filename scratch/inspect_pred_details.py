import pandas as pd
import numpy as np

df = pd.read_csv('results/predictions/all_oos_predictions.csv')

# Check inf values
print("Any inf in Actual:", np.isinf(df['Actual']).sum())
print("Any inf in Prediction:", np.isinf(df['Prediction']).sum())

# Prediction range by model
stats_by_model = df.groupby('Model')[['Actual', 'Prediction']].agg(['min', 'mean', 'max', 'std'])
print("\nStats by Model:")
print(stats_by_model)

# Prediction range by Horizon
stats_by_h = df[df['Model'] != 'Logistic_Regression'].groupby('Horizon')[['Actual', 'Prediction']].agg(['min', 'mean', 'max', 'std'])
print("\nStats by Horizon (Regression Models):")
print(stats_by_h)

# Let's inspect Regime breakdown per Horizon and Fold
print("\nRegime counts by Horizon:")
print(df[df['Model'] == 'Ridge'].groupby(['Horizon', 'Regime'])['Date'].count().unstack())

# Recompute OOS_R2 across Regimes for key models
# Recall formula for OOS_R2 in a subset: 1 - MSE_subset / Var(Actual_subset)
def calc_metrics(g):
    act = g['Actual'].values
    pred = g['Prediction'].values
    mae = np.mean(np.abs(act - pred))
    rmse = np.sqrt(np.mean((act - pred)**2))
    denom = np.mean((act - act.mean())**2)
    r2 = 1.0 - (rmse**2 / denom) if denom > 0 else np.nan
    da = np.mean((act > 0) == (pred > 0))
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'OOS_R2': r2, 'DA': da, 'N': len(g)})

reg_df = df[df['Model'] != 'Logistic_Regression']
regime_perf = reg_df.groupby(['Model', 'Horizon', 'Regime']).apply(calc_metrics).reset_index()
print("\nSample Regime Performance (Ridge 1D):")
print(regime_perf[(regime_perf['Model'] == 'Ridge') & (regime_perf['Horizon'] == '1D')])

# Time Period analysis: how does rq3_time.csv define periods? Let's check rq3_time.csv if it exists
import os
if os.path.exists('results/analysis/rq3_time.csv'):
    df_rq3 = pd.read_csv('results/analysis/rq3_time.csv')
    print("\nRQ3 Time Periods in rq3_time.csv:")
    print(df_rq3['Period'].unique() if 'Period' in df_rq3.columns else df_rq3.columns)
    print(df_rq3.head(5))
