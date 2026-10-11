import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, 'src/training/baselines')
import utils

df_stat = pd.read_csv('results/analysis/statistical_tests.csv')
print("=== Current statistical_tests.csv Summary ===")
print("Total rows:", len(df_stat))
print("Significant tests count:", (df_stat['Significant'] == 'Yes').sum())
print("Non-significant tests count:", (df_stat['Significant'] == 'No').sum())

# Breakdown of Binomial DA Significance by Horizon
print("\nSignificance of Binomial DA Test by Horizon:")
print(pd.crosstab(df_stat['Horizon'], df_stat['Significant']))

# Models that were NOT significant in Binomial DA test at 1D
print("\nModels with p >= 0.05 at 1D:")
print(df_stat[(df_stat['Horizon'] == '1D') & (df_stat['Significant'] == 'No')][['Model', 'Statistic', 'p-value']])

# Now compute Diebold-Mariano tests using utils.diebold_mariano_test
df_pred = pd.read_csv('results/predictions/all_oos_predictions.csv')
bench_df = df_pred[df_pred['Model'] == 'Historical_Mean']
models = [m for m in df_pred['Model'].unique() if m != 'Historical_Mean' and m != 'Logistic_Regression']
horizons = ['1D', '5D', '20D', '60D', '120D']
horizon_steps = {'1D': 1, '5D': 5, '20D': 20, '60D': 60, '120D': 120}

dm_results = []
for mod in models:
    for hrz in horizons:
        sub = df_pred[(df_pred['Model'] == mod) & (df_pred['Horizon'] == hrz)]
        b_sub = bench_df[bench_df['Horizon'] == hrz]
        
        merged = pd.merge(sub[['Date', 'Actual', 'Prediction']],
                          b_sub[['Date', 'Prediction']],
                          on='Date', suffixes=('', '_bench'))
        
        y_t = merged['Actual'].values
        y_m = merged['Prediction'].values
        y_b = merged['Prediction_bench'].values
        h_step = horizon_steps[hrz]
        
        # Test if model error is significantly less than benchmark error
        # loss = 'squared'
        # alternative = 'greater' in utils means benchmark loss > model loss
        dm_stat, dm_p = utils.diebold_mariano_test(y_t, y_m, y_b, h=h_step, loss='squared', alternative='greater')
        
        # Also check MSE difference
        mse_m = np.mean((y_t - y_m)**2)
        mse_b = np.mean((y_t - y_b)**2)
        better = mse_m < mse_b
        
        dm_results.append({
            'Model': mod,
            'Horizon': hrz,
            'DM_Stat': round(float(dm_stat), 4),
            'p_value': round(float(dm_p), 4),
            'Model_MSE_Lower': better,
            'Significant_Superiority': 'Yes' if (dm_p < 0.05 and better) else 'No'
        })

dm_df = pd.DataFrame(dm_results)
print("\n=== Diebold-Mariano Test Results (Independent Recomputation) ===")
print("DM Significant Superiority to Historical Mean:")
print(dm_df[dm_df['Significant_Superiority'] == 'Yes'])

print("\nDM Results Pivot (p-value):")
dm_p_pivot = dm_df.pivot(index='Model', columns='Horizon', values='p_value')[horizons]
print(dm_p_pivot.round(4))

print("\nDM Results Pivot (DM_Stat):")
dm_stat_pivot = dm_df.pivot(index='Model', columns='Horizon', values='DM_Stat')[horizons]
print(dm_stat_pivot.round(4))
