import os
import pandas as pd
import numpy as np

repo = r"e:\Repo Project\IS403.R12-equity-return-predictability"
lan01 = os.path.join(repo, "results", "Lan01")

print("=== CHECKING CSV SUMMARY VALUES ===")
summary_df = pd.read_csv(os.path.join(lan01, "summary", "3.0.3_model_horizon_summary.csv"))
print("Summary models:", sorted(summary_df['Model'].unique().tolist()))
print("Summary horizons:", sorted(summary_df['Horizon'].unique().tolist()))

print("\nTop 5 OOS_R2_Mean in Lan01 Walk-Forward:")
print(summary_df.sort_values(by='OOS_R2_Mean', ascending=False)[['Model', 'Horizon', 'OOS_R2_Mean', 'DA_Mean', 'MAE_Mean', 'RMSE_Mean']].head(8))

print("\nBottom 5 OOS_R2_Mean in Lan01 Walk-Forward:")
print(summary_df.sort_values(by='OOS_R2_Mean', ascending=True)[['Model', 'Horizon', 'OOS_R2_Mean', 'DA_Mean', 'MAE_Mean', 'RMSE_Mean']].head(8))

print("\nARIMA family values in Lan01 Walk-Forward:")
arima_models = summary_df[summary_df['Model'].isin(['ARIMA', 'ARIMAX', 'SARIMA'])]
print(arima_models[['Model', 'Horizon', 'OOS_R2_Mean', 'DA_Mean', 'MAE_Mean', 'RMSE_Mean']])

# 2. Test metrics
test_df = pd.read_csv(os.path.join(lan01, "test_evaluation", "3.0.3_test_metrics_summary.csv"))
print("\nTest evaluation top 8 OOS_R2 on Split 3:")
print(test_df.sort_values(by='OOS_R2', ascending=False)[['Model', 'Horizon', 'OOS_R2', 'DA', 'MAE', 'RMSE', 'DM_Stat', 'DM_pvalue']].head(8))

print("\nTest evaluation bottom 5 OOS_R2 on Split 3:")
print(test_df.sort_values(by='OOS_R2', ascending=True)[['Model', 'Horizon', 'OOS_R2', 'DA', 'MAE', 'RMSE', 'DM_Stat', 'DM_pvalue']].head(5))
