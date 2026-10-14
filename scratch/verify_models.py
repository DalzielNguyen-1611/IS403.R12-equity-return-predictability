import pandas as pd
import glob
import os

repo = r'e:\Repo Project\IS403.R12-equity-return-predictability'
lan01 = os.path.join(repo, 'results', 'Lan01')

print("=== 1. WALK-FORWARD VALIDATION (TRAIN/VAL OOS: 2015-2020) ===")
fold_csv = os.path.join(lan01, 'fold_metrics', '3.0.3_all_fold_metrics.csv')
df_folds = pd.read_csv(fold_csv)
print(f"Total fold metric rows: {len(df_folds)}")
models_folds = sorted(df_folds['Model'].unique().tolist())
print(f"Distinct models ({len(models_folds)}): {models_folds}")
print(f"Distinct horizons: {sorted(df_folds['Horizon'].unique().tolist())}")
print(f"Distinct folds: {sorted(df_folds['Fold'].unique().tolist())}")

summary_csv = os.path.join(lan01, 'summary', '3.0.3_model_horizon_summary.csv')
df_sum = pd.read_csv(summary_csv)
models_sum = sorted(df_sum['Model'].unique().tolist())
print(f"Summary rows: {len(df_sum)} across {len(models_sum)} models: {models_sum}")

print("\n=== 2. INDEPENDENT HOLDOUT TEST (SPLIT 3: 2020-2026) ===")
test_pred_csv = os.path.join(lan01, 'test_evaluation', '3.0.3_test_predictions.csv')
df_test_pred = pd.read_csv(test_pred_csv)
print(f"Test prediction rows: {len(df_test_pred)}")
models_test = sorted(df_test_pred['Model'].unique().tolist())
print(f"Distinct test models ({len(models_test)}): {models_test}")
print(f"Distinct test horizons: {sorted(df_test_pred['Horizon'].unique().tolist())}")
counts = df_test_pred.groupby(['Model', 'Horizon']).size().unstack()
print("Sample counts per (Model, Horizon):")
print(counts)

test_met_csv = os.path.join(lan01, 'test_evaluation', '3.0.3_test_metrics_summary.csv')
df_test_met = pd.read_csv(test_met_csv)
models_test_met = sorted(df_test_met['Model'].unique().tolist())
print(f"\nTest metrics summary rows: {len(df_test_met)} across {len(models_test_met)} regression models: {models_test_met}")

test_clf_csv = os.path.join(lan01, 'test_evaluation', '3.0.3_test_classification_summary.csv')
df_test_clf = pd.read_csv(test_clf_csv)
models_test_clf = sorted(df_test_clf['Model'].unique().tolist())
print(f"Test classification models ({len(models_test_clf)}): {models_test_clf} (rows: {len(df_test_clf)})")

total_unique_test = sorted(list(set(models_test_met + models_test_clf)))
print(f"\nTotal unique models in test evaluation: {len(total_unique_test)}: {total_unique_test}")
