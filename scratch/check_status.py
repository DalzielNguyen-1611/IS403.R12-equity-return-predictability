import os
import pandas as pd

print("=== 1. CHECK LAN01 (WALK-FORWARD VALIDATION) ===")
fm = pd.read_csv("results/Lan01/fold_metrics/3.0.3_all_fold_metrics.csv")
print(f"Fold Metrics: {len(fm)} rows (Expected 350), {fm['Model'].nunique()} models")
print("Models in Fold Metrics:", sorted(fm['Model'].unique().tolist()))

pred = pd.read_csv("results/Lan01/predictions/3.0.3_all_oos_predictions.csv")
print(f"OOS Predictions: {len(pred)} rows (Expected 95200), {pred['Model'].nunique()} models")

sm = pd.read_csv("results/Lan01/summary/3.0.3_model_horizon_summary.csv")
print(f"Summary Metrics: {len(sm)} rows (Expected 70), {sm['Model'].nunique()} models")

tune = pd.read_csv("results/Lan01/tuning/3.0.3_best_hyperparameters.csv")
print(f"Tuning: {len(tune)} rows (Expected 350), {tune['Model'].nunique()} models")

print("\n=== 2. CHECK TEST EVALUATION (SPLIT 3: 2020-2026) ===")
test_pred = pd.read_csv("results/Lan01/test_evaluation/3.0.3_test_predictions.csv")
print(f"Test Predictions: {len(test_pred)} rows (Expected 95340), {test_pred['Model'].nunique()} models")
print("Models in Test Predictions:", sorted(test_pred['Model'].unique().tolist()))

test_m = pd.read_csv("results/Lan01/test_evaluation/3.0.3_test_metrics_summary.csv")
print(f"Test Regression Metrics: {len(test_m)} rows (Expected 65), {test_m['Model'].nunique()} models")

test_c = pd.read_csv("results/Lan01/test_evaluation/3.0.3_test_classification_summary.csv")
print(f"Test Classification: {len(test_c)} rows (Expected 5), {test_c['Model'].nunique()} models")

test_stat = pd.read_csv("results/Lan01/test_evaluation/3.0.3_test_statistical_tests.csv")
print(f"Test Statistical Tests: {len(test_stat)} rows, {test_stat['Model'].nunique()} models")

test_reg = pd.read_csv("results/Lan01/test_evaluation/3.0.3_test_regime_breakdown.csv")
print(f"Test Regime Breakdown: {len(test_reg)} rows, {test_reg['Model'].nunique()} models")
