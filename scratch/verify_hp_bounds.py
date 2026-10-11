import pandas as pd
import ast
import numpy as np

df = pd.read_csv('results/tuning/best_hyperparameters.csv')
print("Total rows:", len(df))

# Check for each model if all parameters are valid
# Models:
# Ridge: alpha in [0.001, 0.01, 0.1, 1, 10, 100]
# Elastic_Net: alpha in [0.001, 0.01, 0.1, 1, 10, 100], l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]
# Logistic_Regression: C in [0.001, 0.01, 0.1, 1, 10, 100], class_weight in [None, 'balanced']
# SVR: kernel='rbf', C in [0.1, 1, 10, 100], gamma in ['scale', 0.001, 0.01, 0.1], epsilon in [0.001, 0.01, 0.05, 0.1]
# ARIMA: order=(p, d, q) where p in [0,1,2,3,5], d in [0,1], q in [0,1,2,3,5]
# ARIMAX: order=(p, d, q)
# SARIMA: order=(p,d,q), seasonal_order=(P,D,Q,s)
# RF: n_estimators in [200, 500], max_depth in [None, 5, 10, 20], min_samples_split in [2, 5, 10], min_samples_leaf in [1, 2, 5], max_features in ['sqrt', 0.5]
# XGBoost: n_estimators in (100, 1000), lr in (0.01, 0.2), max_depth in (2, 10), min_child_weight in (1, 10), subsample in (0.6, 1.0), colsample in (0.6, 1.0), gamma in (0.0, 5.0), reg_alpha in (1e-8, 10.0), reg_lambda in (1e-3, 100.0)
# LightGBM: bounds
# RNN, LSTM, Transformer: lookback, hidden_size, num_layers, dropout, lr, weight_decay, batch_size, epochs, patience

anomalies = []
for idx, row in df.iterrows():
    m = row['Model']
    h = row['Horizon']
    f = row['Fold']
    p_str = row['Best_Params']
    try:
        p_dict = ast.literal_eval(p_str)
    except Exception as e:
        anomalies.append(f"Cannot parse params: {m} {h} {f}: {e}")
        continue
    
    # Specific checks
    if m == 'Ridge':
        if p_dict.get('alpha') not in [0.001, 0.01, 0.1, 1, 10, 100, 1.0, 10.0, 100.0]:
            anomalies.append(f"Ridge alpha outside grid: {p_dict}")
    elif m == 'Elastic_Net':
        if p_dict.get('alpha') not in [0.001, 0.01, 0.1, 1, 10, 100, 1.0, 10.0, 100.0]:
            anomalies.append(f"Elastic_Net alpha outside grid: {p_dict}")
        if p_dict.get('l1_ratio') not in [0.1, 0.3, 0.5, 0.7, 0.9]:
            anomalies.append(f"Elastic_Net l1_ratio outside grid: {p_dict}")
    elif m == 'SVR':
        if p_dict.get('C') not in [0.1, 1, 10, 100, 1.0, 10.0, 100.0]:
            anomalies.append(f"SVR C outside grid: {p_dict}")
    elif m == 'XGBoost':
        if not (100 <= p_dict.get('n_estimators', 0) <= 1000):
            anomalies.append(f"XGB n_estimators outside bounds: {p_dict}")
        if not (0.009 <= p_dict.get('learning_rate', 0) <= 0.201):
            anomalies.append(f"XGB lr outside bounds: {p_dict}")
    elif m == 'LightGBM':
        if not (100 <= p_dict.get('n_estimators', 0) <= 1000):
            anomalies.append(f"LGBM n_estimators outside bounds: {p_dict}")
        if not (0.009 <= p_dict.get('learning_rate', 0) <= 0.201):
            anomalies.append(f"LGBM lr outside bounds: {p_dict}")

print("Search space bounds anomalies found:", len(anomalies))
if anomalies:
    print("Sample anomalies:", anomalies[:5])

# Parameter stability analysis:
print("\n--- Hyperparameter Diversity Across Folds ---")
for m in df['Model'].unique():
    sub = df[df['Model'] == m]
    total_distinct = sub['Best_Params'].nunique()
    print(f"Model {m:20s}: {total_distinct} unique configurations across 25 runs (5 horizons x 5 folds)")

# Check Score_Metric and Best_Val_Score ranges
print("\n--- Validation Score Metrics & Ranges ---")
for sm, grp in df.groupby('Score_Metric'):
    print(f"Metric: {sm:18s} | Count: {len(grp):3d} | Min: {grp['Best_Val_Score'].min():.6f} | Median: {grp['Best_Val_Score'].median():.6f} | Max: {grp['Best_Val_Score'].max():.6f}")
