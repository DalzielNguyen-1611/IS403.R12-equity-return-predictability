import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df_pred = pd.read_csv('results/predictions/all_oos_predictions.csv')
df_fold = pd.read_csv('results/fold_metrics/all_fold_metrics.csv')

print("Pred shape:", df_pred.shape)
print("Models:", df_pred['Model'].unique().tolist())
print("Horizons:", df_pred['Horizon'].unique().tolist())
print("Folds:", df_pred['Fold'].unique().tolist())
print("Regimes:", df_pred['Regime'].value_counts().to_dict())

# Verify Logistic Regression classification metrics
sub_lr_pred = df_pred[df_pred['Model'] == 'Logistic_Regression']
sub_lr_fold = df_fold[df_fold['Model'] == 'Logistic_Regression']

max_acc_diff = 0
max_f1_diff = 0
max_auc_diff = 0

for h in ['1D', '5D', '20D', '60D', '120D']:
    for f in [1, 2, 3, 4, 5]:
        p = sub_lr_pred[(sub_lr_pred['Horizon'] == h) & (sub_lr_pred['Fold'] == f)]
        m = sub_lr_fold[(sub_lr_fold['Horizon'] == h) & (sub_lr_fold['Fold'] == f)].iloc[0]
        
        y_true = p['Actual_Class'].dropna().values
        y_pred = p['Predicted_Class'].dropna().values
        y_prob = p['Predicted_Probability'].dropna().values
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan
        
        acc_diff = abs(acc - m['Accuracy'])
        f1_diff = abs(f1 - m['F1'])
        auc_diff = 0 if (np.isnan(auc) and np.isnan(m['ROC_AUC'])) else abs(auc - m['ROC_AUC'])
        
        max_acc_diff = max(max_acc_diff, acc_diff)
        max_f1_diff = max(max_f1_diff, f1_diff)
        max_auc_diff = max(max_auc_diff, auc_diff)

print(f"Max Logistic Regression metric differences: Acc={max_acc_diff:.2e}, F1={max_f1_diff:.2e}, AUC={max_auc_diff:.2e}")

# Check breakdown by Regime and Time Period
# Time periods: What years/ranges exist?
df_pred['Date'] = pd.to_datetime(df_pred['Date'])
df_pred['Year'] = df_pred['Date'].dt.year

print("\nObservations by Year:")
print(df_pred.groupby('Year')['Date'].agg(['min', 'max', 'count']))

print("\nObservations per Fold (Dates):")
print(df_pred.groupby('Fold')['Date'].agg(['min', 'max', 'count']))

print("\nObservations per Horizon (per Model):")
print(df_pred[df_pred['Model'] == 'Ridge'].groupby('Horizon')['Date'].count())

print("\nObservations per Fold per Horizon (per Model):")
print(df_pred[df_pred['Model'] == 'Ridge'].groupby(['Horizon', 'Fold'])['Date'].count().unstack())
