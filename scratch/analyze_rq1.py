import pandas as pd
import numpy as np

df = pd.read_csv('results/analysis/rq1_horizon.csv')
print("=== Complete rq1_horizon.csv ===")
print(df.to_string())

# Separate Logistic Regression
reg_df = df[df['Model'] != 'Logistic_Regression'].copy()

# Print OOS_R2 matrix across horizons
r2_cols = ['1D_OOS_R2', '5D_OOS_R2', '20D_OOS_R2', '60D_OOS_R2', '120D_OOS_R2']
da_cols = ['1D_DA', '5D_DA', '20D_DA', '60D_DA', '120D_DA']

print("\n=== OOS_R2 across Horizons ===")
print(reg_df[['Model'] + r2_cols].to_string(index=False))

print("\n=== DA across Horizons ===")
print(reg_df[['Model'] + da_cols].to_string(index=False))

# Check monotonicity for each model in OOS_R2:
# Is it strictly decreasing? weakly decreasing? increasing?
def analyze_trajectory(row):
    vals = [row[c] for c in r2_cols]
    diffs = np.diff(vals)
    all_decreasing = all(d <= 0 for d in diffs)
    all_increasing = all(d >= 0 for d in diffs)
    # Check general slope (120D vs 1D)
    net_change = vals[-1] - vals[0]
    return pd.Series({
        'All_Decreasing (Monotonic Decay)': all_decreasing,
        'All_Increasing': all_increasing,
        'Net_Change (120D - 1D)': net_change,
        'Trajectory': ' -> '.join([f"{v:.4f}" for v in vals])
    })

trajectories = reg_df.apply(analyze_trajectory, axis=1)
trajectories['Model'] = reg_df['Model']
print("\n=== Monotonicity & Decay Direction ===")
print(trajectories[['Model', 'All_Decreasing (Monotonic Decay)', 'All_Increasing', 'Net_Change (120D - 1D)', 'Trajectory']].to_string(index=False))
