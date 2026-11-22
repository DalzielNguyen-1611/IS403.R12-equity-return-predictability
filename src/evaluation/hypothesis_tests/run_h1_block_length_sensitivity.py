"""
Script: run_h1_block_length_sensitivity.py
Purpose: Block-length sensitivity analysis for H1 Statistical Robustness
         (Paired Synchronized Moving-Block Bootstrap across b in {20, 40, 60, 80, 120}).
Author: Research Team
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure local imports work when running from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import common
import hac
import bootstrap
import multiple_testing
import h1_multi_horizon

def run_block_length_sensitivity(
    block_lengths=(20, 40, 60, 80, 120),
    n_boot=500,
    hac_bandwidth=20,
    base_seed=42
):
    print("=" * 80)
    print("H1 MULTI-HORIZON STATISTICAL ROBUSTNESS: BLOCK-LENGTH SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Configurations:")
    print(f"  - Block lengths (b): {list(block_lengths)}")
    print(f"  - Bootstrap replications (B): {n_boot}")
    print(f"  - HAC studentization bandwidth (L): {hac_bandwidth}")
    print(f"  - Base random seed: {base_seed}")
    print(f"  - Models: {len(common.ALL_MODELS)} models across {len(common.HORIZONS)} horizons")
    print("=" * 80)

    # 1. Load data
    print("\n[1/4] Loading unified test predictions and loss differentials...")
    df_test_preds = common.load_unified_predictions('test')
    df_test_diff, hm_mse = common.compute_loss_differentials(df_test_preds)
    print(f"  -> Loaded {len(df_test_diff)} rows across 12 models. Synchronized holdout sample.")

    # 2. Compute asymptotic HAC baseline (L = 20)
    print("\n[2/4] Computing asymptotic HAC baseline (L = 20)...")
    df_hac_baseline = h1_multi_horizon.run_h1_hac_wald(df_test_diff, bandwidth=hac_bandwidth)
    hac_lookup = {}
    for _, row in df_hac_baseline.iterrows():
        hac_lookup[row['Model']] = {
            'Wald_Stat': row['Wald_Stat'],
            'df': row['df'],
            'pvalue': row['pvalue'],
            'pvalue_holm': row['pvalue_holm'],
            'Significant_05': row['Significant_05']
        }
    print("  -> HAC baseline computed successfully.")

    # 3. Run Bootstrap for each block length
    print("\n[3/4] Running Paired Synchronized Moving-Block Bootstrap across block lengths...")
    total_tasks = len(block_lengths) * len(common.ALL_MODELS)
    detailed_rows = []

    start_time = time.time()
    with tqdm(total=total_tasks, desc="Overall Progress", unit="model-b") as pbar:
        for b in block_lengths:
            pbar.set_postfix({"Current Block b": b})
            block_rows = []
            
            for m_idx, model in enumerate(common.ALL_MODELS):
                # Retrieve aligned (T, 5) loss differential matrix
                piv = h1_multi_horizon.get_model_loss_matrix(df_test_diff, model)
                X = piv.values  # (T, 5)
                T, K = X.shape

                # Actual sample statistics
                theta_hat = np.mean(X, axis=0)
                omega_act = hac.newey_west_covariance(X, bandwidth=hac_bandwidth)
                act_res = hac.wald_test(theta_hat, omega_act / T, R=h1_multi_horizon.R_CONTRAST)
                W_act = act_res['stat']

                # Enforce H0: subtract column means, add grand mean
                grand_mean = np.mean(X)
                X_h0 = X - theta_hat + grand_mean

                # Consistent random seed control per (block_length, model)
                # This guarantees determinism, reproducibility, and minimal Monte Carlo noise
                seed_for_run = base_seed + int(b) * 1000 + m_idx

                # Paired Moving-Block Bootstrap
                b_stats = []
                for X_star in bootstrap.paired_block_bootstrap_matrix(
                    X_h0, block_length=b, n_boot=n_boot, seed=seed_for_run
                ):
                    th_star = np.mean(X_star, axis=0)
                    om_star = hac.newey_west_covariance(X_star, bandwidth=hac_bandwidth)
                    w_star = hac.wald_test(th_star, om_star / T, R=h1_multi_horizon.R_CONTRAST)['stat']
                    b_stats.append(w_star)

                b_stats = np.array(b_stats)
                boot_pval = float(np.mean(b_stats >= W_act))

                block_rows.append({
                    'Model': model,
                    'T': T,
                    'Block_Length': b,
                    'N_Boot': n_boot,
                    'HAC_Bandwidth': hac_bandwidth,
                    'HAC_Wald_Stat': hac_lookup[model]['Wald_Stat'],
                    'HAC_df': hac_lookup[model]['df'],
                    'HAC_pvalue': hac_lookup[model]['pvalue'],
                    'HAC_pvalue_holm': hac_lookup[model]['pvalue_holm'],
                    'HAC_Reject_05': hac_lookup[model]['Significant_05'],
                    'Bootstrap_Wald_Stat': W_act,
                    'Bootstrap_raw_pvalue': boot_pval,
                    'Seed': seed_for_run
                })
                pbar.update(1)

            # Apply Holm-Bonferroni correction across the 12 models for this block length
            df_block = pd.DataFrame(block_rows)
            df_block = multiple_testing.adjust_dataframe_column(
                df_block,
                pval_col='Bootstrap_raw_pvalue',
                adj_col='Bootstrap_pvalue_holm'
            )
            df_block['Bootstrap_Reject_raw_05'] = df_block['Bootstrap_raw_pvalue'] < 0.05
            df_block['Bootstrap_Reject_holm_05'] = df_block['Bootstrap_pvalue_holm'] < 0.05

            detailed_rows.extend(df_block.to_dict('records'))

    elapsed = time.time() - start_time
    print(f"\n  -> Finished {total_tasks} bootstrap experiments in {elapsed:.2f} seconds.")

    # 4. Consolidate and Save Results
    print("\n[4/4] Generating summary and saving artifacts...")
    df_all = pd.DataFrame(detailed_rows)

    # Output directory
    out_dir = os.path.join(common.REPO_ROOT, 'results', 'hypothesis_tests', 'H1')
    os.makedirs(out_dir, exist_ok=True)

    csv_detailed_path = os.path.join(out_dir, 'h1_block_length_sensitivity.csv')
    df_all.to_csv(csv_detailed_path, index=False)
    print(f"  -> Detailed results saved to: {csv_detailed_path}")

    # Build summary table across block lengths
    summary_rows = []
    for b in block_lengths:
        sub = df_all[df_all['Block_Length'] == b]
        raw_rejects = sub[sub['Bootstrap_Reject_raw_05']]['Model'].tolist()
        holm_rejects = sub[sub['Bootstrap_Reject_holm_05']]['Model'].tolist()
        
        # Check consistency with HAC baseline (ARIMA rejects, others do not)
        hac_holm_rejects = sub[sub['HAC_Reject_05']]['Model'].tolist()
        is_consistent_with_hac = (set(holm_rejects) == set(hac_holm_rejects))

        summary_rows.append({
            'Block_Length': b,
            'Total_Models': len(sub),
            'Raw_Sig_Count_05': len(raw_rejects),
            'Raw_Sig_Models': ", ".join(raw_rejects) if raw_rejects else "None",
            'Holm_Sig_Count_05': len(holm_rejects),
            'Holm_Sig_Models': ", ".join(holm_rejects) if holm_rejects else "None",
            'HAC_Baseline_Holm_Sig_Models': ", ".join(hac_holm_rejects),
            'Consistent_with_HAC_Holm': is_consistent_with_hac
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_summary_path = os.path.join(out_dir, 'h1_block_length_summary.csv')
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"  -> Summary table saved to: {csv_summary_path}")

    # Display console outputs
    print("\n" + "=" * 110)
    print("DETAILED RESULTS (Sample View: ARIMA, TSMixer, TCN, SVR, Ridge)")
    print("=" * 110)
    cols_display = [
        'Model', 'HAC_pvalue', 'HAC_pvalue_holm', 'Block_Length',
        'Bootstrap_raw_pvalue', 'Bootstrap_pvalue_holm', 'Bootstrap_Reject_holm_05'
    ]
    sample_models = ['ARIMA', 'TSMixer', 'TCN', 'SVR', 'Ridge']
    print(df_all[df_all['Model'].isin(sample_models)][cols_display].to_string(index=False))

    print("\n" + "=" * 110)
    print("BLOCK-LENGTH SENSITIVITY SUMMARY TABLE")
    print("=" * 110)
    print(df_summary.to_string(index=False))
    print("=" * 110)
    print("EXECUTION COMPLETED SUCCESSFULLY!\n")

if __name__ == '__main__':
    run_block_length_sensitivity(
        block_lengths=(20, 40, 60, 80, 120),
        n_boot=500,
        hac_bandwidth=20,
        base_seed=42
    )
