"""
Script: run_h1_block_length_sensitivity_B4999.py
Purpose: High-Replication (B = 4,999) Bootstrap Validation for H1 Statistical Robustness
         (Paired Synchronized Moving-Block Bootstrap across b in {20, 40, 60, 80, 120}).
Author: Research Team
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import common
import hac
import bootstrap
import multiple_testing
import h1_multi_horizon

def run_b4999_validation(
    block_lengths=(20, 40, 60, 80, 120),
    n_boot=4999,
    hac_bandwidth=20,
    base_seed=42
):
    print("=" * 90)
    print("TASK: HIGH-REPLICATION BOOTSTRAP VALIDATION FOR H1 (B = 4,999)")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Pre-flight Validation Checklist
    # -------------------------------------------------------------------------
    print("\n[VALIDATION CHECKLIST BEFORE EXECUTION]")
    print("1. Data & Loss Differentials: Identical unified test predictions and normalized loss diffs.")
    print("2. Model Ordering: Identical order in ALL_MODELS (12 models).")
    print(f"   -> {common.ALL_MODELS}")
    print("3. Forecast Horizons: Synchronized 5 horizons ('1D', '5D', '20D', '60D', '120D').")
    print("4. Null Imposition: Identical column-centering + grand mean (X - theta_hat + grand_mean).")
    print("5. Block Resampling: Identical paired synchronized moving-block generator (shared indices).")
    print("6. HAC & Studentization: Identical Bartlett kernel with L = 20 in every replication.")
    print("7. Wald Test: Identical R_CONTRAST (4x5), rank determination, and pinv fallback.")
    print("8. Seed Formula: Seed(b, m) = 42 + 1000 * b + m_idx (strictly preserved).")
    print(f"9. Methodological Change: ONLY B increased from 500 to {n_boot}.")
    print("10. Multiple-Testing: Holm-Bonferroni correction applied independently per block length.")
    print("=" * 90)

    # 1. Load data
    print("\n[1/5] Loading unified test predictions and loss differentials...")
    df_test_preds = common.load_unified_predictions('test')
    df_test_diff, hm_mse = common.compute_loss_differentials(df_test_preds)
    print(f"  -> Loaded {len(df_test_diff)} rows across 12 models. Holdout sample T = 1362.")

    # 2. Compute asymptotic HAC baseline (L = 20)
    print("\n[2/5] Computing asymptotic HAC baseline (L = 20)...")
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

    # 3. Run Bootstrap for B = 4999
    print(f"\n[3/5] Running Paired Synchronized Moving-Block Bootstrap (B = {n_boot})...")
    total_tasks = len(block_lengths) * len(common.ALL_MODELS)
    detailed_rows = []

    start_time = time.time()
    with tqdm(total=total_tasks, desc="Bootstrap Progress (B=4999)", unit="model-b") as pbar:
        for b in block_lengths:
            pbar.set_postfix({"Current Block b": b})
            block_rows = []

            for m_idx, model in enumerate(common.ALL_MODELS):
                piv = h1_multi_horizon.get_model_loss_matrix(df_test_diff, model)
                X = piv.values  # (T, 5)
                T, K = X.shape

                # Sample statistics
                theta_hat = np.mean(X, axis=0)
                omega_act = hac.newey_west_covariance(X, bandwidth=hac_bandwidth)
                act_res = hac.wald_test(theta_hat, omega_act / T, R=h1_multi_horizon.R_CONTRAST)
                W_act = act_res['stat']

                # Enforce H0
                grand_mean = np.mean(X)
                X_h0 = X - theta_hat + grand_mean

                # Exact reproducible seed
                seed_for_run = base_seed + int(b) * 1000 + m_idx

                # Bootstrap loop
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

            # Holm adjustment across 12 models for this block length
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
    print(f"\n  -> Completed {total_tasks} bootstrap experiments in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

    # 4. Save B=4999 Results
    print("\n[4/5] Saving B = 4999 detailed results and summary...")
    out_dir = os.path.join(common.REPO_ROOT, 'results', 'hypothesis_tests', 'H1')
    os.makedirs(out_dir, exist_ok=True)

    df_b4999 = pd.DataFrame(detailed_rows)
    csv_b4999_path = os.path.join(out_dir, 'h1_block_length_sensitivity_B4999.csv')
    df_b4999.to_csv(csv_b4999_path, index=False)
    print(f"  -> Detailed B=4999 results saved to: {csv_b4999_path}")

    # Summary table
    summary_rows = []
    for b in block_lengths:
        sub = df_b4999[df_b4999['Block_Length'] == b]
        raw_rejects = sub[sub['Bootstrap_Reject_raw_05']]['Model'].tolist()
        holm_rejects = sub[sub['Bootstrap_Reject_holm_05']]['Model'].tolist()
        hac_holm_rejects = sub[sub['HAC_Reject_05']]['Model'].tolist()

        summary_rows.append({
            'Block_Length': b,
            'Total_Models': len(sub),
            'Raw_Sig_Count_05': len(raw_rejects),
            'Raw_Sig_Models': ", ".join(raw_rejects) if raw_rejects else "None",
            'Holm_Sig_Count_05': len(holm_rejects),
            'Holm_Sig_Models': ", ".join(holm_rejects) if holm_rejects else "None",
            'HAC_Baseline_Holm_Sig_Models': ", ".join(hac_holm_rejects),
            'Consistent_with_HAC_Holm': (set(holm_rejects) == set(hac_holm_rejects))
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_summary_path = os.path.join(out_dir, 'h1_block_length_summary_B4999.csv')
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"  -> Summary B=4999 results saved to: {csv_summary_path}")

    # Metadata log
    metadata = {
        'task': 'High-Replication Bootstrap Validation for H1',
        'n_boot': n_boot,
        'block_lengths': list(block_lengths),
        'hac_bandwidth': hac_bandwidth,
        'base_seed': base_seed,
        'sample_size_T': 1362,
        'elapsed_seconds': elapsed,
        'models': common.ALL_MODELS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    meta_path = os.path.join(out_dir, 'h1_b4999_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"  -> Metadata saved to: {meta_path}")

    # 5. Comparison with B = 500
    print("\n[5/5] Generating B = 500 vs. B = 4999 Comparison Table...")
    csv_b500_path = os.path.join(out_dir, 'h1_block_length_sensitivity.csv')
    if os.path.exists(csv_b500_path):
        df_b500 = pd.read_csv(csv_b500_path)
        
        # Merge on Model and Block_Length
        df_comp = pd.merge(
            df_b500[['Model', 'Block_Length', 'HAC_Wald_Stat', 'HAC_df', 'HAC_pvalue', 'HAC_pvalue_holm',
                     'Bootstrap_raw_pvalue', 'Bootstrap_pvalue_holm', 'Bootstrap_Reject_holm_05']],
            df_b4999[['Model', 'Block_Length', 'Bootstrap_raw_pvalue', 'Bootstrap_pvalue_holm', 'Bootstrap_Reject_holm_05']],
            on=['Model', 'Block_Length'],
            suffixes=('_B500', '_B4999')
        )
        
        df_comp['Abs_Raw_Diff'] = np.abs(df_comp['Bootstrap_raw_pvalue_B4999'] - df_comp['Bootstrap_raw_pvalue_B500'])
        df_comp['Decision_Changed'] = (df_comp['Bootstrap_Reject_holm_05_B500'] != df_comp['Bootstrap_Reject_holm_05_B4999'])

        comp_path = os.path.join(out_dir, 'h1_block_length_comparison_B500_vs_B4999.csv')
        df_comp.to_csv(comp_path, index=False)
        print(f"  -> Comparison saved to: {comp_path}")
    else:
        print("  -> Warning: B=500 file not found for comparison.")

    print("\n" + "=" * 90)
    print("B = 4999 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == '__main__':
    run_b4999_validation(
        block_lengths=(20, 40, 60, 80, 120),
        n_boot=4999,
        hac_bandwidth=20,
        base_seed=42
    )
