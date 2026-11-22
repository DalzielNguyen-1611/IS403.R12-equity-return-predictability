import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common import (
    load_unified_predictions,
    compute_loss_differentials,
    ALL_MODELS,
    HORIZONS,
    REPO_ROOT
)
import h1_multi_horizon
import h2_regime_cpa
import h3_temporal_instability

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HypothesisTestingRunner")

def setup_directories():
    base_out = os.path.join(REPO_ROOT, 'results', 'hypothesis_tests')
    dirs = {
        'base': base_out,
        'H1': os.path.join(base_out, 'H1'),
        'H2': os.path.join(base_out, 'H2'),
        'H3': os.path.join(base_out, 'H3'),
        'summary': os.path.join(base_out, 'summary'),
        'figures': os.path.join(base_out, 'figures')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def execute_pipeline(quick_mode=False):
    dirs = setup_directories()
    n_boot = 500 if quick_mode else 2000
    
    logger.info("================================================================================")
    logger.info("STARTING RIGOROUS HYPOTHESIS TESTING PIPELINE (H1, H2, H3)")
    logger.info("================================================================================")
    
    # 1. Load Data
    logger.info("Loading unified hold-out test predictions (Lan01 + Lan02)...")
    df_test_preds = load_unified_predictions('test')
    df_test_diff, hm_mse = compute_loss_differentials(df_test_preds)
    logger.info(f"Loaded {len(df_test_diff)} test prediction rows across {len(ALL_MODELS)} models.")
    
    logger.info("Loading unified validation predictions...")
    df_val_preds = load_unified_predictions('val')
    df_val_diff, _ = compute_loss_differentials(df_val_preds)
    
    # -------------------------------------------------------------------------
    # H1: Multi-Horizon Predictive Performance
    # -------------------------------------------------------------------------
    logger.info("\n>>> Executing H1: Multi-Horizon HAC Wald Test...")
    df_h1_wald = h1_multi_horizon.run_h1_hac_wald(df_test_diff, bandwidth=20)
    df_h1_wald.to_csv(os.path.join(dirs['H1'], 'h1_hac_wald_results.csv'), index=False)
    df_h1_wald.to_csv(os.path.join(dirs['H1'], 'h1_holm_adjusted_results.csv'), index=False)
    
    logger.info(">>> Executing H1: Bandwidth Sensitivity (5, 20, 60, 119 lags)...")
    df_h1_sens = h1_multi_horizon.run_h1_bandwidth_sensitivity(df_test_diff, bandwidths=[5, 20, 60, 119])
    df_h1_sens.to_csv(os.path.join(dirs['H1'], 'h1_hac_bandwidth_sensitivity.csv'), index=False)
    
    logger.info(f">>> Executing H1: Paired Moving-Block Bootstrap (B={n_boot}, Block=60)...")
    df_h1_boot = h1_multi_horizon.run_h1_bootstrap(df_test_diff, block_length=60, n_boot=n_boot, bandwidth=20)
    df_h1_boot.to_csv(os.path.join(dirs['H1'], 'h1_block_bootstrap_results.csv'), index=False)
    
    # -------------------------------------------------------------------------
    # H2: Volatility-Regime Dependence and Horizon-Regime Interaction
    # -------------------------------------------------------------------------
    logger.info("\n>>> Executing H2: Regime Breakdown Summary...")
    df_h2_breakdown = h2_regime_cpa.compute_regime_breakdown(df_test_diff)
    df_h2_breakdown.to_csv(os.path.join(dirs['H2'], 'h2_regime_performance.csv'), index=False)
    
    logger.info(">>> Executing H2: Giacomini-White CPA Tests...")
    df_h2_cpa = h2_regime_cpa.run_h2_gw_cpa(df_test_diff, bandwidth=20)
    df_h2_cpa.to_csv(os.path.join(dirs['H2'], 'h2_gw_cpa_results.csv'), index=False)
    df_h2_cpa.to_csv(os.path.join(dirs['H2'], 'h2_holm_adjusted_results.csv'), index=False)
    
    logger.info(">>> Executing H2: Horizon x Regime Interaction Test...")
    df_h2_int = h2_regime_cpa.run_h2_horizon_regime_interaction(df_test_diff, bandwidth=20)
    df_h2_int.to_csv(os.path.join(dirs['H2'], 'h2_horizon_regime_interaction.csv'), index=False)
    
    # -------------------------------------------------------------------------
    # H3: Temporal Instability of Predictive Performance
    # -------------------------------------------------------------------------
    logger.info("\n>>> Executing H3: Fluctuation Instability Tests (Window=252)...")
    df_h3_fluc, df_h3_series = h3_temporal_instability.run_h3_fluctuation_test(
        df_test_diff, window_size=252, bandwidth=20, n_boot=min(n_boot, 500)
    )
    df_h3_fluc.to_csv(os.path.join(dirs['H3'], 'h3_instability_results.csv'), index=False)
    df_h3_fluc.to_csv(os.path.join(dirs['H3'], 'h3_holm_adjusted_results.csv'), index=False)
    df_h3_series.to_csv(os.path.join(dirs['H3'], 'h3_rolling_fluctuation_series.csv'), index=False)
    
    logger.info(">>> Executing H3: Predefined Hold-out Block Wald Test...")
    df_h3_holdout_blocks = h3_temporal_instability.run_h3_block_wald_test(
        df_test_diff, h3_temporal_instability.HOLDOUT_BLOCKS, sample_name='Hold-out', bandwidth=20
    )
    df_h3_holdout_blocks.to_csv(os.path.join(dirs['H3'], 'h3_holdout_block_results.csv'), index=False)
    
    logger.info(">>> Executing H3: Predefined Validation Block Wald Test...")
    df_h3_val_blocks = h3_temporal_instability.run_h3_block_wald_test(
        df_val_diff, h3_temporal_instability.VAL_BLOCKS, sample_name='Validation', bandwidth=20
    )
    df_h3_val_blocks.to_csv(os.path.join(dirs['H3'], 'h3_validation_block_results.csv'), index=False)
    
    # -------------------------------------------------------------------------
    # Consolidated Summary
    # -------------------------------------------------------------------------
    logger.info("\n>>> Generating Consolidated Summary...")
    summary_rows = []
    
    # H1 summary rows
    for _, r in df_h1_wald.iterrows():
        summary_rows.append({
            'Hypothesis': 'H1',
            'Model': r['Model'],
            'Horizon': 'All (1D-120D)',
            'Test_Method': f"HAC Wald Test (bw={r['Bandwidth']})",
            'Statistic': r['Wald_Stat'],
            'df': r['df'],
            'Raw_pvalue': r['pvalue'],
            'Holm_pvalue': r['pvalue_holm'],
            'Significant_05': r['Significant_05'],
            'Sample': 'Hold-out',
            'Sample_Size': r['T'],
            'Conclusion': 'Significant horizon variation' if r['Significant_05'] else 'No significant horizon variation'
        })
        
    # H2 interaction summary rows
    for _, r in df_h2_int.iterrows():
        summary_rows.append({
            'Hypothesis': 'H2',
            'Model': r['Model'],
            'Horizon': 'All (Horizon x Regime)',
            'Test_Method': f"Joint Interaction Wald (bw={r['Bandwidth']})",
            'Statistic': r['Interaction_Wald_Stat'],
            'df': r['df'],
            'Raw_pvalue': r['pvalue'],
            'Holm_pvalue': r['pvalue_holm'],
            'Significant_05': r['Significant_05'],
            'Sample': 'Hold-out',
            'Sample_Size': r['T'],
            'Conclusion': 'Significant horizon-regime interaction' if r['Significant_05'] else 'No significant interaction'
        })
        
    # H3 block holdout summary rows
    for _, r in df_h3_holdout_blocks.iterrows():
        summary_rows.append({
            'Hypothesis': 'H3',
            'Model': r['Model'],
            'Horizon': r['Horizon'],
            'Test_Method': 'Hold-out Block Wald Test',
            'Statistic': r['Wald_Stat'],
            'df': r['df'],
            'Raw_pvalue': r['pvalue'],
            'Holm_pvalue': r['pvalue_holm'],
            'Significant_05': r['Significant_05'],
            'Sample': 'Hold-out (3 Blocks)',
            'Sample_Size': r['T_Total'],
            'Conclusion': 'Significant temporal change across blocks' if r['Significant_05'] else 'No significant block difference'
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(dirs['summary'], 'hypothesis_testing_summary.csv'), index=False)
    
    # Generate Markdown Report
    generate_markdown_report(dirs, df_h1_wald, df_h2_cpa, df_h2_int, df_h3_holdout_blocks, df_h3_val_blocks)
    
    logger.info("================================================================================")
    logger.info("HYPOTHESIS TESTING COMPLETE. Outputs saved in results/hypothesis_tests/")
    logger.info("================================================================================")

def generate_markdown_report(dirs, df_h1, df_h2_cpa, df_h2_int, df_h3_hold, df_h3_val):
    report_path = os.path.join(dirs['summary'], 'hypothesis_testing_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Statistical Hypothesis Testing Report (H1, H2, H3)\n\n")
        f.write("## 1. Overview\n")
        f.write("This report provides rigorous statistical inference for the three research hypotheses evaluating the S&P 500 forecasting models.\n\n")
        
        f.write("## 2. Hypothesis H1: Multi-Horizon Predictive Performance\n")
        f.write("Joint equality null hypothesis: $H_0: \\mu_{1D} = \\mu_{5D} = \\mu_{20D} = \\mu_{60D} = \\mu_{120D}$.\n\n")
        f.write(df_h1[['Model', 'Wald_Stat', 'df', 'pvalue', 'pvalue_holm', 'Significant_05']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Hypothesis H2: Volatility-Regime Interaction\n")
        f.write("Joint interaction null hypothesis: $H_0: \\text{All Horizon } \\times \\text{ Regime Interaction Effects} = 0$.\n\n")
        f.write(df_h2_int[['Model', 'Interaction_Wald_Stat', 'df', 'pvalue', 'pvalue_holm', 'Significant_05']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. Hypothesis H3: Temporal Instability across Hold-out Blocks\n")
        f.write("Block equality null hypothesis: $H_0: \\mu_{2021-2022} = \\mu_{2023-2024} = \\mu_{2025-2026}$.\n\n")
        f.write(df_h3_hold[df_h3_hold['Horizon'] == '120D'][['Model', 'Wald_Stat', 'df', 'pvalue', 'pvalue_holm', 'Significant_05']].to_markdown(index=False))
        f.write("\n")
        
    logger.info(f"Report written to {report_path}")

if __name__ == "__main__":
    quick = '--quick' in sys.argv
    execute_pipeline(quick_mode=quick)
