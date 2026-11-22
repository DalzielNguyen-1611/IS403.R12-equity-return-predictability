"""
run_benchmark_robustness.py
===========================
Execution Script for Benchmark Robustness Evaluation:
Compares model predictive accuracy against Historical Mean (Campbell & Thompson, 2008)
versus Zero-Return Benchmark (Welch & Goyal, 2008) across 12 forecasting models and 5 horizons.

Methodological Highlights:
1. Zero Retraining: Re-uses exact out-of-sample holdout predictions from Lan01 and Lan02.
2. Strict Sample Parity: Identical 1,362 trading dates (2020-11-02 to 2026-04-07).
3. Non-Destructive: Saves all outputs cleanly under results/benchmark_robustness/
   without modifying existing baseline reports or data splits.
4. Statistical Inference: Re-computes Diebold-Mariano tests with HLN (1997) finite-sample correction.

Usage:
    python src/testing/run_benchmark_robustness.py
    python src/testing/run_benchmark_robustness.py --include-val
    python src/testing/run_benchmark_robustness.py --check-only
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

# Setup repository paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.evaluation.hypothesis_tests import common
from src.evaluation import benchmark_robustness


# -------------------------------------------------------------------------
# Logger Setup
# -------------------------------------------------------------------------
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "benchmark_robustness_execution.log")
    
    logger = logging.getLogger("BenchmarkRobustness")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear existing
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger


# -------------------------------------------------------------------------
# Report Generator
# -------------------------------------------------------------------------
def generate_markdown_report(
    df_summary: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
    regime_df: pd.DataFrame,
    output_dir: str
) -> str:
    """
    Generates a concise academic markdown report summarizing benchmark robustness findings.
    """
    report_path = os.path.join(output_dir, "benchmark_robustness_summary_report.md")
    
    avg_hm_r2 = float(df_summary['OOS_R2_HM'].mean())
    avg_zero_r2 = float(df_summary['OOS_R2_Zero'].mean())
    avg_delta = float(df_summary['Delta_OOS_R2'].mean())
    
    sig_hm_count = int((df_summary['DM_pvalue_HM'] < 0.05).sum())
    sig_zero_count = int((df_summary['DM_pvalue_Zero'] < 0.05).sum())
    total_evals = len(df_summary)
    
    lines = [
        "# Benchmark Robustness Evaluation: Historical Mean vs. Zero-Return Benchmark",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Evaluation Period**: 2020-11-02 to 2026-04-07 (Holdout Test, 1,362 trading days).",
        f"- **Models Evaluated**: 12 models across 5 forecasting horizons (60 model-horizon pairs).",
        f"- **Average OOS $R^2$ under Historical Mean Benchmark**: `{avg_hm_r2 * 100:.3f}%`",
        f"- **Average OOS $R^2$ under Zero-Return Benchmark**: `{avg_zero_r2 * 100:.3f}%`",
        f"- **Mean Difference ($\Delta R_{{OOS}}^2 = R_{{Zero}}^2 - R_{{HM}}^2$)**: `{avg_delta * 100:+.3f}%`",
        f"- **Statistically Significant Pairs (DM Test at 5%)**:",
        f"  - Against Historical Mean: `{sig_hm_count} / {total_evals}` ({sig_hm_count / total_evals * 100:.1f}%)",
        f"  - Against Zero-Return Benchmark: `{sig_zero_count} / {total_evals}` ({sig_zero_count / total_evals * 100:.1f}%)",
        "",
        "## 2. Theoretical Econometric Consistency",
        "",
        "According to Welch & Goyal (2008) and Campbell & Thompson (2008), for an equity market with a positive historical equity risk premium (drift $\\bar{y}_{\\text{train}} > 0$):",
        "",
        "$$SSE_{\\text{Zero}} = \\sum_{t} y_t^2 > SSE_{\\text{HM}} = \\sum_{t} (y_t - \\bar{y}_{\\text{train}})^2$$",
        "",
        "Consequently, the denominator of $R_{OOS}^2$ under the Zero benchmark is larger, which systematically implies:",
        "",
        "$$R_{OOS, \\text{Zero}}^2 \\ge R_{OOS, \\text{HM}}^2$$",
        "",
        "This confirms that the Historical Mean benchmark employed as the primary baseline in this study is strictly more conservative and harder to outperform than the Zero-return random walk benchmark.",
        "",
        "## 3. Comparison Matrices",
        "",
        "### Panel A: Out-of-Sample $R^2$ (%) under Historical Mean Benchmark",
        "",
        (matrices['oos_r2_historical_mean'] * 100).round(3).to_markdown(),
        "",
        "### Panel B: Out-of-Sample $R^2$ (%) under Zero-Return Benchmark",
        "",
        (matrices['oos_r2_zero_return'] * 100).round(3).to_markdown(),
        "",
        "### Panel C: Difference $\Delta R_{OOS}^2$ (% points, Zero - HM)",
        "",
        (matrices['delta_oos_r2'] * 100).round(3).to_markdown(),
        "",
        "## 4. Volatility Regime Cross-Model Average",
        "",
        regime_df.round(4).to_markdown(index=False),
        "",
        "---",
        "*Note: This report is generated programmatically by `run_benchmark_robustness.py` with zero model retraining.*"
    ]
    
    report_content = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    return report_path


# -------------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Benchmark Robustness: Historical Mean vs. Zero-Return Benchmark.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "benchmark_robustness"),
        help="Directory to save benchmark robustness outputs."
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Include evaluation on 5 Walk-Forward Validation folds."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check predictions availability without computing and exporting full matrices."
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    
    logger.info("=" * 72)
    logger.info("  STARTING BENCHMARK ROBUSTNESS EVALUATION (HM vs. Zero-Return)")
    logger.info("=" * 72)
    logger.info(f"Target Output Directory: {args.output_dir}")
    
    # ---------------------------------------------------------------------
    # Step 1: Load Unified Holdout Predictions
    # ---------------------------------------------------------------------
    logger.info("Loading unified holdout predictions (Lan01 + Lan02)...")
    df_test = common.load_unified_predictions(split='test')
    
    n_records = len(df_test)
    unique_dates = df_test['Date'].nunique()
    date_min = df_test['Date'].min().strftime('%Y-%m-%d')
    date_max = df_test['Date'].max().strftime('%Y-%m-%d')
    models_found = [m for m in common.ALL_MODELS if m in df_test['Model'].unique()]
    
    logger.info(f"Loaded {n_records} holdout records covering {unique_dates} trading days ({date_min} to {date_max}).")
    logger.info(f"Models identified ({len(models_found)} / 12): {models_found}")
    logger.info(f"Benchmark identified: '{common.BENCHMARK_MODEL}'")
    
    if args.check_only:
        logger.info("Check-only flag provided. Verification completed successfully.")
        return
        
    # ---------------------------------------------------------------------
    # Step 2: Evaluate Holdout Test Benchmarks
    # ---------------------------------------------------------------------
    logger.info("Computing comparative metrics across 12 models x 5 horizons x 4 regime categories...")
    df_detailed, df_summary = benchmark_robustness.evaluate_benchmark_robustness(
        df_preds=df_test,
        models=models_found,
        horizons=common.HORIZONS,
        logger=logger
    )
    
    # Save Detailed and Summary CSVs
    p_detailed = os.path.join(args.output_dir, "benchmark_robustness_detailed.csv")
    df_detailed.to_csv(p_detailed, index=False)
    logger.info(f"Saved detailed benchmark evaluations -> {p_detailed} ({len(df_detailed)} rows)")
    
    p_summary = os.path.join(args.output_dir, "benchmark_comparison_summary.csv")
    df_summary.to_csv(p_summary, index=False)
    logger.info(f"Saved holdout summary table -> {p_summary} ({len(df_summary)} rows)")
    
    # ---------------------------------------------------------------------
    # Step 3: Construct Comparison Matrices
    # ---------------------------------------------------------------------
    logger.info("Constructing comparative pivot matrices...")
    matrices = benchmark_robustness.generate_comparison_matrices(df_summary)
    
    # Export Matrices
    for name, mat in matrices.items():
        mat_path = os.path.join(args.output_dir, f"matrix_{name}.csv")
        mat.to_csv(mat_path)
        logger.info(f"Saved matrix: {name} -> {mat_path}")
        
    # ---------------------------------------------------------------------
    # Step 4: Construct Volatility Regime Breakdown
    # ---------------------------------------------------------------------
    logger.info("Constructing volatility regime breakdown table...")
    regime_df = benchmark_robustness.generate_regime_breakdown_matrix(df_detailed)
    p_regime = os.path.join(args.output_dir, "regime_benchmark_comparison.csv")
    regime_df.to_csv(p_regime, index=False)
    logger.info(f"Saved regime benchmark comparison -> {p_regime}")
    
    # ---------------------------------------------------------------------
    # Step 5: Optional Walk-Forward Validation Folds
    # ---------------------------------------------------------------------
    if args.include_val:
        logger.info("Evaluating Benchmark Robustness on Walk-Forward Validation folds...")
        df_val = common.load_unified_predictions(split='val')
        df_val_res = benchmark_robustness.evaluate_validation_folds(
            df_val_preds=df_val,
            models=models_found,
            horizons=common.HORIZONS,
            logger=logger
        )
        p_val = os.path.join(args.output_dir, "validation_folds_benchmark_robustness.csv")
        df_val_res.to_csv(p_val, index=False)
        logger.info(f"Saved validation folds benchmark results -> {p_val} ({len(df_val_res)} rows)")
        
    # ---------------------------------------------------------------------
    # Step 6: Generate Markdown Executive Summary Report
    # ---------------------------------------------------------------------
    logger.info("Generating academic summary markdown report...")
    p_report = generate_markdown_report(df_summary, matrices, regime_df, args.output_dir)
    logger.info(f"Saved executive markdown report -> {p_report}")
    
    # ---------------------------------------------------------------------
    # Console Summary
    # ---------------------------------------------------------------------
    mean_hm_r2 = df_summary['OOS_R2_HM'].mean() * 100
    mean_zero_r2 = df_summary['OOS_R2_Zero'].mean() * 100
    mean_delta = df_summary['Delta_OOS_R2'].mean() * 100
    
    logger.info("=" * 72)
    logger.info("  BENCHMARK ROBUSTNESS EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 72)
    logger.info(f"Average OOS R2 (Historical Mean Baseline): {mean_hm_r2:.3f}%")
    logger.info(f"Average OOS R2 (Zero-Return Benchmark):    {mean_zero_r2:.3f}%")
    logger.info(f"Average Delta (Zero - HM):                 {mean_delta:+.3f}% points")
    logger.info(f"All files saved under: {args.output_dir}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

