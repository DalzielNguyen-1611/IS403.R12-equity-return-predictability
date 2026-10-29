import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd

# Set environment variable EXP_RUN_NAME to Lan04 if not already specified
if 'EXP_RUN_NAME' not in os.environ:
    os.environ['EXP_RUN_NAME'] = 'Lan04'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
import utils
from ensemble.ensemble_models import run_ensemble, EXPERIMENT_SPECS

LAN04_EXPERIMENTS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

def ensure_benchmark_baseline(logger):
    """
    Ensure Historical_Mean predictions exist in results/Lan04/predictions/
    so that OOS R2 calculations and Diebold-Mariano tests have a baseline to compare against.
    """
    bench_preds_file = config.ALL_PREDICTIONS_PATH
    has_bench = False
    if os.path.exists(bench_preds_file):
        try:
            p_df = pd.read_csv(bench_preds_file)
            if 'Model' in p_df.columns and (p_df['Model'].str.lower() == 'historical_mean').any():
                has_bench = True
        except Exception:
            pass

    if not has_bench:
        logger.info("Initializing Historical_Mean benchmark baseline for Lan04 from Lan01...")
        lan01_preds_file = os.path.join(config.RESULTS_BASE_DIR, 'Lan01', 'predictions', '3.0.3_all_oos_predictions.csv')
        if os.path.exists(lan01_preds_file):
            try:
                lan01_df = pd.read_csv(lan01_preds_file)
                hm_df = lan01_df[lan01_df['Model'] == 'Historical_Mean'].copy()
                if not hm_df.empty:
                    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
                    if os.path.exists(bench_preds_file):
                        cur_df = pd.read_csv(bench_preds_file)
                        comb = pd.concat([hm_df, cur_df], ignore_index=True).drop_duplicates(subset=['Date', 'Model', 'Horizon', 'Fold'])
                    else:
                        comb = hm_df
                    comb.to_csv(bench_preds_file, index=False)
                    logger.info("✓ Copied Historical_Mean benchmark from Lan01 to Lan04.")
                    return
            except Exception as e:
                logger.warning(f"Could not copy Historical_Mean: {e}")

        # Fallback to generating it
        try:
            df = utils.load_dataset()
            import historical_mean.historical_mean as hm_mod
            hm_mod.run_model(df, horizons=config.HORIZONS, logger=logger)
            logger.info("✓ Generated fresh Historical_Mean benchmark for Lan04.")
        except Exception as e:
            logger.warning(f"Could not auto-generate Historical_Mean benchmark: {e}")

def update_timing(model_name, elapsed_seconds, num_folds=25):
    """Save execution timing into benchmark_timing.csv for accurate dashboard tracking."""
    timing_path = config.BENCHMARK_TIMING_PATH
    rec = pd.DataFrame([{
        'Model': model_name,
        'Historical_Total_Seconds': round(elapsed_seconds, 2),
        'Avg_Seconds_Per_Fold': round(elapsed_seconds / max(1, num_folds), 2),
        'Total_Folds': num_folds
    }])
    if os.path.exists(timing_path):
        try:
            ex = pd.read_csv(timing_path)
            ex = ex[ex['Model'] != model_name]
            comb = pd.concat([ex, rec], ignore_index=True)
        except Exception:
            comb = rec
    else:
        comb = rec
    comb.to_csv(timing_path, index=False)

def generate_lan04_summary_report():
    """Generates an executive research synthesis report for Lan04 in results/Lan04/agent_reports/."""
    report_dir = os.path.join(config.RESULTS_DIR, 'agent_reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, '3.0.3_LAN04_ENSEMBLE_REPORT.md')

    summary_file = config.MODEL_HORIZON_SUMMARY_PATH
    if not os.path.exists(summary_file):
        return

    s_df = pd.read_csv(summary_file)
    ens_df = s_df[s_df['Model'].str.startswith('Ensemble_') | s_df['Model'].str.startswith('Baseline_')]

    lines = [
        "# 📊 BÁO CÁO THỰC NGHIỆM LẦN 04 (LAN04) — EQUAL-WEIGHT, HORIZON & REGIME ENSEMBLES",
        f"**Thời gian tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Thiết kế thực nghiệm:** Đánh giá 9 tổ hợp mô hình với **tham số mặc định**:",
        "- **Nhóm A (Fixed-Weight):**",
        "  - `Ensemble_A1`: Toàn bộ 13 mô hình hồi quy (w = 1/13 ≈ 7.69%)",
        "  - `Ensemble_A2`: 4 mô hình cốt lõi Ridge, SVR, LightGBM, SARIMA (w = 1/4 = 25%)",
        "  - `Ensemble_A3`: Trio Ridge, SARIMA, Historical Mean (w = 1/3 ≈ 33.33%)",
        "- **Nhóm B (Horizon-Adaptive):**",
        "  - `Baseline_B1`: Best Single Model per Horizon (argmin MSE trên validation quá khứ)",
        "  - `Baseline_B2`: Normalized Inverse-MSE Weighting (Core 4)",
        "  - `Baseline_B3`: NNLS / Simplex Quadratic Programming (Core 4)",
        "- **Nhóm C (Regime-Adaptive):**",
        "  - `Baseline_C1`: Best Single Model per Horizon and Regime (argmin MSE(m, h, s))",
        "  - `Baseline_C2`: Regime-Shrunk Inverse-MSE Weighting (co rút N_0 = 50)",
        "  - `Baseline_C3`: Defensive Volatility Switch (Low/Normal dùng A2, High dùng Historical Mean)  \n",
        "---",
        "## 1. BẢNG HIỆU NĂNG TỔNG HỢP CÁC MÔ HÌNH (OOS R² & DA)",
        "| Mô hình | Chân trời | OOS R² Trung bình | Độ lệch chuẩn R² | DA (%) | RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, r in ens_df.iterrows():
        r2 = float(r.get('OOS_R2_Mean', 0))
        r2_sd = float(r.get('OOS_R2_Std', 0))
        da = float(r.get('DA_Mean', 0)) * 100
        rmse = float(r.get('RMSE_Mean', 0))
        r2_str = f"**{r2:+.4f}**" if r2 > 0 else f"{r2:+.4f}"
        lines.append(f"| `{r['Model']}` | `{r['Horizon']}` | {r2_str} | ±{r2_sd:.4f} | {da:.2f}% | {rmse:.4f} |")

    lines.extend([
        "\n---",
        "## 2. NHẬN XÉT PHƯƠNG PHÁP LUẬN",
        "1. **Bình đẳng trọng số (1/N DeMiguel et al. 2009):** Kháng quá khớp tham số mạnh mẽ vì không ước lượng trọng số trên mẫu.",
        "2. **Thích ứng theo Chân trời (B1, B2, B3):** Căn chỉnh tỷ trọng dựa trên hiệu năng từng chân trời và giải quy hoạch toàn phương trên simplex.",
        "3. **Thích ứng theo Regime (C1, C2, C3):** Tận dụng thông tin chế độ thị trường (Low, Normal, High) để luân chuyển mô hình hoặc co rút trọng số về mốc chuẩn an toàn.",
        "4. **Mốc so sánh cho Lần 05:** Toàn bộ kết quả ở Lần 04 là baseline tham số mặc định chuẩn xác để đối chứng với Lần 05 (Optuna Adaptive Weights)."
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(description="Run Ensemble & Regime Experiments (Lan04 - Default Parameters)")
    parser.add_argument('--model', type=str.upper, default='ALL',
                        choices=['ALL', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                                 'ENSEMBLE_A1', 'ENSEMBLE_A2', 'ENSEMBLE_A3',
                                 'BASELINE_B1', 'BASELINE_B2', 'BASELINE_B3',
                                 'BASELINE_C1', 'BASELINE_C2', 'BASELINE_C3'],
                        help="Select an experiment ('A1'..'A3', 'B1'..'B3', 'C1'..'C3', or 'all')")
    parser.add_argument('--horizon', type=str, default='all',
                        choices=['all'] + config.HORIZONS,
                        help="Specific forecast horizon (default: all)")
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    log_file = os.path.join(config.RESULTS_DIR, "lan04_execution.log")
    logger = utils.setup_logger("run_lan04", log_file=log_file)
    utils.set_all_seeds(args.seed)

    logger.info("=" * 75)
    logger.info("   STARTING RUN 4 (LAN04) — ENSEMBLES & REGIME BASELINES (DEFAULT PARAMETERS)")
    logger.info(f"   Run Name: {config.RUN_NAME} | Output: {config.RESULTS_DIR}")
    logger.info(f"   Target: {args.model.upper()} | Horizons: {args.horizon}")
    logger.info("=" * 75)

    # Step 0: Ensure Historical Mean benchmark baseline
    ensure_benchmark_baseline(logger)

    horizons = config.HORIZONS if args.horizon == 'all' else [args.horizon]

    # Map model argument
    m_arg = args.model.upper().replace('ENSEMBLE_', '').replace('BASELINE_', '')
    if m_arg == 'ALL':
        exps_to_run = LAN04_EXPERIMENTS
    elif m_arg in LAN04_EXPERIMENTS:
        exps_to_run = [m_arg]
    else:
        exps_to_run = LAN04_EXPERIMENTS

    total_start = time.time()
    failed = []

    for idx, exp_k in enumerate(exps_to_run, 1):
        spec = EXPERIMENT_SPECS[exp_k]
        logger.info(f"\n[{idx}/{len(exps_to_run)}] ======================================================")
        logger.info(f"Executing: {spec['display_name']} ({spec['title']})")
        logger.info(f"Description: {spec.get('description', '')}")
        logger.info("===========================================================================")

        t0 = time.time()
        try:
            m_df, p_df = run_ensemble(
                exp_key=exp_k,
                tune=False,
                horizons=horizons,
                logger=logger,
                seed=args.seed
            )
            elapsed = time.time() - t0
            update_timing(spec['display_name'], elapsed, num_folds=len(m_df))
            logger.info(f"✓ {spec['display_name']} completed in {elapsed:.2f}s.")
        except Exception as e:
            logger.error(f"✗ Failed {spec['display_name']}: {e}")
            logger.error(traceback.format_exc())
            failed.append((exp_k, str(e)))

    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 75)
    logger.info(f"LAN04 RUN COMPLETED IN {total_elapsed:.2f}s")
    if failed:
        logger.warning(f"Failed models ({len(failed)}): {failed}")
    else:
        logger.info("ALL ENSEMBLES & REGIME BASELINES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 75)

    # Post-hoc RQ Analysis & Statistical Tests
    try:
        import generate_rq_analysis
        generate_rq_analysis.run_analysis()
        logger.info("✓ Auto-generated RQ1, RQ2, RQ3 analysis and statistical tests for Lan04.")
    except Exception as e:
        logger.warning(f"Could not generate post-hoc RQ analysis: {e}")

    # Generate synthesis executive report
    try:
        generate_lan04_summary_report()
        logger.info("✓ Auto-generated executive report in results/Lan04/agent_reports/.")
    except Exception as e:
        logger.warning(f"Could not generate summary report: {e}")

if __name__ == "__main__":
    main()
