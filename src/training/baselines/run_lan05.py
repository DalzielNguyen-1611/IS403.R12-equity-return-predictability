import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd

# Set environment variable EXP_RUN_NAME to Lan05 if not already specified
if 'EXP_RUN_NAME' not in os.environ:
    os.environ['EXP_RUN_NAME'] = 'Lan05'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
import utils
from ensemble.ensemble_models import run_ensemble, EXPERIMENT_SPECS

LAN05_EXPERIMENTS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

def ensure_benchmark_baseline(logger):
    """
    Ensure Historical_Mean predictions exist in results/Lan05/predictions/
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
        logger.info("Initializing Historical_Mean benchmark baseline for Lan05 from Lan01...")
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
                    logger.info("✓ Copied Historical_Mean benchmark from Lan01 to Lan05.")
                    return
            except Exception as e:
                logger.warning(f"Could not copy Historical_Mean: {e}")

        # Fallback to generating it
        try:
            df = utils.load_dataset()
            import historical_mean.historical_mean as hm_mod
            hm_mod.run_model(df, horizons=config.HORIZONS, logger=logger)
            logger.info("✓ Generated fresh Historical_Mean benchmark for Lan05.")
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

def generate_lan05_summary_report():
    """Generates an executive research synthesis report for Lan05 in results/Lan05/agent_reports/."""
    report_dir = os.path.join(config.RESULTS_DIR, 'agent_reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, '3.0.3_LAN05_OPTUNA_ENSEMBLE_REPORT.md')

    summary_file = config.MODEL_HORIZON_SUMMARY_PATH
    if not os.path.exists(summary_file):
        return

    s_df = pd.read_csv(summary_file)
    ens_df = s_df[s_df['Model'].str.startswith('Ensemble_') | s_df['Model'].str.startswith('Baseline_')]

    # Check Lan04 summary for comparative delta
    lan04_summary_file = os.path.join(config.RESULTS_BASE_DIR, 'Lan04', 'summary', '3.0.3_model_horizon_summary.csv')
    lan04_map = {}
    if os.path.exists(lan04_summary_file):
        try:
            l4_df = pd.read_csv(lan04_summary_file)
            for _, r in l4_df.iterrows():
                lan04_map[(r['Model'], r['Horizon'])] = float(r.get('OOS_R2_Mean', 0))
        except Exception:
            pass

    lines = [
        "# 🎯 BÁO CÁO THỰC NGHIỆM LẦN 05 (LAN05) — OPTUNA-TUNED ENSEMBLES & ADAPTIVE STRATEGIES",
        f"**Thời gian tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Thiết kế thực nghiệm:** Tối ưu hóa siêu tham số & tỷ trọng bằng **Optuna (TPE Sampler)**:",
        "- **Nhóm A (Fixed-Weight Tuning):**",
        "  - `Ensemble_A1`: Tối ưu 13 trọng số simplex cho toàn bộ 13 mô hình hồi quy",
        "  - `Ensemble_A2`: Tối ưu 4 trọng số cho Core 4 Models (Ridge, SVR, LightGBM, SARIMA)",
        "  - `Ensemble_A3`: Tối ưu 3 trọng số cho Trio (Ridge, SARIMA, Historical Mean)",
        "- **Nhóm B (Horizon-Adaptive Tuning):**",
        "  - `Baseline_B1`: Tối ưu hóa hòa trộn lồi top candidate models theo từng chân trời",
        "  - `Baseline_B2`: Tối ưu độ nhạy lũy thừa gamma của nghịch đảo MSE",
        "  - `Baseline_B3`: Tối ưu quy hoạch toàn phương trên Simplex có điều chuẩn",
        "- **Nhóm C (Regime-Adaptive Tuning):**",
        "  - `Baseline_C1`: Tối ưu lựa chọn mô hình & hòa trộn theo từng Volatility Regime",
        "  - `Baseline_C2`: Tối ưu tham số co rút James-Stein N_0 và độ nhạy gamma của Inverse-MSE",
        "  - `Baseline_C3`: Tối ưu mức phòng thủ alpha trong chế độ High Volatility và phân bổ Core trong Low/Normal",
        "- **Zero-Data-Leakage:** Tối ưu hóa trên tập OOS tích lũy quá khứ (Expanding Window Walk-Forward)  \n",
        "---",
        "## 1. BẢNG HIỆU NĂNG TỔNG HỢP (SO SÁNH LAN05 OPTUNA VS LAN04 MẶC ĐỊNH)",
        "| Mô hình | Chân trời | OOS R² Lan05 (Optuna) | OOS R² Lan04 (Mặc định) | Δ OOS R² | DA (%) | RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, r in ens_df.iterrows():
        m_name = r['Model']
        hrz = r['Horizon']
        r2_l5 = float(r.get('OOS_R2_Mean', 0))
        r2_l4 = lan04_map.get((m_name, hrz), np.nan)
        da = float(r.get('DA_Mean', 0)) * 100
        rmse = float(r.get('RMSE_Mean', 0))

        delta_str = f"**{r2_l5 - r2_l4:+.4f}**" if not np.isnan(r2_l4) else "-"
        r2_l4_str = f"{r2_l4:+.4f}" if not np.isnan(r2_l4) else "-"
        r2_l5_str = f"**{r2_l5:+.4f}**" if r2_l5 > 0 else f"{r2_l5:+.4f}"

        lines.append(f"| `{m_name}` | `{hrz}` | {r2_l5_str} | {r2_l4_str} | {delta_str} | {da:.2f}% | {rmse:.4f} |")

    lines.extend([
        "\n---",
        "## 2. Ý NGHĨA KHOA HỌC & KINH TẾ LƯỢNG",
        "1. **Tối ưu hóa Simplex & Tham số Co rút:** Optuna TPE khám phá không gian tỷ trọng lồi và tìm mức co rút N_0 tối ưu, cân bằng giữa tín hiệu cục bộ và mốc chuẩn toàn cục.",
        "2. **Thích ứng Động theo Regime:** Chuyển đổi và tinh chỉnh tỷ trọng tự động khi thị trường rơi vào pha biến động mạnh (High Volatility).",
        "3. **Kiểm chứng So Sánh:** Đo lường chính xác giá trị thặng dư (Δ OOS R²) do việc tối ưu hóa tham số mang lại so với quy tắc mặc định (Lần 04)."
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(description="Run Optuna-Tuned Ensemble & Regime Experiments (Lan05)")
    parser.add_argument('--model', type=str.upper, default='ALL',
                        choices=['ALL', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                                 'ENSEMBLE_A1', 'ENSEMBLE_A2', 'ENSEMBLE_A3',
                                 'BASELINE_B1', 'BASELINE_B2', 'BASELINE_B3',
                                 'BASELINE_C1', 'BASELINE_C2', 'BASELINE_C3'],
                        help="Select an experiment ('A1'..'A3', 'B1'..'B3', 'C1'..'C3', or 'all')")
    parser.add_argument('--horizon', type=str, default='all',
                        choices=['all'] + config.HORIZONS,
                        help="Specific forecast horizon (default: all)")
    parser.add_argument('--n_trials', type=int, default=50,
                        help="Number of Optuna TPE trials per fold (default: 50)")
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    log_file = os.path.join(config.RESULTS_DIR, "lan05_execution.log")
    logger = utils.setup_logger("run_lan05", log_file=log_file)
    utils.set_all_seeds(args.seed)

    logger.info("=" * 75)
    logger.info("   STARTING RUN 5 (LAN05) — OPTUNA ENSEMBLES & REGIME STRATEGIES (A1..A3, C1..C3)")
    logger.info(f"   Run Name: {config.RUN_NAME} | Output: {config.RESULTS_DIR}")
    logger.info(f"   Target: {args.model.upper()} | Horizons: {args.horizon} | Trials: {args.n_trials}")
    logger.info("=" * 75)

    # Step 0: Ensure Historical Mean benchmark baseline
    ensure_benchmark_baseline(logger)

    horizons = config.HORIZONS if args.horizon == 'all' else [args.horizon]

    # Map model argument
    m_arg = args.model.upper().replace('ENSEMBLE_', '').replace('BASELINE_', '')
    if m_arg == 'ALL':
        exps_to_run = LAN05_EXPERIMENTS
    elif m_arg in LAN05_EXPERIMENTS:
        exps_to_run = [m_arg]
    else:
        exps_to_run = LAN05_EXPERIMENTS

    total_start = time.time()
    failed = []

    for idx, exp_k in enumerate(exps_to_run, 1):
        spec = EXPERIMENT_SPECS[exp_k]
        logger.info(f"\n[{idx}/{len(exps_to_run)}] ======================================================")
        logger.info(f"Executing Optuna Tuning: {spec['display_name']} ({spec['title']})")
        logger.info(f"Description: {spec.get('description', '')}")
        logger.info(f"Optuna Trials: {args.n_trials} (TPESampler seed={args.seed})")
        logger.info("===========================================================================")

        t0 = time.time()
        try:
            m_df, p_df = run_ensemble(
                exp_key=exp_k,
                tune=True,
                horizons=horizons,
                n_trials=args.n_trials,
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
    logger.info(f"LAN05 (OPTUNA) RUN COMPLETED IN {total_elapsed:.2f}s")
    if failed:
        logger.warning(f"Failed models ({len(failed)}): {failed}")
    else:
        logger.info("ALL OPTUNA-TUNED ENSEMBLES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 75)

    # Post-hoc RQ Analysis & Statistical Tests
    try:
        import generate_rq_analysis
        generate_rq_analysis.run_analysis()
        logger.info("✓ Auto-generated RQ1, RQ2, RQ3 analysis and statistical tests for Lan05.")
    except Exception as e:
        logger.warning(f"Could not generate post-hoc RQ analysis: {e}")

    # Generate synthesis executive report
    try:
        generate_lan05_summary_report()
        logger.info("✓ Auto-generated comparative report in results/Lan05/agent_reports/.")
    except Exception as e:
        logger.warning(f"Could not generate summary report: {e}")

if __name__ == "__main__":
    main()
