import os
import sys
import time
import argparse
import traceback
import pandas as pd

# Set environment variable EXP_RUN_NAME to Lan03 if not already specified
if 'EXP_RUN_NAME' not in os.environ:
    os.environ['EXP_RUN_NAME'] = 'Lan03'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
import utils

# The 7 deep learning architectures for Run 3 (Lan03 - Optuna Hyperparameter Optimization)
LAN03_MODELS = [
    'tcn',
    'nbeats',
    'nhits',
    'patchtst',
    'tsmixer',
    'itransformer',
    'cnn'
]

def ensure_benchmark_baseline(df, logger):
    """
    Ensure Historical_Mean benchmark predictions exist in results/Lan03/
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
        logger.info("Initializing Historical_Mean benchmark baseline for Lan03...")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("historical_mean", os.path.join(CURRENT_DIR, "historical_mean", "historical_mean.py"))
            bench_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bench_mod)
            bench_mod.run_model(df, horizons=config.HORIZONS, logger=logger)
            logger.info("✓ Benchmark Historical_Mean ready for Lan03.")
        except Exception as e:
            logger.warning(f"Could not auto-generate Historical_Mean benchmark: {e}")

def update_timing(model_name, elapsed_seconds, num_folds=25):
    """Save execution timing into benchmark_timing.csv for accurate ETA tracking."""
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

def main():
    parser = argparse.ArgumentParser(description="Run 7 Advanced Deep Learning Time-Series Models with Optuna Tuning (Lan03)")
    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + LAN03_MODELS,
                        help="Select a specific model or 'all' to run all 7 models")
    parser.add_argument('--horizon', type=str, default='all',
                        choices=['all'] + config.HORIZONS,
                        help="Specific forecast horizon (default: all)")
    parser.add_argument('--no-tune', dest='tune', action='store_false',
                        help="Disable Optuna (defaults to TRUE for Lan03)")
    parser.set_defaults(tune=True)
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    log_file = os.path.join(config.RESULTS_DIR, "lan03_execution.log")
    logger = utils.setup_logger("run_lan03", log_file=log_file)
    utils.set_all_seeds(args.seed)

    logger.info("=" * 75)
    logger.info(f"   STARTING RUN 3 (LAN03) — 7 DL MODELS WITH OPTUNA HYPERPARAMETER TUNING")
    logger.info(f"   Device: {config.DEVICE} | Run Name: {config.RUN_NAME} | Output: {config.RESULTS_DIR}")
    logger.info(f"   Models: {args.model.upper()} | Horizons: {args.horizon} | Optuna Tuning: {args.tune}")
    logger.info("=" * 75)

    df = utils.load_dataset()
    logger.info(f"Loaded dataset: {config.DATA_PATH} ({len(df):,} trading days, 16 features).")

    # Step 0: Ensure Historical Mean benchmark baseline
    ensure_benchmark_baseline(df, logger)

    horizons = config.HORIZONS if args.horizon == 'all' else [args.horizon]
    models_to_run = LAN03_MODELS if args.model == 'all' else [args.model]

    import importlib.util
    import inspect
    failed = []
    total_start = time.time()

    for idx, m_key in enumerate(models_to_run, 1):
        folder, pyfile = config.MODEL_REGISTRY[m_key]
        model_script = os.path.join(CURRENT_DIR, folder, pyfile)
        logger.info(f"\n[{idx}/{len(models_to_run)}] ======================================================")
        logger.info(f"Launching Model: {m_key.upper()} with Optuna Tuning from {folder}/{pyfile}")
        logger.info("===========================================================================")

        t0 = time.time()
        try:
            spec = importlib.util.spec_from_file_location(f"lan03_{m_key}", model_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            sig = inspect.signature(mod.run_model)
            if 'tune' in sig.parameters:
                m_df, p_df = mod.run_model(df, horizons=horizons, tune=args.tune, logger=logger)
            else:
                m_df, p_df = mod.run_model(df, horizons=horizons, logger=logger)

            elapsed = time.time() - t0
            model_display_name = getattr(mod, 'MODEL_NAME', m_key.upper())
            update_timing(model_display_name, elapsed, num_folds=len(m_df))
            logger.info(f"✓ Model {m_key.upper()} completed in {elapsed:.1f}s ({elapsed/60:.2f} mins).")
        except Exception as e:
            logger.error(f"✗ Failed {m_key}: {e}")
            logger.error(traceback.format_exc())
            failed.append((m_key, str(e)))

    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 75)
    logger.info(f"LAN03 (OPTUNA) RUN COMPLETED IN {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)")
    if failed:
        logger.warning(f"Failed models ({len(failed)}): {failed}")
    else:
        logger.info("ALL 7 ADVANCED DEEP LEARNING ARCHITECTURES COMPLETED OPTUNA TUNING SUCCESSFULLY!")
    logger.info("=" * 75)

    # Post-hoc RQ Analysis & Statistical Tests
    try:
        import generate_rq_analysis
        generate_rq_analysis.run_analysis()
        logger.info("✓ Auto-generated RQ1, RQ2, RQ3 analysis and statistical tests for Lan03.")
    except Exception as e:
        logger.warning(f"Could not generate post-hoc RQ analysis: {e}")

if __name__ == "__main__":
    main()
