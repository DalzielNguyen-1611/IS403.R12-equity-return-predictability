import os
import sys
import argparse
import importlib.util
import traceback
import pandas as pd

BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

def get_model_module(model_key):
    """Dynamically import run_model function from model's folder."""
    folder, pyfile = config.MODEL_REGISTRY[model_key]
    model_path = os.path.join(BASELINES_DIR, folder, pyfile)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    spec = importlib.util.spec_from_file_location(f"model_{model_key}", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def aggregate_summary(logger=None):
    """Aggregate latest_metrics.csv across all model directories into baseline_summary.csv."""
    reg_rows = []
    cls_rows = []
    
    for m_key, (folder, _) in config.MODEL_REGISTRY.items():
        metrics_file = os.path.join(BASELINES_DIR, folder, "latest_metrics.csv")
        if not os.path.exists(metrics_file):
            continue
            
        m_df = pd.read_csv(metrics_file)
        if m_key == 'logistic_regression':
            for h, grp in m_df.groupby('horizon'):
                cls_rows.append({
                    'Model': 'Logistic_Regression',
                    'Horizon': h,
                    'Mean_Accuracy': grp['Accuracy'].mean(),
                    'Std_Accuracy': grp['Accuracy'].std(),
                    'Mean_Precision': grp['Precision'].mean(),
                    'Std_Precision': grp['Precision'].std(),
                    'Mean_Recall': grp['Recall'].mean(),
                    'Std_Recall': grp['Recall'].std(),
                    'Mean_F1': grp['F1'].mean(),
                    'Std_F1': grp['F1'].std(),
                    'Mean_ROC_AUC': grp['ROC_AUC'].mean(),
                    'Std_ROC_AUC': grp['ROC_AUC'].std(),
                })
        else:
            model_name = m_df['model'].iloc[0] if 'model' in m_df.columns else m_key
            for h, grp in m_df.groupby('horizon'):
                reg_rows.append({
                    'Model': model_name,
                    'Horizon': h,
                    'Mean_MAE': grp['MAE'].mean(),
                    'Std_MAE': grp['MAE'].std(),
                    'Mean_RMSE': grp['RMSE'].mean(),
                    'Std_RMSE': grp['RMSE'].std(),
                    'Mean_R2': grp['R2'].mean(),
                    'Std_R2': grp['R2'].std(),
                    'Mean_Directional_Accuracy': grp['Directional_Accuracy'].mean(),
                    'Std_Directional_Accuracy': grp['Directional_Accuracy'].std(),
                })
                
    if reg_rows:
        reg_df = pd.DataFrame(reg_rows)
        reg_df.to_csv(config.SUMMARY_PATH, index=False)
        if logger:
            logger.info(f"Updated regression summary at: {config.SUMMARY_PATH}")
            
    if cls_rows:
        cls_path = os.path.join(BASELINES_DIR, "classification_summary.csv")
        cls_df = pd.DataFrame(cls_rows)
        cls_df.to_csv(cls_path, index=False)
        if logger:
            logger.info(f"Updated classification summary at: {cls_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Models")
    parser.add_argument('--model', type=str, default='all', choices=['all'] + config.MODELS,
                        help="Specific model to run (default: all)")
    parser.add_argument('--horizon', type=str, default='all', choices=['all'] + config.HORIZONS,
                        help="Forecast horizon to run (default: all)")
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    parser.add_argument('--analyze-only', action='store_true',
                        help="Only run post-hoc RQ analysis & statistical tests on existing predictions")
    parser.add_argument('--tune', action='store_true',
                        help="Enable Optuna hyperparameter optimization for models that support it")
    args = parser.parse_args()
    
    import generate_rq_analysis
    if args.analyze_only:
        generate_rq_analysis.run_analysis()
        return

    log_file = os.path.join(BASELINES_DIR, "baseline_run.log")
    logger = utils.setup_logger("baseline_runner", log_file=log_file)
    utils.set_all_seeds(args.seed)
    
    logger.info("=" * 70)
    logger.info("STARTING BASELINES BENCHMARK SUITE")
    logger.info(f"Device: {config.DEVICE} | Horizons: {args.horizon} | Tuning: {args.tune}")
    logger.info("=" * 70)
    
    df = utils.load_dataset()
    horizons = config.HORIZONS if args.horizon == 'all' else [args.horizon]
    models_to_run = config.MODELS if args.model == 'all' else [args.model]
        
    import inspect
    failed = []
    for idx, m_key in enumerate(models_to_run, 1):
        folder, _ = config.MODEL_REGISTRY[m_key]
        logger.info(f"\n[{idx}/{len(models_to_run)}] Running model: {m_key.upper()} ({folder}/)...")
        try:
            mod = get_model_module(m_key)
            sig = inspect.signature(mod.run_model)
            if 'tune' in sig.parameters:
                mod.run_model(df, horizons=horizons, tune=args.tune, logger=logger)
            else:
                mod.run_model(df, horizons=horizons, logger=logger)
            logger.info(f"✓ Model {m_key.upper()} completed successfully.")
        except Exception as e:
            logger.error(f"✗ Failed {m_key}: {e}")
            logger.error(traceback.format_exc())
            failed.append((m_key, str(e)))
            
    aggregate_summary(logger=logger)
    
    # Auto-generate RQ1, RQ2, RQ3 and Statistical tests
    try:
        generate_rq_analysis.run_analysis()
    except Exception as e:
        logger.warning(f"Could not auto-generate RQ analysis: {e}")
    
    logger.info("\n" + "=" * 70)
    if failed:
        logger.warning(f"Finished with {len(failed)} failures: {failed}")
    else:
        logger.info("ALL REQUESTED BASELINE RUNS FINISHED SUCCESSFULLY!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
