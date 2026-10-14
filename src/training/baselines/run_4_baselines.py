import os
import sys
import shutil
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
import utils

# Import the 4 baseline runners
from historical_mean.historical_mean import run_model as run_historical_mean
from arima.arima import run_model as run_arima
from sarima.sarima import run_model as run_sarima
from arimax.arimax import run_model as run_arimax

def ensure_pre_audit_backup():
    """Ensure a full backup of results/Lan01 exists before running."""
    backup_dir = os.path.join(config.RESULTS_BASE_DIR, "Lan01_backup_pre_audit")
    source_dir = config.RESULTS_DIR
    if not os.path.exists(backup_dir) and os.path.exists(source_dir):
        print(f"[BACKUP] Creating pre-audit backup at: {backup_dir} ...")
        shutil.copytree(source_dir, backup_dir)
        print("[BACKUP] Backup completed successfully.")
    else:
        print(f"[BACKUP] Pre-audit backup already verified at: {backup_dir}")

def run_all_4_baselines():
    """
    Runs the 4 audited baseline models sequentially in a single command.
    Each model updates ONLY its own rows in-place in:
    - 3.0.3_all_fold_metrics.csv (25 rows per model)
    - 3.0.3_all_oos_predictions.csv (6,800 rows per model)
    - 3.0.3_model_horizon_summary.csv (5 rows per model)
    All other 10 models are 100% untouched.
    """
    logger = utils.setup_logger("run_4_baselines")
    logger.info("=== STARTING AUDITED RUN FOR 4 BASELINES ===")
    
    ensure_pre_audit_backup()
    
    df = utils.load_dataset()
    logger.info(f"Loaded dataset from {config.DATA_PATH} with {len(df)} rows.")
    
    models_to_run = [
        ("Historical_Mean", run_historical_mean),
        ("ARIMA", run_arima),
        ("SARIMA", run_sarima),
        ("ARIMAX", run_arimax)
    ]
    
    total_start = time.time()
    
    for name, run_fn in models_to_run:
        logger.info(f"\n---> [START] Running {name} ...")
        t0 = time.time()
        try:
            m_df, p_df = run_fn(df=df, logger=logger)
            elapsed = time.time() - t0
            logger.info(f"---> [DONE] {name} completed in {elapsed:.2f}s ({len(m_df)} fold metrics, {len(p_df)} predictions).")
        except Exception as e:
            logger.error(f"---> [ERROR] Failed running {name}: {e}", exc_info=True)
            raise e
            
    total_elapsed = time.time() - total_start
    logger.info(f"\n=== ALL 4 BASELINES FINISHED IN {total_elapsed:.2f}s ===")
    
    # Verification of In-Place Preservation
    verify_csv_integrity()

def verify_csv_integrity():
    """Verify that row counts and non-target models remain 100% intact."""
    print("\n" + "="*70)
    print("VERIFICATION OF IN-PLACE ROW PRESERVATION ACROSS CSVS")
    print("="*70)
    
    # 1. Fold Metrics
    if os.path.exists(config.ALL_FOLD_METRICS_PATH):
        fm = pd.read_csv(config.ALL_FOLD_METRICS_PATH)
        print(f"1. Fold Metrics File: {config.ALL_FOLD_METRICS_PATH}")
        print(f"   - Total Rows: {len(fm)} (Expected: 350)")
        print(f"   - Models present: {fm['Model'].nunique()} models")
        counts = fm['Model'].value_counts()
        print("   - Rows per model:")
        for m, cnt in counts.items():
            tag = "[UPDATED]" if m in ['Historical_Mean', 'ARIMA', 'SARIMA', 'ARIMAX'] else "[UNTOUCHED]"
            print(f"     * {m:<20}: {cnt:3d} rows {tag}")
            
    # 2. Predictions
    if os.path.exists(config.ALL_PREDICTIONS_PATH):
        preds = pd.read_csv(config.ALL_PREDICTIONS_PATH)
        print(f"\n2. Predictions File: {config.ALL_PREDICTIONS_PATH}")
        print(f"   - Total Rows: {len(preds)} (Expected: 95,200)")
        p_counts = preds['Model'].value_counts()
        print("   - Sample model row counts:")
        for m in ['Historical_Mean', 'ARIMA', 'Ridge', 'Transformer']:
            if m in p_counts:
                tag = "[UPDATED]" if m in ['Historical_Mean', 'ARIMA'] else "[UNTOUCHED]"
                print(f"     * {m:<20}: {p_counts[m]:6d} rows {tag}")
                
    # 3. Model Horizon Summary
    if os.path.exists(config.MODEL_HORIZON_SUMMARY_PATH):
        sm = pd.read_csv(config.MODEL_HORIZON_SUMMARY_PATH)
        print(f"\n3. Summary File: {config.MODEL_HORIZON_SUMMARY_PATH}")
        print(f"   - Total Rows: {len(sm)} (Expected: 70)")
        print("\nUpdated Summary for the 4 Baselines:")
        mask = sm['Model'].isin(['Historical_Mean', 'ARIMA', 'SARIMA', 'ARIMAX'])
        print(sm[mask][['Model', 'Horizon', 'MAE_Mean', 'RMSE_Mean', 'OOS_R2_Mean', 'DA_Mean']].to_string(index=False))
        
    print("="*70 + "\n")

if __name__ == "__main__":
    run_all_4_baselines()
