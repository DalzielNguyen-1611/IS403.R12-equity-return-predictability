"""
================================================================================
Ablation Study Runner: Feature Group Contribution Analysis
Project: Equity Return Predictability (IS403.R12)
Evaluates marginal contribution of 6 feature groups across forecast horizons
using representative models (Linear: Ridge, Non-linear ML: XGBoost, Deep Learning: LSTM).
================================================================================
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import test_config as config
import test_engine as engine

# ==============================================================================
# FEATURE GROUPS DEFINITION (from Paper Section 3 & Tables 1 & 2)
# ==============================================================================
FEATURE_GROUPS = {
    'Lagged_Returns': ['Return_lag_1', 'Return_lag_5', 'Return_lag_20'],
    'Momentum': ['Momentum_5D', 'Momentum_20D', 'Momentum_60D'],
    'Historical_Volatility': ['Volatility_5D', 'Volatility_20D', 'Volatility_60D'],
    'VIX_Indicators': ['VIXCLS', 'VIX_lag_1', 'VIX_change_1D', 'VIX_pct_change_1D'],
    'Intraday_Indicators': ['HL_range', 'OC_return'],
    'Trading_Activity': ['Volume_change']
}

ABLATION_CONFIGS = {
    'Full_Model': config.FEATURE_COLS,
    'w/o Lagged Returns': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['Lagged_Returns']],
    'w/o Momentum': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['Momentum']],
    'w/o Historical Volatility': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['Historical_Volatility']],
    'w/o VIX': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['VIX_Indicators']],
    'w/o Intraday': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['Intraday_Indicators']],
    'w/o Trading Volume': [c for c in config.FEATURE_COLS if c not in FEATURE_GROUPS['Trading_Activity']],
}

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'ablation_study')


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("ablation_study")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(output_dir, 'ablation_execution.log'), encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def compute_metrics(y_true, y_pred, y_bench):
    """Calculates MAE, RMSE, Campbell-Thompson OOS_R2, and DA."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    sse_m = float(np.sum((y_true - y_pred) ** 2))
    sse_b = float(np.sum((y_true - y_bench) ** 2))
    oos_r2 = float(1.0 - (sse_m / (sse_b + 1e-12)))
    
    da = float(np.mean((y_true > 0) == (y_pred > 0)))
    return {
        'MAE': mae,
        'RMSE': rmse,
        'OOS_R2': oos_r2,
        'DA': da
    }


def run_ablation(models=None, horizons=None, device=config.DEVICE, output_dir=DEFAULT_OUTPUT_DIR):
    logger = setup_logger(output_dir)
    config.set_run('Lan01')

    if models is None:
        models = ['ridge', 'xgboost', 'lstm']
    if horizons is None or horizons == ['all']:
        horizons = config.HORIZONS

    logger.info("=" * 80)
    logger.info("STARTING ABLATION STUDY: FEATURE GROUP CONTRIBUTION ANALYSIS")
    logger.info(f"Models: {models}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {device}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)

    # 1. Load dataset & splits
    df = engine.load_dataset()
    train_df = df[df['Split'].isin([1, 2])].copy()
    test_df = df[df['Split'] == 3].copy()
    logger.info(f"Loaded dataset: Train rows={len(train_df)}, Test rows={len(test_df)}")

    # Load tuned parameters if available
    try:
        tuned_dict = config.load_best_hyperparameters()
    except Exception:
        tuned_dict = {}

    total_runs = len(models) * len(horizons) * len(ABLATION_CONFIGS)
    run_idx = 0
    start_time = time.time()

    records = []

    for model_name in models:
        model_key = model_name.lower().strip()
        logger.info(f"\n>>> Running Model: {model_name.upper()} <<<")

        for horizon in horizons:
            # Baseline benchmark predictions (Historical Mean) for OOS R2
            bench_data = engine.prepare_test_data(train_df, test_df, horizon, scale=False, purge=True)
            y_bench = engine.eval_historical_mean(bench_data, horizon)
            y_true = bench_data['y_test']

            full_r2 = None

            for config_name, feature_subset in ABLATION_CONFIGS.items():
                run_idx += 1
                iter_start = time.time()

                # Scale features for Ridge and LSTM; unscaled for tree-based XGBoost
                needs_scale = (model_key in ['ridge', 'lstm', 'elastic_net', 'svr', 'transformer'])
                data = engine.prepare_test_data(
                    train_df, test_df, horizon, 
                    feature_cols=feature_subset, 
                    scale=needs_scale, 
                    purge=True
                )

                # Fetch tuned params
                display_name = {'ridge': 'Ridge', 'xgboost': 'XGBoost', 'lstm': 'LSTM'}.get(model_key, model_name)
                params = tuned_dict.get((display_name, horizon), None)

                # Fit and predict
                if model_key == 'ridge':
                    y_pred = engine.eval_ridge(data, horizon, params=params)
                elif model_key == 'xgboost':
                    y_pred = engine.eval_xgboost(data, horizon, params=params)
                elif model_key == 'lstm':
                    y_pred = engine.train_and_eval_dl_model(engine.LSTMModel, data, horizon, params=params, device=device)
                else:
                    raise ValueError(f"Unsupported ablation model: {model_key}")

                # Metrics
                m = compute_metrics(y_true, y_pred, y_bench)
                if config_name == 'Full_Model':
                    full_r2 = m['OOS_R2']
                    delta_r2 = 0.0
                else:
                    delta_r2 = m['OOS_R2'] - (full_r2 if full_r2 is not None else 0.0)

                elapsed = time.time() - iter_start

                logger.info(
                    f"[{run_idx:02d}/{total_runs}] Model: {model_name:7s} | Hor: {horizon:4s} | "
                    f"Config: {config_name:25s} ({len(feature_subset):02d} feats) | "
                    f"R2: {m['OOS_R2']:+.4f} | ΔR2: {delta_r2:+.4f} | DA: {m['DA']*100:.1f}% | {elapsed:.2f}s"
                )

                records.append({
                    'Model': model_name.upper(),
                    'Horizon': horizon,
                    'Configuration': config_name,
                    'Num_Features': len(feature_subset),
                    'OOS_R2': round(m['OOS_R2'], 4),
                    'Delta_R2': round(delta_r2, 4),
                    'DA': round(m['DA'], 4),
                    'MAE': round(m['MAE'], 4),
                    'RMSE': round(m['RMSE'], 4),
                    'Elapsed_Sec': round(elapsed, 2)
                })

    # Save results
    res_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'ablation_metrics_summary.csv')
    res_df.to_csv(csv_path, index=False)
    logger.info(f"\n[DONE] Saved raw ablation metrics to: {csv_path}")

    # Generate Markdown Summary Report
    report_path = os.path.join(output_dir, 'ablation_study_report.md')
    generate_markdown_report(res_df, report_path)
    logger.info(f"[DONE] Saved Markdown summary report to: {report_path}")

    total_elapsed = time.time() - start_time
    logger.info(f"\n🎉 ALL ABLATION EXPERIMENTS COMPLETED IN {total_elapsed/60:.2f} MINUTES!")


def generate_markdown_report(df, report_path):
    """Generates an academic markdown report summarizing feature contributions."""
    lines = []
    lines.append("# ABLATION STUDY: FEATURE GROUP MARGINAL CONTRIBUTIONS")
    lines.append(f"*Generated automatically on:* `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
    lines.append("> [!NOTE]")
    lines.append("> $\\Delta R^2_{\\mathrm{OOS}} = R^2_{\\mathrm{OOS}}(\\text{Ablated}) - R^2_{\\mathrm{OOS}}(\\text{Full})$.")
    lines.append("> A **negative $\\Delta R^2$** indicates that removing the feature group hurts predictive performance (i.e. the feature group has positive marginal predictive value).\n")

    for model, m_df in df.groupby('Model'):
        lines.append(f"## 1. Model: {model}\n")
        
        # Pivot table for Delta R2
        pivot_delta = m_df.pivot(index='Configuration', columns='Horizon', values='Delta_R2')
        # Reindex rows to keep Full_Model on top
        cfg_order = list(ABLATION_CONFIGS.keys())
        pivot_delta = pivot_delta.reindex([c for c in cfg_order if c in pivot_delta.index])
        
        lines.append("### $\\Delta R^2_{\\mathrm{OOS}}$ Comparison (Drop vs Full)")
        lines.append(pivot_delta.to_markdown())
        lines.append("\n")

        # Pivot table for raw OOS R2
        pivot_r2 = m_df.pivot(index='Configuration', columns='Horizon', values='OOS_R2')
        pivot_r2 = pivot_r2.reindex([c for c in cfg_order if c in pivot_r2.index])
        lines.append("### Raw $R^2_{\\mathrm{OOS}}$ Metrics")
        lines.append(pivot_r2.to_markdown())
        lines.append("\n---\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Feature Group Ablation Study")
    parser.add_argument('--models', type=str, default='ridge,xgboost,lstm',
                        help="Comma-separated list of models to ablate (e.g. 'ridge,xgboost,lstm' or 'xgboost')")
    parser.add_argument('--horizons', type=str, default='all',
                        help="Comma-separated horizons (e.g. 'all' or '1D,5D,20D,60D,120D')")
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help="PyTorch device ('cuda' or 'cpu')")
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save ablation CSV and reports")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    models = [m.strip().lower() for m in args.models.split(',')]
    horizons = [h.strip() for h in args.horizons.split(',')]
    run_ablation(models=models, horizons=horizons, device=args.device, output_dir=args.out_dir)
