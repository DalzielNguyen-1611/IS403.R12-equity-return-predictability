import os
import sys
import time
import inspect
import importlib
import argparse
import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import test_config as config
import test_engine as engine
import test_analysis as analysis

BASELINES_DIR = os.path.join(SRC_DIR, 'training', 'baselines')
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

try:
    from src.training.baselines import config as baselines_pkg_config
    from src.training.baselines import utils as baselines_pkg_utils
except Exception:
    baselines_pkg_config = None
    baselines_pkg_utils = None

try:
    baselines_flat_config = importlib.import_module('config')
    baselines_flat_utils = importlib.import_module('utils')
except Exception:
    baselines_flat_config = None
    baselines_flat_utils = None


DEFAULT_SEEDS_POOL = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

AUTO_RUN_MAP = {
    'rnn': 'Lan01',
    'lstm': 'Lan01',
    'transformer': 'Lan01',
    'cnn': 'Lan02',
    'patchtst': 'Lan02',
    'nhits': 'Lan02',
    'nbeats': 'Lan02',
    'itransformer': 'Lan02',
    'tsmixer': 'Lan02',
    'tcn': 'Lan02',
}

HOLDOUT_BLOCKS = {
    '2021_2022': ('2021-01-01', '2022-12-31'),
    '2023_2024': ('2023-01-01', '2024-12-31'),
    '2025_2026': ('2025-01-01', '2026-04-07'),
}


def set_global_seed(seed):
    """Synchronize seed across test_config, baselines config (pkg & flat), and PyTorch/NumPy."""
    seed = int(seed)
    config.RANDOM_SEED = seed
    engine.set_all_seeds(seed)
    for cfg_mod in (baselines_pkg_config, baselines_flat_config):
        if cfg_mod is not None and hasattr(cfg_mod, 'RANDOM_SEED'):
            cfg_mod.RANDOM_SEED = seed
    for util_mod in (baselines_pkg_utils, baselines_flat_utils):
        if util_mod is not None and hasattr(util_mod, 'set_all_seeds'):
            util_mod.set_all_seeds(seed)
    if 'config' in sys.modules and hasattr(sys.modules['config'], 'RANDOM_SEED'):
        sys.modules['config'].RANDOM_SEED = seed
    if 'utils' in sys.modules and hasattr(sys.modules['utils'], 'set_all_seeds'):
        sys.modules['utils'].set_all_seeds(seed)


def invoke_model_fn(fn: Any, data, horizon, params, logger):
    """Call dispatch function cleanly whether or not it accepts a logger kwarg."""
    sig_params = inspect.signature(fn).parameters
    accepts_logger = (
        'logger' in sig_params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    )
    if accepts_logger:
        return fn(data, horizon, params, logger=logger)
    return fn(data, horizon, params)


def setup_logger(log_path):
    logger = logging.getLogger("multi_seed_evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def compute_oos_r2(y_true, y_pred, y_bench):
    if len(y_true) == 0:
        return np.nan
    sse_m = float(np.sum((y_true - y_pred) ** 2))
    sse_b = float(np.sum((y_true - y_bench) ** 2))
    return float(1.0 - (sse_m / (sse_b + 1e-12)))


def evaluate_single_prediction_frame(sub_df, elapsed_sec=np.nan):
    sub_df = sub_df.sort_values('Date').reset_index(drop=True)
    y_true = sub_df['Actual'].values.astype(float)
    y_pred = sub_df['Prediction'].values.astype(float)
    y_bench = sub_df['Benchmark'].values.astype(float)
    horizon = str(sub_df['Horizon'].iloc[0])
    model = str(sub_df['Model'].iloc[0])
    seed = int(sub_df['Seed'].iloc[0])
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)

    oos_r2 = compute_oos_r2(y_true, y_pred, y_bench)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    da, sign_p = analysis.directional_binomial_test(y_true, y_pred)
    naive_pos_da = float(np.mean(y_true > 0)) if len(y_true) > 0 else 0.5
    excess_da = float(da - naive_pos_da)
    dm_stat, dm_p = analysis.diebold_mariano_test(y_true, y_pred, y_bench, h=h_step)

    row = {
        'Model': model,
        'Horizon': horizon,
        'Seed': seed,
        'N_Obs': int(len(sub_df)),
        'OOS_R2': round(oos_r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'DA': round(float(da), 4),
        'Excess_DA': round(excess_da, 4),
        'DM_Stat': round(float(dm_stat), 4),
        'DM_pvalue': round(float(dm_p), 4),
        'Sign_pvalue': round(float(sign_p), 4),
    }

    for reg in ('Low', 'Normal', 'High'):
        reg_sub = sub_df[sub_df['Regime'] == reg]
        row[f'OOS_R2_{reg}'] = (
            round(compute_oos_r2(reg_sub['Actual'].values, reg_sub['Prediction'].values, reg_sub['Benchmark'].values), 4)
            if len(reg_sub) > 0 else np.nan
        )

    dates_str = sub_df['Date'].astype(str)
    for blk_name, (start_dt, end_dt) in HOLDOUT_BLOCKS.items():
        blk_sub = sub_df[(dates_str >= start_dt) & (dates_str <= end_dt)]
        row[f'OOS_R2_{blk_name}'] = (
            round(compute_oos_r2(blk_sub['Actual'].values, blk_sub['Prediction'].values, blk_sub['Benchmark'].values), 4)
            if len(blk_sub) > 0 else np.nan
        )

    row['Elapsed_Seconds'] = round(float(elapsed_sec), 2) if not np.isnan(elapsed_sec) else np.nan
    return row


def rebuild_summaries(preds_df, per_run_path, summary_path, timings=None):
    if preds_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    timings = timings or {}
    if os.path.exists(per_run_path):
        try:
            old_runs = pd.read_csv(per_run_path)
            for _, r in old_runs.iterrows():
                key = (str(r['Model']), str(r['Horizon']), int(r['Seed']))
                if key not in timings and 'Elapsed_Seconds' in r and pd.notna(r['Elapsed_Seconds']):
                    timings[key] = float(r['Elapsed_Seconds'])
        except Exception:
            pass

    run_rows = []
    for (model, horizon, seed), grp in preds_df.groupby(['Model', 'Horizon', 'Seed']):
        elapsed = timings.get((str(model), str(horizon), int(seed)), np.nan)
        run_rows.append(evaluate_single_prediction_frame(grp, elapsed_sec=elapsed))

    df_runs = pd.DataFrame(run_rows)
    horizon_order = {h: idx for idx, h in enumerate(config.HORIZONS)}
    df_runs['_h_ord'] = df_runs['Horizon'].map(lambda h: horizon_order.get(h, 999))
    df_runs = df_runs.sort_values(['Model', '_h_ord', 'Seed']).drop(columns=['_h_ord']).reset_index(drop=True)
    df_runs.to_csv(per_run_path, index=False)

    agg_cols = [
        'OOS_R2', 'MAE', 'RMSE', 'DA', 'Excess_DA', 'DM_Stat',
        'OOS_R2_Low', 'OOS_R2_Normal', 'OOS_R2_High',
        'OOS_R2_2021_2022', 'OOS_R2_2023_2024', 'OOS_R2_2025_2026',
    ]

    summary_rows = []
    for (model, horizon), grp in df_runs.groupby(['Model', 'Horizon']):
        s_row = {
            'Model': model,
            'Horizon': horizon,
            'N_Seeds': int(len(grp)),
            'Seeds': ",".join(str(int(s)) for s in grp['Seed'].tolist()),
            'OOS_R2_Mean': round(float(grp['OOS_R2'].mean()), 4),
            'OOS_R2_Std': round(float(grp['OOS_R2'].std(ddof=1)) if len(grp) > 1 else 0.0, 4),
            'OOS_R2_Median': round(float(grp['OOS_R2'].median()), 4),
            'OOS_R2_Min': round(float(grp['OOS_R2'].min()), 4),
            'OOS_R2_Max': round(float(grp['OOS_R2'].max()), 4),
            'Positive_Seeds': int((grp['OOS_R2'] > 0).sum()),
            'DM_Sig5pct_Seeds': int((grp['DM_pvalue'] < 0.05).sum()),
        }
        for col in agg_cols[1:]:
            vals = grp[col].dropna()
            s_row[f'{col}_Mean'] = round(float(vals.mean()), 4) if len(vals) > 0 else np.nan
            s_row[f'{col}_Std'] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
        summary_rows.append(s_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary['_h_ord'] = df_summary['Horizon'].map(lambda h: horizon_order.get(h, 999))
    df_summary = df_summary.sort_values(['Model', '_h_ord']).drop(columns=['_h_ord']).reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False)
    return df_runs, df_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Holdout Test Evaluation Runner (Split 3: 2020-2026, without re-running Optuna)"
    )
    parser.add_argument(
        '--models', type=str, required=True,
        help="Comma-separated model keys (e.g. 'tsmixer' or 'patchtst,lstm,transformer')"
    )
    parser.add_argument(
        '--horizons', type=str, default='all',
        help="Comma-separated horizons or 'all' (e.g. '120D' or '1D,5D,20D,60D,120D')"
    )
    parser.add_argument(
        '--num-seeds', type=int, default=5,
        help="Number of seeds from DEFAULT_SEEDS_POOL when --seeds is not given (max 10 in default pool; auto-extends if >10)"
    )
    parser.add_argument(
        '--seeds', type=str, default=None,
        help="Explicit comma-separated integer seeds (e.g. '42,101,202,303,404'). Overrides --num-seeds."
    )
    parser.add_argument(
        '--run', type=str, default='auto', choices=['auto', 'Lan01', 'Lan02', 'Lan03'],
        help="Hyperparameter source run ('auto' uses Lan01 for LSTM/Transformer/RNN and Lan02 for TSMixer/PatchTST/etc.)"
    )
    parser.add_argument(
        '--device', type=str, default=config.DEVICE,
        help="Device for PyTorch models ('cuda' or 'cpu')"
    )
    parser.add_argument(
        '--tag', type=str, default='default_run',
        help="Subdirectory name under results/multi_seed/ to store predictions and summary tables"
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help="Re-run even if (Model, Horizon, Seed) already exists in multi_seed_predictions.csv"
    )
    parser.add_argument(
        '--summarize-only', action='store_true',
        help="Skip training and recompute summary CSVs from existing multi_seed_predictions.csv"
    )
    return parser.parse_args()


def resolve_seeds(args):
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    n = max(1, int(args.num_seeds))
    if n <= len(DEFAULT_SEEDS_POOL):
        return DEFAULT_SEEDS_POOL[:n]
    pool = list(DEFAULT_SEEDS_POOL)
    cur = pool[-1] + 101
    while len(pool) < n:
        pool.append(cur)
        cur += 101
    return pool


def main():
    args = parse_args()
    out_dir = os.path.join(REPO_ROOT, 'results', 'multi_seed', args.tag)
    os.makedirs(out_dir, exist_ok=True)

    preds_path = os.path.join(out_dir, 'multi_seed_predictions.csv')
    per_run_path = os.path.join(out_dir, 'multi_seed_per_run_metrics.csv')
    summary_path = os.path.join(out_dir, 'multi_seed_summary.csv')
    log_path = os.path.join(out_dir, 'multi_seed_execution.log')
    logger = setup_logger(log_path)

    if args.summarize_only:
        if not os.path.exists(preds_path):
            logger.error(f"Predictions file not found: {preds_path}")
            sys.exit(1)
        preds_df = pd.read_csv(preds_path)
        df_runs, df_summary = rebuild_summaries(preds_df, per_run_path, summary_path)
        logger.info(f"Rebuilt summaries from {preds_path}: {len(df_runs)} runs -> {summary_path}")
        print("\n" + df_summary[['Model', 'Horizon', 'N_Seeds', 'OOS_R2_Mean', 'OOS_R2_Std', 'OOS_R2_Min', 'OOS_R2_Max', 'Positive_Seeds']].to_string(index=False))
        return

    config.DEVICE = args.device
    for cfg_mod in (baselines_pkg_config, baselines_flat_config):
        if cfg_mod is not None and hasattr(cfg_mod, 'DEVICE'):
            cfg_mod.DEVICE = args.device

    models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    horizons = config.HORIZONS if args.horizons.lower() == 'all' else [config.normalize_horizon(h) for h in args.horizons.split(',') if h.strip()]
    seeds = resolve_seeds(args)

    logger.info("=" * 72)
    logger.info("MULTI-SEED HOLDOUT TEST EVALUATION (SPLIT 3: 2020-2026)")
    logger.info(f"Output Dir : {out_dir}")
    logger.info(f"Models     : {models}")
    logger.info(f"Horizons   : {horizons}")
    logger.info(f"Seeds ({len(seeds)}): {seeds}")
    logger.info(f"Run Source : {args.run}")
    logger.info(f"Device     : {args.device}")
    logger.info("=" * 72)

    df = engine.load_dataset()
    train_df = df[df['Split'].isin([1, 2])].copy()
    test_df = df[df['Split'] == 3].copy()
    data_by_horizon = {h: engine.prepare_test_data(train_df, test_df, h, scale=True, purge=True) for h in horizons}

    tuning_cache = {}
    for run_name in ('Lan01', 'Lan02', 'Lan03'):
        t_path = config.get_run_paths(run_name)['tuning_path']
        tuning_cache[run_name] = config.load_best_hyperparameters(t_path) if t_path else {}

    existing_df = None
    if not args.no_resume and os.path.exists(preds_path):
        try:
            existing_df = pd.read_csv(preds_path)
        except Exception:
            existing_df = None

    timings = {}
    total_jobs = len(models) * len(horizons) * len(seeds)
    done_jobs = 0

    for model_key in models:
        m_display = config.MODEL_NAME_MAP.get(model_key, model_key)
        fn = engine.MODEL_EVAL_DISPATCH.get(model_key)
        if fn is None:
            logger.warning(f"Unknown model key '{model_key}', skipping.")
            continue

        run_src = AUTO_RUN_MAP.get(model_key, 'Lan02') if args.run == 'auto' else args.run
        best_params_map = tuning_cache.get(run_src, {})

        for horizon in horizons:
            data = data_by_horizon[horizon]
            test_dates = data['test_dates']
            test_regimes = data['test_regimes']
            y_test = data['y_test']
            y_bench = data['y_bench']
            params = best_params_map.get((m_display, horizon), {})

            for seed in seeds:
                done_jobs += 1
                if existing_df is not None and not existing_df.empty:
                    sub = existing_df[
                        (existing_df['Model'] == m_display) &
                        (existing_df['Horizon'] == horizon) &
                        (existing_df['Seed'] == seed)
                    ]
                    if len(sub) == len(test_dates):
                        logger.info(f"[{done_jobs}/{total_jobs}] Skipping {m_display} [{horizon}] seed={seed} (already saved)")
                        continue

                logger.info(f"[{done_jobs}/{total_jobs}] Running {m_display} [{horizon}] seed={seed} (params_src={run_src})")
                set_global_seed(seed)

                t0 = time.perf_counter()
                try:
                    preds = invoke_model_fn(fn, data, horizon, params, logger)
                except Exception as exc:
                    logger.error(f"Failed {m_display} [{horizon}] seed={seed}: {exc}", exc_info=True)
                    continue
                elapsed = time.perf_counter() - t0
                timings[(m_display, horizon, seed)] = elapsed

                cur_df = pd.DataFrame({
                    'Date': test_dates,
                    'Model': m_display,
                    'Horizon': horizon,
                    'Seed': int(seed),
                    'Actual': y_test.astype(float),
                    'Prediction': np.asarray(preds, dtype=float),
                    'Benchmark': y_bench.astype(float),
                    'Regime': test_regimes,
                })

                if os.path.exists(preds_path):
                    try:
                        disk_df = pd.read_csv(preds_path)
                        mask = (
                            (disk_df['Model'] == m_display) &
                            (disk_df['Horizon'] == horizon) &
                            (disk_df['Seed'] == seed)
                        )
                        disk_df = pd.concat([disk_df[~mask], cur_df], ignore_index=True)
                    except Exception:
                        disk_df = cur_df
                else:
                    disk_df = cur_df

                disk_df.to_csv(preds_path, index=False)
                existing_df = disk_df

                run_metrics = evaluate_single_prediction_frame(cur_df, elapsed_sec=elapsed)
                logger.info(
                    f"   ✓ Saved {m_display} [{horizon}] seed={seed} | "
                    f"OOS_R2={run_metrics['OOS_R2']:.4f} | DA={run_metrics['DA']:.4f} | {elapsed:.1f}s"
                )
                rebuild_summaries(existing_df, per_run_path, summary_path, timings=timings)

    if os.path.exists(preds_path):
        final_preds = pd.read_csv(preds_path)
        _, df_summary = rebuild_summaries(final_preds, per_run_path, summary_path, timings=timings)
        logger.info("=" * 72)
        logger.info(f"COMPLETED! Summary saved to: {summary_path}")
        logger.info("=" * 72)
        print("\n" + df_summary[['Model', 'Horizon', 'N_Seeds', 'OOS_R2_Mean', 'OOS_R2_Std', 'OOS_R2_Min', 'OOS_R2_Max', 'Positive_Seeds']].to_string(index=False))


if __name__ == '__main__':
    main()

