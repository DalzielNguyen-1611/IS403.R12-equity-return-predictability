import os
import sys
import numpy as np
import pandas as pd
import optuna

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

# ==============================================================================
# ENSEMBLE SPECIFICATIONS (From Research Plan & Group Proposal)
# ==============================================================================
# A1: All 13 Regression Models Equal-Weight / Weighted Ensemble
# A2: Core 4 Models (Ridge, SVR, LightGBM, SARIMA)
# A3: Trio (Ridge, SARIMA, Historical_Mean)
# ==============================================================================

CONSTITUENTS_A1 = [
    'Historical_Mean',
    'Ridge',
    'Elastic_Net',
    'ARIMA',
    'ARIMAX',
    'SARIMA',
    'LightGBM',
    'SVR',
    'Random_Forest',
    'XGBoost',
    'RNN',
    'LSTM',
    'Transformer'
]

CONSTITUENTS_A2 = [
    'Ridge',
    'SVR',
    'LightGBM',
    'SARIMA'
]

CONSTITUENTS_A3 = [
    'Ridge',
    'SARIMA',
    'Historical_Mean'
]

EXPERIMENT_SPECS = {
    'A1': {
        'model_key': 'ensemble_a1',
        'display_name': 'Ensemble_A1',
        'title': 'All 13 Models Equal-Weight Ensemble',
        'description': 'Combines all 13 regression baselines across linear, tree, kernel, statistical, and deep learning families.',
        'models': CONSTITUENTS_A1,
        'default_weights': {m: 1.0 / len(CONSTITUENTS_A1) for m in CONSTITUENTS_A1}
    },
    'A2': {
        'model_key': 'ensemble_a2',
        'display_name': 'Ensemble_A2',
        'title': 'Core 4 Models Equal-Weight Ensemble',
        'description': 'Diverse 4-model core ensemble: Ridge (Regularized Linear) + SVR (Kernel) + LightGBM (Gradient Boosting) + SARIMA (Statistical TS).',
        'models': CONSTITUENTS_A2,
        'default_weights': {m: 0.25 for m in CONSTITUENTS_A2}
    },
    'A3': {
        'model_key': 'ensemble_a3',
        'display_name': 'Ensemble_A3',
        'title': 'Ridge + SARIMA + Historical Mean',
        'description': 'Parsimonious 3-model ensemble: Ridge + SARIMA + Historical Mean baseline.',
        'models': CONSTITUENTS_A3,
        'default_weights': {m: 1.0 / 3.0 for m in CONSTITUENTS_A3}
    }
}

def load_constituent_predictions(source_path=None):
    """
    Loads pre-computed out-of-sample predictions for all constituent models.
    Filters out classification (Logistic_Regression) to maintain strictly continuous return targets.
    Returns:
        pvt_df: DataFrame pivoted by ['Date', 'Horizon', 'Fold', 'Actual', 'Regime']
                with one column per regression model.
    """
    if source_path is None or not os.path.exists(source_path):
        # Look in Lan01 predictions first
        lan01_preds = os.path.join(config.RESULTS_BASE_DIR, 'Lan01', 'predictions', '3.0.3_all_oos_predictions.csv')
        central_preds = os.path.join(config.RESULTS_BASE_DIR, 'predictions', '3.0.3_all_oos_predictions.csv')
        current_preds = config.ALL_PREDICTIONS_PATH

        if os.path.exists(lan01_preds):
            source_path = lan01_preds
        elif os.path.exists(central_preds):
            source_path = central_preds
        elif os.path.exists(current_preds):
            source_path = current_preds
        else:
            raise FileNotFoundError(
                f"Cannot locate constituent predictions file. Tried:\n"
                f" - {lan01_preds}\n - {central_preds}\n - {current_preds}"
            )

    df = pd.read_csv(source_path)
    # Strictly exclude classification models
    reg_df = df[df['Model'] != 'Logistic_Regression'].copy()

    # Verify presence of all 13 models
    unique_models = set(reg_df['Model'].unique())
    missing = set(CONSTITUENTS_A1) - unique_models
    if missing:
        raise ValueError(f"Missing constituent models in predictions: {missing}")

    # Pivot table to align predictions by timestamp
    pvt_df = reg_df.pivot_table(
        index=['Date', 'Horizon', 'Fold', 'Actual', 'Regime'],
        columns='Model',
        values='Prediction'
    ).reset_index()

    pvt_df['Date'] = pvt_df['Date'].astype(str)
    return pvt_df

def optimize_simplex_weights(val_df, models, n_trials=50, seed=config.RANDOM_SEED):
    """
    Optimizes ensemble weights over the simplex (w_m >= 0, sum(w_m) = 1) using Optuna TPE Sampler.
    Objective: Minimize Validation RMSE (strictly equivalent to maximizing OOS R2).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    y_val = val_df['Actual'].values
    pred_matrix = np.column_stack([val_df[m].values for m in models])

    def objective(trial):
        raw_weights = np.array([trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in models])
        s = raw_weights.sum()
        if s <= 1e-12:
            weights = np.full(len(models), 1.0 / len(models))
        else:
            weights = raw_weights / s
        
        pred = pred_matrix @ weights
        rmse = np.sqrt(np.mean((y_val - pred) ** 2))
        return float(rmse)

    study.optimize(objective, n_trials=n_trials)

    # Normalize best weights
    best_raw = [study.best_params[f'w_{m}'] for m in models]
    s_best = sum(best_raw)
    if s_best <= 1e-12:
        norm_weights = {m: 1.0 / len(models) for m in models}
    else:
        norm_weights = {m: round(float(w / s_best), 6) for m, w in zip(models, best_raw)}

    return norm_weights, float(study.best_value)

def run_ensemble(exp_key='A1', tune=False, horizons=config.HORIZONS, n_trials=50, logger=None, seed=config.RANDOM_SEED):
    """
    Executes walk-forward evaluation of an ensemble experiment (A1, A2, or A3).
    - If tune=False (Lan04): Uses fixed theoretical Equal-Weights (tham số mặc định: 1/13, 1/4, 1/3).
    - If tune=True (Lan05): Optimizes simplex weights via Optuna (TPE Sampler) on expanding historical OOS.
    """
    if exp_key not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown ensemble experiment: {exp_key}. Choose from {list(EXPERIMENT_SPECS.keys())}")

    spec = EXPERIMENT_SPECS[exp_key]
    model_name = spec['display_name']
    constituents = spec['models']
    default_w = spec['default_weights']

    if logger:
        logger.info(f"▶ Initializing {model_name} ({spec['title']})")
        logger.info(f"  Constituent Models ({len(constituents)}): {', '.join(constituents)}")
        logger.info(f"  Mode: {'Optuna Adaptive Weighting (Lan05)' if tune else 'Fixed Equal-Weighting (Lan04)'}")

    # Load and pivot constituent predictions
    pvt_df = load_constituent_predictions()

    # Load fold date boundaries
    fold_def_df = pd.read_csv(config.FOLD_DEF_PATH)
    meta_by_fold = {}
    for _, row in fold_def_df.iterrows():
        f = int(row['Fold'])
        meta_by_fold[f] = {
            'train_start': str(row['Train_Start']),
            'train_end': str(row['Train_End']),
            'val_start': str(row['Val_Start']),
            'val_end': str(row['Val_End'])
        }

    fold_metrics = []
    oos_preds = []

    for horizon in horizons:
        h_steps = config.HORIZON_TO_STEPS.get(horizon, 1)

        for fold in range(1, 6):
            # Target test fold slice
            test_slice = pvt_df[(pvt_df['Horizon'] == horizon) & (pvt_df['Fold'] == fold)].copy()
            if test_slice.empty:
                continue

            # Determine weights
            if not tune:
                # Mode 1: Equal-weight default parameters (Lan04)
                weights_dict = default_w.copy()
                val_score = 0.0
                score_metric = "Default_Equal_Weights"
                best_params_record = {
                    'mode': 'equal_weight',
                    'formula': f"1/{len(constituents)} * sum({model_name})",
                    'weights': weights_dict
                }
            else:
                # Mode 2: Optuna-tuned weights (Lan05)
                # Zero-leakage walk-forward:
                # For fold > 1: Optimize weights on expanding historical OOS from folds 1 to fold-1
                # For fold 1: Cold start prior = 1/N equal weights (or inner validation split)
                hist_slice = pvt_df[(pvt_df['Horizon'] == horizon) & (pvt_df['Fold'] < fold)].copy()
                if not hist_slice.empty and len(hist_slice) >= 100:
                    weights_dict, val_score = optimize_simplex_weights(
                        hist_slice, constituents, n_trials=n_trials, seed=seed
                    )
                    score_metric = "Hist_OOS_RMSE"
                    best_params_record = {
                        'mode': 'optuna_tpe',
                        'tuning_folds': f"1..{fold-1}",
                        'tuning_samples': len(hist_slice),
                        'weights': weights_dict
                    }
                else:
                    # Cold-start prior: standard 1/N equal weights
                    weights_dict = default_w.copy()
                    val_score = 0.0
                    score_metric = "Cold_Start_Prior"
                    best_params_record = {
                        'mode': 'cold_start_prior',
                        'reason': 'No prior OOS fold available for Fold 1',
                        'weights': weights_dict
                    }

            # Save tuning record into results/tuning/3.0.3_best_hyperparameters.csv
            utils.save_tuning_record(
                model_name, horizon, fold, best_params_record, val_score, score_metric=score_metric
            )

            # Compute ensemble prediction: \hat{R} = sum_m w_m * \hat{R}^(m)
            ensemble_pred = np.zeros(len(test_slice), dtype=float)
            for m in constituents:
                ensemble_pred += weights_dict[m] * test_slice[m].values

            # Calculate metrics
            y_true = test_slice['Actual'].values
            y_bench = test_slice['Historical_Mean'].values
            m = utils.calc_regression_metrics(y_true, ensemble_pred, y_bench=y_bench, h=h_steps)

            meta = meta_by_fold.get(fold, {
                'train_start': '-', 'train_end': '-', 'val_start': '-', 'val_end': '-'
            })

            fold_metrics.append({
                'model': model_name,
                'horizon': horizon,
                'fold': fold,
                'train_start': meta['train_start'],
                'train_end': meta['train_end'],
                'val_start': meta['val_start'],
                'val_end': meta['val_end'],
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'R2': m['OOS_R2'],
                'Directional_Accuracy': m['DA'],
                'Sign_pvalue': m.get('Sign_pvalue', np.nan),
                'DM_Stat': m.get('DM_Stat', np.nan),
                'DM_pvalue': m.get('DM_pvalue', np.nan),
                'tuning_mode': 'Optuna' if tune else 'Equal'
            })

            # Record predictions
            dates = test_slice['Date'].values
            regimes = test_slice['Regime'].values
            for dt, act, pr, reg in zip(dates, y_true, ensemble_pred, regimes):
                oos_preds.append({
                    'Date': dt,
                    'Model': model_name,
                    'Horizon': horizon,
                    'Fold': fold,
                    'Actual': act,
                    'Prediction': pr,
                    'Regime': reg
                })

            if logger:
                logger.info(
                    f"  ✓ {model_name} [{horizon} | Fold {fold}/5]: "
                    f"OOS R² = {m['OOS_R2']:+.4f} | DA = {m['DA']*100:.1f}% | RMSE = {m['RMSE']:.4f}"
                )

    m_df = pd.DataFrame(fold_metrics)
    p_df = pd.DataFrame(oos_preds)

    # Save outputs following strict in-place update into central Tier 1 & 2 files
    utils.save_model_latest_results(CURRENT_DIR, m_df, p_df)
    return m_df, p_df
