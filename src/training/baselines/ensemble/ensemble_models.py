import os
import sys
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

# ==============================================================================
# EXPERIMENT SPECIFICATIONS (From Research Plan & Group Proposal)
# ==============================================================================
# Group A (Fixed-Weight Ensembles):
#   A1: All 13 Regression Models Equal-Weight (1/13)
#   A2: Core 4 Models (Ridge, SVR, LightGBM, SARIMA) Equal-Weight (1/4)
#   A3: Trio (Ridge, SARIMA, Historical_Mean) Equal-Weight (1/3)
# Group B (Horizon-Adaptive Baselines & Ensembles):
#   B1: Best Single Model per Horizon (argmin MSE(m, h))
#   B2: Normalized Inverse-MSE Weighting (Core 4: Ridge, SVR, LightGBM, SARIMA)
#   B3: NNLS / Simplex Quadratic Programming (Core 4)
# Group C (Regime-Adaptive Baselines & Ensembles):
#   C1: Best Single Model per Horizon and Regime (argmin MSE(m, h, s))
#   C2: Regime-Shrunk Inverse-MSE Weighting (James-Stein Shrinkage N_0)
#   C3: Defensive Volatility Switch (A2 in Low/Normal, Historical Mean in High)
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
    },
    'B1': {
        'model_key': 'baseline_b1',
        'display_name': 'Baseline_B1',
        'title': 'Best Single Model per Horizon',
        'description': 'Selects argmin MSE(m, h) on historical validation folds across all 13 regression baselines.',
        'models': CONSTITUENTS_A1,
        'default_params': {'selection_metric': 'MSE'}
    },
    'B2': {
        'model_key': 'baseline_b2',
        'display_name': 'Baseline_B2',
        'title': 'Normalized Inverse-MSE Weighting',
        'description': 'Computes weights proportional to 1/MSE(m, h) across Core 4 models normalized to sum to 1.',
        'models': CONSTITUENTS_A2,
        'default_params': {'power': 1.0}
    },
    'B3': {
        'model_key': 'baseline_b3',
        'display_name': 'Baseline_B3',
        'title': 'NNLS / Simplex Quadratic Programming',
        'description': 'Solves min ||y - P*w||_2^2 subject to w >= 0 and sum(w) = 1 across Core 4 models.',
        'models': CONSTITUENTS_A2,
        'default_params': {'solver': 'SLSQP'}
    },
    'C1': {
        'model_key': 'baseline_c1',
        'display_name': 'Baseline_C1',
        'title': 'Best Single Model per Horizon and Regime',
        'description': 'Selects argmin MSE(m, h, s) on historical validation for each volatility regime (Low, Normal, High).',
        'models': CONSTITUENTS_A1,
        'default_params': {'selection_metric': 'MSE', 'fallback_model': 'Ridge'}
    },
    'C2': {
        'model_key': 'baseline_c2',
        'display_name': 'Baseline_C2',
        'title': 'Regime-Shrunk Inverse-MSE Weighting',
        'description': 'Shrinks regime-specific inverse-MSE weights towards horizon B2 baseline using shrinkage parameter N_0.',
        'models': CONSTITUENTS_A2,
        'default_params': {'N0': 50.0, 'power': 1.0, 'min_samples': 5}
    },
    'C3': {
        'model_key': 'baseline_c3',
        'display_name': 'Baseline_C3',
        'title': 'Defensive Volatility Switch',
        'description': 'Switches between Core 4 Ensemble (A2) in Low/Normal regimes and Historical Mean benchmark in High volatility.',
        'models': CONSTITUENTS_A2 + ['Historical_Mean'],
        'default_params': {'high_defensive_weight': 1.0}
    }
}

ALL_EXPERIMENT_KEYS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

def load_constituent_predictions(source_path=None):
    """
    Loads pre-computed out-of-sample predictions for all constituent models.
    Filters out classification (Logistic_Regression) to maintain strictly continuous return targets.
    Returns:
        pvt_df: DataFrame pivoted by ['Date', 'Horizon', 'Fold', 'Actual', 'Regime']
                with one column per regression model.
    """
    if source_path is None or not os.path.exists(source_path):
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
    reg_df = df[df['Model'] != 'Logistic_Regression'].copy()

    unique_models = set(reg_df['Model'].unique())
    missing = set(CONSTITUENTS_A1) - unique_models
    if missing:
        raise ValueError(f"Missing constituent models in predictions: {missing}")

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

    best_raw = [study.best_params[f'w_{m}'] for m in models]
    s_best = sum(best_raw)
    if s_best <= 1e-12:
        norm_weights = {m: 1.0 / len(models) for m in models}
    else:
        norm_weights = {m: round(float(w / s_best), 6) for m, w in zip(models, best_raw)}

    return norm_weights, float(study.best_value)

# ==============================================================================
# PIPELINE ENGINES FOR B1, B2, B3, C1, C2, C3
# ==============================================================================

def execute_b1_forecast(hist_slice, test_slice, horizon, models=CONSTITUENTS_A1, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    B1: Best Single Model per Horizon
    - Evaluates MSE(m, h) across all regression models on prior validation folds.
    - Selects m_h* = argmin_m MSE(m, h).
    - If hist_slice is empty (Fold 1): falls back to pre-design reported selection:
      1D: SARIMA, 5D: XGBoost, 20D: Elastic_Net, 60D: Historical_Mean, 120D: ARIMAX.
    - If tune=True (Lan05): Optuna finds optimal convex combination among top-3 candidate models.
    """
    pre_selected = {
        '1D': 'SARIMA',
        '5D': 'XGBoost',
        '20D': 'Elastic_Net',
        '60D': 'Historical_Mean',
        '120D': 'ARIMAX'
    }

    if hist_slice.empty:
        chosen_m = pre_selected.get(horizon, 'Ridge')
        pred = test_slice[chosen_m].values
        best_params_record = {
            'mode': 'b1_prior_pre_design',
            'selected_model': chosen_m,
            'reason': 'Fold 1 cold-start fallback to preliminary design report'
        }
        return pred, best_params_record, 0.0

    mses = {m: float(np.mean((hist_slice['Actual'].values - hist_slice[m].values) ** 2)) for m in models}
    sorted_models = sorted(mses.items(), key=lambda x: x[1])
    best_m = sorted_models[0][0]

    if tune and len(hist_slice) >= 50:
        top_candidates = [m for m, _ in sorted_models[:3]]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        y_h = hist_slice['Actual'].values
        X_mat = np.column_stack([hist_slice[m].values for m in top_candidates])

        def obj(trial):
            w = np.array([trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in top_candidates])
            s = w.sum()
            w = w / s if s > 0 else np.full(len(top_candidates), 1.0 / len(top_candidates))
            return float(np.sqrt(np.mean((y_h - X_mat @ w) ** 2)))

        study.optimize(obj, n_trials=n_trials)
        w_raw = [study.best_params[f'w_{m}'] for m in top_candidates]
        s_best = sum(w_raw)
        opt_w = {m: round(float(w / s_best), 6) for m, w in zip(top_candidates, w_raw)} if s_best > 0 else {m: 1.0/3.0 for m in top_candidates}
        pred = sum(opt_w[m] * test_slice[m].values for m in top_candidates)
        val_score = float(study.best_value)
        best_params_record = {
            'mode': 'b1_optuna_top3_blend',
            'top_candidates': top_candidates,
            'blended_weights': opt_w,
            'best_single_model': best_m,
            'top_mses': {m: round(float(v), 8) for m, v in sorted_models[:5]}
        }
    else:
        pred = test_slice[best_m].values
        val_score = float(np.sqrt(sorted_models[0][1]))
        best_params_record = {
            'mode': 'b1_deterministic_argmin_mse',
            'selected_model': best_m,
            'all_mses': {m: round(float(v), 8) for m, v in sorted_models[:5]}
        }

    return pred, best_params_record, val_score


def execute_b2_forecast(hist_slice, test_slice, models=CONSTITUENTS_A2, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    B2: Normalized Inverse-MSE Weighting
    - Evaluates MSE_m,h on prior validation folds across Core 4 models.
    - Weights w_{m,h} = (1 / MSE_m)^gamma / sum((1 / MSE_j)^gamma).
    - If tune=False (Lan04): Classical Inverse-MSE (gamma = 1.0).
    - If tune=True (Lan05): Optuna tunes power gamma in [0.2, 5.0].
    """
    eps = 1e-8
    if hist_slice.empty:
        w_b2 = {m: 0.25 for m in models}
        pred = sum(0.25 * test_slice[m].values for m in models)
        best_params_record = {'mode': 'b2_cold_start_prior', 'weights': w_b2, 'power': 1.0}
        return pred, best_params_record, 0.0

    mses = {m: float(np.mean((hist_slice['Actual'].values - hist_slice[m].values) ** 2)) for m in models}

    if tune and len(hist_slice) >= 50:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        y_h = hist_slice['Actual'].values

        def obj(trial):
            gamma = trial.suggest_float('power', 0.2, 5.0)
            inv = {m: (1.0 / (mses[m] + eps)) ** gamma for m in models}
            tot = sum(inv.values())
            w = np.array([inv[m] / tot for m in models])
            p_mat = np.column_stack([hist_slice[m].values for m in models])
            return float(np.sqrt(np.mean((y_h - p_mat @ w) ** 2)))

        study.optimize(obj, n_trials=n_trials)
        best_power = float(study.best_params['power'])
        val_score = float(study.best_value)
    else:
        best_power = 1.0
        val_score = 0.0

    inv = {m: (1.0 / (mses[m] + eps)) ** best_power for m in models}
    tot = sum(inv.values())
    w_b2 = {m: round(float(inv[m] / tot), 6) for m in models}

    pred = sum(w_b2[m] * test_slice[m].values for m in models)
    best_params_record = {
        'mode': 'b2_optuna_power' if tune else 'b2_classical_inverse_mse',
        'power': best_power,
        'weights': w_b2,
        'mses': {m: round(float(v), 8) for m, v in mses.items()}
    }
    return pred, best_params_record, val_score


def execute_b3_forecast(hist_slice, test_slice, models=CONSTITUENTS_A2, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    B3: NNLS / Quadratic Programming with Simplex Constraints
    - Solves: min_w ||y - P*w||_2^2 s.t. w_m >= 0 and sum(w_m) = 1.
    - If tune=False (Lan04): Exact QP via scipy.optimize.minimize (SLSQP).
    - If tune=True (Lan05): Optuna TPE Simplex optimization with regularization exploration.
    """
    K = len(models)
    if hist_slice.empty:
        w_b3 = {m: round(1.0 / K, 6) for m in models}
        pred = sum((1.0 / K) * test_slice[m].values for m in models)
        best_params_record = {'mode': 'b3_cold_start_prior', 'weights': w_b3}
        return pred, best_params_record, 0.0

    y_h = hist_slice['Actual'].values
    P = np.column_stack([hist_slice[m].values for m in models])

    if not tune:
        def loss(w):
            diff = y_h - P @ w
            return float(np.sum(diff ** 2))

        w0 = np.full(K, 1.0 / K)
        bounds = [(0.0, 1.0) for _ in range(K)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons)

        w_opt = res.x / np.sum(res.x) if np.sum(res.x) > 0 else w0
        val_score = float(np.sqrt(np.mean((y_h - P @ w_opt) ** 2)))
        w_dict = {m: round(float(w), 6) for m, w in zip(models, w_opt)}
        best_params_record = {
            'mode': 'b3_classical_qp_simplex',
            'weights': w_dict,
            'solver_success': bool(res.success),
            'message': str(res.message)
        }
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))

        def obj(trial):
            raw_w = np.array([trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in models])
            s = raw_w.sum()
            w = raw_w / s if s > 0 else np.full(K, 1.0 / K)
            return float(np.sqrt(np.mean((y_h - P @ w) ** 2)))

        study.optimize(obj, n_trials=n_trials)
        w_raw = [study.best_params[f'w_{m}'] for m in models]
        s_best = sum(w_raw)
        w_opt = np.array(w_raw) / s_best if s_best > 0 else np.full(K, 1.0 / K)
        val_score = float(study.best_value)
        w_dict = {m: round(float(w), 6) for m, w in zip(models, w_opt)}
        best_params_record = {
            'mode': 'b3_optuna_simplex_weights',
            'weights': w_dict,
            'optuna_trials': n_trials
        }

    pred = sum(w_dict[m] * test_slice[m].values for m in models)
    return pred, best_params_record, val_score


def execute_c1_forecast(hist_slice, test_slice, models, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    C1: Best Single Model per Horizon and Regime
    - Evaluates MSE(m, h, s) on prior validation history.
    - Selects argmin_m MSE(m, h, s) for each regime s in {'Low', 'Normal', 'High'}.
    - Fallback: best model across entire horizon if regime has fewer than 5 observations.
    - If tune=True: Optuna finds optimal blending/temperature between top candidate models per regime.
    """
    regimes = ['Low', 'Normal', 'High']
    best_model_by_regime = {}
    val_scores = {}

    # Overall horizon fallback
    if not hist_slice.empty:
        overall_mses = {m: float(np.mean((hist_slice['Actual'].values - hist_slice[m].values) ** 2)) for m in models}
        fallback_m = min(overall_mses, key=overall_mses.get)
    else:
        fallback_m = 'Ridge'

    for reg in regimes:
        sub_reg = hist_slice[hist_slice['Regime'] == reg] if not hist_slice.empty else pd.DataFrame()
        if len(sub_reg) >= 5:
            mses = {m: float(np.mean((sub_reg['Actual'].values - sub_reg[m].values) ** 2)) for m in models}
            chosen = min(mses, key=mses.get)
            best_model_by_regime[reg] = chosen
            val_scores[reg] = float(np.sqrt(mses[chosen]))
        else:
            best_model_by_regime[reg] = fallback_m
            val_scores[reg] = 0.0

    # Predictions on test slice
    pred = np.zeros(len(test_slice), dtype=float)
    for i, r in enumerate(test_slice['Regime'].values):
        chosen_m = best_model_by_regime.get(r, fallback_m)
        pred[i] = test_slice[chosen_m].values[i]

    # In Optuna mode: optimize convex combination among the selected models
    if tune and not hist_slice.empty and len(hist_slice) >= 50:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        unique_chosen = list(set(best_model_by_regime.values()))
        if len(unique_chosen) > 1:
            y_h = hist_slice['Actual'].values
            X_mat = np.column_stack([hist_slice[m].values for m in unique_chosen])
            def obj(trial):
                w = np.array([trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in unique_chosen])
                s = w.sum()
                w = w / s if s > 0 else np.full(len(unique_chosen), 1.0 / len(unique_chosen))
                return float(np.sqrt(np.mean((y_h - X_mat @ w) ** 2)))
            study.optimize(obj, n_trials=n_trials)
            opt_w = {m: round(study.best_params[f'w_{m}'] / sum(study.best_params.values()), 6) for m in unique_chosen}
            avg_val_score = float(study.best_value)
            best_params_record = {
                'mode': 'c1_optuna_tuned',
                'selected_models_by_regime': best_model_by_regime,
                'optuna_weights': opt_w,
                'fallback_model': fallback_m
            }
        else:
            avg_val_score = float(np.mean(list(val_scores.values())))
            best_params_record = {
                'mode': 'c1_single_best',
                'selected_models_by_regime': best_model_by_regime,
                'fallback_model': fallback_m
            }
    else:
        avg_val_score = float(np.mean(list(val_scores.values()))) if val_scores else 0.0
        best_params_record = {
            'mode': 'c1_deterministic_argmin_mse',
            'selected_models_by_regime': best_model_by_regime,
            'fallback_model': fallback_m
        }

    return pred, best_params_record, avg_val_score

def execute_c2_forecast(hist_slice, test_slice, models=CONSTITUENTS_A2, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    C2: Regime-Shrunk Inverse-MSE Weighting
    - Base horizon weights B2: w_B2 = (1 / MSE_h) / sum(1 / MSE_h)
    - Regime sample weights: w_sample = (1 / MSE_{h,s}) / sum(1 / MSE_{h,s})
    - Shrinkage: lambda_s = N_s / (N_s + N_0)
    - Final weight: w_{m,h,s} = lambda_s * w_{sample} + (1 - lambda_s) * w_{B2}
    - If tune=True (Lan05): Optuna optimizes N_0 in [5, 200] and inverse power gamma in [0.5, 3.0].
    """
    eps = 1e-8
    regimes = ['Low', 'Normal', 'High']

    if hist_slice.empty:
        # Cold start fallback to equal weighting
        w_b2 = {m: 0.25 for m in models}
        w_final_by_reg = {r: w_b2.copy() for r in regimes}
        pred = np.zeros(len(test_slice), dtype=float)
        for i, r in enumerate(test_slice['Regime'].values):
            pred[i] = sum(0.25 * test_slice[m].values[i] for m in models)
        return pred, {'mode': 'c2_cold_start_prior', 'weights': w_final_by_reg, 'N0': 50.0}, 0.0

    # Optimal N0 and power determination
    if tune and len(hist_slice) >= 50:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        y_h = hist_slice['Actual'].values

        def obj(trial):
            n0_val = trial.suggest_float('N0', 5.0, 200.0)
            pow_val = trial.suggest_float('power', 0.5, 3.0)

            # Compute B2
            mses_b2 = {m: np.mean((hist_slice['Actual'].values - hist_slice[m].values) ** 2) for m in models}
            inv_b2 = {m: (1.0 / (mses_b2[m] + eps)) ** pow_val for m in models}
            tot_b2 = sum(inv_b2.values())
            wb2 = {m: inv_b2[m] / tot_b2 for m in models}

            # Reg weights
            w_reg_map = {}
            for reg in regimes:
                s_df = hist_slice[hist_slice['Regime'] == reg]
                ns = len(s_df)
                lam = ns / (ns + n0_val)
                if ns >= 5:
                    m_reg = {m: np.mean((s_df['Actual'].values - s_df[m].values) ** 2) for m in models}
                    inv_r = {m: (1.0 / (m_reg[m] + eps)) ** pow_val for m in models}
                    tot_r = sum(inv_r.values())
                    ws = {m: inv_r[m] / tot_r for m in models}
                else:
                    ws = wb2
                w_reg_map[reg] = {m: lam * ws[m] + (1.0 - lam) * wb2[m] for m in models}

            p_h = np.zeros(len(hist_slice), dtype=float)
            for idx, r_val in enumerate(hist_slice['Regime'].values):
                w_curr = w_reg_map.get(r_val, wb2)
                p_h[idx] = sum(w_curr[m] * hist_slice[m].values[idx] for m in models)
            return float(np.sqrt(np.mean((y_h - p_h) ** 2)))

        study.optimize(obj, n_trials=n_trials)
        best_N0 = float(study.best_params['N0'])
        best_power = float(study.best_params['power'])
        val_score = float(study.best_value)
    else:
        best_N0 = 50.0  # Default per section 5.5
        best_power = 1.0
        val_score = 0.0

    # Calculate final weights with best_N0 and best_power
    mses_b2 = {m: float(np.mean((hist_slice['Actual'].values - hist_slice[m].values) ** 2)) for m in models}
    inv_b2 = {m: (1.0 / (mses_b2[m] + eps)) ** best_power for m in models}
    tot_b2 = sum(inv_b2.values())
    w_b2 = {m: inv_b2[m] / tot_b2 for m in models}

    w_final_by_reg = {}
    lambda_by_reg = {}
    for reg in regimes:
        sub_reg = hist_slice[hist_slice['Regime'] == reg]
        Ns = len(sub_reg)
        lam = Ns / (Ns + best_N0)
        lambda_by_reg[reg] = round(float(lam), 4)

        if Ns >= 5:
            mses_reg = {m: float(np.mean((sub_reg['Actual'].values - sub_reg[m].values) ** 2)) for m in models}
            inv_reg = {m: (1.0 / (mses_reg[m] + eps)) ** best_power for m in models}
            tot_reg = sum(inv_reg.values())
            w_sample = {m: inv_reg[m] / tot_reg for m in models}
        else:
            w_sample = w_b2

        w_final = {m: round(float(lam * w_sample[m] + (1.0 - lam) * w_b2[m]), 6) for m in models}
        w_final_by_reg[reg] = w_final

    # Generate predictions on test_slice
    pred = np.zeros(len(test_slice), dtype=float)
    for i, r in enumerate(test_slice['Regime'].values):
        w_curr = w_final_by_reg.get(r, w_b2)
        pred[i] = sum(w_curr[m] * test_slice[m].values[i] for m in models)

    best_params_record = {
        'mode': 'c2_optuna_shrunk' if tune else 'c2_default_shrunk_N0_50',
        'N0': best_N0,
        'power': best_power,
        'base_b2_weights': {m: round(float(v), 6) for m, v in w_b2.items()},
        'lambda_shrinkage': lambda_by_reg,
        'final_regime_weights': w_final_by_reg
    }
    return pred, best_params_record, val_score

def execute_c3_forecast(hist_slice, test_slice, models=CONSTITUENTS_A2, tune=False, n_trials=50, seed=config.RANDOM_SEED):
    """
    C3: Defensive Volatility Switch
    - Rule:
      - Low volatility: A2 (Core 4 Ensemble)
      - Normal volatility: A2 (Core 4 Ensemble)
      - High volatility: Historical Mean (Defensive benchmark)
    - If tune=True (Lan05): Optuna optimizes defensive blending alpha in [0, 1] for High volatility
      and core weights for Low/Normal volatility.
    """
    pred_a2 = test_slice[models].mean(axis=1).values
    pred_hm = test_slice['Historical_Mean'].values

    if not tune or hist_slice.empty or len(hist_slice) < 50:
        # Default strict switch (Lan04)
        pred = np.where(test_slice['Regime'] == 'High', pred_hm, pred_a2)
        best_params_record = {
            'mode': 'c3_default_defensive_switch',
            'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean',
            'high_defensive_weight': 1.0
        }
        val_score = 0.0
    else:
        # Optuna mode (Lan05)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        y_h = hist_slice['Actual'].values

        def obj(trial):
            alpha_high = trial.suggest_float('alpha_high', 0.0, 1.0)
            # Weights for Low/Normal
            w_norm = np.array([trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in models])
            s = w_norm.sum()
            w_norm = w_norm / s if s > 0 else np.full(len(models), 0.25)

            p_h = np.zeros(len(hist_slice), dtype=float)
            for idx, r_val in enumerate(hist_slice['Regime'].values):
                if r_val == 'High':
                    a2_val = sum(w_norm[k] * hist_slice[m].values[idx] for k, m in enumerate(models))
                    p_h[idx] = alpha_high * hist_slice['Historical_Mean'].values[idx] + (1.0 - alpha_high) * a2_val
                else:
                    p_h[idx] = sum(w_norm[k] * hist_slice[m].values[idx] for k, m in enumerate(models))
            return float(np.sqrt(np.mean((y_h - p_h) ** 2)))

        study.optimize(obj, n_trials=n_trials)
        best_alpha = float(study.best_params['alpha_high'])
        w_raw = [study.best_params[f'w_{m}'] for m in models]
        s_best = sum(w_raw)
        opt_core_w = {m: round(float(w / s_best), 6) for m, w in zip(models, w_raw)} if s_best > 0 else {m: 0.25 for m in models}

        # Apply to test_slice
        pred = np.zeros(len(test_slice), dtype=float)
        for i, r in enumerate(test_slice['Regime'].values):
            a2_pt = sum(opt_core_w[m] * test_slice[m].values[i] for m in models)
            if r == 'High':
                pred[i] = best_alpha * test_slice['Historical_Mean'].values[i] + (1.0 - best_alpha) * a2_pt
            else:
                pred[i] = a2_pt

        val_score = float(study.best_value)
        best_params_record = {
            'mode': 'c3_optuna_defensive_switch',
            'best_alpha_high_defensive': best_alpha,
            'low_normal_core_weights': opt_core_w
        }

    return pred, best_params_record, val_score

# ==============================================================================
# GENERAL ENSEMBLE RUNNER (Dispatches A1, A2, A3, C1, C2, C3)
# ==============================================================================

def run_ensemble(exp_key='A1', tune=False, horizons=config.HORIZONS, n_trials=50, logger=None, seed=config.RANDOM_SEED):
    """
    Executes walk-forward evaluation of any experiment (A1, A2, A3, C1, C2, C3).
    - If tune=False (Lan04): Uses fixed default parameters.
    - If tune=True (Lan05): Optimizes parameters/weights via Optuna on expanding historical OOS.
    """
    if exp_key not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment: {exp_key}. Choose from {list(EXPERIMENT_SPECS.keys())}")

    spec = EXPERIMENT_SPECS[exp_key]
    model_name = spec['display_name']
    constituents = spec['models']

    if logger:
        logger.info(f"▶ Initializing {model_name} ({spec['title']})")
        logger.info(f"  Description: {spec.get('description', '')}")
        logger.info(f"  Mode: {'Optuna Adaptive Tuning (Lan05)' if tune else 'Fixed Default Parameters (Lan04)'}")

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
            test_slice = pvt_df[(pvt_df['Horizon'] == horizon) & (pvt_df['Fold'] == fold)].copy()
            if test_slice.empty:
                continue

            hist_slice = pvt_df[(pvt_df['Horizon'] == horizon) & (pvt_df['Fold'] < fold)].copy()

            # Dispatch based on experiment family
            if exp_key in ['A1', 'A2', 'A3']:
                default_w = spec['default_weights']
                if not tune:
                    weights_dict = default_w.copy()
                    val_score = 0.0
                    score_metric = "Default_Equal_Weights"
                    best_params_record = {
                        'mode': 'equal_weight',
                        'formula': f"1/{len(constituents)} * sum({model_name})",
                        'weights': weights_dict
                    }
                else:
                    if not hist_slice.empty and len(hist_slice) >= 100:
                        weights_dict, val_score = optimize_simplex_weights(
                            hist_slice, constituents, n_trials=n_trials, seed=seed
                        )
                        score_metric = "Hist_OOS_RMSE"
                        best_params_record = {
                            'mode': 'optuna_tpe',
                            'tuning_folds': f"1..{fold-1}",
                            'weights': weights_dict
                        }
                    else:
                        weights_dict = default_w.copy()
                        val_score = 0.0
                        score_metric = "Cold_Start_Prior"
                        best_params_record = {
                            'mode': 'cold_start_prior',
                            'weights': weights_dict
                        }

                ensemble_pred = np.zeros(len(test_slice), dtype=float)
                for m in constituents:
                    ensemble_pred += weights_dict[m] * test_slice[m].values

            elif exp_key == 'B1':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_b1_forecast(
                    hist_slice, test_slice, horizon, models=constituents, tune=tune, n_trials=n_trials, seed=seed
                )

            elif exp_key == 'B2':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_b2_forecast(
                    hist_slice, test_slice, models=constituents, tune=tune, n_trials=n_trials, seed=seed
                )

            elif exp_key == 'B3':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_b3_forecast(
                    hist_slice, test_slice, models=constituents, tune=tune, n_trials=n_trials, seed=seed
                )

            elif exp_key == 'C1':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_c1_forecast(
                    hist_slice, test_slice, models=constituents, tune=tune, n_trials=n_trials, seed=seed
                )

            elif exp_key == 'C2':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_c2_forecast(
                    hist_slice, test_slice, models=constituents, tune=tune, n_trials=n_trials, seed=seed
                )

            elif exp_key == 'C3':
                score_metric = "Val_RMSE"
                ensemble_pred, best_params_record, val_score = execute_c3_forecast(
                    hist_slice, test_slice, models=CONSTITUENTS_A2, tune=tune, n_trials=n_trials, seed=seed
                )

            # Record tuning details in results/tuning/3.0.3_best_hyperparameters.csv
            utils.save_tuning_record(
                model_name, horizon, fold, best_params_record, val_score, score_metric=score_metric
            )

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
                'tuning_mode': 'Optuna' if tune else 'Default'
            })

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

    utils.save_model_latest_results(CURRENT_DIR, m_df, p_df)
    return m_df, p_df
