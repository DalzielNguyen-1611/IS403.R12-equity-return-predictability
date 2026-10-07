import os
import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import torch
try:
    from . import config
except (ImportError, ValueError):
    import config

def setup_logger(name="baseline_runner", log_file=None):
    """Setup console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def set_all_seeds(seed=config.RANDOM_SEED):
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset():
    """Load and validate the clean walk-forward dataset."""
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def get_walk_forward_splits(df):
    """
    Generator yielding (fold, train_df, val_df, meta_dict).
    Strict Walk-Forward:
      - Training: Expanding window starting from row 0 to before the current fold.
      - Validation: Fixed ~272 rows for the current fold.
    """
    for fold in range(1, 6):
        val_mask = (df['Val_Fold'] == fold)
        val_df = df[val_mask].copy()
        if len(val_df) == 0:
            raise ValueError(f"No validation records found for Val_Fold == {fold}")
        
        val_start_idx = val_df.index[0]
        train_df = df.iloc[:val_start_idx].copy()
        
        meta = {
            'fold': fold,
            'train_start': str(train_df['Date'].iloc[0].date()),
            'train_end': str(train_df['Date'].iloc[-1].date()),
            'val_start': str(val_df['Date'].iloc[0].date()),
            'val_end': str(val_df['Date'].iloc[-1].date()),
            'train_rows': len(train_df),
            'val_rows': len(val_df)
        }
        yield fold, train_df, val_df, meta

def prepare_fold_data(train_df, val_df, horizon, feature_cols=None, scale=True):
    """
    Extracts features and target for a fold with strict leakage control.
    Scaler is fit ONLY on train_df, then transforms both train_df and val_df.
    """
    if feature_cols is None:
        feature_cols = config.FEATURE_COLS
    
    target_col = config.HORIZON_TO_TARGET[horizon]
    
    train_valid = train_df.dropna(subset=[target_col]).copy()
    val_valid = val_df.dropna(subset=[target_col]).copy()
    
    X_train = train_valid[feature_cols].values
    y_train = train_valid[target_col].values
    
    X_val = val_valid[feature_cols].values
    y_val = val_valid[target_col].values
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    
    y_train_dir = (y_train > 0).astype(int)
    y_val_dir = (y_val > 0).astype(int)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_dir': y_train_dir,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_dir': y_val_dir,
        'val_dates': val_valid['Date'].dt.strftime('%Y-%m-%d').values,
        'val_regimes': val_valid['Volatility_Regime'].values,
        'scaler': scaler
    }

def create_sequences(X_train, y_train, X_val, lookback=20):
    """
    Construct lookback sequences for RNN/LSTM/Transformer.
    Val sequences use the historical tail of X_train so that exactly all
    validation days have a full sequence without looking ahead.
    """
    full_X = np.concatenate([X_train, X_val], axis=0)
    N_train = len(X_train)
    N_val = len(X_val)
    
    train_seqs = [X_train[i - lookback + 1 : i + 1] for i in range(lookback - 1, N_train)]
    train_targets = y_train[lookback - 1 : N_train]
    val_seqs = [full_X[i - lookback + 1 : i + 1] for i in range(N_train, N_train + N_val)]
    
    return np.array(train_seqs, dtype=np.float32), np.array(train_targets, dtype=np.float32), np.array(val_seqs, dtype=np.float32)

def diebold_mariano_test(y_true, y_model, y_bench, h=1, loss='squared', alternative='greater'):
    """
    Diebold-Mariano test with Harvey, Leybourne, and Newbold (HLN, 1997) finite-sample correction.
    Testing H0: Model and Benchmark have equal predictive accuracy vs H1: Model is more accurate.
    Loss differential d_t = Loss(bench) - Loss(model). Positive d_t means model has smaller error.
    """
    e_m = y_true - y_model
    e_b = y_true - y_bench
    d = (e_b**2 - e_m**2) if loss == 'squared' else (np.abs(e_b) - np.abs(e_m))
    T = len(d)
    if T <= 1:
        return 0.0, 1.0
    
    mean_d = np.mean(d)
    gamma = [np.var(d, ddof=0)]
    for lag in range(1, int(h)):
        gamma.append(np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d)))
    
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_d <= 1e-12:
        return 0.0, 1.0
    
    dm_stat = mean_d / np.sqrt(var_d)
    hln_factor = np.sqrt(max(1e-6, (T + 1 - 2*h + h*(h-1)/T) / T))
    dm_stat_adj = dm_stat * hln_factor
    
    df = max(1, T - 1)
    from scipy import stats
    if alternative == 'greater':
        p_val = 1.0 - stats.t.cdf(dm_stat_adj, df=df)
    elif alternative == 'less':
        p_val = stats.t.cdf(dm_stat_adj, df=df)
    else:
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat_adj), df=df))
    return float(dm_stat_adj), float(p_val)

def directional_binomial_test(y_true, y_pred, alternative='greater'):
    """
    Binomial / Sign test for Directional Accuracy.
    Tests H0: Directional Accuracy <= 0.5 (no predictive skill / coin flip)
    vs H1: Directional Accuracy > 0.5 (statistically significant directional skill).
    """
    sign_true = np.where(y_true > 0, 1, -1)
    sign_pred = np.where(y_pred > 0, 1, -1)
    k = int(np.sum(sign_true == sign_pred))
    n = int(len(y_true))
    if n == 0:
        return 0.5, 1.0
    from scipy.stats import binomtest
    res = binomtest(k=k, n=n, p=0.5, alternative=alternative)
    return float(k / n), float(res.pvalue)

def calc_regression_metrics(y_true, y_pred, y_bench=None, h=1):
    """
    Calculate metrics based on the official Research Metric Framework:
    - Primary: OOS_R2
    - Secondary: MAE, RMSE, DA (Directional Accuracy)
    - Statistical Inference: DM_Stat, DM_pvalue, Sign_Test_pvalue
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # OOS R2
    if y_bench is not None:
        sse_model = np.sum((y_true - y_pred) ** 2)
        sse_bench = np.sum((y_true - y_bench) ** 2)
        oos_r2 = 1.0 - (sse_model / (sse_bench + 1e-12))
    else:
        oos_r2 = r2_score(y_true, y_pred)
        
    # Directional Accuracy & Sign/Binomial Test
    da, sign_pval = directional_binomial_test(y_true, y_pred, alternative='greater')
    
    res = {
        'OOS_R2': float(oos_r2),
        'R2': float(oos_r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'DA': float(da),
        'Directional_Accuracy': float(da),
        'Sign_pvalue': float(sign_pval)
    }
    
    # DM Test (if benchmark is supplied)
    if y_bench is not None:
        dm_stat, dm_pval = diebold_mariano_test(y_true, y_pred, y_bench, h=h, loss='squared', alternative='greater')
        res['DM_Stat'] = float(dm_stat)
        res['DM_pvalue'] = float(dm_pval)
        
    return res

def calc_classification_metrics(y_true, y_prob, y_pred=None):
    """
    Calculate classification metrics for Logistic Regression:
    - Primary: ROC_AUC
    - Secondary: Accuracy, Precision, Recall, F1, Balanced_Accuracy
    """
    from sklearn.metrics import balanced_accuracy_score
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    return {
        'ROC_AUC': float(auc),
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'Balanced_Accuracy': float(bal_acc)
    }

def save_model_latest_results(model_dir, metrics_df, preds_df):
    """
    Saves outputs following the 3-tier architecture:
    1. Local copy in model directory: latest_metrics.csv, latest_predictions.csv
    2. Central Tier 1: results/fold_metrics/all_fold_metrics.csv and results/summary/model_horizon_summary.csv
    3. Central Tier 2: results/predictions/all_oos_predictions.csv
    """
    # Standardize column names
    col_map = {
        'model': 'Model', 'horizon': 'Horizon', 'fold': 'Fold',
        'train_start': 'Train_Start', 'train_end': 'Train_End',
        'val_start': 'Val_Start', 'val_end': 'Val_End',
        'Directional_Accuracy': 'DA', 'R2': 'OOS_R2'
    }
    m_clean = metrics_df.rename(columns=col_map).copy()
    p_clean = preds_df.rename(columns=col_map).copy()
    
    # 1. Local files in model directory
    metrics_path = os.path.join(model_dir, "latest_metrics.csv")
    preds_path = os.path.join(model_dir, "latest_predictions.csv")
    m_clean.to_csv(metrics_path, index=False)
    p_clean.to_csv(preds_path, index=False)
    
    # 2. Central Tier 1: all_fold_metrics.csv (upsert)
    all_folds_path = config.ALL_FOLD_METRICS_PATH
    if os.path.exists(all_folds_path):
        try:
            existing_m = pd.read_csv(all_folds_path)
            # Filter out existing entries for the same model and horizons present in m_clean
            cond = ~((existing_m['Model'].isin(m_clean['Model'])) & (existing_m['Horizon'].isin(m_clean['Horizon'])))
            combined_m = pd.concat([existing_m[cond], m_clean], ignore_index=True)
        except Exception:
            combined_m = m_clean
    else:
        combined_m = m_clean
    combined_m.to_csv(all_folds_path, index=False)
    
    # 3. Central Tier 2: all_oos_predictions.csv (upsert)
    all_preds_path = config.ALL_PREDICTIONS_PATH
    if os.path.exists(all_preds_path):
        try:
            existing_p = pd.read_csv(all_preds_path)
            cond_p = ~((existing_p['Model'].isin(p_clean['Model'])) & (existing_p['Horizon'].isin(p_clean['Horizon'])))
            combined_p = pd.concat([existing_p[cond_p], p_clean], ignore_index=True)
        except Exception:
            combined_p = p_clean
    else:
        combined_p = p_clean
    combined_p.to_csv(all_preds_path, index=False)
    
    # 4. Central Tier 1: model_horizon_summary.csv (aggregate mean & std across folds)
    if 'MAE' in combined_m.columns and 'RMSE' in combined_m.columns:
        reg_df = combined_m[combined_m['Model'] != 'logistic_regression'].copy()
        if not reg_df.empty:
            summary_rows = []
            for (mod, hrz), grp in reg_df.groupby(['Model', 'Horizon']):
                row = {
                    'Model': mod,
                    'Horizon': hrz,
                    'MAE_Mean': grp['MAE'].mean(),
                    'MAE_Std': grp['MAE'].std(),
                    'RMSE_Mean': grp['RMSE'].mean(),
                    'RMSE_Std': grp['RMSE'].std(),
                    'OOS_R2_Mean': grp['OOS_R2'].mean() if 'OOS_R2' in grp.columns else np.nan,
                    'OOS_R2_Std': grp['OOS_R2'].std() if 'OOS_R2' in grp.columns else np.nan,
                    'DA_Mean': grp['DA'].mean() if 'DA' in grp.columns else np.nan,
                    'DA_Std': grp['DA'].std() if 'DA' in grp.columns else np.nan,
                }
                summary_rows.append(row)
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(config.MODEL_HORIZON_SUMMARY_PATH, index=False)
            
    # 5. Auto-update DASHBOARD.md directly
    try:
        update_dashboard_direct()
    except Exception:
        pass

    return metrics_path, preds_path

def save_tuning_record(model_name, horizon, fold, best_params, best_score, score_metric='Val_RMSE'):
    """
    Save hyperparameter tuning evidence into results/tuning/best_hyperparameters.csv
    Columns: Model, Horizon, Fold, Best_Params, Best_Val_Score, Score_Metric
    """
    rec = pd.DataFrame([{
        'Model': model_name,
        'Horizon': horizon,
        'Fold': fold,
        'Best_Params': str(best_params),
        'Best_Val_Score': round(float(best_score), 6),
        'Score_Metric': score_metric
    }])
    path = config.BEST_HYPERPARAMS_PATH
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            cond = ~((existing['Model'] == model_name) & (existing['Horizon'] == horizon) & (existing['Fold'] == fold))
            combined = pd.concat([existing[cond], rec], ignore_index=True)
        except Exception:
            combined = rec
    else:
        combined = rec
    combined.to_csv(path, index=False)

def get_internal_train_val_split(X_train, y_train, val_ratio=0.15):
    """
    Splits Fold k training data chronologically into internal train and internal validation.
    Guarantees strict zero-leakage: fold k validation set is NEVER seen during tuning.
    """
    n_train = len(X_train)
    split_idx = int(n_train * (1.0 - val_ratio))
    X_tr_inner = X_train[:split_idx]
    y_tr_inner = y_train[:split_idx]
    X_val_inner = X_train[split_idx:]
    y_val_inner = y_train[split_idx:]
    return X_tr_inner, y_tr_inner, X_val_inner, y_val_inner

def load_checkpoint(model_dir):
    """
    Load in-progress fold metrics and predictions from local checkpoint files.
    Returns:
        completed_set: set of (str(horizon), int(fold))
        fold_metrics: list of metric dicts
        oos_preds: list of prediction dicts
    """
    ckpt_m_path = os.path.join(model_dir, "checkpoint_metrics.csv")
    ckpt_p_path = os.path.join(model_dir, "checkpoint_predictions.csv")
    if os.path.exists(ckpt_m_path) and os.path.exists(ckpt_p_path):
        try:
            m_df = pd.read_csv(ckpt_m_path)
            p_df = pd.read_csv(ckpt_p_path)
            completed_set = set(zip(m_df['horizon'].astype(str), m_df['fold'].astype(int)))
            return completed_set, m_df.to_dict('records'), p_df.to_dict('records')
        except Exception:
            pass
    return set(), [], []

def save_checkpoint_step(model_dir, fold_metrics, oos_preds):
    """
    Save current fold progress to checkpoint files after each fold finishes.
    And automatically update DASHBOARD.md in real-time.
    """
    ckpt_m_path = os.path.join(model_dir, "checkpoint_metrics.csv")
    ckpt_p_path = os.path.join(model_dir, "checkpoint_predictions.csv")
    try:
        pd.DataFrame(fold_metrics).to_csv(ckpt_m_path, index=False)
        pd.DataFrame(oos_preds).to_csv(ckpt_p_path, index=False)
    except Exception:
        pass

    # Real-time Dashboard auto-refresh directly inside model execution code
    try:
        update_dashboard_direct()
    except Exception:
        pass

def update_dashboard_direct():
    """
    Directly generates and writes DASHBOARD.md inside the model training process.
    Zero external scripts needed. Updates in real-time after every fold and every model.
    """
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dashboard_path = os.path.join(config.REPO_ROOT, "DASHBOARD.md")
    timing_path = os.path.join(config.RESULTS_DIR, "benchmark_timing.csv")
    
    all_models_meta = [
        {"name": "Historical_Mean", "category": "Benchmark", "total_folds": 25, "folder": "historical_mean"},
        {"name": "Ridge", "category": "Linear Regression", "total_folds": 25, "folder": "ridge"},
        {"name": "Logistic_Regression", "category": "Classification", "total_folds": 25, "folder": "logistic_regression"},
        {"name": "Elastic_Net", "category": "Linear Regression", "total_folds": 25, "folder": "elastic_net"},
        {"name": "ARIMA", "category": "Statistical TS", "total_folds": 25, "folder": "arima"},
        {"name": "ARIMAX", "category": "Statistical TS", "total_folds": 25, "folder": "arimax"},
        {"name": "SARIMA", "category": "Statistical TS", "total_folds": 25, "folder": "sarima"},
        {"name": "LightGBM", "category": "Tree Ensemble", "total_folds": 25, "folder": "lightgbm"},
        {"name": "SVR", "category": "Kernel Method", "total_folds": 25, "folder": "svr"},
        {"name": "Random_Forest", "category": "Tree Ensemble", "total_folds": 25, "folder": "random_forest"},
        {"name": "XGBoost", "category": "Tree Ensemble", "total_folds": 25, "folder": "xgboost"},
        {"name": "RNN", "category": "Deep Learning", "total_folds": 25, "folder": "rnn"},
        {"name": "LSTM", "category": "Deep Learning", "total_folds": 25, "folder": "lstm"},
        {"name": "Transformer", "category": "Deep Learning", "total_folds": 25, "folder": "transformer"},
    ]
    
    def fmt_dur(sec):
        if sec is None or np.isnan(sec):
            return "-"
        if sec < 60:
            return f"{sec:.1f}s"
        m = int(sec // 60)
        s = int(sec % 60)
        if m < 60:
            return f"{m:02d}m {s:02d}s"
        h = int(m // 60)
        m = int(m % 60)
        return f"{h:02d}h {m:02d}m"

    # 1. Load completed fold counts from central all_fold_metrics.csv
    completed_counts = {}
    fm_df = None
    if os.path.exists(config.ALL_FOLD_METRICS_PATH):
        try:
            fm_df = pd.read_csv(config.ALL_FOLD_METRICS_PATH)
            if 'Model' in fm_df.columns:
                completed_counts = fm_df['Model'].value_counts().to_dict()
        except Exception:
            pass
            
    # 2. Check for active in-progress fold checkpoint in model directories
    active_model_name = None
    active_ckpt_df = None
    active_elapsed = 0.0
    active_speed = 0.0
    
    for m_info in all_models_meta:
        m_dir = os.path.join(config.BASELINES_DIR, m_info["folder"])
        ckpt_file = os.path.join(m_dir, "checkpoint_metrics.csv")
        if os.path.exists(ckpt_file):
            try:
                c_df = pd.read_csv(ckpt_file)
                if not c_df.empty:
                    active_model_name = m_info["name"]
                    active_ckpt_df = c_df
                    c_ctime = os.path.getctime(ckpt_file)
                    c_mtime = os.path.getmtime(ckpt_file)
                    active_elapsed = max(c_mtime - c_ctime, 1.0)
                    active_speed = active_elapsed / len(c_df)
                    break
            except Exception:
                pass
                
    # 3. Load historical timings
    timing_data = {}
    if os.path.exists(timing_path):
        try:
            t_df = pd.read_csv(timing_path)
            for _, r in t_df.iterrows():
                timing_data[str(r['Model'])] = {
                    'total': float(r['Historical_Total_Seconds']) if pd.notnull(r['Historical_Total_Seconds']) else None,
                    'avg': float(r['Avg_Seconds_Per_Fold']) if pd.notnull(r['Avg_Seconds_Per_Fold']) else None,
                }
        except Exception:
            pass

    # Determine highlighted model
    is_active = (active_model_name is not None and active_ckpt_df is not None)
    if is_active:
        focus_name = active_model_name
        focus_df = active_ckpt_df
        last_r = active_ckpt_df.iloc[-1]
        done_folds_act = len(active_ckpt_df)
        completed_counts[active_model_name] = done_folds_act
        status_banner = f"🟡 Đang huấn luyện: Fold {last_r.get('fold', done_folds_act)}/5 (Horizon {last_r.get('horizon', '-')}) — Cập nhật Real-Time"
        focus_dur_str = fmt_dur(active_elapsed)
        focus_speed_str = f"{active_speed:.2f}s/fold"
        focus_eta_str = f"~{fmt_dur((25 - done_folds_act) * active_speed)} (còn {25 - done_folds_act} folds)"
    else:
        focus_name = "LightGBM"
        if fm_df is not None and not fm_df.empty:
            focus_name = str(fm_df.iloc[-1]['Model'])
            focus_df = fm_df[fm_df['Model'] == focus_name]
            last_r = focus_df.iloc[-1]
        else:
            last_r = {}
            focus_df = pd.DataFrame()
        status_banner = "🟢 Đã hoàn thành 25/25 Folds — Chờ lệnh chạy mô hình tiếp theo"
        t_info = timing_data.get(focus_name, {})
        focus_dur_str = fmt_dur(t_info.get('total'))
        focus_speed_str = f"{t_info.get('avg', 0):.2f}s/fold" if t_info.get('avg') else "-"
        focus_eta_str = "Đã xong 100% (còn 0 folds)"

    latest_fold_info = {
        'horizon': str(last_r.get('horizon', last_r.get('Horizon', '-'))),
        'fold': str(last_r.get('fold', last_r.get('Fold', '-'))),
        'oos_r2': f"{float(last_r.get('R2', last_r.get('OOS_R2', 0))):.4f}" if ('R2' in last_r or 'OOS_R2' in last_r) and pd.notnull(last_r.get('R2', last_r.get('OOS_R2'))) else "-",
        'da': f"{float(last_r.get('Directional_Accuracy', last_r.get('DA', 0))) * 100:.2f}%" if ('Directional_Accuracy' in last_r or 'DA' in last_r) and pd.notnull(last_r.get('Directional_Accuracy', last_r.get('DA'))) else "-",
        'rmse': f"{float(last_r.get('RMSE', 0)):.4f}" if 'RMSE' in last_r and pd.notnull(last_r.get('RMSE')) else "-",
        'mae': f"{float(last_r.get('MAE', 0)):.4f}" if 'MAE' in last_r and pd.notnull(last_r.get('MAE')) else "-",
    }

    # Best record across all historical data
    best_r2_val = "-"
    best_r2_model = "-"
    if fm_df is not None and 'OOS_R2' in fm_df.columns:
        valid_r2 = fm_df.dropna(subset=['OOS_R2'])
        if not valid_r2.empty:
            best_idx = valid_r2['OOS_R2'].idxmax()
            best_row = valid_r2.loc[best_idx]
            best_r2_val = f"{best_row['OOS_R2']:.4f}"
            best_r2_model = f"{best_row['Model']} (H={best_row['Horizon']}, Fold={best_row['Fold']})"

    # Table rows
    tot_models = len(all_models_meta)
    done_models = 0
    tot_folds = tot_models * 25
    done_folds_all = 0
    tot_logged_sec = 0.0
    table_rows = []
    
    for idx, m_info in enumerate(all_models_meta, 1):
        m_name = m_info["name"]
        cat = m_info["category"]
        cnt = completed_counts.get(m_name, 0)
        done_folds_all += cnt
        
        t_info = timing_data.get(m_name, {})
        hist_total = t_info.get('total')
        avg_fold = t_info.get('avg')
        
        if cnt >= 25:
            done_models += 1
            status = "✅ Hoàn thành"
            past_time_str = fmt_dur(hist_total) if hist_total else "Hoàn tất"
            avg_fold_str = f"{avg_fold:.2f}s/fold" if avg_fold else "-"
            eta_str = "Đã xong (100%)"
            if hist_total:
                tot_logged_sec += hist_total
        elif cnt > 0:
            status = f"🔄 Đang chạy ({cnt}/25)"
            rem = 25 - cnt
            if m_name == active_model_name and active_speed > 0:
                past_time_str = fmt_dur(active_elapsed)
                avg_fold_str = f"{active_speed:.2f}s/fold"
                eta_str = f"~{fmt_dur(rem * active_speed)} (còn {rem} folds)"
            elif avg_fold and avg_fold > 0:
                past_time_str = fmt_dur(cnt * avg_fold)
                avg_fold_str = f"{avg_fold:.2f}s/fold"
                eta_str = f"~{fmt_dur(rem * avg_fold)} (còn {rem} folds)"
            else:
                past_time_str = f"{cnt}/25 folds"
                avg_fold_str = "Đang đo..."
                eta_str = f"Ước lượng sau khi xong fold (còn {rem} folds)"
        else:
            status = "⏳ Chờ chạy"
            avg_fold_str = "-"
            if hist_total and hist_total > 0:
                past_time_str = f"Lịch sử: {fmt_dur(hist_total)}"
                avg_fold_str = f"{avg_fold:.2f}s/fold" if avg_fold else "-"
                eta_str = f"~{fmt_dur(hist_total)} (theo lịch sử chạy trước của chính nó)"
            else:
                past_time_str = "Chưa có lượt chạy"
                eta_str = "*Chờ fold 1 của chính model này (Không ước lượng từ model khác)*"
                
        table_rows.append({
            "idx": idx, "name": m_name, "cat": cat, "progress": f"{cnt}/25",
            "status": status, "past_time": past_time_str, "avg_fold": avg_fold_str, "eta": eta_str
        })
        
    overall_pct = (done_folds_all / tot_folds) * 100.0
    filled = int(round(24 * overall_pct / 100.0))
    overall_bar = f"`[{'█' * filled + '░' * (24 - filled)}] {overall_pct:.1f}%`"
    
    # Section 2 recent folds (most recent completed steps)
    recent_folds_rows = []
    if not focus_df.empty:
        total_focus_folds = len(focus_df)
        start_idx = max(1, total_focus_folds - 4)
        for step_num, (_, r) in enumerate(focus_df.tail(5).iterrows(), start=start_idx):
            hz = r.get('horizon', r.get('Horizon', '-'))
            fd = r.get('fold', r.get('Fold', '-'))
            v_start = r.get('val_start', r.get('Val_Start', ''))
            v_end = r.get('val_end', r.get('Val_End', ''))
            val_period = f"{v_start} → {v_end}" if v_start and v_end else f"Fold {fd}/5"
            r2_val = r.get('R2', r.get('OOS_R2'))
            da_val = r.get('Directional_Accuracy', r.get('DA'))
            rmse_val = r.get('RMSE')
            mae_val = r.get('MAE')
            recent_folds_rows.append({
                "step": f"#{step_num}",
                "horizon": hz, "fold": f"Fold {fd}/5",
                "val_period": val_period,
                "status": "✅ Hoàn thành",
                "r2": f"{float(r2_val):.4f}" if pd.notnull(r2_val) else "-",
                "da": f"{float(da_val)*100:.2f}%" if pd.notnull(da_val) else "-",
                "rmse": f"{float(rmse_val):.4f}" if pd.notnull(rmse_val) else "-",
                "mae": f"{float(mae_val):.4f}" if pd.notnull(mae_val) else "-"
            })

    md = f"""# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** {focus_name} | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** {now_str}  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`{focus_name}`** | **Fold vừa hoàn tất** | Fold {latest_fold_info.get('fold', '-')}/5 (Horizon {latest_fold_info.get('horizon', '-')}) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `{latest_fold_info.get('oos_r2', '-')}` |
| **Tiến trình Mô hình** | **`{done_models}/{tot_models} Models ({overall_pct:.1f}%)`** | **Directional Accuracy (DA)** | `{latest_fold_info.get('da', '-')}` |
| **Tiến trình Folds tổng** | **`{done_folds_all}/{tot_folds} Folds ({overall_pct:.1f}%)`** | **RMSE Fold vừa xong** | `{latest_fold_info.get('rmse', '-')}` |
| **Kỷ lục OOS R² cao nhất** | `{best_r2_val}` | **Loss / MAE Fold vừa xong** | `{latest_fold_info.get('mae', '-')}` |
| **Mô hình giữ Kỷ lục** | {best_r2_model} | **Best Horizon của Model** | Horizon {latest_fold_info.get('horizon', '-')} |
| **Thời gian đã chạy (chính nó)** | **`{focus_dur_str}`** | **Tốc độ chạy (chính nó)** | **`{focus_speed_str}`** |
| **Tiến độ Nhóm Linear & Stats** | ✅ ĐÃ HOÀN TẤT 7/7 MODELS | **Trạng thái Nhóm 1** | Đã lưu Fold Metrics & OOS Preds |
| **Dự kiến XONG mô hình này** | {focus_eta_str} | **Thời gian còn lại (chính nó)** | {focus_eta_str} |
| **Trạng thái mô hình** | {status_banner} | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Code tự ghi vào file) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`{focus_name}`)

> 💡 **Quy luật huấn luyện Time-Series:** Mô hình chạy tuần tự theo Horizon: **`1D` (Fold 1→2→3→4→5)** rồi mới chuyển sang **`5D` (Fold 1→2→3→4→5)**, kế tiếp là `20D`, `60D`, `120D`.  
> Bảng dưới đây trích xuất **5 bước vừa chạy xong gần nhất** theo đúng dòng thời gian:

| BƯỚC | HORIZON | FOLD | GIAI ĐOẠN VALIDATION | TRẠNG THÁI | VAL OOS R² | VAL DA | VAL RMSE | VAL MAE |
| :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for rf in recent_folds_rows:
        md += f"| **{rf['step']}** | **{rf['horizon']}** | {rf['fold']} | `{rf['val_period']}` | {rf['status']} | `{rf['r2']}` | `{rf['da']}` | `{rf['rmse']}` | `{rf['mae']}` |\n"


    md += f"""
---

## 📋 3. TIẾN ĐỘ TỔNG HỢP 14 MÔ HÌNH & DỰ BÁO HOÀN THÀNH

> ⚠️ **Nguyên tắc vàng:** Thời gian ước lượng (ETA) **bắt buộc tính dựa trên tốc độ và thời gian quá khứ của CHÍNH model đó**, tuyệt đối không suy đoán chéo giữa các model khác nhau.

| # | Mô hình | Nhóm thuật toán | Tiến độ | Trạng thái | Thời gian thực tế của chính nó | Tốc độ 1 fold | Dự báo hoàn thành (ETA của chính model) |
| :-: | :--- | :--- | :-: | :-: | :---: | :---: | :--- |
"""
    for r in table_rows:
        md += f"| {r['idx']} | **`{r['name']}`** | {r['cat']} | `{r['progress']}` | {r['status']} | {r['past_time']} | `{r['avg_fold']}` | {r['eta']} |\n"

    md += f"""
---

## ⚡ 4. THỨ TỰ & LỆNH ĐIỀU PHỐI MÔ HÌNH TIẾP THEO

Chạy toàn bộ 6 mô hình còn lại tự động trong 1 lệnh duy nhất:
```powershell
python run_remaining.py
```

---
*Bảng điều khiển được cập nhật **hoàn toàn tự động từ bên trong code huấn luyện** sau mỗi Fold.*
"""
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(md)



def finalize_checkpoint(model_dir):
    """
    Clean up checkpoint files once all horizons & folds have completed.
    """
    for fname in ["checkpoint_metrics.csv", "checkpoint_predictions.csv"]:
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

def load_saved_tuning_params(model_name):
    """
    Check results/tuning/best_hyperparameters.csv to recover any previously tuned params for this model.
    Returns dict: {(str(horizon), int(fold)): (params_dict, val_score)}
    """
    import ast
    path = config.BEST_HYPERPARAMS_PATH
    cached = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            sub = df[df['Model'] == model_name]
            for _, r in sub.iterrows():
                h = str(r['Horizon'])
                f = int(r['Fold'])
                raw_params = r['Best_Params']
                try:
                    params_dict = ast.literal_eval(raw_params)
                except Exception:
                    params_dict = raw_params
                val_score = float(r['Best_Val_Score']) if pd.notnull(r.get('Best_Val_Score')) else 0.0
                cached[(h, f)] = (params_dict, val_score)
        except Exception:
            pass
    return cached


