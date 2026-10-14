import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from . import test_config as config
except (ImportError, ValueError):
    import test_config as config

# Re-use baselines utils if available for sequence generation and seed setting
BASELINES_DIR = os.path.join(config.SRC_DIR, 'training', 'baselines')
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

# Import model architectures
try:
    from rnn.rnn_model import VanillaRNN
    from lstm.lstm_model import LSTMModel
    from transformer.transformer_model import TimeSeriesTransformer
except Exception:
    # Standalone definitions in case baseline path is not directly resolvable
    class VanillaRNN(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, input_size=16, lookback=20, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.input_proj(x) + self.pos_emb
            out = self.transformer(h)
            return self.fc(out[:, -1, :]).squeeze(-1)


def set_all_seeds(seed=config.RANDOM_SEED):
    import random
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
    """Load and validate the clean dataset."""
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Defensive sanitization
    inf_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[inf_cols]).any().any():
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan).ffill()
    return df


def get_purged_train_df(train_df, horizon):
    """
    Marcos López de Prado Purging Protocol.
    For horizon h > 1, drops last h - 1 observations of train_df to eliminate overlap into test set.
    """
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)
    if h_step > 1:
        purge_gap = h_step - 1
        return train_df.iloc[:-purge_gap].copy()
    return train_df.copy()


def prepare_test_data(train_df, test_df, horizon, feature_cols=None, scale=True, purge=True):
    """
    Extracts features and target for test evaluation with strict zero-leakage:
    - Purges last h-1 days of train_df (boundary label overlap elimination).
    - Fits StandardScaler ONLY on purged train_df.
    - Transforms test_df using the fitted scaler.
    - Calculates Campbell-Thompson Historical Mean benchmark from train_df.
    """
    if feature_cols is None:
        feature_cols = config.FEATURE_COLS
    target_col = config.HORIZON_TO_TARGET[horizon]
    
    train_valid = train_df.dropna(subset=[target_col]).copy()
    test_valid = test_df.dropna(subset=[target_col]).copy()
    
    if purge:
        train_valid = get_purged_train_df(train_valid, horizon)
        
    X_train = train_valid[feature_cols].values
    y_train = train_valid[target_col].values
    
    X_test = test_valid[feature_cols].values
    y_test = test_valid[target_col].values
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    y_train_dir = (y_train > 0).astype(int)
    y_test_dir = (y_test > 0).astype(int)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_dir': y_train_dir,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_dir': y_test_dir,
        'test_dates': test_valid['Date'].dt.strftime('%Y-%m-%d').values,
        'test_regimes': test_valid['Volatility_Regime'].values,
        'y_bench': np.full(len(y_test), np.mean(y_train)),
        'train_mean': float(np.mean(y_train)),
        'scaler': scaler,
        'train_valid': train_valid,
        'test_valid': test_valid
    }


def create_sequences(X_train, y_train, X_test, lookback=20):
    """
    Construct lookback sequences for RNN/LSTM/Transformer.
    Test sequences use the historical tail of X_train so that exactly all
    test days have full sequences without looking into test-period future.
    """
    full_X = np.concatenate([X_train, X_test], axis=0)
    N_train = len(X_train)
    N_test = len(X_test)
    
    train_seqs = [X_train[i - lookback + 1 : i + 1] for i in range(lookback - 1, N_train)]
    train_targets = y_train[lookback - 1 : N_train]
    test_seqs = [full_X[i - lookback + 1 : i + 1] for i in range(N_train, N_train + N_test)]
    
    return np.array(train_seqs, dtype=np.float32), np.array(train_targets, dtype=np.float32), np.array(test_seqs, dtype=np.float32)


# ==============================================================================
# Model Evaluation Functions
# ==============================================================================

def eval_historical_mean(data, horizon, params=None):
    """Historical Mean Campbell-Thompson benchmark."""
    y_pred = data['y_bench']
    return y_pred


def eval_ridge(data, horizon, params=None):
    alpha = params.get('alpha', 100.0) if params else 100.0
    model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_elastic_net(data, horizon, params=None):
    alpha = params.get('alpha', 0.01) if params else 0.01
    l1_ratio = params.get('l1_ratio', 0.5) if params else 0.5
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=config.RANDOM_SEED, max_iter=2000)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_svr(data, horizon, params=None):
    C = params.get('C', 1.0) if params else 1.0
    epsilon = params.get('epsilon', 0.01) if params else 0.01
    gamma = params.get('gamma', 'scale') if params else 'scale'
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_random_forest(data, horizon, params=None):
    rf_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': config.RANDOM_SEED,
        'n_jobs': -1
    }
    if params:
        rf_params.update(params)
    model = RandomForestRegressor(**rf_params)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_xgboost(data, horizon, params=None):
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': config.RANDOM_SEED,
        'n_jobs': -1
    }
    if params:
        xgb_params.update(params)
    model = XGBRegressor(**xgb_params)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_lightgbm(data, horizon, params=None):
    lgb_params = {
        'n_estimators': 300,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': config.RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    if params:
        lgb_params.update(params)
    model = LGBMRegressor(**lgb_params)
    model.fit(data['X_train'], data['y_train'])
    return model.predict(data['X_test'])


def eval_logistic_regression(data, horizon, params=None):
    C = params.get('C', 1.0) if params else 1.0
    cw = params.get('class_weight', None) if params else None
    model = LogisticRegression(C=C, class_weight=cw, random_state=config.RANDOM_SEED, max_iter=1000)
    model.fit(data['X_train'], data['y_train_dir'])
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    return y_prob


def eval_arima(data, horizon, params=None):
    train_y = data['y_train']
    test_y = data['y_test']
    full_y = np.concatenate([train_y, test_y])
    
    order = (1, 0, 1)
    if params and 'order' in params:
        order = params['order']
        
    try:
        mod = ARIMA(train_y, order=order)
        res = mod.fit()
    except Exception:
        order = (1, 0, 0)
        mod = ARIMA(train_y, order=order)
        res = mod.fit()
        
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)
    if h_step == 1:
        applied = res.apply(full_y)
        n_train, n_test = len(train_y), len(test_y)
        preds = applied.predict(start=n_train, end=n_train + n_test - 1)
    else:
        # Rolling-origin direct multi-horizon forecasting (Strict zero-leakage)
        n_test = len(test_y)
        preds_static = res.forecast(steps=n_test)
        mu = float(res.params[0]) if (len(res.params) > 0 and 'const' in res.param_names) else float(np.mean(train_y))
        phi = float(res.arparams[0]) if len(getattr(res, 'arparams', [])) > 0 else 0.0
        
        preds_list = []
        for t in range(n_test):
            if t < h_step:
                preds_list.append(preds_static[t])
            else:
                y_obs = test_y[t - h_step]
                y_hat = mu + (phi ** h_step) * (y_obs - mu)
                preds_list.append(y_hat)
        preds = np.array(preds_list)
    return preds


def eval_sarima(data, horizon, params=None):
    train_y = data['y_train']
    test_y = data['y_test']
    full_y = np.concatenate([train_y, test_y])
    
    order = (1, 0, 0)
    seasonal_order = (1, 0, 0, 5)
    if params:
        order = params.get('order', order)
        seasonal_order = params.get('seasonal_order', seasonal_order)
        
    try:
        mod = ARIMA(train_y, order=order, seasonal_order=seasonal_order)
        res = mod.fit()
    except Exception:
        try:
            mod = ARIMA(train_y, order=(1, 0, 0))
            res = mod.fit()
        except Exception:
            return np.full_like(test_y, np.mean(train_y))
            
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)
    if h_step == 1:
        applied = res.apply(full_y)
        n_train, n_test = len(train_y), len(test_y)
        preds = applied.predict(start=n_train, end=n_train + n_test - 1)
    else:
        # Rolling-origin direct multi-horizon forecasting (Strict zero-leakage)
        n_test = len(test_y)
        preds_static = res.forecast(steps=n_test)
        mu = float(res.params[0]) if (len(res.params) > 0 and 'const' in res.param_names) else float(np.mean(train_y))
        phi = float(res.arparams[0]) if len(getattr(res, 'arparams', [])) > 0 else 0.0
        
        preds_list = []
        for t in range(n_test):
            if t < h_step:
                preds_list.append(preds_static[t])
            else:
                y_obs = test_y[t - h_step]
                y_hat = mu + (phi ** h_step) * (y_obs - mu)
                preds_list.append(y_hat)
        preds = np.array(preds_list)
    return preds


def eval_arimax(data, horizon, params=None):
    train_valid = data['train_valid']
    test_valid = data['test_valid']
    exog_cols = config.ARIMAX_EXOG_COLS
    
    train_y = data['y_train']
    test_y = data['y_test']
    full_y = np.concatenate([train_y, test_y])
    
    scaler = StandardScaler()
    train_exog = scaler.fit_transform(train_valid[exog_cols].values)
    test_exog = scaler.transform(test_valid[exog_cols].values)
    full_exog = np.concatenate([train_exog, test_exog], axis=0)
    
    order = (1, 0, 1)
    if params and 'order' in params:
        order = params['order']
        
    try:
        mod = ARIMA(train_y, exog=train_exog, order=order)
        res = mod.fit()
    except Exception:
        order = (1, 0, 0)
        mod = ARIMA(train_y, exog=train_exog, order=order)
        res = mod.fit()
        
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)
    if h_step == 1:
        applied = res.apply(full_y, exog=full_exog)
        n_train, n_test = len(train_y), len(test_y)
        preds = applied.predict(start=n_train, end=n_train + n_test - 1, exog=test_exog)
    else:
        # Rolling-origin direct multi-horizon forecasting with exogenous variables
        n_test = len(test_y)
        preds_static = res.forecast(steps=n_test, exog=test_exog)
        phi = float(res.arparams[0]) if len(getattr(res, 'arparams', [])) > 0 else 0.0
        
        preds_list = []
        for t in range(n_test):
            if t < h_step:
                preds_list.append(preds_static[t])
            else:
                y_obs = test_y[t - h_step]
                y_hat = preds_static[t] + (phi ** h_step) * (y_obs - preds_static[t - h_step])
                preds_list.append(y_hat)
        preds = np.array(preds_list)
    return preds


def train_and_eval_dl_model(model_cls, data, horizon, params=None, device=config.DEVICE):
    """
    Trains PyTorch deep learning architecture (RNN/LSTM/Transformer) on train set
    with internal train-val early stopping, and predicts on test set.
    """
    set_all_seeds(config.RANDOM_SEED)
    p = {
        'lookback': 20,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 25,
        'patience': 5
    }
    if params:
        p.update(params)
        
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    
    # Internal chronological split for early stopping
    n_train = len(X_train)
    split_idx = int(n_train * 0.85)
    h_step = config.HORIZON_TO_STEPS.get(horizon, 1)
    purge_gap = max(0, h_step - 1)
    end_tr_idx = max(0, split_idx - purge_gap)
    
    X_tr_in = X_train[:end_tr_idx]
    y_tr_in = y_train[:end_tr_idx]
    X_val_in = X_train[split_idx:]
    y_val_in = y_train[split_idx:]
    
    lookback = p['lookback']
    tr_seq, tr_t, val_in_seq = create_sequences(X_tr_in, y_tr_in, X_val_in, lookback=lookback)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_t)), batch_size=p['batch_size'], shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_in_seq), torch.from_numpy(y_val_in.astype(np.float32))), batch_size=p['batch_size'], shuffle=False)
    
    if model_cls == TimeSeriesTransformer:
        model = TimeSeriesTransformer(
            input_size=X_train.shape[1],
            lookback=lookback,
            d_model=p.get('d_model', 64),
            nhead=p.get('nhead', 4),
            num_layers=p.get('num_layers', 2),
            dim_feedforward=p.get('dim_feedforward', 128),
            dropout=p.get('dropout', 0.1)
        ).to(device)
    else:
        model = model_cls(
            input_size=X_train.shape[1],
            hidden_size=p.get('hidden_size', 64),
            num_layers=p.get('num_layers', 1),
            dropout=p.get('dropout', 0.1)
        ).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p.get('learning_rate', 0.001))
    
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    for epoch in range(p['epochs']):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vloss += criterion(model(bx), by).item() * len(by)
        vloss /= len(y_val_in)
        
        if vloss < best_loss:
            best_loss = vloss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= p['patience']:
            break
            
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    # Sequence generation for full test set
    _, _, test_seq = create_sequences(X_train, y_train, X_test, lookback=lookback)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_seq)), batch_size=p['batch_size'], shuffle=False)
    
    model.eval()
    preds = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            preds.extend(model(bx).cpu().numpy().tolist())
            
    return np.array(preds)


MODEL_EVAL_DISPATCH = {
    'historical_mean': eval_historical_mean,
    'ridge': eval_ridge,
    'elastic_net': eval_elastic_net,
    'svr': eval_svr,
    'random_forest': eval_random_forest,
    'xgboost': eval_xgboost,
    'lightgbm': eval_lightgbm,
    'logistic_regression': eval_logistic_regression,
    'arima': eval_arima,
    'sarima': eval_sarima,
    'arimax': eval_arimax,
    'rnn': lambda data, h, p: train_and_eval_dl_model(VanillaRNN, data, h, p),
    'lstm': lambda data, h, p: train_and_eval_dl_model(LSTMModel, data, h, p),
    'transformer': lambda data, h, p: train_and_eval_dl_model(TimeSeriesTransformer, data, h, p),
}


def run_test_evaluation(models=None, horizons=None, logger=None, device=config.DEVICE):
    """
    Master test execution engine:
    1. Loads dataset, splits into Train (Split 1+2) and Test (Split 3).
    2. Runs selected models across selected horizons.
    3. Saves 3.0.3_test_predictions.csv.
    """
    set_all_seeds(config.RANDOM_SEED)
    df = load_dataset()
    train_df = df[df['Split'].isin([1, 2])].copy()
    test_df = df[df['Split'] == 3].copy()
    
    if models is None:
        models = config.MODELS
    if horizons is None:
        horizons = config.HORIZONS
        
    best_params_map = config.load_best_hyperparameters()
    
    all_preds = []
    
    if logger:
        logger.info(f"Loaded Train Set (Splits 1+2): {len(train_df)} rows ({train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()})")
        logger.info(f"Loaded Test Set (Split 3): {len(test_df)} rows ({test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()})")
        logger.info(f"Target Models: {models}")
        logger.info(f"Target Horizons: {horizons}")
        
    for horizon in horizons:
        if logger:
            logger.info(f"\n================ Processing Horizon {horizon} ================")
        data = prepare_test_data(train_df, test_df, horizon, scale=True, purge=True)
        test_dates = data['test_dates']
        test_regimes = data['test_regimes']
        y_test = data['y_test']
        
        for model_key in models:
            m_display = config.MODEL_NAME_MAP.get(model_key, model_key)
            params = best_params_map.get((m_display, horizon), {})
            
            if logger:
                logger.info(f"Evaluating {m_display} [{horizon}] with params: {params}")
                
            fn = MODEL_EVAL_DISPATCH.get(model_key)
            if not fn:
                if logger:
                    logger.warning(f"No dispatch function for model: {model_key}")
                continue
                
            try:
                preds = fn(data, horizon, params)
                for dt, act, pr, reg in zip(test_dates, y_test, preds, test_regimes):
                    all_preds.append({
                        'Date': dt,
                        'Model': m_display,
                        'Horizon': horizon,
                        'Actual': float(act),
                        'Prediction': float(pr),
                        'Regime': reg
                    })
            except Exception as e:
                if logger:
                    logger.error(f"Error evaluating {m_display} [{horizon}]: {e}", exc_info=True)
                    
    new_preds_df = pd.DataFrame(all_preds)
    
    # In-place row preservation: If predictions file already exists, preserve non-targeted models/horizons
    if os.path.exists(config.TEST_PREDICTIONS_PATH):
        try:
            existing_df = pd.read_csv(config.TEST_PREDICTIONS_PATH)
            target_display_models = [config.MODEL_NAME_MAP.get(m, m) for m in models]
            mask_overwrite = existing_df['Model'].isin(target_display_models) & existing_df['Horizon'].isin(horizons)
            retained_df = existing_df[~mask_overwrite]
            preds_df = pd.concat([retained_df, new_preds_df], ignore_index=True)
            preds_df = preds_df.sort_values(['Horizon', 'Model', 'Date']).reset_index(drop=True)
        except Exception as e:
            if logger:
                logger.warning(f"Could not merge with existing predictions: {e}. Overwriting with current run.")
            preds_df = new_preds_df
    else:
        preds_df = new_preds_df
        
    os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)
    preds_df.to_csv(config.TEST_PREDICTIONS_PATH, index=False)
    if logger:
        logger.info(f"Saved {len(preds_df)} total test predictions ({len(new_preds_df)} newly evaluated) to {config.TEST_PREDICTIONS_PATH}")
    return preds_df
