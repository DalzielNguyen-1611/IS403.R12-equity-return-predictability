import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

MODEL_NAME = "Transformer"

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

def train_fold(X_train, y_train, X_val, params=None, tune=False, n_trials=30, meta=None, logger=None):
    utils.set_all_seeds(config.RANDOM_SEED)
    p = config.TRANSFORMER_PARAMS.copy()
    if params:
        p.update(params)
        
    # Internal chronological split of training set (15% internal validation)
    X_tr_in, y_tr_in, X_val_in, y_val_in = utils.get_internal_train_val_split(X_train, y_train, val_ratio=0.15)
    
    if tune and not params:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            t_lookback = trial.suggest_categorical('lookback', config.TRANSFORMER_OPTUNA_SPACE['lookback'])
            t_d_model = trial.suggest_categorical('d_model', config.TRANSFORMER_OPTUNA_SPACE['d_model'])
            t_nhead = trial.suggest_categorical('nhead', config.TRANSFORMER_OPTUNA_SPACE['nhead'])
            t_layers = trial.suggest_categorical('num_encoder_layers', config.TRANSFORMER_OPTUNA_SPACE['num_encoder_layers'])
            t_dim_ff = trial.suggest_categorical('dim_feedforward', config.TRANSFORMER_OPTUNA_SPACE['dim_feedforward'])
            t_drop = trial.suggest_float('dropout', config.TRANSFORMER_OPTUNA_SPACE['dropout'][0], config.TRANSFORMER_OPTUNA_SPACE['dropout'][1])
            t_lr = trial.suggest_float('learning_rate', config.TRANSFORMER_OPTUNA_SPACE['learning_rate'][0], config.TRANSFORMER_OPTUNA_SPACE['learning_rate'][1], log=True)
            t_batch = trial.suggest_categorical('batch_size', config.TRANSFORMER_OPTUNA_SPACE['batch_size'])
            
            tr_seq, tr_t, val_seq = utils.create_sequences(X_tr_in, y_tr_in, X_val_in, lookback=t_lookback)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_t)), batch_size=t_batch, shuffle=False)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(y_val_in.astype(np.float32))), batch_size=t_batch, shuffle=False)
            
            m = TimeSeriesTransformer(
                input_size=X_train.shape[1],
                lookback=t_lookback,
                d_model=t_d_model,
                nhead=t_nhead,
                num_layers=t_layers,
                dim_feedforward=t_dim_ff,
                dropout=t_drop
            ).to(config.DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=t_lr)
            crit = nn.MSELoss()
            
            best_inner_loss = float('inf')
            for ep in range(15):
                m.train()
                for bx, by in tr_loader:
                    bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
                    opt.zero_grad()
                    crit(m(bx), by).backward()
                    opt.step()
                m.eval()
                vloss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
                        vloss += crit(m(bx), by).item() * len(by)
                vloss /= len(y_val_in)
                if vloss < best_inner_loss:
                    best_inner_loss = vloss
            return float(np.sqrt(best_inner_loss))
            
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials)
        p.update(study.best_params)
        
    lookback = p['lookback']
    tr_seq, tr_t, val_in_seq = utils.create_sequences(X_tr_in, y_tr_in, X_val_in, lookback=lookback)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_t)), batch_size=p['batch_size'], shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_in_seq), torch.from_numpy(y_val_in.astype(np.float32))), batch_size=p['batch_size'], shuffle=False)
    
    model = TimeSeriesTransformer(
        input_size=X_train.shape[1],
        lookback=lookback,
        d_model=p['d_model'],
        nhead=p['nhead'],
        num_layers=p['num_encoder_layers'] if 'num_encoder_layers' in p else p['num_layers'],
        dim_feedforward=p['dim_feedforward'],
        dropout=p['dropout']
    ).to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'])
    
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    for epoch in range(p['epochs']):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
                vloss += criterion(model(bx), by).item() * len(by)
        vloss /= len(y_val_in)
        
        if vloss < best_loss:
            best_loss = vloss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if (epoch + 1) % 5 == 0 or (epoch + 1) == p['epochs'] or no_improve >= p['patience']:
            end_yr = meta['train_end'][:4] if meta else "Train"
            if logger:
                logger.info(f"   ↳ [Epoch {epoch+1:02d}/{p['epochs']}] Dữ liệu học đến năm {end_yr} | Inner Val Loss: {vloss:.5f} (Best: {best_loss:.5f})")
                
        if no_improve >= p['patience']:
            break
                
    if best_weights:
        model.load_state_dict(best_weights)
        
    # Predict strictly on test fold validation set
    _, _, final_val_seq = utils.create_sequences(X_train, y_train, X_val, lookback=lookback)
    final_val_loader = DataLoader(TensorDataset(torch.from_numpy(final_val_seq)), batch_size=p['batch_size'], shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for (bx,) in final_val_loader:
            bx = bx.to(config.DEVICE)
            preds.extend(model(bx).cpu().numpy().tolist())
    return np.array(preds), p, float(np.sqrt(best_loss))

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    if df is None:
        df = utils.load_dataset()
    
    # 1. Load checkpoint if interrupted previously
    completed_keys, fold_metrics, oos_preds = utils.load_checkpoint(CURRENT_DIR)
    cached_tuning = utils.load_saved_tuning_params(MODEL_NAME)
    
    if logger:
        logger.info(f"Running {MODEL_NAME} on device '{config.DEVICE}' across horizons: {horizons}")
        if completed_keys:
            logger.info(f"🔄 Found checkpoint with {len(completed_keys)} completed folds. Resuming...")
        if cached_tuning:
            logger.info(f"⚡ Found {len(cached_tuning)} cached tuning records in results/tuning/best_hyperparameters.csv.")
        
    for horizon in horizons:
        for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
            if (str(horizon), int(fold)) in completed_keys:
                if logger:
                    logger.info(f"⏩ [CHECKPOINT] Skipping already completed {MODEL_NAME} | Horizon: {horizon} | Fold: {fold}")
                continue
            
            tr_span = f"{meta['train_start'][:4]}–{meta['train_end'][:4]}"
            val_span = f"{meta['val_start'][:4]}–{meta['val_end'][:4]}"
            if logger:
                logger.info(f"\n▶ [{MODEL_NAME}] Horizon {horizon} | Fold {fold}/5: Đang xử lý Train: {meta['train_start']} → {meta['train_end']} ({tr_span}, {meta['train_rows']:,} ngày) | Test OOS: {meta['val_start']} → {meta['val_end']} ({val_span}, {meta['val_rows']} ngày)...")
            
            data = utils.prepare_fold_data(train_df, val_df, horizon, scale=True)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            
            cached_key = (str(horizon), int(fold))
            cached_params = None
            if cached_key in cached_tuning:
                cached_params, _ = cached_tuning[cached_key]
                if logger:
                    logger.info(f"⚡ [CACHED] Đã nạp tham số tối ưu Horizon {horizon} Fold {fold}. Bỏ qua Optuna!")
            elif tune and logger:
                logger.info(f"🔍 [OPTUNA] Bắt đầu tối ưu 30 trials cho giai đoạn {tr_span}...")
            
            n_trials = config.OPTUNA_TRIALS.get('transformer', 30)
            val_preds, best_p, best_score = train_fold(
                X_train, y_train, X_val, params=cached_params, tune=tune, n_trials=n_trials,
                meta=meta, logger=logger
            )
            utils.save_tuning_record(MODEL_NAME, horizon, fold, best_p, best_score, score_metric='Internal_Val_RMSE')
            
            m = utils.calc_regression_metrics(y_val, val_preds)
            if logger:
                logger.info(f"✓ [HOÀN TẤT FOLD {fold} ({val_span})] Val OOS R²: {m['R2']:.4f} | DA: {m['Directional_Accuracy']*100:.2f}% | RMSE: {m['RMSE']:.4f}")
            
            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_params': str(best_p),
                'best_val_rmse': round(best_score, 6),
                'lookback': best_p['lookback'], 'device': config.DEVICE,
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
                'Directional_Accuracy': m['Directional_Accuracy']
            })
            
            val_dates = data['val_dates']
            val_regimes = data['val_regimes']
            for dt, act, pr, reg in zip(val_dates, y_val, val_preds, val_regimes):
                oos_preds.append({
                    'Date': dt, 'Model': MODEL_NAME, 'Horizon': horizon,
                    'Fold': fold, 'Actual': act, 'Prediction': pr, 'Regime': reg
                })
                
            # Save checkpoint after each fold
            utils.save_checkpoint_step(CURRENT_DIR, fold_metrics, oos_preds)
            completed_keys.add((str(horizon), int(fold)))
            if logger:
                logger.info(f"✓ Completed & check-pointed {MODEL_NAME} | Horizon: {horizon} | Fold: {fold}")
                
    m_df = pd.DataFrame(fold_metrics)
    p_df = pd.DataFrame(oos_preds)
    utils.save_model_latest_results(CURRENT_DIR, m_df, p_df)
    utils.finalize_checkpoint(CURRENT_DIR)
    if logger:
        logger.info(f"Completed {MODEL_NAME}.")
    return m_df, p_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help="Run Optuna hyperparameter optimization")
    args = parser.parse_args()
    logger = utils.setup_logger(MODEL_NAME)
    run_model(tune=args.tune, logger=logger)
