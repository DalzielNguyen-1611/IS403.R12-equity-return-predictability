import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
import utils

MODEL_NAME = "N-HiTS"

class NHiTSBlock(nn.Module):
    def __init__(self, lookback, num_features, pool_kernel=1, theta_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.pool_kernel = pool_kernel
        
        # Calculate pooled length
        if pool_kernel > 1:
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
            self.pooled_len = int(np.ceil(lookback / pool_kernel))
        else:
            self.pool = nn.Identity()
            self.pooled_len = lookback
            
        self.input_dim = self.num_features * self.pooled_len
        self.fc_stack = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.theta_b = nn.Linear(hidden_dim, theta_dim)
        self.theta_f = nn.Linear(hidden_dim, theta_dim)
        self.backcast_basis = nn.Linear(theta_dim, self.input_dim)
        self.forecast_basis = nn.Linear(theta_dim, 1)

    def forward(self, x):
        # x: (batch_size, lookback, num_features)
        batch_size = x.size(0)
        # Transpose for 1D pooling along time dimension: (batch_size, num_features, lookback)
        x_trans = x.transpose(1, 2)
        x_pooled = self.pool(x_trans)
        # Flatten: (batch_size, num_features * pooled_len)
        x_flat = x_pooled.reshape(batch_size, -1)

        h = self.fc_stack(x_flat)
        b_coef = self.theta_b(h)
        f_coef = self.theta_f(h)

        b_proj = self.backcast_basis(b_coef).reshape(batch_size, self.num_features, self.pooled_len)
        if self.pooled_len != self.lookback:
            # Hierarchical linear interpolation back to original lookback dimension
            b_interp = F.interpolate(b_proj, size=self.lookback, mode='linear', align_corners=False)
        else:
            b_interp = b_proj
        backcast = b_interp.transpose(1, 2) # (batch_size, lookback, num_features)

        forecast = self.forecast_basis(f_coef).squeeze(-1)
        return backcast, forecast

class NHiTSNet(nn.Module):
    def __init__(self, lookback=20, num_features=16, pooling_kernels=(4, 2, 1), num_blocks_per_stack=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for k in pooling_kernels:
            for _ in range(num_blocks_per_stack):
                self.blocks.append(
                    NHiTSBlock(
                        lookback=lookback,
                        num_features=num_features,
                        pool_kernel=k,
                        theta_dim=hidden_dim // 2,
                        hidden_dim=hidden_dim,
                        dropout=dropout
                    )
                )

    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        forecast = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            b, f = block(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast

def train_fold(X_train, y_train, X_val, params=None, tune=False, n_trials=30, meta=None, logger=None, horizon=None, fold=None):
    utils.set_all_seeds(config.RANDOM_SEED)
    p = config.NHITS_PARAMS.copy()
    if params:
        p.update(params)

    X_tr_in, y_tr_in, X_val_in, y_val_in = utils.get_internal_train_val_split(X_train, y_train, val_ratio=0.15, horizon=horizon)

    if tune and not params:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            t_lookback = trial.suggest_categorical('lookback', config.NHITS_OPTUNA_SPACE['lookback'])
            t_hidden = trial.suggest_categorical('hidden_dim', config.NHITS_OPTUNA_SPACE['hidden_dim'])
            t_pool_mode = trial.suggest_categorical('pooling_mode', [0, 1])
            pools = (8, 4, 1) if t_pool_mode == 0 else (4, 2, 1)
            t_blocks = trial.suggest_categorical('num_blocks_per_stack', config.NHITS_OPTUNA_SPACE['num_blocks_per_stack'])
            t_drop = trial.suggest_float('dropout', config.NHITS_OPTUNA_SPACE['dropout'][0], config.NHITS_OPTUNA_SPACE['dropout'][1])
            t_lr = trial.suggest_float('learning_rate', config.NHITS_OPTUNA_SPACE['learning_rate'][0], config.NHITS_OPTUNA_SPACE['learning_rate'][1], log=True)
            t_batch = trial.suggest_categorical('batch_size', config.NHITS_OPTUNA_SPACE['batch_size'])

            tr_seq, tr_t, val_seq = utils.create_sequences(X_tr_in, y_tr_in, X_val_in, lookback=t_lookback)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_t)), batch_size=t_batch, shuffle=False)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(y_val_in.astype(np.float32))), batch_size=t_batch, shuffle=False)

            m = NHiTSNet(
                lookback=t_lookback,
                num_features=X_train.shape[1],
                pooling_kernels=pools,
                num_blocks_per_stack=t_blocks,
                hidden_dim=t_hidden,
                dropout=t_drop
            ).to(config.DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=t_lr, weight_decay=1e-4)
            crit = nn.MSELoss()

            best_inner_loss = float('inf')
            for ep in range(15):
                m.train()
                for bx, by in tr_loader:
                    bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
                    opt.zero_grad()
                    crit(m(bx), by).backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
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
        p['pooling_kernels'] = (8, 4, 1) if p.get('pooling_mode', 0) == 0 else (4, 2, 1)

    if 'pooling_kernels' not in p:
        p['pooling_kernels'] = (4, 2, 1)

    lookback = p['lookback']
    tr_seq, tr_t, val_in_seq = utils.create_sequences(X_tr_in, y_tr_in, X_val_in, lookback=lookback)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_t)), batch_size=p['batch_size'], shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_in_seq), torch.from_numpy(y_val_in.astype(np.float32))), batch_size=p['batch_size'], shuffle=False)

    model = NHiTSNet(
        lookback=lookback,
        num_features=X_train.shape[1],
        pooling_kernels=p['pooling_kernels'],
        num_blocks_per_stack=p['num_blocks_per_stack'],
        hidden_dim=p['hidden_dim'],
        dropout=p['dropout']
    ).to(config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p.get('weight_decay', 1e-4))

    best_loss = float('inf')
    best_weights = None
    no_improve = 0

    for epoch in range(p['epochs']):
        model.train()
        ep_t0 = time.time()
        for bx, by in train_loader:
            bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        ep_duration = time.time() - ep_t0
        end_yr = meta['train_end'][:4] if meta else "Train"
        if logger:
            logger.info(f"   ↳ [Epoch {epoch+1:02d}/{p['epochs']}] Dữ liệu học đến năm {end_yr} | Inner Val Loss: {vloss:.5f} (Best: {best_loss:.5f}) | {ep_duration:.2f}s/ep")
        if horizon and fold:
            utils.save_epoch_progress(MODEL_NAME, horizon, fold, epoch + 1, p['epochs'], vloss, best_loss, epoch_duration=ep_duration)

        if no_improve >= p['patience']:
            break

    if best_weights:
        model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_weights.items()})

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

            fold_t0 = time.time()
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

            n_trials = config.OPTUNA_TRIALS.get('nhits', 30)
            val_preds, best_p, best_score = train_fold(
                X_train, y_train, X_val, params=cached_params, tune=tune, n_trials=n_trials,
                meta=meta, logger=logger, horizon=horizon, fold=fold
            )
            utils.save_tuning_record(MODEL_NAME, horizon, fold, best_p, best_score, score_metric='Internal_Val_RMSE')

            m = utils.calc_regression_metrics(y_val, val_preds, y_bench=data['y_bench'], h=config.HORIZON_TO_STEPS.get(horizon, 1))
            fold_duration = round(time.time() - fold_t0, 2)
            if logger:
                logger.info(f"✓ [HOÀN TẤT FOLD {fold} ({val_span})] Val OOS R²: {m['R2']:.4f} | DA: {m['Directional_Accuracy']*100:.2f}% | RMSE: {m['RMSE']:.4f} | Thời gian: {fold_duration:.1f}s")

            fold_metrics.append({
                'model': MODEL_NAME, 'horizon': horizon, 'fold': fold,
                'train_start': meta['train_start'], 'train_end': meta['train_end'],
                'val_start': meta['val_start'], 'val_end': meta['val_end'],
                'best_params': str(best_p),
                'best_val_rmse': round(best_score, 6),
                'lookback': best_p['lookback'], 'device': config.DEVICE,
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
                'Directional_Accuracy': m['Directional_Accuracy'],
                'duration_seconds': fold_duration,
                'Duration_Seconds': fold_duration
            })

            val_dates = data['val_dates']
            val_regimes = data['val_regimes']
            for dt, act, pr, reg in zip(val_dates, y_val, val_preds, val_regimes):
                oos_preds.append({
                    'Date': dt, 'Model': MODEL_NAME, 'Horizon': horizon,
                    'Fold': fold, 'Actual': act, 'Prediction': pr, 'Regime': reg
                })

            utils.save_checkpoint_step(CURRENT_DIR, fold_metrics, oos_preds, model_name=MODEL_NAME, fold_duration=fold_duration)
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
