import os
import sys
import datetime
import time
import pandas as pd
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
BASELINES_DIR = os.path.join(WORKSPACE_DIR, "src", "training", "baselines")
FOLD_METRICS_PATH = os.path.join(RESULTS_DIR, "fold_metrics", "all_fold_metrics.csv")
TIMING_PATH = os.path.join(RESULTS_DIR, "benchmark_timing.csv")
DASHBOARD_PATH = os.path.join(WORKSPACE_DIR, "DASHBOARD.md")

ALL_MODELS = [
    {"name": "Historical_Mean", "category": "Benchmark", "total_folds": 25, "script": "src/training/baselines/historical_mean/historical_mean.py"},
    {"name": "Ridge", "category": "Linear Regression", "total_folds": 25, "script": "src/training/baselines/ridge/ridge.py"},
    {"name": "Logistic_Regression", "category": "Classification", "total_folds": 25, "script": "src/training/baselines/logistic_regression/logistic_regression.py"},
    {"name": "Elastic_Net", "category": "Linear Regression", "total_folds": 25, "script": "src/training/baselines/elastic_net/elastic_net.py"},
    {"name": "ARIMA", "category": "Statistical TS", "total_folds": 25, "script": "src/training/baselines/arima/arima.py"},
    {"name": "ARIMAX", "category": "Statistical TS", "total_folds": 25, "script": "src/training/baselines/arimax/arimax.py"},
    {"name": "SARIMA", "category": "Statistical TS", "total_folds": 25, "script": "src/training/baselines/sarima/sarima.py"},
    {"name": "LightGBM", "category": "Tree Ensemble", "total_folds": 25, "script": "src/training/baselines/lightgbm/lightgbm_model.py"},
    {"name": "SVR", "category": "Kernel Method", "total_folds": 25, "script": "src/training/baselines/svr/svr.py"},
    {"name": "Random_Forest", "category": "Tree Ensemble", "total_folds": 25, "script": "src/training/baselines/random_forest/random_forest.py"},
    {"name": "XGBoost", "category": "Tree Ensemble", "total_folds": 25, "script": "src/training/baselines/xgboost/xgboost_model.py"},
    {"name": "RNN", "category": "Deep Learning", "total_folds": 25, "script": "src/training/baselines/rnn/rnn_model.py"},
    {"name": "LSTM", "category": "Deep Learning", "total_folds": 25, "script": "src/training/baselines/lstm/lstm_model.py"},
    {"name": "Transformer", "category": "Deep Learning", "total_folds": 25, "script": "src/training/baselines/transformer/transformer_model.py"},
]

def format_duration(seconds):
    if seconds is None or np.isnan(seconds):
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes:02d}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}h {mins:02d}m"

def render_progress_bar(percentage, length=20):
    filled = int(round(length * percentage / 100.0))
    bar = "█" * filled + "░" * (length - filled)
    return f"`[{bar}] {percentage:.1f}%`"

def generate_dashboard():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Load completed fold metrics from all_fold_metrics.csv
    completed_counts = {}
    fm_df = None
    if os.path.exists(FOLD_METRICS_PATH):
        try:
            fm_df = pd.read_csv(FOLD_METRICS_PATH)
            if 'Model' in fm_df.columns:
                completed_counts = fm_df['Model'].value_counts().to_dict()
        except Exception as e:
            print(f"Warning reading fold metrics: {e}")
            
    # 2. Check for ACTIVE IN-PROGRESS model checkpoint
    active_model_name = None
    active_ckpt_df = None
    active_elapsed = 0.0
    active_avg_speed = 0.0
    
    for m_info in ALL_MODELS:
        m_dir = os.path.join(WORKSPACE_DIR, os.path.dirname(m_info["script"]))
        ckpt_file = os.path.join(m_dir, "checkpoint_metrics.csv")
        if os.path.exists(ckpt_file):
            try:
                c_df = pd.read_csv(ckpt_file)
                if not c_df.empty:
                    active_model_name = m_info["name"]
                    active_ckpt_df = c_df
                    # Compute duration of active model
                    c_ctime = os.path.getctime(ckpt_file)
                    c_mtime = os.path.getmtime(ckpt_file)
                    active_elapsed = max(c_mtime - c_ctime, 1.0)
                    active_avg_speed = active_elapsed / len(c_df)
                    break
            except Exception:
                pass
                
    # 3. Load historical timing database
    timing_data = {}
    if os.path.exists(TIMING_PATH):
        try:
            t_df = pd.read_csv(TIMING_PATH)
            for _, row in t_df.iterrows():
                timing_data[str(row['Model'])] = {
                    'historical_total': float(row['Historical_Total_Seconds']) if pd.notnull(row['Historical_Total_Seconds']) else None,
                    'avg_per_fold': float(row['Avg_Seconds_Per_Fold']) if pd.notnull(row['Avg_Seconds_Per_Fold']) else None,
                    'last_run': str(row['Last_Run_Timestamp']) if pd.notnull(row['Last_Run_Timestamp']) else None,
                }
        except Exception as e:
            print(f"Warning reading timing data: {e}")

    # Determine which model to highlight in Header and Section 1
    is_actively_training = (active_model_name is not None and active_ckpt_df is not None)
    
    if is_actively_training:
        focus_model_name = active_model_name
        focus_df = active_ckpt_df
        last_row = active_ckpt_df.iloc[-1]
        active_folds_done = len(active_ckpt_df)
        completed_counts[active_model_name] = active_folds_done
        
        status_banner = f"🟡 Đang huấn luyện: Fold {last_row.get('fold', active_folds_done)}/5 (Horizon {last_row.get('horizon', '-')}) — Cập nhật Real-Time"
        model_eta_str = f"~{format_duration((25 - active_folds_done) * active_avg_speed)} (còn {25 - active_folds_done} folds)"
        model_dur_str = f"{format_duration(active_elapsed)}"
        model_speed_str = f"{active_avg_speed:.2f}s/fold"
    else:
        # Last completed model
        focus_model_name = "LightGBM"
        if fm_df is not None and not fm_df.empty:
            focus_model_name = str(fm_df.iloc[-1]['Model'])
            focus_df = fm_df[fm_df['Model'] == focus_model_name]
            last_row = focus_df.iloc[-1]
        else:
            last_row = {}
            focus_df = pd.DataFrame()
            
        status_banner = "🟢 Đã hoàn thành 25/25 Folds — Chờ lệnh chạy mô hình tiếp theo"
        t_info = timing_data.get(focus_model_name, {})
        model_dur_str = format_duration(t_info.get('historical_total'))
        model_speed_str = f"{t_info.get('avg_per_fold', 0):.2f}s/fold" if t_info.get('avg_per_fold') else "-"
        model_eta_str = "Đã xong 100% (còn 0 folds)"

    latest_fold_info = {
        'model': focus_model_name,
        'horizon': str(last_row.get('horizon', last_row.get('Horizon', '-'))),
        'fold': str(last_row.get('fold', last_row.get('Fold', '-'))),
        'oos_r2': f"{float(last_row.get('R2', last_row.get('OOS_R2', 0))):.4f}" if ('R2' in last_row or 'OOS_R2' in last_row) and pd.notnull(last_row.get('R2', last_row.get('OOS_R2'))) else "-",
        'da': f"{float(last_row.get('Directional_Accuracy', last_row.get('DA', 0))) * 100:.2f}%" if ('Directional_Accuracy' in last_row or 'DA' in last_row) and pd.notnull(last_row.get('Directional_Accuracy', last_row.get('DA'))) else "-",
        'rmse': f"{float(last_row.get('RMSE', 0)):.4f}" if 'RMSE' in last_row and pd.notnull(last_row.get('RMSE')) else "-",
        'mae': f"{float(last_row.get('MAE', 0)):.4f}" if 'MAE' in last_row and pd.notnull(last_row.get('MAE')) else "-",
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

    # Compile table of all 14 models
    total_models = len(ALL_MODELS)
    completed_models = 0
    total_folds_sum = total_models * 25
    completed_folds_sum = 0
    total_logged_seconds = 0.0
    
    table_rows = []
    
    for idx, m_info in enumerate(ALL_MODELS, 1):
        m_name = m_info["name"]
        cat = m_info["category"]
        tot_folds = m_info["total_folds"]
        done_folds = completed_counts.get(m_name, 0)
        completed_folds_sum += done_folds
        
        t_info = timing_data.get(m_name, {})
        hist_total = t_info.get('historical_total')
        avg_fold = t_info.get('avg_per_fold')
        
        if done_folds >= tot_folds:
            completed_models += 1
            status = "✅ Hoàn thành"
            if hist_total:
                past_time_str = format_duration(hist_total)
                avg_fold_str = f"{avg_fold:.2f}s/fold" if avg_fold else "-"
                eta_str = "Đã xong (100%)"
                total_logged_seconds += hist_total
            else:
                past_time_str = "Hoàn tất"
                avg_fold_str = "-"
                eta_str = "Đã xong"
        elif done_folds > 0:
            status = f"🔄 Đang chạy ({done_folds}/25)"
            rem_folds = tot_folds - done_folds
            if m_name == active_model_name and active_avg_speed > 0:
                past_time_str = format_duration(active_elapsed)
                avg_fold_str = f"{active_avg_speed:.2f}s/fold"
                eta_str = f"~{format_duration(rem_folds * active_avg_speed)} (còn {rem_folds} folds)"
            elif avg_fold and avg_fold > 0:
                est_rem_sec = rem_folds * avg_fold
                past_time_str = format_duration(done_folds * avg_fold)
                avg_fold_str = f"{avg_fold:.2f}s/fold"
                eta_str = f"~{format_duration(est_rem_sec)} (còn {rem_folds} folds)"
            else:
                past_time_str = f"{done_folds}/{tot_folds} folds"
                avg_fold_str = "Đang đo..."
                eta_str = f"Ước lượng sau khi xong fold (còn {rem_folds} folds)"
        else:
            status = "⏳ Chờ chạy"
            avg_fold_str = "-"
            if hist_total and hist_total > 0:
                past_time_str = f"Lịch sử: {format_duration(hist_total)}"
                avg_fold_str = f"{avg_fold:.2f}s/fold" if avg_fold else "-"
                eta_str = f"~{format_duration(hist_total)} (theo lịch sử chạy trước của chính nó)"
            else:
                past_time_str = "Chưa có lượt chạy"
                eta_str = "*Chờ fold 1 của chính model này (Không ước lượng từ model khác)*"
                
        table_rows.append({
            "idx": idx,
            "name": m_name,
            "cat": cat,
            "progress": f"{done_folds}/{tot_folds}",
            "status": status,
            "past_time": past_time_str,
            "avg_fold": avg_fold_str,
            "eta": eta_str
        })
        
    overall_pct = (completed_folds_sum / total_folds_sum) * 100.0
    overall_bar = render_progress_bar(overall_pct, length=24)
    
    # Extract recent folds to display in Section 2 (most recent completed steps)
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
                "horizon": hz,
                "fold": f"Fold {fd}/5",
                "val_period": val_period,
                "status": "✅ Hoàn thành",
                "r2": f"{float(r2_val):.4f}" if pd.notnull(r2_val) else "-",
                "da": f"{float(da_val)*100:.2f}%" if pd.notnull(da_val) else "-",
                "rmse": f"{float(rmse_val):.4f}" if pd.notnull(rmse_val) else "-",
                "mae": f"{float(mae_val):.4f}" if pd.notnull(mae_val) else "-"
            })

    # Render Markdown
    md = f"""# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** {focus_model_name} | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** {now_str}  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`{focus_model_name}`** | **Fold vừa hoàn tất** | Fold {latest_fold_info.get('fold', '-')}/5 (Horizon {latest_fold_info.get('horizon', '-')}) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `{latest_fold_info.get('oos_r2', '-')}` |
| **Tiến trình Mô hình** | **`{completed_models}/{total_models} Models ({overall_pct:.1f}%)`** | **Directional Accuracy (DA)** | `{latest_fold_info.get('da', '-')}` |
| **Tiến trình Folds tổng** | **`{completed_folds_sum}/{total_folds_sum} Folds ({overall_pct:.1f}%)`** | **RMSE Fold vừa xong** | `{latest_fold_info.get('rmse', '-')}` |
| **Kỷ lục OOS R² cao nhất** | `{best_r2_val}` | **Loss / MAE Fold vừa xong** | `{latest_fold_info.get('mae', '-')}` |
| **Mô hình giữ Kỷ lục** | {best_r2_model} | **Best Horizon của Model** | Horizon {latest_fold_info.get('horizon', '-')} |
| **Thời gian đã chạy (chính nó)** | **`{model_dur_str}`** | **Tốc độ chạy (chính nó)** | **`{model_speed_str}`** |
| **Tiến độ Nhóm Linear & Stats** | ✅ ĐÃ HOÀN TẤT 7/7 MODELS | **Trạng thái Nhóm 1** | Đã lưu Fold Metrics & OOS Preds |
| **Dự kiến XONG mô hình này** | {model_eta_str} | **Thời gian còn lại (chính nó)** | {model_eta_str} |
| **Trạng thái mô hình** | {status_banner} | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Zero manual action) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`{focus_model_name}`)

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
*Bảng điều khiển được cập nhật **hoàn toàn tự động** sau mỗi Fold bằng hook nội bộ.*
"""

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"Dashboard successfully generated at: {DASHBOARD_PATH}")

if __name__ == "__main__":
    generate_dashboard()
