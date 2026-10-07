import os
import sys
import subprocess
import time
import pandas as pd

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.join(WORKSPACE_DIR, "src", "training", "baselines")
FOLD_METRICS_PATH = os.path.join(WORKSPACE_DIR, "results", "fold_metrics", "all_fold_metrics.csv")

ALL_MODELS = [
    {
        "name": "Historical_Mean",
        "script": os.path.join(BASELINES_DIR, "historical_mean", "historical_mean.py"),
        "args": []
    },
    {
        "name": "Ridge",
        "script": os.path.join(BASELINES_DIR, "ridge", "ridge.py"),
        "args": ["--tune"]
    },
    {
        "name": "Logistic_Regression",
        "script": os.path.join(BASELINES_DIR, "logistic_regression", "logistic_regression.py"),
        "args": ["--tune"]
    },
    {
        "name": "Elastic_Net",
        "script": os.path.join(BASELINES_DIR, "elastic_net", "elastic_net.py"),
        "args": ["--tune"]
    },
    {
        "name": "ARIMA",
        "script": os.path.join(BASELINES_DIR, "arima", "arima.py"),
        "args": ["--tune"]
    },
    {
        "name": "ARIMAX",
        "script": os.path.join(BASELINES_DIR, "arimax", "arimax.py"),
        "args": ["--tune"]
    },
    {
        "name": "SARIMA",
        "script": os.path.join(BASELINES_DIR, "sarima", "sarima.py"),
        "args": ["--tune"]
    },
    {
        "name": "LightGBM",
        "script": os.path.join(BASELINES_DIR, "lightgbm", "lightgbm_model.py"),
        "args": ["--tune"]
    },
    {
        "name": "SVR",
        "script": os.path.join(BASELINES_DIR, "svr", "svr.py"),
        "args": []
    },
    {
        "name": "Random_Forest",
        "script": os.path.join(BASELINES_DIR, "random_forest", "random_forest.py"),
        "args": ["--tune"]
    },
    {
        "name": "XGBoost",
        "script": os.path.join(BASELINES_DIR, "xgboost", "xgboost_model.py"),
        "args": ["--tune"]
    },
    {
        "name": "RNN",
        "script": os.path.join(BASELINES_DIR, "rnn", "rnn_model.py"),
        "args": ["--tune"]
    },
    {
        "name": "LSTM",
        "script": os.path.join(BASELINES_DIR, "lstm", "lstm_model.py"),
        "args": ["--tune"]
    },
    {
        "name": "Transformer",
        "script": os.path.join(BASELINES_DIR, "transformer", "transformer_model.py"),
        "args": ["--tune"]
    }
]

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU KIỂM TRA VÀ CHẠY TOÀN BỘ 14 MÔ HÌNH BASELINE TRONG HỆ THỐNG")
    print("=" * 80)
    
    start_total = time.time()
    
    for idx, item in enumerate(ALL_MODELS, 1):
        m_name = item["name"]
        script_path = item["script"]
        extra_args = item["args"]
        
        # Check current progress from all_fold_metrics.csv
        cnt = 0
        if os.path.exists(FOLD_METRICS_PATH):
            try:
                fm_df = pd.read_csv(FOLD_METRICS_PATH)
                if 'Model' in fm_df.columns:
                    cnt = (fm_df['Model'].astype(str).str.lower() == m_name.lower()).sum()
            except Exception:
                cnt = 0
                
        if cnt >= 25:
            print(f"[{idx:02d}/14] ⏩ {m_name.upper():<20} ĐÃ HOÀN TẤT ({cnt}/25 Folds) -> Bỏ qua!")
            continue
        elif cnt > 0:
            print(f"\n[{idx:02d}/14] 🔄 {m_name.upper():<20} Đang dở dang ({cnt}/25 Folds) -> Tiếp tục huấn luyện...")
        else:
            print(f"\n[{idx:02d}/14] ▶ {m_name.upper():<20} Chưa chạy (0/25 Folds) -> Bắt đầu huấn luyện...")
            
        print(f"       Script: {script_path}")
        
        cmd = [sys.executable, script_path] + extra_args
        t0 = time.time()
        try:
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"✓ {m_name.upper()} HOÀN TẤT THÀNH CÔNG trong {mins:02d}m {secs:02d}s!")
        except subprocess.CalledProcessError as e:
            print(f"✗ LỖI khi chạy {m_name}: Mã thoát {e.returncode}")
            print("Dừng chuỗi tự động để bạn kiểm tra.")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print(f"\n⚠️ Người dùng đã tạm dừng ({m_name}). Tiến trình fold đã được checkpoint an toàn.")
            sys.exit(1)
            
    # Auto-generate RQ Analysis and statistical tests once all models complete
    print("\n" + "=" * 80)
    print("🎉 TOÀN BỘ 14 MÔ HÌNH ĐÃ HOÀN TẤT 100%! ĐANG TẠO BÁO CÁO NGHIÊN CỨU...")
    print("=" * 80)
    try:
        rq_script = os.path.join(BASELINES_DIR, "generate_rq_analysis.py")
        subprocess.run([sys.executable, rq_script], check=True)
        print("✓ Báo cáo RQ1, RQ2, RQ3 và Statistical Tests đã tạo thành công trong results/analysis/!")
    except Exception as e:
        print(f"Cảnh báo khi tạo báo cáo RQ: {e}")

    # Final Dashboard Refresh
    try:
        import update_dashboard
        update_dashboard.generate_dashboard()
        print("✓ Dashboard DASHBOARD.md đã làm mới!")
    except Exception:
        pass

    total_time = time.time() - start_total
    print(f"\n✨ TỔNG THỜI GIAN THỰC THI: {int(total_time // 60)}m {int(total_time % 60)}s.")

if __name__ == "__main__":
    main()
