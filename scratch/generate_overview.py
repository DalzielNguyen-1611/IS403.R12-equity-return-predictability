import os
import sys
import json
import ast
import pandas as pd
import numpy as np

# Load Lan01 tuning data
lan01_tuning_file = 'results/Lan01/tuning/3.0.3_best_hyperparameters.csv'
df_lan01 = pd.read_csv(lan01_tuning_file) if os.path.exists(lan01_tuning_file) else pd.DataFrame()

# Definition of all 40 models across 5 experimental runs
ALL_MODELS_DEF = [
    # --- Lan01 ---
    {'run': 'Lan01', 'name': 'Historical_Mean', 'family': 'Benchmark', 'strategy': 'None (Parameter-free)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'Ridge', 'family': 'Linear Model', 'strategy': 'Grid Search (alpha)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Grid)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'Logistic_Regression', 'family': 'Classification', 'strategy': 'Grid Search (C, class_weight)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Grid)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'Elastic_Net', 'family': 'Linear Model', 'strategy': 'Grid Search (alpha, l1_ratio)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Grid)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'ARIMA', 'family': 'Statistical TS', 'strategy': 'Grid Search (p, d, q AIC)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Grid)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'ARIMAX', 'family': 'Statistical TS', 'strategy': 'Fixed Spec (1,0,1 + Exog)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Fixed)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'SARIMA', 'family': 'Statistical TS', 'strategy': 'Fixed Spec (1,0,0)x(1,0,0,5)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Fixed)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'SVR', 'family': 'Kernel Method', 'strategy': 'Grid Search (C, gamma, eps)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Grid)', 'is_optuna': False},
    {'run': 'Lan01', 'name': 'Random_Forest', 'family': 'Tree Ensemble', 'strategy': 'Optuna TPE (CV_RMSE)', 'optuna_trials': '30', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},
    {'run': 'Lan01', 'name': 'XGBoost', 'family': 'Tree Ensemble', 'strategy': 'Optuna TPE (CV_RMSE)', 'optuna_trials': '50', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},
    {'run': 'Lan01', 'name': 'LightGBM', 'family': 'Tree Ensemble', 'strategy': 'Optuna TPE (CV_RMSE)', 'optuna_trials': '50', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},
    {'run': 'Lan01', 'name': 'RNN', 'family': 'Deep Learning', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},
    {'run': 'Lan01', 'name': 'LSTM', 'family': 'Deep Learning', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},
    {'run': 'Lan01', 'name': 'Transformer', 'family': 'Deep Learning', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Đã hoàn tất Optuna', 'is_optuna': True},

    # --- Lan02 ---
    {'run': 'Lan02', 'name': 'TCN', 'family': 'Advanced DL (Temporal CNN)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'N-BEATS', 'family': 'Advanced DL (Basis Expansion)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'N-HiTS', 'family': 'Advanced DL (Hierarchical MLP)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'PatchTST', 'family': 'Advanced DL (Patch Transformer)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'TSMixer', 'family': 'Advanced DL (All-MLP)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'iTransformer', 'family': 'Advanced DL (Inverted Attn)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan02', 'name': 'CNN', 'family': 'Advanced DL (1D Conv Baseline)', 'strategy': 'Default / Pre-set', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},

    # --- Lan03 ---
    {'run': 'Lan03', 'name': 'TCN', 'family': 'Advanced DL (Temporal CNN)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'N-BEATS', 'family': 'Advanced DL (Basis Expansion)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'N-HiTS', 'family': 'Advanced DL (Hierarchical MLP)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'PatchTST', 'family': 'Advanced DL (Patch Transformer)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'TSMixer', 'family': 'Advanced DL (All-MLP)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'iTransformer', 'family': 'Advanced DL (Inverted Attn)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan03', 'name': 'CNN', 'family': 'Advanced DL (1D Conv Baseline)', 'strategy': 'Optuna TPE (Val_Loss)', 'optuna_trials': '30', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},

    # --- Lan04 ---
    {'run': 'Lan04', 'name': 'Ensemble_A1', 'family': 'Ensemble (All 13 Models)', 'strategy': 'Equal-Weight (1/13)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan04', 'name': 'Ensemble_A2', 'family': 'Ensemble (Core 4 Models)', 'strategy': 'Equal-Weight (1/4)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan04', 'name': 'Ensemble_A3', 'family': 'Ensemble (Ridge, SARIMA, HM)', 'strategy': 'Equal-Weight (1/3)', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan04', 'name': 'Baseline_C1', 'family': 'Regime Strategy (Best Model)', 'strategy': 'Argmin MSE per Regime', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan04', 'name': 'Baseline_C2', 'family': 'Regime Strategy (Shrunk Inv-MSE)', 'strategy': 'Shrinkage N0=50.0 fixed', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},
    {'run': 'Lan04', 'name': 'Baseline_C3', 'family': 'Regime Strategy (Defensive Switch)', 'strategy': 'A2 / HM hard switch', 'optuna_trials': '-', 'optuna_status': 'Không dùng Optuna (Mặc định)', 'is_optuna': False},

    # --- Lan05 ---
    {'run': 'Lan05', 'name': 'Ensemble_A1', 'family': 'Ensemble (All 13 Models)', 'strategy': 'Optuna Simplex Weights', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan05', 'name': 'Ensemble_A2', 'family': 'Ensemble (Core 4 Models)', 'strategy': 'Optuna Simplex Weights', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan05', 'name': 'Ensemble_A3', 'family': 'Ensemble (Ridge, SARIMA, HM)', 'strategy': 'Optuna Simplex Weights', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan05', 'name': 'Baseline_C1', 'family': 'Regime Strategy (Best Model)', 'strategy': 'Optuna Regime Selection/Blend', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan05', 'name': 'Baseline_C2', 'family': 'Regime Strategy (Shrunk Inv-MSE)', 'strategy': 'Optuna N0 & Power gamma', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
    {'run': 'Lan05', 'name': 'Baseline_C3', 'family': 'Regime Strategy (Defensive Switch)', 'strategy': 'Optuna Alpha & Core Weights', 'optuna_trials': '50', 'optuna_status': 'Chưa Optuna', 'is_optuna': True},
]

# Build master overview table rows
rows = []
for idx, m in enumerate(ALL_MODELS_DEF, 1):
    run_name = m['run']
    mod_name = m['name']
    
    best_params_str = ""
    best_val_score_str = ""
    metric_str = ""
    
    if run_name == 'Lan01' and m['is_optuna']:
        # Extract from Lan01 tuning data
        sub = df_lan01[df_lan01['Model'] == mod_name]
        if not sub.empty:
            # Get typical Fold 5 params or average
            f5 = sub[sub['Fold'] == 5]
            if not f5.empty:
                f5_1d = f5[f5['Horizon'] == '1D']
                if not f5_1d.empty:
                    rec = f5_1d.iloc[0]
                    best_params_str = str(rec['Best_Params'])
                    best_val_score_str = f"{float(rec['Best_Val_Score']):.6f}"
                    metric_str = str(rec['Score_Metric'])
                else:
                    rec = f5.iloc[0]
                    best_params_str = str(rec['Best_Params'])
                    best_val_score_str = f"{float(rec['Best_Val_Score']):.6f}"
                    metric_str = str(rec['Score_Metric'])
    elif run_name == 'Lan01' and not m['is_optuna'] and mod_name != 'Historical_Mean':
        # Grid search models in Lan01: user requested "đã potuna, chưa potuna thì để trống"
        # So leave best_params blank as it's not optuna!
        best_params_str = ""
        best_val_score_str = ""
        metric_str = m['strategy']
    else:
        # Not optuna-tuned yet -> leave blank
        best_params_str = ""
        best_val_score_str = ""
        metric_str = ""

    rows.append({
        'STT': idx,
        'Run': run_name,
        'Model_Name': mod_name,
        'Architecture_Family': m['family'],
        'Tuning_Method': m['strategy'],
        'Optuna_Trials': m['optuna_trials'],
        'Optuna_Status': m['optuna_status'],
        'Best_Hyperparameters_Optuna': best_params_str,
        'Best_Val_Score': best_val_score_str,
        'Metric': metric_str
    })

overview_df = pd.DataFrame(rows)
os.makedirs('results', exist_ok=True)
csv_path = 'results/3.0.3_models_hyperparameters_overview.csv'
overview_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Saved CSV overview to {csv_path} ({len(overview_df)} rows)")

# Now generate comprehensive Markdown file MODELS_OVERVIEW_HYPERPARAMS.md
md_lines = [
    "# 📋 BẢNG TỔNG HỢP TOÀN BỘ MÔ HÌNH VÀ SIÊU THAM SỐ / TRỌNG SỐ TỐI ƯU HÓA",
    "",
    "> **Dự án nghiên cứu:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  ",
    "> **Mục tiêu tệp:** Danh mục tổng hợp toàn bộ 40 đối tượng thực nghiệm qua 5 đợt huấn luyện (Lan01 → Lan05), hiển thị chi tiết siêu tham số/trọng số tốt nhất đã tìm qua Optuna. Các mô hình **chưa Optuna** được để trống theo đúng yêu cầu.",
    "",
    "---",
    "",
    "## 1. BẢNG OVERVIEW TỔNG THỂ (MASTER MODEL & HYPERPARAMETER MATRIX)",
    "",
    "| STT | Đợt Chạy | Tên Mô Hình | Họ Kiến Trúc | Phương Pháp Tuning | Số Trials | Trạng Thái Optuna | Siêu Tham Số / Trọng Số Tốt Nhất (Optuna) | Điểm Thẩm Định Tốt Nhất (Val Loss) |",
    "| :-: | :---: | :--- | :--- | :--- | :-: | :--- | :--- | :-: |"
]

for _, r in overview_df.iterrows():
    p_display = f"`{r['Best_Hyperparameters_Optuna']}`" if r['Best_Hyperparameters_Optuna'] else ""
    s_display = f"`{r['Best_Val_Score']}`" if r['Best_Val_Score'] else ""
    md_lines.append(
        f"| {r['STT']} | **{r['Run']}** | `{r['Model_Name']}` | {r['Architecture_Family']} | {r['Tuning_Method']} | {r['Optuna_Trials']} | {r['Optuna_Status']} | {p_display} | {s_display} |"
    )

md_lines.extend([
    "",
    "---",
    "",
    "## 2. CHI TIẾT SIÊU THAM SỐ CÁC MÔ HÌNH ĐÃ OPTUNA (LẦN 01 — LAN01)",
    "",
    "Tại Lần 01, 6 mô hình phi tuyến và học sâu đã hoàn tất 100% quá trình tinh chỉnh Bayes thông qua Optuna TPE Sampler trên 5 chân trời và 5 Folds kiểm định ngoại mẫu:",
    ""
])

optuna_models = ['Random_Forest', 'XGBoost', 'LightGBM', 'RNN', 'LSTM', 'Transformer']
for m in optuna_models:
    sub = df_lan01[df_lan01['Model'] == m]
    trials = 50 if m in ['XGBoost', 'LightGBM'] else 30
    md_lines.extend([
        f"### 🔹 Mô Hình: `{m}` (Optuna {trials} Trials/Fold)",
        "| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |",
        "| :---: | :---: | :---: | :--- |"
    ])
    for _, row in sub.iterrows():
        md_lines.append(f"| `{row['Horizon']}` | Fold {row['Fold']} | `{row['Best_Val_Score']:.6f}` | `{row['Best_Params']}` |")
    md_lines.append("")

md_lines.extend([
    "---",
    "",
    "## 3. KHÔNG GIAN TÌM KIẾM CỦA CÁC MÔ HÌNH CHƯA OPTUNA (SẼ CHẠY Ở LẦN 03 VÀ LẦN 05)",
    "",
    "Các mô hình dưới đây hiện tại **để trống** giá trị tham số tối ưu vì chưa bước vào phiên huấn luyện Optuna:",
    "",
    "### 3.1 Nhóm Deep Learning Chuỗi Thời Gian Tiên Tiến (Lần 03 — Lan03)",
    "- **Mô hình:** `TCN`, `N-BEATS`, `N-HiTS`, `PatchTST`, `TSMixer`, `iTransformer`, `CNN`.",
    "- **Số lượng trials:** 30 trials / fold / horizon.",
    "- **Không gian siêu tham số dự kiến:**",
    "  - `lookback`: $[20, 40, 60]$",
    "  - `hidden_dim` / `d_model`: $[32, 64, 128]$",
    "  - `num_layers`: $[1, 2, 3]$",
    "  - `dropout`: $[0.1, 0.5]$",
    "  - `learning_rate`: $[10^{-4}, 10^{-2}]$ (log scale)",
    "  - `batch_size`: $[16, 32, 64]$",
    "",
    "### 3.2 Nhóm Tổ Hợp Ensemble & Chiến Lược Thích Ứng Regime (Lần 05 — Lan05)",
    "- **Mô hình:** `Ensemble_A1`, `Ensemble_A2`, `Ensemble_A3`, `Baseline_C1`, `Baseline_C2`, `Baseline_C3`.",
    "- **Số lượng trials:** 50 trials / fold / horizon.",
    "- **Không gian tối ưu hóa tỷ trọng dự kiến:**",
    "  - `A1, A2, A3`: Véc-tơ trọng số lồi trên không gian Simplex ($w_m \\ge 0, \\sum w_m = 1$).",
    "  - `C1`: Tỷ trọng hòa trộn giữa các mô hình dẫn đầu theo từng chế độ biến động.",
    "  - `C2`: Hệ số co rút James-Stein $N_0 \\in [5.0, 200.0]$ và độ nhạy nghịch đảo phương sai $\\gamma \\in [0.5, 3.0]$.",
    "  - `C3`: Tỷ trọng phòng thủ $\\alpha_{\\text{High}} \\in [0, 1]$ khi thị trường High Volatility và phân bổ Core $A2$ trong Low/Normal.",
    "",
    "---",
    "",
    "## 4. GIẢI ĐÁP: CÁC FILE KẾT QUẢ SAU HUẤN LUYỆN CÓ ĐƯỢC PUSH LÊN GITHUB KHÔNG?",
    "",
    "> [!IMPORTANT]",
    "> **CÓ, toàn bộ tệp kết quả nghiên cứu khoa học (CSVs, metrics, predictions, summary, logs, markdown reports, dashboard) ĐỀU ĐƯỢC PUSH lên GitHub đầy đủ!**  ",
    "> **CHỈ CÓ các file trọng số nhị phân siêu nặng (`.pt`, `.pth`, `.pkl`, `.h5`, `.joblib`) là KHÔNG PUSH lên GitHub.**",
    "",
    "### Chi tiết chính sách kiểm soát tệp theo `.gitignore` của Repository:",
    "",
    "| Loại Tệp / Thư Mục | Trạng Thái Trên GitHub | Lý Do & Vai Trò Học Thuật |",
    "| :--- | :---: | :--- |",
    "| **Kết quả số liệu OOS Predictions** (`results/*/predictions/*.csv`) |  **ĐƯỢC PUSH** | Minh chứng dữ liệu dự báo ngoại mẫu từng ngày (Lan01 có 95,200 dòng đã push). |",
    "| **Chỉ số đánh giá từng Fold** (`results/*/fold_metrics/*.csv`) |  **ĐƯỢC PUSH** | Bằng chứng kiểm định $R^2$, DA, RMSE, MAE, Diebold-Mariano test cho 350 folds. |",
    "| **Tóm tắt hiệu năng mô hình** (`results/*/summary/*.csv`) |  **ĐƯỢC PUSH** | Tổng hợp trung bình và độ lệch chuẩn theo từng chân trời. |",
    "| **Nhật ký siêu tham số** (`results/*/tuning/*.csv`) |  **ĐƯỢC PUSH** | Lưu lại 100% lịch sử Optuna và Grid Search để tái lập kết quả (Replication). |",
    "| **Báo cáo nghiên cứu & Phân tích RQ** (`results/*/agent_reports/*.md`) |  **ĐƯỢC PUSH** | Tài liệu học thuật tổng hợp giải đáp RQ1, RQ2, RQ3. |",
    "| **Bảng điều khiển trực quan** (`dashboard.html`, `DASHBOARD.md`) |  **ĐƯỢC PUSH** | Giao diện đồ họa giám sát thời gian thực. |",
    "| **File trọng số nhị phân mô hình** (`*.pt`, `*.pth`, `*.pkl`, `*.h5`) | ❌ **BỊ CHẶN (.gitignore)** | Các file mạng nơ-ron nhị phân dung lượng từ vài trăm MB đến hàng GB. GitHub có giới hạn 100MB/file; việc chặn binary weights là quy chuẩn quốc tế để tránh hỏng repository. |",
    "| **Tệp rác / cache trung gian** (`__pycache__`, `latest_metrics.csv`, `.env`) | ❌ **BỊ CHẶN (.gitignore)** | Tránh xung đột môi trường và giữ kho lưu trữ gọn gàng, bảo mật. |"
])

md_content = '\n'.join(md_lines)
with open('MODELS_OVERVIEW_HYPERPARAMS.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Saved MODELS_OVERVIEW_HYPERPARAMS.md ({len(md_lines)} lines)")
