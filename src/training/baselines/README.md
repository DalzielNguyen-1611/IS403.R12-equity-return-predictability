# 14 Baseline Models for Equity Return Predictability Research

> **Research Title:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

Hệ thống mã nguồn chứa **14 mô hình cơ sở (Baseline Models)** được tổ chức thành các nhóm chuyên biệt theo đúng phân loại học thuật, phục vụ đánh giá tính suy giảm của khả năng dự báo lợi suất cổ phiếu (S&P 500) qua các khung thời gian ($h \in \{1D, 5D, 20D, 60D, 120D\}$) và các chế độ biến động thị trường (`Low`, `Normal`, `High`).

---

## 📂 1. Cấu trúc thư mục

Mỗi mô hình được đặt trong một thư mục mang chính tên mô hình đó. **Mỗi folder chỉ gồm file code (`.py`) và 1 file markdown (`README.md`) lưu thông tin của mô hình**:

```text
src/training/baselines/
├── historical_mean/              # 1. Historical Expanding Mean (Benchmark)
│   ├── historical_mean.py
│   └── README.md
│
├── ridge/                        # 2. Ridge Regression (L2)
│   ├── ridge.py
│   └── README.md
│
├── elastic_net/                  # 3. Elastic Net (L1 + L2)
│   ├── elastic_net.py
│   └── README.md
│
├── arima/                        # 4. Univariate ARIMA (AIC Order Selection)
│   ├── arima.py
│   └── README.md
│
├── arimax/                       # 5. ARIMAX (Exogenous VIX & Price Signals)
│   ├── arimax.py
│   └── README.md
│
├── sarima/                       # 6. Seasonal ARIMA (Weekly s=5)
│   ├── sarima.py
│   └── README.md
│
├── svr/                          # 7. Support Vector Regression (RBF Kernel)
│   ├── svr.py
│   └── README.md
│
├── random_forest/                # 8. Random Forest Regression (Bagging)
│   ├── random_forest.py
│   └── README.md
│
├── xgboost/                      # 9. Extreme Gradient Boosting (XGBoost)
│   ├── xgboost_model.py
│   └── README.md
│
├── lightgbm/                     # 10. Light Gradient Boosting Machine (LightGBM)
│   ├── lightgbm_model.py
│   └── README.md
│
├── rnn/                          # 11. Vanilla Recurrent Neural Network (PyTorch CUDA)
│   ├── rnn_model.py
│   └── README.md
│
├── lstm/                         # 12. Long Short-Term Memory Network (PyTorch CUDA)
│   ├── lstm_model.py
│   └── README.md
│
├── transformer/                  # 13. Time-Series Self-Attention Transformer (PyTorch CUDA)
│   ├── transformer_model.py
│   └── README.md
│
├── logistic_regression/          # 14. Logistic Regression Direction Classifier
│   ├── logistic_regression.py
│   └── README.md
│
├── config.py                 # Cấu hình danh mục mô hình, biến mục tiêu, đặc trưng
├── utils.py                  # Walk-forward splits, scaling, DM test, Binomial test, metrics
├── fold_definition.csv       # Bảng thông số chi tiết 5 Folds Walk-Forward
├── run_all_baselines.py      # Script điều phối CLI master
├── generate_rq_analysis.py   # Script trích xuất bảng nghiên cứu RQ1, RQ2, RQ3 và Statistical Tests
└── README.md                 # Tài liệu tổng quan
```

### 🗂️ Cấu trúc Lưu trữ Kết quả 3 Tầng (`results/`)
Toàn bộ dữ liệu xuất ra được quản lý tập trung và phân tầng theo đúng chu trình nghiên cứu định lượng:
$$\boxed{\text{OOS Predictions} \longrightarrow \text{Fold Metrics} \longrightarrow \text{Summary} \longrightarrow \text{RQ1 / RQ2 / RQ3} \longrightarrow \text{Statistical Tests}}$$

```text
results/
├── fold_metrics/
│   └── all_fold_metrics.csv        # Tầng 1: Chi tiết từng fold (Model, Horizon, Fold, MAE, RMSE, OOS_R2, DA)
├── summary/
│   └── model_horizon_summary.csv   # Tầng 1 (tổng hợp): Mean & Std các chỉ số qua 5 folds cho từng Horizon
├── predictions/
│   └── all_oos_predictions.csv     # Tầng 2 (Dữ liệu gốc): Date, Model, Horizon, Fold, Actual, Prediction, Regime
└── analysis/                       # Tầng 3 (Thesis / Paper Ready):
    ├── rq1_horizon.csv             # RQ1: Ma trận suy giảm khả năng dự báo theo Horizon (1D..120D)
    ├── rq2_regime.csv              # RQ2: Hiệu năng phân tách theo Volatility Regime (Low, Normal, High)
    ├── rq3_time.csv                # RQ3: Tính ổn định theo chu kỳ thời gian (Time Periods: 2015-2016, 2017-2018,...)
    └── statistical_tests.csv       # Kiểm định thống kê: Model, Horizon, Test, Benchmark, Statistic, p-value, Significant
└── tuning/                         # Bằng chứng thực nghiệm siêu tham số (Evidence):
    └── best_hyperparameters.csv    # Ghi nhận Best Params, Best Val RMSE của từng Fold & Horizon
```

---

## ⚙️ 2. Khung Tinh chỉnh Siêu tham số (Hyperparameter Tuning Strategy)

Để đảm bảo tính **công bằng học thuật (fair comparison)** và **khả năng giải thích (interpretability)**, hệ thống phân chia 14 mô hình thành 3 nhóm chiến lược chuẩn mực:

| Nhóm | Mô hình | Phương pháp | Không gian / Số tổ hợp | Số Trial Optuna |
| :--- | :--- | :---: | :--- | :---: |
| **Benchmark** | `Historical Mean` | **Không tuning** | Trung bình lịch sử mở rộng trên Train của Fold | - |
| **Linear / Reg** | `Logistic Regression` | **Grid Search** | $C \in [10^{-3}..10^2] \times \text{class\_weight} \in [None, \text{'balanced'}]$ (12 tổ hợp) | - |
| **Linear / Reg** | `Ridge Regression` | **Grid Search** | $\alpha \in [10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100]$ (6 tổ hợp) | - |
| **Linear / Reg** | `Elastic Net` | **Grid Search** | $\alpha \in [10^{-3}..10^2] \times l_1\text{-ratio} \in [0.1, 0.3, 0.5, 0.7, 0.9]$ (30 tổ hợp) | - |
| **Statistical** | `ARIMA` | **Grid Search** | $p, q \in [0, 1, 2, 3, 5], d \in [0, 1]$ (AIC / Train RMSE) | - |
| **Statistical** | `ARIMAX` | **Grid Search** | $p, q \in [0, 1, 2, 3, 5], d \in [0, 1]$ + 10 biến ngoại sinh VIX & thị trường | - |
| **Statistical** | `SARIMA` | **Grid Search** | $p, q \in [0, 1, 2], d \in [0, 1], P, Q \in [0, 1], D \in [0, 1], s=5$ (chu kỳ tuần) | - |
| **Kernel** | `SVR` | **Grid Search** | RBF kernel, $C \in [0.1..100] \times \gamma \in [\text{'scale'}..0.1] \times \epsilon \in [10^{-3}..0.1]$ (64 tổ hợp) | - |
| **Tree Ensemble** | `Random Forest` | **Optuna** | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | 30 trials |
| **Tree Ensemble** | `XGBoost` | **Optuna** | `lr`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `alpha`, `lambda` | 50 trials |
| **Tree Ensemble** | `LightGBM` | **Optuna** | `lr`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `alpha`, `lambda`| 50 trials |
| **Deep Learning** | `RNN` | **Optuna** | `lookback` $[10..60]$, `hidden_size` $[32..128]$, `num_layers` $[1..3]$, `dropout`, `lr`, `batch` | 30 trials |
| **Deep Learning** | `LSTM` | **Optuna** | `lookback` $[10..60]$, `hidden_size` $[32..128]$, `dropout`, `lr`, `batch` (Unidirectional) | 30 trials |
| **Deep Learning** | `Transformer` | **Optuna** | `lookback` $[20..60]$, `d_model` $[32, 64]$, `nhead` $[2, 4]$, `layers` $[1, 2]$, `ffn`, `lr` | 30 trials |

### 🛡️ Nguyên tắc vàng: Chống rò rỉ dữ liệu khi Tuning (Zero-Leakage Nested CV)
1. **Tuyệt đối không chạm vào Validation của Fold:** Quá trình Grid Search / Optuna **chỉ được phép nhìn thấy dữ liệu Training của chính Fold đó** ($1990 \to \text{Train\_End}$).
2. **Internal Time-Series Split:** Dữ liệu Training của Fold $k$ được tách nội bộ (*internal train* & *internal validation*).
3. **Mục tiêu tối ưu:** Cực tiểu hóa **RMSE trên internal validation** (đảm bảo tính ổn định tối đa).
4. **Retrain & Predict:** Sau khi tìm được bộ tham số tối ưu, mô hình được **huấn luyện lại trên toàn bộ Training của Fold $k$**, rồi mới dự báo duy nhất 1 lần trên tập Validation ngoài mẫu của Fold $k$.
5. **Lưu vết khoa học:** Mọi siêu tham số được chọn cùng điểm số nội bộ đều được lưu tự động vào `results/tuning/best_hyperparameters.csv` làm bằng chứng thực nghiệm khi bảo vệ luận văn.

---

## 📊 3. Khung Tiêu chuẩn Đánh giá (Official Metric Framework)

Toàn bộ hệ thống đánh giá tuân thủ chuẩn kinh tế lượng tài chính thực nghiệm (Empirical Asset Pricing):

### A. Continuous Regression (13 Models)
- **Primary Metric:**
  $$\boxed{OOS\ R^2 = 1 - \frac{\sum_{t=1}^T (r_t - \hat{r}_t)^2}{\sum_{t=1}^T (r_t - \bar{r}_t)^2}}$$
  *(Đo lường mức vượt trội về sai số dự báo so với chuẩn Historical Mean benchmark $\bar{r}_t$. $R^2_{OOS} > 0$ chứng minh mô hình có năng lực dự báo vượt trội).*
- **Secondary Metrics:**
  - $MAE$: Sai số tuyệt đối trung bình.
  - $RMSE$: Căn bậc hai sai số bình phương trung bình.
  - $DA$ (*Directional Accuracy*): Tỷ lệ dự báo đúng chiều biến động $\text{sign}(\hat{y}_t) == \text{sign}(y_t)$.
- **Statistical Inference (Kiểm định thống kê):**
  - $\boxed{\text{Diebold-Mariano (DM) Test}}$: Kiểm định giả thuyết $H_0$ sai số của mô hình không khác biệt so với Historical Mean benchmark, có hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold (HLN, 1997) cho đa kỳ dự báo $h$.
  - $\boxed{\text{Sign / Binomial Test}}$: Kiểm định xem tỷ lệ đoán đúng chiều $DA$ có vượt trội có ý nghĩa thống kê so với tung đồng xu ngẫu nhiên ($H_0: p \le 0.5$ vs $H_1: p > 0.5$).

### B. Classification Side-Task (Logistic Regression)
- **Primary Metric:**
  $$\boxed{\text{ROC-AUC}}$$
  *(Đo lường năng lực phân loại xác suất tổng quát không phụ thuộc ngưỡng cutoff).*
- **Secondary Metrics:**
  - $Accuracy$: Tỷ lệ dự báo đúng nhãn.
  - $Precision$: Độ chuẩn xác khi dự báo thị trường Tăng.
  - $Recall$: Tỷ lệ bắt trọn các phiên thực tế Tăng.
  - $F1\text{-Score}$: Trung bình điều hòa giữa Precision và Recall.
  - $Balanced\ Accuracy$: Độ chính xác cân bằng giữa 2 lớp.

---

## 🎯 4. Danh mục 14 Mô hình và Tham số

| STT | Nhóm (Category) | Mô hình | Code | Đặc điểm chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Benchmark** | `Historical Mean` | `historical_mean.py` | Trung bình mở rộng lịch sử $\hat{y}_t = \frac{1}{t-1}\sum_{\tau<t} y_\tau$. |
| **2** | **Classical / Statistical** | `Ridge Regression` | `ridge.py` | Điều chuẩn L2, tune $\alpha \in [0.01..100]$ bằng `TimeSeriesSplit`. |
| **3** | **Classical / Statistical** | `Elastic Net` | `elastic_net.py` | Kết hợp L1 + L2, tune $\alpha$ và `l1_ratio` $[0.1..0.9]$. |
| **4** | **Classical / Statistical** | `ARIMA` | `arima.py` | Đơn biến, chọn bậc tối ưu bằng AIC trên Train. |
| **5** | **Classical / Statistical** | `ARIMAX` | `arimax.py` | Tích hợp 10 biến ngoại sinh VIX và thị trường. |
| **6** | **Classical / Statistical** | `SARIMA` | `sarima.py` | Chu kỳ mùa $s=5$ ngày giao dịch mỗi tuần. |
| **7** | **Kernel-based** | `SVR` | `svr.py` | Nhân RBF, tune $C, \gamma, \epsilon$ bằng `TimeSeriesSplit`. |
| **8** | **Tree-based / Ensemble** | `Random Forest` | `random_forest.py` | 300 cây, `max_depth` 10, `max_features` sqrt. |
| **9** | **Tree-based / Ensemble** | `XGBoost` | `xgboost_model.py` | 300 cây, `max_depth` 4, `learning_rate` 0.03. |
| **10** | **Tree-based / Ensemble** | `LightGBM` | `lightgbm_model.py` | 300 cây, `max_depth` 5, thuật toán lá `leaf-wise`. |
| **11** | **Deep Learning** | `RNN` | `rnn_model.py` | Lookback 20 ngày, 64 hidden units, early stopping, GPU CUDA. |
| **12** | **Deep Learning** | `LSTM` | `lstm_model.py` | Lookback 20 ngày, 64 hidden units, early stopping, GPU CUDA. |
| **13** | **Deep Learning** | `Transformer` | `transformer_model.py` | 4 heads attention, 2 layers encoder, Positional Encoding, GPU. |
| **14** | **Classification** | `Logistic Regression`| `logistic_regression.py`| Phân loại hướng đi $Direction \in \{0, 1\}$, lưu xác suất và ROC-AUC. |

---

## 🛡️ 4. Quy chuẩn chống Data Leakage
1. **Walk-Forward Expanding Window:** 
   - Fold 1: Train 6,351 ngày (`1990-03-27` → `2015-06-09`) $\rightarrow$ Val 272 ngày (`2015-06-10` → `2016-07-07`).
   - Fold 2: Train 6,623 ngày $\rightarrow$ Val 272 ngày (`2016-07-08` → `2017-08-04`).
   - ...
   - Fold 5: Train 7,439 ngày $\rightarrow$ Val 272 ngày (`2019-10-04` → `2020-10-30`).
2. **Khóa tập Test:** Giai đoạn `2020-11-02` → `2026-04-07` tuyệt đối không tham gia huấn luyện hay tinh chỉnh.
3. **Chuẩn hóa Scaler:** `StandardScaler` chỉ fit trên `X_train` của từng Fold.
4. **Không đưa `Volatility_Regime` vào ma trận $X$:** Đảm bảo đây là mô hình tổng quát (Global Model). Nhãn Regime được lưu kèm dự báo OOS để phân tích cho **RQ2**.

---

## 💻 5. Hướng dẫn chạy thử nghiệm độc lập hoặc toàn diện

### Cách 1: Chạy trực tiếp từng mô hình riêng lẻ
Mỗi mô hình có thể chạy độc lập, tự động đọc dữ liệu và ghi kết quả `latest_metrics.csv` cùng `latest_predictions.csv` vào chính thư mục của nó:
```bash
# Ví dụ chạy Historical Mean:
python src/training/baselines/historical_mean/historical_mean.py

# Ví dụ chạy Ridge:
python src/training/baselines/ridge/ridge.py

# Ví dụ chạy LightGBM:
python src/training/baselines/lightgbm/lightgbm_model.py

# Ví dụ chạy Transformer:
python src/training/baselines/transformer/transformer_model.py
```

### Cách 2: Chạy thông qua Bộ điều phối Master (`run_all_baselines.py`)
```bash
# Chạy theo 1 mô hình cụ thể:
python src/training/baselines/run_all_baselines.py --model ridge
python src/training/baselines/run_all_baselines.py --model lightgbm

# Chạy theo 1 horizon cụ thể:
python src/training/baselines/run_all_baselines.py --horizon 20D

# Chạy toàn bộ 14 mô hình trên tất cả 5 horizons:
python src/training/baselines/run_all_baselines.py
```
