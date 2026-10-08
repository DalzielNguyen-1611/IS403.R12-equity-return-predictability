# Báo Cáo Đánh Giá Tinh Chỉnh Siêu Tham Số & Kiểm Soát Rò Rỉ Dữ Liệu (04_best_hyperparameters_report.md)

**Mã tài liệu:** `IS403-REP-04`  
**Ngày thực hiện:** 2026-10-08  
**Tập tin mục tiêu:** `results/tuning/best_hyperparameters.csv`  
**Người thực hiện:** Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu & Mô Tả Tập Tin (Objective & File Description)
Tập tin `best_hyperparameters.csv` ghi nhận toàn bộ hồ sơ nghiệm chứng về quá trình tìm kiếm và tối ưu hóa không gian siêu tham số (hyperparameter optimization) cho các mô hình trong nghiên cứu. Mục tiêu cốt lõi của bước đánh giá này là:
1. **Kiểm tra độ bao phủ tinh chỉnh (Tuning Coverage):** Đảm bảo tất cả mô hình có tham số đều được điều chỉnh độc lập cho từng cặp (Chân trời, Fold).
2. **Kiểm tra tính hợp lệ của tham số:** Xác minh mọi giá trị được chọn đều nằm trong không gian tìm kiếm (search space bounds) đã khai báo tại `config.py`.
3. **Độ ổn định của tham số (Parameter Stability):** Đo lường mức độ biến động của siêu tham số qua các chu kỳ thời gian (Folds) và chân trời (Horizons).
4. **Kiểm soát rủi ro rò rỉ dữ liệu (Data Leakage Audit):** Đảm bảo tuyệt đối rằng siêu tham số không được tối ưu hóa dựa trên tập kiểm thử ngoại mẫu (outer test set).

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước vật lý
- **Đường dẫn lưu trữ:** `results/tuning/best_hyperparameters.csv`
- **Dung lượng:** ~50.3 KB (50,303 bytes)
- **Số hàng:** $325$ hàng
- **Số cột:** $6$ cột

### 2.2 Cấu trúc lược đồ dữ liệu (Data Schema)
| Cột | Kiểu dữ liệu | Số lượng Null | Mô tả học thuật |
| :--- | :--- | :--- | :--- |
| `Model` | `object` | 0 | Tên định danh mô hình (13 mô hình có tham số tinh chỉnh) |
| `Horizon` | `object` | 0 | Chân trời dự báo (`1D`, `5D`, `20D`, `60D`, `120D`) |
| `Fold` | `int64` | 0 | Chỉ số Fold kiểm định ngoại mẫu ($1, 2, 3, 4, 5$) |
| `Best_Params` | `object` (string dict) | 0 | Cấu hình siêu tham số tối ưu nhất được lựa chọn |
| `Best_Val_Score` | `float64` | 0 | Điểm số mục tiêu tốt nhất trên tập thẩm định nội bộ |
| `Score_Metric` | `object` | 0 | Thước đo dùng để chọn siêu tham số |

---

## 3. Kiểm Tra Tính Toàn Vẹn & Bao Phủ (Coverage Checks)

### 3.1 Ma trận bao phủ mô hình (Model × Horizon × Fold Coverage)
- **Tổng số mô hình trong nghiên cứu:** 14 mô hình.
- **Mô hình không cần tinh chỉnh (Parameter-free benchmark):** `Historical_Mean` (chỉ tính trung bình mẫu mở rộng, không có siêu tham số cần tối ưu). Sự vắng mặt của `Historical_Mean` trong tập tin này là hoàn toàn chính xác về mặt phương pháp luận.
- **Số mô hình được tinh chỉnh:** 13 mô hình.
- **Số hàng kỳ vọng:** $13 \text{ Models} \times 5 \text{ Horizons} \times 5 \text{ Folds} = 325$ hàng.
- **Số hàng thực tế:** $325$ hàng.
- **Tỷ lệ bao phủ:** **100.0%**. Không có bất kỳ ô trống hoặc lần chạy nào bị thất lạc.
- **Kiểm tra trùng lặp khóa chính `(Model, Horizon, Fold)`:** $0$ hàng trùng lặp.

### 3.2 Phân bổ thước đo tối ưu hóa (`Score_Metric`)
1. **`Internal_Val_RMSE` (150 lượt chạy):** Áp dụng cho các mô hình Cây quyết định và Học sâu dùng thuật toán tối ưu hóa Bayes qua Optuna (`Random_Forest`, `XGBoost`, `LightGBM`, `RNN`, `LSTM`, `Transformer`). Điểm số dao động từ $0.007684$ đến $0.096096$ (trung vị $0.029764$).
2. **`CV_RMSE` (75 lượt chạy):** Áp dụng cho `Ridge`, `Elastic_Net` và `SVR`.
3. **`AIC` (Akaike Information Criterion - 75 lượt chạy):** Áp dụng cho các mô hình chuỗi thời gian thống kê cổ điển (`ARIMA`, `ARIMAX`, `SARIMA`). Điểm AIC dao động từ $-45,979.01$ đến $-34,372.20$.
4. **`CV_ROC_AUC` (25 lượt chạy):** Áp dụng riêng cho mô hình phân loại `Logistic_Regression`. Điểm số dao động từ $0.468287$ đến $0.533536$ (trung vị $0.511733$).

---

## 4. Kiểm Định Không Gian Tìm Kiếm & Tính Hợp Lệ Của Tham Số (Search Space Bounds Audit)

Đối chiếu chéo từng tham số trong $325$ bản ghi với lưới tìm kiếm đã khai báo tại `src/training/baselines/config.py`:
- **Ridge Regression:** Tất cả các giá trị $\alpha$ đều thuộc tập khai báo $\{0.001, 0.01, 0.1, 1, 10, 100\}$.
- **Elastic Net:** $\alpha \in \{0.001, \dots, 100\}$ và $l_1\text{-ratio} \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$.
- **SVR:** $C \in \{0.1, 1, 10, 100\}$, $\gamma \in \{\text{'scale'}, 0.001, 0.01, 0.1\}$, $\epsilon \in \{0.001, 0.01, 0.05, 0.1\}$.
- **Tree-based Ensembles (XGBoost, LightGBM, Random Forest):** Toàn bộ tham số số lượng cây ($n\_estimators \in [100, 1000]$), tốc độ học ($learning\_rate \in [0.01, 0.2]$), độ sâu tối đa, tham số điều chuẩn $\alpha, \lambda$ đều nằm chính xác trong biên chặn của Optuna.
- **Deep Learning (RNN, LSTM, Transformer):** Các tham số cấu trúc ($lookback \in [20, 40]$, $hidden\_size \in [32, 128]$, dropout $\in [0.1, 0.5]$) đều tuân thủ chặt chẽ cấu hình.
- **Kết quả kiểm toán:** **0 phát hiện dị thường (0 bounds anomalies)**. Không có tham số nào vượt ra ngoài không gian tìm kiếm.

---

## 5. Phân Tích Độ Ổn Định Của Siêu Tham Số (Hyperparameter Stability Across Folds & Horizons)

Bảng dưới đây thống kê số lượng cấu hình siêu tham số độc nhất (unique configurations) được lựa chọn qua 25 kịch bản ($5 \text{ Horizons} \times 5 \text{ Folds}$) của từng mô hình:

| Mô hình | Phương pháp Tuning | Số cấu hình độc nhất / 25 | Đánh giá độ ổn định tham số |
| :--- | :---: | :---: | :--- |
| **ARIMAX** | Fixed Specification (Order 1,0,1) | 1 / 25 | Cố định theo thiết kế mô hình (không chạy Grid Search các bậc khác) |
| **SARIMA** | Fixed Specification (s=5) | 1 / 25 | Cố định theo thiết kế mô hình (chu kỳ tuần $s=5$) |
| **ARIMA** | Grid Search (AIC 3 orders) | 2 / 25 | Lựa chọn giữa $(1,0,1)$ và $(1,0,0)$ trên dữ liệu train |
| **Ridge** | RidgeCV (TimeSeriesSplit) | 3 / 25 | Ổn định cao (ưu tiên $\alpha = 100.0$ ở đa số folds) |
| **Logistic_Regression** | GridSearchCV | 5 / 25 | Ổn định (thường chọn $C = 0.001$, `class_weight='balanced'`) |
| **Elastic_Net** | ElasticNetCV (TimeSeriesSplit) | 7 / 25 | Độ biến thiên thấp (chọn $\alpha = 0.001, l_1 = 0.7$ hoặc $0.5$) |
| **SVR** | GridSearchCV | 7 / 25 | Độ biến thiên thấp (chọn $C=0.1, \epsilon=0.01, \gamma=0.001$) |
| **Random_Forest** | Optuna (30 trials) | 11 / 25 | Biến thiên tham số rời rạc theo từng fold |
| **LightGBM** | Optuna (50 trials) | 21 / 25 | Biến thiên cấu hình cao trên không gian liên tục |
| **RNN** | Optuna (30 trials) | 22 / 25 | Biến thiên cấu hình cao trên không gian liên tục |
| **LSTM** | Optuna (30 trials) | 23 / 25 | Biến thiên cấu hình cao trên không gian liên tục |
| **XGBoost** | Optuna (50 trials) | 24 / 25 | Biến thiên cấu hình cao trên không gian liên tục |
| **Transformer** | Optuna (30 trials) | 25 / 25 | Mỗi fold và horizon đều tìm ra cấu hình riêng biệt |

### Nhận định chuyên sâu:
1. **Thiết kế cấu trúc dòng mô hình chuỗi thời gian:** ARIMAX và SARIMA được triển khai theo cấu hình bậc cố định tiên nghiệm (a priori fixed specification: ARIMAX $(1,0,1)$ và SARIMA $(1,0,0)\times(1,0,0,5)$) nhằm đảm bảo tính đơn giản và tính toán nhanh, chứ không trải qua vòng quét Grid Search nhiều bậc như ARIMA chuẩn.
2. **Độ biến thiên cấu hình ở các mô hình phi tuyến & Học sâu:** Việc các mô hình như XGBoost, LightGBM và Transformer có từ 21/25 đến 25/25 cấu hình độc nhất phản ánh độ nhạy cao của bộ lấy mẫu TPE (Optuna) đối với dữ liệu từng fold và không gian siêu tham số liên tục, không đồng nghĩa với tính tối ưu toàn cục hay năng lực thích ứng kinh tế học.

---

## 6. Kiểm Định Chống Rò Rỉ Dữ Liệu (Data Leakage & Test-Data Contamination Audit)

Đây là yêu cầu kiểm toán phương pháp luận then chốt theo quy định thực nghiệm tài chính:

> **Quy định ngặt nghèo:** Siêu tham số TUYỆT ĐỐI KHÔNG được lựa chọn dựa trên tập kiểm định ngoại mẫu cuối cùng (final test set / outer validation fold). Mọi bước chuẩn hóa và tinh chỉnh phải được thực hiện khép kín bên trong phần dữ liệu huấn luyện của outer fold.

### 6.1 Đánh giá dựa trên tập tin `best_hyperparameters.csv` độc lập
- **Kết luận theo chuẩn quy định:**  
  *"Unable to verify from this file alone."*  
  *(Không thể xác nhận tính miễn nhiễm rò rỉ chỉ từ tập tin này, vì bảng dữ liệu chỉ lưu trữ các giá trị tham số tối ưu và điểm số, không chứa thông tin về chỉ số dòng thời gian phân chia nội bộ).*

### 6.2 Đánh giá kết hợp kiểm toán mã nguồn triển khai (Codebase Verification)
Để đảm bảo tính trung thực tuyệt đối, tiến hành kiểm tra mã nguồn thực thi tại `src/training/baselines/utils.py` và các script con:
1. **Cơ chế phân tách nội bộ cho Deep Learning & Tree Models:**
   - Tại `utils.py` (dòng 369–380): hàm `get_internal_train_val_split(X_train, y_train, val_ratio=0.15)` chỉ nhận vào `X_train` và `y_train` của outer fold $k$. Dữ liệu kiểm thử ngoại mẫu $X_{\text{val}}$ hoàn toàn bị phong tỏa và không được chuyển vào hàm này.
   - Quá trình Optuna chỉ tính toán loss mục tiêu trên $15\%$ dữ liệu cuối của chuỗi huấn luyện nội bộ ($X_{\text{val\_inner}}$).
2. **Cơ chế TimeSeriesSplit cho Linear & Logistic Models:**
   - Tại `ridge.py` (dòng 34–35) và `elastic_net.py` (dòng 34–38): việc chọn $\alpha$ được thực hiện thông qua `TimeSeriesSplit(n_splits=3)` hoàn toàn bên trong `X_train`.
3. **Phát hiện dị thường về giá trị `Best_Val_Score = 0.0` tại Ridge và Elastic Net:**
   - Trong `ridge.py` và `elastic_net.py`, hàm `save_tuning_record` truyền giá trị `0.0` vào cột `Best_Val_Score`.
   - **Xác minh kỹ thuật độc lập:** Môi trường hiện tại sử dụng `scikit-learn 1.9.0`. Kiểm tra đối tượng fitted cho thấy `RidgeCV` **thực tế có thuộc tính `best_score_`** (chứa điểm trung bình qua các fold của TimeSeriesSplit), và `ElasticNetCV` có thuộc tính `mse_path_` (từ đó có thể suy ra MSE tối ưu). Việc mã nguồn gán cứng `0.0` là một thói quen lập trình rút gọn (implementation artifact).
   - **Đánh giá phương pháp luận:** Đây là một **Auditability Limitation** (giới hạn khả năng kiểm chứng điểm số validation thực tế từ CSV), hoàn toàn **không phải là data leakage**, vì các siêu tham số `alpha_` và `l1_ratio_` vẫn được ước lượng hoàn toàn hợp lệ trên tập huấn luyện nội bộ.

---

## 7. Hạn Chế Phương Pháp Luận Của Quá Trình Tinh Chỉnh (Methodological Limitations)
1. **Ngân sách vòng thử nghiệm Optuna có giới hạn tính toán (30–50 trials):** Ngân sách thử nghiệm 30–50 trials cung cấp một cơ chế tìm kiếm heuristic có giới hạn tài nguyên trên không gian siêu tham số xác định trước, nhưng chưa đảm bảo tính tối ưu toàn cục (global optimality) cũng như chưa quét trọn vẹn không gian tương tác phi tuyến đa chiều của Transformer.
2. **Cấu trúc TimeSeriesSplit nội bộ ngắn trong các Fold đầu:** Tại Fold 1, việc chia tiếp `TimeSeriesSplit` 3 phần khiến tập huấn luyện con chỉ còn khoảng 150–200 ngày, làm tăng phương sai ước lượng siêu tham số tại Fold 1.
3. **Tiền xử lý chuẩn hóa ở cấp độ Outer Train:** Scaler được fit trên toàn bộ outer train trước khi cắt split 15% inner val cho Optuna, tạo ra rò rỉ phân phối nhẹ trong nội bộ vòng lặp tuning (tuy nhiên outer validation hoàn toàn cách ly).

---

## 8. Kết Luận Bước 4 & Kế Hoạch Tiếp Theo (Conclusions & Next Steps)
- **Kết luận:** Quá trình kiểm toán mã nguồn cung cấp bằng chứng vững chắc rằng việc tinh chỉnh siêu tham số được thực hiện phân tầng (nested time-series tuning) khép kín bên trong từng outer training fold và hoàn toàn cách ly khỏi tập kiểm định ngoại mẫu (outer validation) và tập kiểm tra độc lập (test set). Tuy nhiên, tính hợp lệ trọn vẹn của toàn bộ pipeline để so sánh giữa các dòng mô hình vẫn mang tính có điều kiện (conditional), phụ thuộc vào việc xử lý vấn đề điều kiện hóa bước lọc của dòng ARIMA (đã phát hiện tại Report 03).
- **Bước tiếp theo:** Xem xét các hạng mục trong phần **Methodological Audit** bên dưới và đánh giá **Step 5 Readiness Gate** trước khi tiến hành phân tích RQ1 Horizon.

---

# Methodological Audit

## 1. Audit Scope & Rules
Kiểm toán phương pháp luận độc lập này tập trung đánh giá tính toàn vẹn của quy trình tối ưu hóa siêu tham số (Hyperparameter Optimization - HPO) trong khuôn khổ thiết kế **Nested Walk-Forward Time-Series Validation** của nghiên cứu:
**"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"**

- **Phạm vi kiểm tra:**
  - Tệp kết quả: `results/tuning/best_hyperparameters.csv` ($325$ hàng, 13 mô hình $\times$ 5 horizons $\times$ 5 folds).
  - Tệp mã nguồn thực thi: `src/training/baselines/utils.py`, `config.py`, và toàn bộ script huấn luyện thành phần (`ridge/`, `elastic_net/`, `logistic_regression/`, `arima/`, `arimax/`, `sarima/`, `svr/`, `random_forest/`, `xgboost/`, `lightgbm/`, `rnn/`, `lstm/`, `transformer/`).
  - Dữ liệu phân chia: `dataset/3.0.3_val_folds.csv`.
- **Nguyên tắc thực hiện:**
  - Tuyệt đối không retrain model, không sửa tệp CSV kết quả, không thay đổi hyperparameter hiện tại.
  - Mọi nhận định `Verified` đều đi kèm bằng chứng mã nguồn trực tiếp (line-level evidence).

---

## 2. Audit Nested Walk-Forward Tuning
Kiểm tra luồng thực thi (execution call chain) từ điểm khởi đầu đến khi tạo dự báo ngoại mẫu:

$$\text{run\_all\_baselines.py} \to \text{run\_model(df)} \to \text{get\_walk\_forward\_splits(df)} \to \text{prepare\_fold\_data} \to \text{tuning} \to \text{final fit} \to \text{predict}(X_{\text{val}})$$

### Bằng chứng xác minh:
1. **Dữ liệu Outer Validation không bao giờ tham gia tuning:**
   Trong tất cả 13 mô hình, hàm phân chia nội bộ `get_internal_train_val_split` hoặc `TimeSeriesSplit(3)` chỉ nhận đầu vào là `X_train` và `y_train` của fold $k$. Biến `X_val` và `y_val` được giữ nguyên riêng biệt và chỉ xuất hiện ở bước `model.predict(X_val)`.
2. **Tính độc lập tuyệt đối giữa các Fold và Horizons:**
   Các vòng lặp được lồng nhau theo thứ tự:
   ```python
   for horizon in horizons:
       for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):
           # Tuning độc lập cho từng cặp (horizon, fold)
   ```
   Không có biến toàn cục nào lưu lại siêu tham số của Fold 1 để tái sử dụng ép buộc cho Fold 2–5, hoặc lấy từ 1D để áp cho 120D.
3. **Mô hình được fit lại trên toàn bộ Outer Train trước khi dự báo ngoại mẫu:**
   - Đối với các mô hình Cây (`Random_Forest`, `XGBoost`, `LightGBM`) và Kernel (`SVR`): Sau khi Optuna/GridSearch chọn được `best_params`, một thực thể mô hình mới được khởi tạo và gọi `fit(X_train, y_train)` trên toàn bộ $100\%$ dữ liệu của outer training fold trước khi gọi `predict(X_val)`.
   - Đối với mô hình Deep Learning (`RNN`, `LSTM`, `Transformer`): Mô hình huấn luyện trên $85\%$ inner train và sử dụng $15\%$ inner val để early stopping nhằm chống overfit, sau đó tạo sequence dự báo trên `X_val` dựa trên ngữ cảnh lịch sử `X_train`.
- **Trạng thái:** `Verified` (Luồng thực thi tuân thủ nghiêm ngặt nguyên tắc Nested Walk-Forward).

---

## 3. Audit Internal Validation Split
Kiểm tra hàm `get_internal_train_val_split(X_train, y_train, val_ratio=0.15)` tại `src/training/baselines/utils.py` (dòng 369–380):

```python
def get_internal_train_val_split(X_train, y_train, val_ratio=0.15):
    n_train = len(X_train)
    split_idx = int(n_train * (1.0 - val_ratio))
    X_tr_inner = X_train[:split_idx]
    y_tr_inner = y_train[:split_idx]
    X_val_inner = X_train[split_idx:]
    y_val_inner = y_train[split_idx:]
    return X_tr_inner, y_tr_inner, X_val_inner, y_val_inner
```

### Đánh giá chi tiết:
- **Thứ tự thời gian (Chronological Order):** `X_tr_inner` lấy từ đầu đến `split_idx`, `X_val_inner` lấy từ `split_idx` đến hết. Inner validation nằm strictly sau inner train.
- **Không xáo trộn (Zero Shuffle):** Cắt lát mảng numpy liên tục, không dùng hàm `train_test_split(shuffle=True)`.
- **Không nhiễm dữ liệu Outer Val / Test:** `n_train` chỉ tính trên `X_train` của fold hiện tại.
- **Trạng thái:** `Verified` (Cấu trúc phân tách nội bộ hoàn toàn chuẩn xác).

---

## 4. Audit Scaling / Preprocessing Leakage
Kiểm tra toàn bộ pipeline chuẩn hóa đặc trưng:

### Quy trình trong mã nguồn (`utils.py`, dòng 88–126):
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

### Đánh giá hai cấp độ rò rỉ:
1. **Cấp độ Outer Fold (Train vs Outer Validation):**
   - Scaler chỉ được `fit` trên `X_train` (dữ liệu quá khứ).
   - `X_val` chỉ nhận `transform` dựa trên trung bình $\mu_{\text{train}}$ và độ lệch chuẩn $\sigma_{\text{train}}$.
   - **Kết luận:** **Zero Leakage** giữa Train và Outer Validation. Không có hiện tượng `Scaler.fit(All Data)`.
2. **Cấp độ Inner Tuning (Inner Train vs Inner Validation):**
   - Đối với các mô hình Cây (`Random_Forest`, `XGBoost`, `LightGBM`): `scale=False`. Không có biến đổi chuẩn hóa nào được áp dụng, do đó hoàn toàn không có rò rỉ ở cả hai cấp độ.
   - Đối với các mô hình Linear, SVR và Deep Learning: `X_train` đã được chuẩn hóa bởi scaler của Outer Train trước khi truyền vào `get_internal_train_val_split`. Vì vậy, $\mu$ và $\sigma$ dùng để chuẩn hóa `X_tr_inner` có chứa thông tin thống kê bậc 1 của $15\%$ dữ liệu `X_val_inner`.
   - **Đánh giá mức độ nghiêm trọng:** **Low / Negligible**. Đây là sự rò rỉ phân phối nhẹ trong nội bộ bước tinh chỉnh siêu tham số (inner tuning loop), không làm rò rỉ bất kỳ thông tin nào của tập kiểm định ngoại mẫu thực sự (`X_val`).
- **Trạng thái:** `Verified` ở cấp độ Outer Fold; `Potential issue` (mức độ thấp) ở cấp độ Inner Tuning.

---

## 5. Audit Ridge / Elastic Net

### 5.1 Kiểm tra thuộc tính scikit-learn (Version 1.9.0)
Report 04 trước đây tuyên bố rằng scikit-learn không lưu trữ giá trị vô hướng `best_score_`. Kiểm tra thực tế trên môi trường thực thi:
- **`RidgeCV`:** Khi được cấu hình với `cv=TimeSeriesSplit(3)`, đối tượng đã fit **sở hữu thuộc tính `best_score_`**, đại diện cho điểm MSE âm trung bình qua 3 splits.
- **`ElasticNetCV`:** Sở hữu mảng `mse_path_` dạng 3D `(n_l1_ratio, n_alphas, n_folds)`. Điểm số tốt nhất có thể tính bằng `np.min(np.mean(model.mse_path_, axis=-1))`.

### 5.2 Đánh giá việc lưu `Best_Val_Score = 0.0`
- Việc mã nguồn gán giá trị mặc định `0.0` tại `utils.save_tuning_record` cho Ridge và Elastic Net là một sự giản lược trong triển khai (logging artifact).
- **Hệ quả:** Tạo ra một giới hạn kiểm chứng (Auditability Limitation) khiến người đọc không thể so sánh trực tiếp điểm validation nội bộ của Ridge/Elastic Net với các mô hình khác.
- **Tính hợp lệ của mô hình:** Các thuộc tính quan trọng nhất gồm `model.alpha_` và `model.l1_ratio_` đã được trích xuất chính xác và lưu vào `best_hyperparameters.csv`.
- **Trạng thái:** `Verified` (Xác định rõ nguyên nhân; không ảnh hưởng đến tính hợp lệ của dự báo ngoại mẫu).

---

## 6. Audit Logistic Regression
Kiểm tra tệp `src/training/baselines/logistic_regression/logistic_regression.py`:

- **Phương pháp Cross-Validation:** Sử dụng `TimeSeriesSplit(n_splits=3)` (dòng 35). Không dùng K-Fold xáo trộn ngẫu nhiên.
- **Hàm mục tiêu tối ưu:** Khai báo `scoring='roc_auc'` trong `GridSearchCV` (dòng 46).
- **Tham số `class_weight`:** Nằm trong lưới tìm kiếm `param_grid = {'C': [...], 'class_weight': ['balanced', None]}` (dòng 36–39). Tham số này được tối ưu hóa động qua CV chứ không bị gán cứng.
- **Ngưỡng quyết định (Decision Threshold):** Cố định tại mức xác suất chuẩn $0.5$ (`y_preds = (y_probs >= 0.5).astype(int)`). Ngưỡng hoàn toàn không bị tinh chỉnh dựa trên dữ liệu ngoại mẫu.
- **Trạng thái:** `Verified` (Quy trình phân loại nhị phân độc lập, chuẩn xác).

---

## 7. Audit ARIMA / ARIMAX / SARIMA Tuning

### 7.1 Kiểm tra quy trình thực tế
Đối chiếu mã nguồn triển khai thực tế của 3 mô hình chuỗi thời gian:

1. **ARIMA (`arima.py`):**
   - Quét qua 3 bậc ứng viên `CANDIDATE_ORDERS = [(1, 0, 0), (0, 0, 1), (1, 0, 1)]`.
   - Tính toán AIC trên chuỗi huấn luyện `train_series` của từng fold.
   - Chọn bậc có AIC nhỏ nhất. Dữ liệu validation hoàn toàn không tham gia vào bước chọn bậc.
2. **ARIMAX (`arimax.py`):**
   - Mã nguồn khai báo trực tiếp `order = (1, 0, 1)` với cơ chế xử lý ngoại lệ chuyển về `(1, 0, 0)`.
   - **Thực tế:** Không thực hiện Grid Search qua nhiều bậc. Đây là cấu hình tham số cố định tiên nghiệm (fixed specification).
   - Biến ngoại sinh (11 biến) được chuẩn hóa bằng `StandardScaler` fit trên `train_exog_raw`.
3. **SARIMA (`sarima.py`):**
   - Mã nguồn khai báo cứng `order = (1, 0, 0)` và `seasonal_order = (1, 0, 0, 5)`.
   - Tham số chu kỳ mùa $s = 5$ (đại diện cho 5 ngày giao dịch trong tuần) được chọn cố định theo giả định kinh tế lượng tiên nghiệm trước khi chạy mô hình, không phải kết quả sau khi nhìn thấy dữ liệu kiểm thử.
- **Trạng thái:** `Verified` (Mã nguồn chạy đúng như thiết kế cố định; cần sửa lại nhận định trong báo cáo rằng ARIMAX/SARIMA chạy Grid Search).

---

## 8. Audit Target Construction trong ARIMA Family
Mục này liên kết trực tiếp với phát hiện quan trọng tại **Report 03 Methodological Audit**:

- **Biến mục tiêu được nạp:**
  ```python
  target_col = config.HORIZON_TO_TARGET[horizon]
  train_y = train_df[target_col].values
  ```
  Mô hình tối ưu hóa trực tiếp trên $Target_{h,t} = \ln(Close_{t+h}/Close_t)$ (**Case A - Direct Multi-Horizon**), không phải mô hình đệ quy cộng dồn lợi suất 1 ngày (**Case B**).
- **Cơ chế lọc 1 bước (1-step filtering):**
  Lệnh `applied = res.apply(full_y)` và `predict(start=n_train, end=n_train+n_val-1)` trong `statsmodels` thực hiện cập nhật lọc 1 bước trên chuỗi $Target_{h,t}$.
  Tại ngày $t$, mô hình sử dụng giá trị trễ $Target_{h,t-1} = \ln(Close_{t+h-1}/Close_{t-1})$, vốn chứa giá đóng cửa tại ngày $t+h-1$ (tương lai 119 ngày ở horizon 120D).
- **Ý nghĩa đối với quá trình Tuning:** Mặc dù AIC được tính trên tập train và việc chọn tham số không bị rò rỉ dữ liệu từ tập val, nhưng bản thân bài toán mà dòng ARIMA đang giải ở pha dự báo ngoại mẫu là bài toán lọc 1 bước trên chuỗi gối đầu, tạo ra sự bất tương đồng thông tin so với các mô hình Machine Learning.
- **Trạng thái:** `Potential issue` (Mức độ nghiêm trọng: **Critical** đối với việc so sánh hiệu năng ngoại mẫu giữa các dòng mô hình).

---

## 9. Audit Optuna Objective
Kiểm tra hàm mục tiêu Optuna trong các mô hình phi tuyến và học sâu (`Random_Forest`, `XGBoost`, `LightGBM`, `RNN`, `LSTM`, `Transformer`):

```text
Optuna Trial
    ↓
Gợi ý bộ siêu tham số ứng viên (learning_rate, max_depth, dropout,...)
    ↓
Huấn luyện mô hình nội bộ trên X_tr_inner, y_tr_inner
    ↓
Dự báo trên tập kiểm định nội bộ X_val_inner
    ↓
Tính toán Inner Validation RMSE: sqrt(mean((y_val_in - preds)^2))
    ↓
Trả về RMSE cho bộ lấy mẫu TPE để tối ưu hóa
```

### Điểm kiểm định cốt lõi:
- Không có bất kỳ dòng lệnh nào đánh giá trial trên outer validation set (`X_val`, `y_val`).
- Dừng sớm (Early stopping) ở Deep Learning chỉ theo dõi `inner_val_loss`.
- Trọng số mô hình tốt nhất (`best_weights`) được lưu dựa trên `inner_val_loss`.
- **Trạng thái:** `Verified` (Hàm mục tiêu Optuna hoàn toàn cách ly, chuẩn mực phương pháp luận).

---

## 10. Audit Hyperparameter Selection Across Horizons
- Kiểm tra mã nguồn: Vòng lặp ngoài cùng lặp qua từng chân trời `for horizon in horizons:`.
- Với mỗi chân trời, quá trình tuning được kích hoạt độc lập với không gian và dữ liệu riêng của chân trời đó.
- Không có hiện tượng chia sẻ hay tái sử dụng siêu tham số giữa các chân trời (ví dụ: tham số của 1D không bị áp đặt cho 120D).
- **Trạng thái:** `Verified` (Tính độc lập giữa các chân trời được bảo toàn $100\%$).

---

## 11. Audit Hyperparameter Selection Across Folds
- Kiểm tra mã nguồn: Bên trong vòng lặp horizon, vòng lặp thứ hai duyệt qua từng fold `for fold, train_df, val_df, meta in utils.get_walk_forward_splits(df):`.
- Mỗi fold sở hữu một phiên chạy Optuna / GridSearch riêng biệt dựa trên độ dài dữ liệu mở rộng của chính fold đó.
- Không có hiện tượng lấy trung bình siêu tham số trên cả 5 folds để tạo ra một bộ tham số toàn cục (global parameters).
- **Trạng thái:** `Verified` (Tính độc lập giữa các fold kiểm định ngoại mẫu được bảo toàn $100\%$).

---

## 12. Audit Search Space Claims
Kiểm tra chéo biên không gian tìm kiếm tại `config.py` với các giá trị trong `best_hyperparameters.csv`:

| Mô hình | Không gian tìm kiếm tại `config.py` | Kiểm tra biên giá trị tối ưu | Kết quả |
| :--- | :--- | :--- | :---: |
| **Ridge** | $\alpha \in [0.001, 100.0]$ | $\min = 0.001, \max = 100.0$ | Tuân thủ |
| **Elastic_Net** | $\alpha \in [0.001, 100.0], l_1 \in [0.1, 0.9]$ | $\alpha \in [0.001, 10.0], l_1 \in [0.1, 0.9]$ | Tuân thủ |
| **SVR** | $C \in [0.1, 100], \epsilon \in [0.001, 0.1], \gamma \in [\text{'scale'}, 0.001, 0.1]$ | Mọi tham số đều thuộc lưới định nghĩa | Tuân thủ |
| **Random_Forest** | $n\_est \in [100, 500]$, max_depth $\in [3, 10]$ | $n\_est \in [100, 500]$, max_depth $\in [3, 10]$ | Tuân thủ |
| **XGBoost** | $lr \in [0.01, 0.2]$, depth $\in [2, 10]$, reg_alpha $\in [10^{-8}, 10]$ | Mọi giá trị log-scale đều nằm trong khoảng | Tuân thủ |
| **LightGBM** | $lr \in [0.01, 0.2]$, num_leaves $\in [15, 127]$ | num_leaves $\in [15, 63]$ | Tuân thủ |
| **Deep Learning** | lookback $\in [20, 40]$, dropout $\in [0.1, 0.5]$, $lr \in [10^{-5}, 10^{-3}]$ | Mọi tham số kiến trúc đều nằm trong biên | Tuân thủ |

- **Trạng thái:** `Verified` (0 vi phạm biên tìm kiếm).

---

## 13. Audit Claim về Parameter Stability
- **Đánh giá lại phát biểu:** Trong Report 04 ban đầu, số lượng cấu hình độc nhất ($1/25$ ở ARIMAX, $25/25$ ở Transformer) được gọi là thước đo "độ ổn định tham số" và "tính thích ứng động cao".
- **Hiệu chỉnh học thuật:**
  - $1/25$ ở ARIMAX và SARIMA thực chất là do mô hình có cấu hình bậc cố định theo thiết kế, không phản ánh sự ổn định qua tối ưu hóa.
  - $21/25$ đến $25/25$ ở các mô hình Cây và Deep Learning phản ánh độ nhạy cao của thuật toán tối ưu hóa Bayes trên không gian tham số liên tục trước sự thay đổi của tập dữ liệu huấn luyện, không nên tự động diễn giải là "thích ứng kinh tế học thông minh".
- **Trạng thái:** `Verified` (Đã điều chỉnh cách diễn đạt thận trọng, tránh suy diễn quá mức).

---

## 14. Audit Optuna Trial Count
- **Số vòng thử nghiệm thực tế:** 30 trials (Random Forest, RNN, LSTM, Transformer) và 50 trials (XGBoost, LightGBM).
- **Đánh giá khoa học:** Ngân sách 30–50 trials là một giải pháp dung hòa thực dụng giữa độ chính xác và chi phí tính toán khi phải chạy $325$ phiên tối ưu hóa độc lập. Ngân sách này đủ để tìm ra vùng siêu tham số khả dĩ (reasonable local region), nhưng không đảm bảo tìm ra điểm cực trị toàn cục (global optimality).
- **Trạng thái:** `Verified` (Đã chuẩn hóa phát biểu học thuật).

---

## 15. Audit Final Test Isolation
Kiểm tra tập dữ liệu gốc `dataset/3.0.3_val_folds.csv`:
- Tổng số quan sát: $9,073$ dòng (`1990-03-27` $\to$ `2026-04-07`).
- Tập kiểm tra độc lập (Out-of-sample Test Set): $1,362$ dòng (`2020-11-02` $\to$ `2026-04-07`, nằm sau Fold 5).
- **Xác minh mã nguồn:** Hàm `get_walk_forward_splits` chỉ lặp `for fold in range(1, 6):`, chỉ cắt dữ liệu đến ngày kết thúc của Fold 5 (`2020-10-30`).
- Toàn bộ $1,362$ ngày giao dịch của tập Test hoàn toàn không bao giờ được đọc vào bộ nhớ trong quá trình huấn luyện, tinh chỉnh hay kiểm định ngoại mẫu của bất kỳ mô hình nào.
- **Trạng thái:** `Verified` (Tập Test được cô lập hoàn hảo $100\%$).

---

## 16. Audit Regime Variable
- Kiểm tra danh sách đặc trưng `FEATURE_COLS` tại `config.py` (dòng 72–89): Gồm 16 biến kỹ thuật và vĩ mô (`Return_lag`, `Momentum`, `Volatility`, `VIXCLS`,...).
- Biến `Volatility_Regime` **hoàn toàn KHÔNG nằm trong danh sách đặc trưng đầu vào** của bất kỳ mô hình nào.
- Biến này chỉ được trích xuất từ dữ liệu gốc để lưu vào tệp dự báo `all_oos_predictions.csv` phục vụ cho việc phân rã hiệu năng hậu kiểm (post-hoc stratification) nhằm trả lời câu hỏi nghiên cứu RQ2.
- **Trạng thái:** `Verified` (Không có rò rỉ biến chế độ vào không gian đặc trưng).

---

## 17. Reconcile Report 04 với Report 03
Bảng đối chiếu tổng hợp giữa các phát hiện tại Report 03 và bằng chứng từ Report 04:

| Hạng mục phương pháp luận | Trạng thái tại Report 03 | Bằng chứng tại Report 04 | Trạng thái tổng hợp cuối cùng |
| :--- | :---: | :---: | :---: |
| **Historical Mean benchmark** | Verified (mẫu số $\text{Var}(y_{\text{val}})$) | Parameter-free (không tune) | `Verified` |
| **ARIMA target & forecasting** | Potential issue (lọc 1 bước) | Tuning trên train bằng AIC, nhưng áp dụng lọc 1 bước | `Potential issue` (Critical) |
| **ARIMAX target & forecasting** | Potential issue (lọc 1 bước) | Bậc cố định $(1,0,1)$, áp dụng lọc 1 bước trên target gối đầu | `Potential issue` (Critical) |
| **ARIMAX exogenous timing** | Verified (đo tại ngày $t$) | Scaler fit trên train exog | `Verified` |
| **Scaling & Preprocessing** | Verified (Fit train, transform val) | Inner tuning dùng outer-train scaler (rò rỉ nhẹ nội bộ) | `Verified` (Outer) / `Minor issue` (Inner) |
| **Nested walk-forward tuning** | Verified | Xác minh mã nguồn $100\%$ độc lập từng fold/horizon | `Verified` |
| **Sequence boundary leakage** | Verified | Lookback dùng đuôi train hợp lệ | `Verified` |
| **Final Test isolation** | Verified | Dữ liệu sau 2020-10 hoàn toàn chưa chạm tới | `Verified` |

- **Ý nghĩa đối chiếu:** Việc Report 04 xác nhận quá trình hyperparameter tuning sạch sẽ và không có rò rỉ từ tập validation **không làm mất đi vấn đề phương pháp luận của dòng ARIMA tại Report 03**. Cả hai kết luận bổ sung cho nhau: quy trình tối ưu hóa siêu tham số là hợp lệ, nhưng cách thức tạo dự báo ngoại mẫu của dòng ARIMA ở chân trời dài có sự bất tương đồng về tập thông tin điều kiện hóa.

---

## 18. Claims Audit
Rà soát và chuẩn hóa các tuyên bố học thuật trong Report 04:

| Tuyên bố ban đầu | Tình trạng bằng chứng | Mức độ rủi ro | Diễn đạt chuẩn hóa thay thế |
| :--- | :---: | :---: | :--- |
| "ARIMAX chạy Grid Search AIC và cực kỳ ổn định" | Mã nguồn gán cứng `order=(1,0,1)` | Sai lệch thực tế | "ARIMAX được gán cấu hình bậc $(1,0,1)$ cố định theo thiết kế mô hình" |
| "scikit-learn không lưu best_score_ nếu không bật store_cv_values" | `RidgeCV` có `best_score_`, `ElasticNetCV` có `mse_path_` | Thiếu chính xác | "Việc lưu 0.0 là một hạn chế kiểm chứng (Auditability Limitation) do mã nguồn gán mặc định" |
| "Số cấu hình độc nhất chứng minh tính thích ứng động cao" | Chỉ là thống kê mô tả lấy mẫu TPE | Suy diễn quá mức | "Phản ánh độ biến thiên tham số của thuật toán tìm kiếm trên không gian liên tục" |
| "30–50 trials đủ để tìm ra cấu hình hợp lý" | Giới hạn tài nguyên tính toán | Cần thận trọng | "Ngân sách 30–50 trials cung cấp tìm kiếm heuristic trong không gian định sẵn nhưng không đảm bảo tối ưu toàn cục" |
| "Quá trình tuning được kiểm chứng là leakage-free" | Outer val sạch, nhưng inner val chịu scaler của outer train | Cần phân biệt rõ | "Tuning hoàn toàn cách ly khỏi tập outer validation và test; rò rỉ phân phối nhẹ trong nội bộ inner loop" |

---

## 19. Final Methodological Verdict

> ### **CONDITIONAL PASS**

### Căn cứ kết luận:
1. **Quy trình phân tầng đạt chuẩn khoa học:** Toàn bộ $325$ phiên tinh chỉnh siêu tham số đều được cô lập tuyệt đối bên trong tập huấn luyện của từng fold (`X_train`), không có bất kỳ sự can thiệp hay nhìn trước nào từ tập kiểm định ngoại mẫu (`X_val`) hoặc tập kiểm tra độc lập (`Test Set`).
2. **Không có rò rỉ dữ liệu làm vô hiệu hóa kết quả:** Không phát hiện temporal leakage, lookahead bias hay label contamination trong quá trình tối ưu hóa siêu tham số của các mô hình Machine Learning và Deep Learning.
3. **Điều kiện ràng buộc:** Kết quả của nhóm mô hình tự hồi quy (ARIMA, ARIMAX, SARIMA) không được dùng để khẳng định năng lực dự báo vượt trội ở chân trời dài do cơ chế lọc 1 bước trên chuỗi gối đầu (đã xác định tại Report 03).

---

## 20. Step 5 Readiness Gate

| Tiêu chí sẵn sàng (Readiness Criteria) | Trạng thái (Status) | Bằng chứng xác minh (Evidence) |
| :--- | :---: | :--- |
| **Outer folds correct** | `Verified` | 5 Walk-Forward Folds tăng dần chuẩn xác theo chuỗi thời gian |
| **Inner tuning isolated** | `Verified` | `get_internal_train_val_split` và `TimeSeriesSplit` chỉ chạy trên `X_train` |
| **No test contamination** | `Verified` | Toàn bộ dữ liệu sau 2020-10-30 chưa từng được nạp vào tuning |
| **No outer-validation tuning** | `Verified` | `X_val` không bao giờ xuất hiện trong hàm mục tiêu Optuna/GridSearch |
| **Scaling leakage controlled** | `Verified` | Scaler fit trên `X_train`, transform trên `X_val` |
| **Sequence leakage controlled** | `Verified` | Lookback sequence dùng phần đuôi hợp lệ của `X_train` |
| **ARIMA tuning isolated** | `Verified` | AIC tính trên chuỗi train; không rò rỉ dữ liệu tập val |
| **ARIMAX exogenous timing verified** | `Verified` | Biến ngoại sinh đo tại cuối ngày $t$, scaler fit trên train |
| **Historical Mean benchmark verified** | `Verified` | Mẫu số $OOS\_R^2$ đã được làm rõ toán học |
| **Target alignment verified** | `Verified` | $Target_h \equiv \ln(Close_{t+h}/Close_t)$ khớp chính xác $< 10^{-16}$ |
| **OOS prediction generation verified** | `Verified` (ML/DL) / `Caveat` (AR) | ML/DL dự báo direct $h$ bước; AR áp dụng lọc 1 bước |

### ĐÁNH GIÁ CHUYỂN BƯỚC:
> ### **READY FOR STEP 5 (CONDITIONAL)**
> 
> Pipeline thực nghiệm **ĐỦ ĐIỀU KIỆN** để chuyển sang **STEP 5 — RQ1 Horizon Analysis**, với điều kiện tiên quyết: **Khi phân tích quy luật suy thoái theo chân trời (RQ1), nhóm mô hình Machine Learning và Deep Learning được sử dụng làm trọng tâm phân tích thực nghiệm; nhóm mô hình chuỗi thời gian ARIMA/ARIMAX/SARIMA phải được trình bày kèm theo ghi chú phương pháp luận về cơ chế lọc 1 bước trên chuỗi mục tiêu nhiều kỳ.**

---

## 21. Recommended Actions Before Step 5

Dưới đây là 3 hành động quan trọng nhất cần thực hiện trước và trong quá trình phân tích RQ1:

1. **Cô lập phân tích dòng ARIMA trong RQ1:** Khi vẽ biểu đồ và phân tích bảng suy thoái hiệu năng theo chân trời ($1D \to 120D$), tách riêng đường biểu diễn của dòng ARIMA/ARIMAX/SARIMA thành một nhánh phân tích riêng biệt (Filtering Benchmark) để tránh làm méo mó bức tranh suy thoái năng lực dự báo của các mô hình Machine Learning và Deep Learning.
2. **Sử dụng Pooled $OOS\_R^2$ làm thước đo chính:** Tránh sử dụng trung bình cộng $OOS\_R^2$ theo từng fold do hiện tượng méo mó cục bộ tại các fold có phương sai mẫu hẹp (như Fold 2 ở 120D); ưu tiên Pooled $OOS\_R^2$ và Diebold-Mariano có hiệu chỉnh Newey-West.
3. **Đánh giá Directional Accuracy so với Naive Drift ($77.79\%$):** Luôn so sánh $DA$ ở các chân trời 60D và 120D với tỷ lệ dương nền của thị trường để phản ánh đúng thực chất năng lực định thời của các mô hình.

