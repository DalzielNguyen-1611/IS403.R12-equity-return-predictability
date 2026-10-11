# Báo Cáo Kiểm Toán Dữ Liệu: Tính Toàn Vẹn Thực Nghiệm & Đo Lường Cấp Fold (01_all_fold_metrics_report.md)

**Tập tin nguồn:** `results/fold_metrics/all_fold_metrics.csv`
**Thời điểm thực hiện kiểm toán:** 2026-10-08 16:15:00 UTC+7
**Vai trò:** Research Data Analyst (Financial Econometrics & Machine Learning)
**Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*
**Trạng thái phiên bản:** Đã kiểm toán phương pháp luận (Methodologically Audited)

---

## 1. Mô Tả Tập Tin & Mục Đích Kiểm Toán (File Description & Audit Purpose)

Tập tin `results/fold_metrics/all_fold_metrics.csv` là bản ghi chi tiết cấp cơ sở của toàn bộ quy trình kiểm định ngoại mẫu mở rộng (walk-forward expanding window). Tập tin lưu trữ các chỉ số hiệu năng riêng lẻ, các siêu tham số được chọn, ranh giới chu kỳ huấn luyện/kiểm định và cấu hình phần cứng cho từng tổ hợp:

$$\text{Model} \times \text{Horizon} \times \text{Fold}$$

Mục tiêu của cuộc kiểm toán này là thiết lập **Tính Toàn Vẹn Thực Nghiệm (Experimental Integrity)** và rà soát lại các nhận định ban đầu nhằm tránh diễn giải quá mức (over-interpretation):
1. Xác minh mức độ bao phủ đầy đủ $100\%$ không bị thất lạc kịch bản ($14 \times 5 \times 5 = 350$ lượt chạy).
2. Kiểm toán ranh giới thời gian giữa các fold để phát hiện rò rỉ dữ liệu chuỗi thời gian (temporal leakage).
3. Kiểm tra tính hợp lệ về mặt số học của các thước đo ($MAE \ge 0$, $RMSE \ge MAE$, $0 \le DA \le 1$).
4. Kiểm toán độc lập các giá trị cực đoan của $OOS\_R^2$ ($< -3.0$ và $> 0.90$).
5. Xác lập các cờ cảnh báo phương pháp luận (Critical Methodological Flags) đối với họ mô hình Tự hồi quy (ARIMA, ARIMAX, SARIMA).

> **Lưu ý phương pháp luận:** Báo cáo này tuyệt đối không đưa ra kết luận nghiên cứu sau cùng về các câu hỏi RQ1, RQ2, RQ3 mà chỉ đánh giá tính hợp thức của tập dữ liệu cơ sở.

---

## 2. Kích Thước, Cấu Trúc & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước tập tin & Số lượng quan sát
- **Số hàng thực tế:** 350 hàng dữ liệu (cộng 1 dòng tiêu đề = 351 dòng).
- **Số hàng kỳ vọng:** 350 hàng ($14 \text{ Mô hình} \times 5 \text{ Chân trời} \times 5 \text{ Folds}$).
- **Độ lệch hàng:** Chính xác 0 hàng thiếu; 0 hàng thừa.
- **Dòng trùng lặp:** Đúng 0 dòng trùng lặp trên bộ khóa `(Model, Horizon, Fold)`.
- **Số lượng cột:** 27 cột.

### 2.2 Định nghĩa lược đồ dữ liệu (Schema Definitions)
| Tên cột | Kiểu dữ liệu | Mô tả học thuật | Miền giá trị kỳ vọng |
| :--- | :---: | :--- | :--- |
| `Model` | `str` | Tên định danh thuật toán cơ sở | 14 mô hình đăng ký |
| `Horizon` | `str` | Chân trời mục tiêu dự báo | `['1D', '5D', '20D', '60D', '120D']` |
| `Fold` | `int` | Chỉ số fold kiểm định ngoại mẫu mở rộng | `[1, 2, 3, 4, 5]` |
| `Train_Start` | `str` | Mốc bắt đầu cửa sổ huấn luyện | Cố định: `1990-03-27` |
| `Train_End` | `str` | Mốc kết thúc cửa sổ huấn luyện | Mở rộng tịnh tiến (`2015-06-09` đến `2019-10-03`) |
| `Val_Start` | `str` | Mốc bắt đầu kiểm định ngoại mẫu | Ngày giao dịch tiếp theo liền kề sau `Train_End` |
| `Val_End` | `str` | Mốc kết thúc kiểm định ngoại mẫu | Khoảng 272 ngày giao dịch không chồng lấn |
| `MAE` | `float` | Sai số tuyệt đối trung bình ngoại mẫu | $[0, +\infty)$, chỉ áp dụng cho hồi quy |
| `RMSE` | `float` | Căn bậc hai sai số bình phương trung bình | $[0, +\infty)$, chỉ áp dụng cho hồi quy |
| `OOS_R2` | `float` | Chỉ số $R^2$ ngoại mẫu được ghi nhận | $(-\infty, 1.0]$, chỉ áp dụng cho hồi quy |
| `DA` | `float` | Độ chính xác định hướng (Directional Accuracy) | $[0.0, 1.0]$, chỉ áp dụng cho hồi quy |
| `ROC_AUC` | `float` | Diện tích dưới đường cong ROC | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Accuracy` | `float` | Độ chính xác phân loại nhị phân | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Precision` | `float` | Độ chuẩn xác dương tính | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Recall` | `float` | Độ nhạy (True Positive Rate) | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `F1` | `float` | Điểm F1 trung bình điều hòa | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Balanced_Accuracy` | `float` | Độ chính xác cân bằng | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `best_alpha`, `best_C`, ... | `mixed` | Tham số tối ưu được lưu trữ theo mô hình | Không gian tham số tương ứng |
| `device` | `str` | Phần cứng tính toán cho mô hình mạng | `cuda` (Deep Learning) / NaN (CPU) |

---

## 3. Ma Trận Bao Phủ Thực Nghiệm & Kiểm Toán Rò Rỉ Dữ Liệu (Coverage & Leakage Audit)

### 3.1 Ma trận bao phủ thiết kế thực nghiệm ($14 \times 5 \times 5$)
Bảng dưới đây xác nhận mức độ bao phủ đầy đủ 100% qua 350 lượt chạy:

| Mô hình | 1D | 5D | 20D | 60D | 120D | Tổng số Fold | Trạng thái bao phủ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** (Benchmark) | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **Ridge** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **Elastic_Net** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **ARIMA** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **ARIMAX** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **SARIMA** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **LightGBM** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **SVR** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **Random_Forest** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **XGBoost** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **RNN** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **LSTM** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **Transformer** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **Logistic_Regression** (Phân loại) | 5 | 5 | 5 | 5 | 5 | 25 | ✅ PASS |
| **TỔNG CỘNG** | **70** | **70** | **70** | **70** | **70** | **350** | ✅ **100% COMPLETE** |

### 3.2 Kiểm toán ranh giới thời gian & Xác minh rò rỉ dữ liệu (Data Leakage Audit)
Ranh giới thời gian của 5 Folds mở rộng (expanding windows):
- **Fold 1:** Train `1990-03-27` $\to$ `2015-06-09` ($N=6,351$) $\implies$ Val `2015-06-10` $\to$ `2016-07-07` ($N=272$)
- **Fold 2:** Train `1990-03-27` $\to$ `2016-07-07` ($N=6,623$) $\implies$ Val `2016-07-08` $\to$ `2017-08-04` ($N=272$)
- **Fold 3:** Train `1990-03-27` $\to$ `2017-08-04` ($N=6,895$) $\implies$ Val `2017-08-07` $\to$ `2018-09-04` ($N=272$)
- **Fold 4:** Train `1990-03-27` $\to$ `2018-09-04` ($N=7,167$) $\implies$ Val `2018-09-05` $\to$ `2019-10-03` ($N=272$)
- **Fold 5:** Train `1990-03-27` $\to$ `2019-10-03` ($N=7,439$) $\implies$ Val `2019-10-04` $\to$ `2020-10-30` ($N=272$)

#### Kết luận kiểm toán rò rỉ dữ liệu (Audit Conclusion on Leakage):
> **Tuyên bố phương pháp luận thận trọng:**
> *"No evidence of temporal leakage was identified from fold boundaries in this file."*
> *(Không tìm thấy bằng chứng về rò rỉ thời gian từ các ranh giới fold trong tập tin này).*

Tuy nhiên, **tuyệt đối không được kết luận "zero data leakage" hay "hoàn toàn không có rò rỉ"** đối với toàn bộ pipeline thực nghiệm chỉ dựa trên các cột ngày tháng `Train_Start`, `Train_End`, `Val_Start`, `Val_End`.

#### Các loại rò rỉ dữ liệu CHƯA THỂ XÁC MINH từ tập tin này một mình:
1. **Scaler fitting leakage:** Liệu các phép chuẩn hóa (StandardScaler, MinMaxScaler) có được `fit` riêng biệt chỉ trên tập huấn luyện hay bị fit gộp toàn bộ tập dữ liệu?
2. **Hyperparameter tuning leakage:** Quá trình tìm kiếm siêu tham số tối ưu (Optuna, GridSearchCV) có sử dụng tập validation ngoại mẫu hay chỉ dùng cross-validation nội bộ bên trong tập train?
3. **Feature selection leakage:** Việc lựa chọn đặc trưng hoặc biến trễ có sử dụng thông tin thống kê của toàn bộ chu kỳ hay không?
4. **Sequence construction leakage:** Các chuỗi trượt (rolling sequence) cho mạng RNN/LSTM/Transformer có bị tạo trước khi chia fold, dẫn đến điểm biên bị chồng lấn qua ranh giới train/val hay không?
5. **Regime construction leakage:** Nhãn chế độ biến động (`Regime`) có được xác định chỉ từ thông tin lịch sử trễ tại thời điểm $t$ hay dùng ngưỡng phân vị tính trên toàn bộ mẫu?
6. **Exogenous-variable availability:** Đối với các biến ngoại sinh (như VIX, High-Low range trong ARIMAX), dữ liệu có thực sự sẵn có tại thời điểm đưa ra dự báo hay sử dụng giá trị cùng thời điểm với mục tiêu tương lai?

*Tất cả các rủi ro này bắt buộc phải được kiểm toán chi tiết ở các tập tin kế tiếp (báo cáo siêu tham số và báo cáo dự báo).*

---

## 4. Kiểm Tra Tính Hợp Lệ Của Các Thước Đo (Metric Sanity Checks)

### 4.1 Kiểm định miền giá trị các thước đo hồi quy (13 Mô hình, 325 Folds)
1. **$MAE \ge 0$:** Đạt yêu cầu $100\%$ ($Min = 0.003450$, 0 vi phạm).
2. **$RMSE \ge 0$:** Đạt yêu cầu $100\%$ ($Min = 0.004900$, 0 vi phạm).
3. **Đặc tính lý thuyết $RMSE \ge MAE$:** Theo Bất đẳng thức Jensen, căn bậc hai của kỳ vọng bình phương luôn lớn hơn hoặc bằng kỳ vọng giá trị tuyệt đối. Trên toàn bộ 325 folds: $\sum \mathbb{I}(RMSE < MAE) = 0$ (Đạt $100\%$, 0 vi phạm).
4. **$0.0 \le DA \le 1.0$:** Đạt yêu cầu $100\%$ ($Min = 0.301471$, $Max = 1.000000$).
5. **Giá trị vô cùng (Infinite values):** Đúng 0 trường hợp xuất hiện `Inf` hoặc `-Inf`.

### 4.2 Giải trình các ô khuyết thiếu (Missing Value Audit)
- **25 ô khuyết thiếu ở `MAE`, `RMSE`, `OOS_R2`, `DA`:**
  Tất cả 25 ô này thuộc về `Logistic_Regression` ($5 \text{ horizons} \times 5 \text{ folds}$). Đây là đặc tính kỹ thuật **hoàn toàn kỳ vọng và chính xác (expected)**, vì Logistic Regression là mô hình phân loại hướng đi nhị phẩn, không dự báo mức lợi suất thực tế liên tục.
- **1 ô khuyết thiếu ở `ROC_AUC` (`Logistic_Regression`, `Horizon=120D`, `Fold=2`):**
  Trong Fold 2 ở chân trời 120D (`2016-07-08` $\to$ `2017-08-04`), nhãn thực tế $y_{\text{val}}$ dương $100\%$ ($272/272$ ngày có lợi suất tích lũy $> 0$). Khi tập kiểm định chỉ chứa duy nhất một lớp (single-class ground truth), mẫu số trong công thức tích phân ROC-AUC bị triệt tiêu ($\frac{0}{0}$ không xác định). Giá trị `NaN` ở đây là **kết quả toán học hoàn toàn chuẩn xác (mathematically expected)**. Tuyệt đối **không được thay thế NaN bằng 0.5** vì điều đó sẽ tạo ra giả định sai lệch về năng lực phân loại.

---

## 5. Thống Kê Mô Tả Dữ Liệu Cấp Fold (Descriptive Statistics)

### 5.1 Thống kê mô tả các thước đo hồi quy (325 Folds)
| Chỉ số | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAE** | 0.030658 | 0.029248 | 0.003450 | 0.008217 | 0.016978 | 0.046489 | 0.120588 |
| **RMSE** | 0.039653 | 0.035928 | 0.004900 | 0.011260 | 0.023690 | 0.057788 | 0.147946 |
| **$OOS\_R^2$** | -0.177106 | 1.067223 | -8.673887 | -0.186930 | -0.013213 | 0.010318 | 0.981548 |
| **Directional Accuracy (DA)** | 65.88% | 17.57% | 30.15% | 52.57% | 59.93% | 79.41% | 100.00% |

### 5.2 Thống kê mô tả bài toán phân loại (`Logistic_Regression`, 25 Folds)
| Chỉ số | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.97% | 14.13% | 42.65% | 51.84% | 57.72% | 68.38% | 100.00% |
| **ROC_AUC** | 0.454485 | 0.119714 | 0.258274 | 0.394213 | 0.452555 | 0.515226 | 0.772244 |
| **Balanced Accuracy** | 49.98% | 11.09% | 36.52% | 46.81% | 49.34% | 50.00% | 100.00% |
| **F1 Score** | 0.723289 | 0.149727 | 0.404959 | 0.669927 | 0.731935 | 0.812227 | 1.000000 |

---

## 6. Kiểm Toán Các Giá Trị Cực Đoan Của $OOS\_R^2$ (Audit of Extreme Values)

### 6.1 Kiểm toán các trường hợp $OOS\_R^2 < -3.0$
Trong tập dữ liệu, có đúng 10 trường hợp ghi nhận $OOS\_R^2 < -3.0$. Dưới đây là bảng tái tính toán độc lập các thành phần sai số:

| Model | Horizon | Fold | Reported $OOS\_R^2$ | Model MSE | Denom Variance | Bench MSE | Recomputed $R^2_{\text{var}}$ | Recomputed $R^2_{\text{bench}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 120D | 2 | -3.1175 | 0.002368 | 0.000575 | 0.002368 | -3.1175 | 0.0000 |
| **Ridge** | 120D | 2 | -5.6602 | 0.003831 | 0.000575 | 0.002368 | -5.6602 | -0.6176 |
| **Elastic_Net** | 120D | 2 | -3.2449 | 0.002441 | 0.000575 | 0.002368 | -3.2449 | -0.0309 |
| **LightGBM** | 120D | 1 | -3.0177 | 0.015222 | 0.003789 | 0.003827 | -3.0177 | -2.9775 |
| **XGBoost** | 120D | 2 | -3.1410 | 0.002382 | 0.000575 | 0.002368 | -3.1410 | -0.0057 |
| **RNN** | 120D | 2 | -6.8896 | 0.004538 | 0.000575 | 0.002368 | -6.8896 | -0.9161 |
| **LSTM** | 120D | 2 | -4.8062 | 0.003339 | 0.000575 | 0.002368 | -4.8062 | -0.4101 |
| **Transformer** | 1D | 2 | -8.6739 | 0.000235 | 0.000024 | 0.000024 | -8.6739 | -8.6416 |
| **Transformer** | 5D | 2 | -3.5654 | 0.000440 | 0.000096 | 0.000098 | -3.5654 | -3.5078 |
| **Transformer** | 120D | 2 | -5.3721 | 0.003665 | 0.000575 | 0.002368 | -5.3721 | -0.5476 |

#### Phân tích khách quan dựa thuần túy trên số liệu:
1. **Công thức toán học thực tế được sử dụng trong tập tin:**
   Giá trị `OOS_R2` trong file được tính theo công thức:

$$R^2 = 1 - \frac{\text{Model\_MSE}}{\text{Var}(y_{\text{val}})}$$

   với mẫu số là phương sai mẫu của tập validation ($\text{Var}(y_{\text{val}})$), thay vì công thức Campbell-Thompson chuẩn mực sử dụng sai số của mô hình benchmark ($1 - \frac{\text{Model\_MSE}}{\text{Bench\_MSE}}$). Điều này được minh chứng khi cột `Reported OOS_R2` khớp chính xác 100% với `Recomputed R2_var`.
2. **Hiện tượng tập trung tại Fold 2:**
   9/10 trường hợp âm sâu tập trung tại **Fold 2**.
   *Mô tả thuần dữ liệu:* **"Fold 2 exhibits unusually low realized target variance"** (Phương sai thực tế của lợi suất tại Fold 2 rất thấp: $\text{Var}(y) = 0.000575$ tại 120D so với mức $0.003789 - 0.015636$ ở các Fold khác).
   > **Kết luận kiểm toán:**
   > *"Potentially explained by a small benchmark-error denominator, but this cannot be established from the current file alone."*
   > *(Hiện tượng này có khả năng được giải thích bởi mẫu số sai số chuẩn quá nhỏ, nhưng điều này chưa thể khẳng định chắc chắn chỉ từ tập tin hiện tại mà cần đối chiếu chuỗi dự báo chi tiết).*

---

### 6.2 Kiểm toán các trường hợp $OOS\_R^2 > 0.90$ (Tập trung ở ARIMA / ARIMAX / SARIMA)
Trong toàn bộ 350 lượt chạy, có đúng **33 trường hợp** đạt $OOS\_R^2 > 0.90$. Toàn bộ 33 trường hợp này thuộc về **3 mô hình Tự hồi quy**:
- **ARIMA (10 trường hợp):** 60D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9234, 0.9547]$); 120D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9178, 0.9618]$).
- **ARIMAX (13 trường hợp):** 20D (Folds 1, 4, 5: $R^2 \in [0.9211, 0.9322]$); 60D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9492, 0.9754]$); 120D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9400, 0.9815]$).
- **SARIMA (10 trường hợp):** 60D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9232, 0.9548]$); 120D (Folds 1, 2, 3, 4, 5: $R^2 \in [0.9164, 0.9605]$).
- Không có bất kỳ mô hình Machine Learning hoặc Deep Learning nào khác đạt $OOS\_R^2 > 0.90$.

Dưới đây là bảng trích mẫu các trường hợp tiêu biểu:

| Model | Horizon | Fold | Reported $OOS\_R^2$ | Model MSE | Denom Variance | Bench MSE | Recomputed $R^2_{\text{var}}$ | Recomputed $R^2_{\text{bench}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA** | 60D | 1 | 0.9432 | 0.000193 | 0.003404 | 0.003449 | 0.9432 | 0.9440 |
| **ARIMA** | 60D | 2 | 0.9234 | 0.000047 | 0.000619 | 0.000841 | 0.9234 | 0.9436 |
| **ARIMA** | 120D | 1 | 0.9495 | 0.000191 | 0.003789 | 0.003827 | 0.9495 | 0.9500 |
| **ARIMA** | 120D | 2 | 0.9178 | 0.000047 | 0.000575 | 0.002368 | 0.9178 | 0.9800 |
| **ARIMA** | 120D | 5 | 0.9618 | 0.000542 | 0.014204 | 0.017659 | 0.9618 | 0.9693 |
| **ARIMAX**| 60D | 1 | 0.9711 | 0.000098 | 0.003404 | 0.003449 | 0.9711 | 0.9715 |
| **ARIMAX**| 120D | 5 | 0.9815 | 0.000262 | 0.014204 | 0.017659 | 0.9815 | 0.9852 |
| **SARIMA**| 120D | 5 | 0.9605 | 0.000561 | 0.014204 | 0.017659 | 0.9605 | 0.9682 |

> **Cảnh báo học thuật nghiêm ngặt:**
> **Tuyệt đối KHÔNG kết luận rằng đây là năng lực dự báo thực chất (genuine predictability)** trước khi kiểm tra công thức xây dựng mục tiêu và cơ chế tạo dự báo.
> Cần làm rõ:
> - **ARIMA target:** Biến mục tiêu đưa vào mô hình là gì?
> - **ARIMAX target:** Biến mục tiêu và biến ngoại sinh được xử lý như thế nào?
> - **SARIMA target:** Chu kỳ mùa vụ và mục tiêu được thiết lập ra sao?
> - **Forecast construction for 1D/5D/20D/60D/120D:** Mô hình dự báo là direct forecast hay recursive/filtering forecast?
>
> **Kết luận kiểm toán:**
> *"Target formulation cannot be verified from all_fold_metrics.csv alone."*
> *(Công thức xây dựng mục tiêu không thể xác minh chỉ từ tập tin all_fold_metrics.csv mà bắt buộc phải kiểm toán mã nguồn và dữ liệu dự báo ở các bước tiếp theo).*

---

## 7. Cờ Cảnh Báo Phương Pháp Luận Then Chốt (Critical Methodological Flags)

Phần này ghi nhận các vấn đề phương pháp luận trọng yếu cần được gắn cờ theo dõi đặc biệt:

### 🚩 FLAG 1: ARIMA / ARIMAX / SARIMA Horizon Comparability — `[HIGH PRIORITY AUDIT]`
- **Mức độ ưu tiên:** **CAO NHẤT (HIGH PRIORITY)**.
- **Vấn đề:** Các mô hình kinh tế lượng chuỗi thời gian đạt $OOS\_R^2 > 0.95$ ở các chân trời dài hạn 60D và 120D. Nếu mô hình được áp dụng cơ chế 1-step filtering trên chuỗi lợi suất gộp chồng lấn $y_{t} = \ln(Close_{t+h}/Close_t)$, giá trị trễ $y_{t-1} = \ln(Close_{t+h-1}/Close_{t-1})$ sẽ chứa thông tin về mức giá tương lai $Close_{t+h-1}$ vốn chưa xuất hiện tại thời điểm dự báo $t$.
- **Hành động bắt buộc:** Phải tiến hành kiểm toán chuyên sâu mã nguồn huấn luyện tại `src/training/baselines/arima/` và chuỗi dự báo trong `all_oos_predictions.csv` trước khi thừa nhận kết quả này.

### 🚩 FLAG 2: Mẫu số tính $OOS\_R^2$ không đồng nhất với chuẩn Campbell-Thompson
- **Vấn đề:** Cột `OOS_R2` trong file sử dụng $\text{Var}(y_{\text{val}})$ làm mẫu số, khiến các mô hình bị phạt âm rất nặng khi phương sai tập kiểm định nhỏ. Cần tính toán song song chỉ số Campbell-Thompson chuẩn tắc sử dụng $\text{MSE}(\text{Historical\_Mean})$ làm chuẩn so sánh trong các báo cáo phân tích tiếp theo.

---

## 8. Hạn Chế Của Dữ Liệu Cấp Fold (Limitations of Granular Fold Data)

1. **Tính biến thiên của mẫu số giữa các Fold:** Phương sai của chuỗi thực tế biến động mạnh giữa các fold (từ $0.000575$ ở Fold 2 lên $0.015636$ ở Fold 5), khiến cho giá trị $R^2$ đơn lẻ tại một fold không thể dùng để so sánh trực tiếp độ khó giữa các thời kỳ.
2. **Không thể xác minh tính toàn vẹn của chuỗi đặc trưng:** Tập tin chỉ ghi nhận kết quả cuối cùng, không phản ánh các bước xử lý biến đổi chuỗi đầu vào.

---

## 9. Kết Luận Kiểm Toán & Kế Hoạch Bước Tiếp Theo (Audit Conclusions)

- Tập tin `all_fold_metrics.csv` đã vượt qua các kiểm định về tính toàn vẹn cấu trúc ($350/350$ lượt chạy đầy đủ, không trùng lặp, giá trị số học hợp lệ).
- Báo cáo đã điều chỉnh toàn bộ các nhận định mang tính phỏng đoán thị trường, thay thế bằng các mô tả thuần số liệu khách quan.
- Báo cáo đã xác lập cờ kiểm toán ưu tiên cao nhất (`HIGH PRIORITY AUDIT`) cho họ mô hình Tự hồi quy và chuyển giao vấn đề này sang các bước kiểm toán tiếp theo.

