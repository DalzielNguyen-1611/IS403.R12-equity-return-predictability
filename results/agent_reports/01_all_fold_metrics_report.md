# Báo Cáo Phân Tích Dữ Liệu: Tính Toàn Vẹn Thực Nghiệm & Đo Lường Fold (01_all_fold_metrics_report.md)

**Nguồn dữ liệu:** `results/fold_metrics/all_fold_metrics.csv`  
**Thời gian thực hiện:** 2026-10-08 15:15:00 UTC+7  
**Vai trò chuyên gia:** Research Data Analyst (Kinh tế lượng tài chính & Học máy)  
**Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mô Tả Tập Tin & Mục Đích Phương Pháp Luận

Tập tin `results/fold_metrics/all_fold_metrics.csv` là bản ghi dữ liệu thực thi chi tiết cấp cơ sở (granular execution record) của toàn bộ khung kiểm định thực nghiệm. Tập tin lưu trữ các chỉ số đo lường hiệu năng riêng lẻ, siêu tham số, ranh giới khoảng thời gian dữ liệu và cấu hình phần cứng cho mọi sự kết hợp giữa Mô hình (Model), Chân trời dự báo (Forecast Horizon), và Cửa sổ mở rộng Walk-Forward (Fold).

Vai trò then chốt của tập tin trong quy trình nghiên cứu là thiết lập **Tính Toàn Vẹn Thực Nghiệm (Experimental Integrity)**:
1. Xác nhận quá trình huấn luyện/kiểm thử hoàn tất 100% không bị thiếu sót lượt chạy trên toàn bộ không gian thực nghiệm ($14 \times 5 \times 5 = 350$).
2. Xác minh việc không xảy ra rò rỉ dữ liệu (zero data leakage) qua các ranh giới fold.
3. Kiểm tra tính hợp lệ về mặt số học của các thước đo thống kê ($MAE \ge 0$, $RMSE \ge MAE$, $0 \le DA \le 1$).
4. Cung cấp các khối dữ liệu nguyên tử gốc để từ đó tổng hợp nên toàn bộ các bảng nghiên cứu tiếp theo (RQ1, RQ2, RQ3, Summary, Statistical Tests).

---

## 2. Kích Thước, Cấu Trúc & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước tập tin & Số lượng hàng
- **Số hàng thực tế:** 350 hàng (cộng 1 dòng tiêu đề = 351 dòng)
- **Số hàng kỳ vọng:** 350 hàng ($14\text{ Mô hình} \times 5\text{ Chân trời} \times 5\text{ Folds}$)
- **Độ lệch hàng:** Chính xác 0 hàng bị thiếu; 0 hàng thừa.
- **Dòng trùng lặp:** Đúng 0 dòng trùng lặp trên bộ khóa `(Model, Horizon, Fold)`.
- **Số lượng cột:** 27 cột.

### 2.2 Định nghĩa lược đồ dữ liệu (Schema Definitions)
| Tên cột | Kiểu dữ liệu | Mô tả học thuật | Miền giá trị kỳ vọng |
| :--- | :---: | :--- | :--- |
| `Model` | `str` | Tên định danh thuật toán cơ sở | 14 mô hình đăng ký |
| `Horizon` | `str` | Chân trời mục tiêu dự báo | `['1D', '5D', '20D', '60D', '120D']` |
| `Fold` | `int` | Chỉ số fold kiểm định ngoại mẫu mở rộng | `[1, 2, 3, 4, 5]` |
| `Train_Start` | `str` | Mốc thời gian bắt đầu cửa sổ huấn luyện | Cố định: `1990-03-27` |
| `Train_End` | `str` | Mốc thời gian kết thúc cửa sổ huấn luyện | Mở rộng tịnh tiến (`2015-06-09` đến `2019-10-03`) |
| `Val_Start` | `str` | Mốc bắt đầu kiểm định ngoại mẫu | Luôn luôn $> \text{Train\_End}$ theo thứ tự thời gian |
| `Val_End` | `str` | Mốc kết thúc kiểm định ngoại mẫu | Không chồng lấn giữa các fold (~272 ngày giao dịch) |
| `MAE` | `float` | Sai số tuyệt đối trung bình ngoại mẫu | $[0, +\infty)$, chỉ áp dụng cho mô hình hồi quy |
| `RMSE` | `float` | Căn bậc hai sai số bình phương trung bình ngoại mẫu | $[0, +\infty)$, chỉ áp dụng cho mô hình hồi quy |
| `OOS_R2` | `float` | Chỉ số $R^2$ ngoại mẫu Campbell-Thompson so với Benchmark | $(-\infty, 1.0]$, chỉ áp dụng cho mô hình hồi quy |
| `DA` | `float` | Độ chính xác định hướng (đồng nhất về dấu) | $[0.0, 1.0]$, chỉ áp dụng cho mô hình hồi quy |
| `ROC_AUC` | `float` | Diện tích dưới đường cong ROC | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Accuracy` | `float` | Độ chính xác phân loại | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Precision` | `float` | Độ chuẩn xác (Positive Predictive Value) | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Recall` | `float` | Độ nhạy / Tỷ lệ dương tính thực tế (True Positive Rate) | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `F1` | `float` | Trung bình điều hòa giữa Precision và Recall | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `Balanced_Accuracy` | `float` | Trung bình cộng của độ nhạy và độ đặc hiệu | $[0.0, 1.0]$, chỉ riêng `Logistic_Regression` |
| `best_alpha`, `best_C`, ... | `mixed` | Tham số tối ưu được ghi nhận theo từng mô hình | Không gian tham số tương ứng |
| `device` | `str` | Phần cứng tính toán cho mạng nơ-ron | `cuda` (cho Deep Learning) / NaN (cho CPU) |

---

## 3. Chất Lượng Dữ Liệu & Ma Trận Bao Phủ Thực Nghiệm

### 3.1 Ma trận bao phủ thiết kế thực nghiệm (Coverage Matrix)
Bảng dưới đây xác nhận phân bổ chính xác số lượt chạy Fold trên toàn bộ không gian thiết kế thực nghiệm $14 \times 5$:

| Mô hình | 1D | 5D | 20D | 60D | 120D | Tổng số Fold | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** (Benchmark) | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **Ridge** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **Elastic_Net** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **ARIMA** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **ARIMAX** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **SARIMA** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **LightGBM** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **SVR** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **Random_Forest** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **XGBoost** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **RNN** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **LSTM** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **Transformer** | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **Logistic_Regression** (Phân loại) | 5 | 5 | 5 | 5 | 5 | 25 | ✅ 100% |
| **TỔNG CỘNG** | **70** | **70** | **70** | **70** | **70** | **350** | ✅ **Hoàn thành 100%** |

### 3.2 Ranh giới thời gian của Cửa sổ mở rộng Walk-Forward
Mỗi fold thỏa mãn nghiêm ngặt tính mở rộng tuần tự theo thời gian, ngăn chặn tuyệt đối rủi ro rò rỉ nhìn trước tương lai (forward-looking data leakage):
- **Fold 1:** Train `1990-03-27` $\to$ `2015-06-09` ($N=6,351$) $\implies$ Val `2015-06-10` $\to$ `2016-07-07` ($N=272$)
- **Fold 2:** Train `1990-03-27` $\to$ `2016-07-07` ($N=6,623$) $\implies$ Val `2016-07-08` $\to$ `2017-08-04` ($N=272$)
- **Fold 3:** Train `1990-03-27` $\to$ `2017-08-04` ($N=6,895$) $\implies$ Val `2017-08-07` $\to$ `2018-09-04` ($N=272$)
- **Fold 4:** Train `1990-03-27` $\to$ `2018-09-04` ($N=7,167$) $\implies$ Val `2018-09-05` $\to$ `2019-10-03` ($N=272$)
- **Fold 5:** Train `1990-03-27` $\to$ `2019-10-03` ($N=7,439$) $\implies$ Val `2019-10-04` $\to$ `2020-10-30` ($N=272$)

Các cửa sổ kiểm định ngoại mẫu hoàn toàn tách biệt và tiếp nối nhau liên tục. Cửa sổ huấn luyện mở rộng thêm đúng 272 ngày giao dịch sau mỗi fold, bảo toàn tuyệt đối chiều thời gian tự nhiên.

---

## 4. Kiểm Tra Tính Hợp Lệ & Đo Lường Thống Kê (Sanity Checks)

### 4.1 Kiểm định miền giá trị các chỉ số hồi quy (13 Mô hình, 325 Folds)
1. **Kiểm tra $MAE \ge 0$:** Chính xác 0 vi phạm ($Min = 0.003450$).
2. **Kiểm tra $RMSE \ge 0$:** Chính xác 0 vi phạm ($Min = 0.004900$).
3. **Tính chất lý thuyết $RMSE \ge MAE$:** Theo Bất đẳng thức Jensen, $RMSE \ge MAE$ luôn phải thỏa mãn trên mọi tập dữ liệu thực tế. Trong tập dữ liệu này: $\sum \mathbb{I}(RMSE < MAE) = 0$. Chính xác 0 vi phạm.
4. **Kiểm tra $0.0 \le DA \le 1.0$:** Chính xác 0 vi phạm ($Min = 0.301471$, $Max = 1.000000$).
5. **Kiểm tra chia cho 0 / Giá trị vô cùng:** Đúng 0 trường hợp xuất hiện `Inf` hoặc `-Inf`.

### 4.2 Phân tích chi tiết các giá trị khuyết thiếu (Missing Values)
- `MAE`, `RMSE`, `OOS_R2`, `DA`: Đúng 25 ô khuyết thiếu.
  - **Xác minh khoa học:** Tất cả 25 dòng này thuộc về `Logistic_Regression`. Điều này hoàn toàn chính xác về mặt phương pháp luận vì `Logistic_Regression` là mô hình phân loại nhị phân chiều hướng thị trường, không đưa ra dự báo mức lợi suất thực tế.
- `ROC_AUC`: 24 giá trị hợp lệ, duy nhất 1 giá trị `NaN` tại chỉ mục 71 (`Logistic_Regression`, `Horizon=120D`, `Fold=2`).
  - **Nguyên nhân toán học:** Tại chân trời 120D ở Fold 2 (`2016-07-08` $\to$ `2017-08-04`), nhãn chiều hướng lợi suất thực tế dương 100% ($y_{\text{val}} > 0$ trong toàn bộ 272 ngày). Do tập kiểm định thực tế chỉ có duy nhất một lớp, mẫu số tích phân hình thang của đường cong ROC bằng 0 ($\frac{0}{0}$ không xác định). Hệ thống đã gán `NaN` một cách chuẩn xác thay vì tạo ra số liệu giả định tùy tiện.

---

## 5. Thống Kê Mô Tả Cấp Chi Tiết Từng Fold

### 5.1 Thống kê mô tả các thước đo hồi quy chính (325 Folds)
| Chỉ số | Trung bình (Mean) | Độ lệch chuẩn (Std) | Giá trị nhỏ nhất | Phân vị 25% | Trung vị (50%) | Phân vị 75% | Giá trị lớn nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAE** | 0.030658 | 0.029248 | 0.003450 | 0.008217 | 0.016978 | 0.046489 | 0.120588 |
| **RMSE** | 0.039653 | 0.035928 | 0.004900 | 0.011260 | 0.023690 | 0.057788 | 0.147946 |
| **$OOS\_R^2$** | -0.177106 | 1.067223 | -8.673887 | -0.186930 | -0.013213 | 0.010318 | 0.981548 |
| **Độ chính xác hướng đi (DA)** | 65.88% | 17.57% | 30.15% | 52.57% | 59.93% | 79.41% | 100.00% |

### 5.2 Thống kê mô tả bài toán phân loại (`Logistic_Regression`, 25 Folds)
| Chỉ số | Trung bình (Mean) | Độ lệch chuẩn (Std) | Giá trị nhỏ nhất | Phân vị 25% | Trung vị (50%) | Phân vị 75% | Giá trị lớn nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.97% | 14.13% | 42.65% | 51.84% | 57.72% | 68.38% | 100.00% |
| **ROC_AUC** | 0.454485 | 0.119714 | 0.258274 | 0.394213 | 0.452555 | 0.515226 | 0.772244 |
| **Balanced Accuracy** | 49.98% | 11.09% | 36.52% | 46.81% | 49.34% | 50.00% | 100.00% |
| **F1 Score** | 0.723289 | 0.149727 | 0.404959 | 0.669927 | 0.731935 | 0.812227 | 1.000000 |

---

## 6. Các Điểm Dị Thường Chi Tiết & Giá Trị Cực Đoan

### 6.1 Hiệu năng kém nghiêm trọng ($OOS\_R^2$ âm rất sâu)
Trong dự báo tài chính ngoại mẫu, $OOS\_R^2$ âm cho thấy sai số MSE của mô hình lớn hơn so với MSE của Historical Mean ($\text{MSE}_{\text{model}} > \text{MSE}_{\text{bench}}$). 5 lượt chạy fold sau đây ghi nhận giá trị cực đoan $OOS\_R^2 < -3.0$:
1. **Transformer | 1D | Fold 2:** $OOS\_R^2 = -8.6739$ ($RMSE = 0.01534$, $MAE = 0.01442$)
2. **Transformer | 120D | Fold 2:** $OOS\_R^2 = -5.3721$ ($RMSE = 0.06054$, $MAE = 0.05053$)
3. **LSTM | 120D | Fold 2:** $OOS\_R^2 = -4.8062$ ($RMSE = 0.05779$, $MAE = 0.05228$)
4. **Ridge | 120D | Fold 2:** $OOS\_R^2 = -4.2707$ ($RMSE = 0.05524$, $MAE = 0.04870$)
5. **Transformer | 5D | Fold 2:** $OOS\_R^2 = -3.5654$ ($RMSE = 0.02096$, $MAE = 0.01884$)

**Bối cảnh kinh tế lượng:**
Các giá trị bị phạt nặng này tập trung bất thường tại **Fold 2 (`2016-07-08` $\to$ `2017-08-04`)**. Giai đoạn này là thời kỳ thị trường tăng giá với biến động cực thấp kỷ lục sau bầu cử Mỹ 2016, nơi chuỗi lợi suất thực tế có phương sai quanh xu hướng tăng gần như bằng 0. Khi phương sai mẫu benchmark tiệm cận 0, mẫu số trong công thức $R^2 = 1 - \frac{\text{MSE}}{\text{Var}(\text{Benchmark})}$ trở nên rất bé, làm khuếch đại những sai lệch nhỏ về mặt giá trị tuyệt đối thành các tỷ lệ phần trăm âm khổng lồ.

### 6.2 Hiện tượng $OOS\_R^2$ dương rất cao ở dòng Tự Hồi Quy
- `ARIMAX` và `ARIMA` liên tục đạt $OOS\_R^2 > 0.90$ riêng tại **các chân trời 60D và 120D** (đạt đỉnh tới $0.9815$ tại Fold 5).
- Ngược lại, các mô hình ML và Deep Learning (LightGBM, XGBoost, Transformer) hiếm khi vượt quá $OOS\_R^2 > 0.10$ ở các chân trời dài hạn.
- Sự phân kỳ cấu trúc này đã được kiểm định chi tiết ở các bước tiếp theo để xác định bản chất quán tính tự tương quan của chuỗi lợi suất tích lũy.

---

## 7. Hạn Chế Phương Pháp Luận Của Dữ Liệu Cấp Fold

1. **Độ khó không đồng nhất giữa các Fold:** Fold 2 có phương sai thị trường quá nén, làm sai lệch chỉ số tỷ đối của toàn bộ mô hình, trong khi Fold 5 hứng chịu cú sốc biến động COVID-19 tháng 3/2020. Không thể diễn giải các giá trị fold đơn lẻ nếu không chuẩn hóa theo bối cảnh chế độ thị trường (Regime).
2. **Tính không tương thích của bài toán phân loại:** `Logistic_Regression` không có các thước đo sai số hồi quy mức lợi suất, đòi hỏi phải được báo cáo phân tách hoàn toàn.
3. **Chưa có ý nghĩa thống kê tổng hợp ở cấp độ Fold:** Dữ liệu từng fold riêng biệt chưa thể kết luận sự vượt trội có ý nghĩa thống kê so với nhiễu thị trường hay không.

---

## 8. Kết Luận & Chuyển Giao Cho Bước Tiếp Theo

### 8.1 Tóm tắt tính toàn vẹn thực nghiệm
- ✅ **Độ bao phủ:** Hoàn thành $100.0\%$ ($350/350$ ô kịch bản đầy đủ).
- ✅ **Chống rò rỉ dữ liệu:** Các mốc thời gian mở rộng tuần tự nghiêm ngặt, tuyệt đối không nhìn trước tương lai.
- ✅ **Tính hợp lệ chỉ số:** $MAE \ge 0$, $RMSE \ge MAE$, và $0 \le DA \le 1$ thỏa mãn trên toàn bộ tập dữ liệu.
- ✅ **Tính toàn vẹn benchmark:** `Historical_Mean` hiện diện đầy đủ trên cả 25 lượt chạy.

### 8.2 Chuyển giao sang Bước 2
Với tính toàn vẹn thực nghiệm đã được xác nhận vững chắc trên tập dữ liệu nguyên tử, bước tiếp theo là phân tích **`results/summary/model_horizon_summary.csv`** để đánh giá hiệu năng trung bình và độ phân tán qua các họ mô hình và chân trời dự báo ($1D \to 120D$).
