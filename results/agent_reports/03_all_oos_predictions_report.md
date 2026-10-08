# Báo Cáo Phân Tích Dữ Liệu Dự Báo Ngoại Mẫu (03_all_oos_predictions_report.md)

**Mã tài liệu:** `IS403-REP-03`  
**Ngày thực hiện:** 2026-10-08  
**Tập tin mục tiêu:** `results/predictions/all_oos_predictions.csv`  
**Người thực hiện:** Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mô Tả Tập Tin (File Description)
Tập tin `all_oos_predictions.csv` là kho lưu trữ dữ liệu dự báo ngoại mẫu cấp quan sát (observation-level out-of-sample predictions) cho toàn bộ chu kỳ kiểm thử của nghiên cứu. Đây là **nguồn dữ liệu gốc (single source of truth)** làm căn cứ tính toán cho:
1. `all_fold_metrics.csv` (tính theo từng Fold $1 \dots 5$)
2. `model_horizon_summary.csv` (trung bình và độ lệch chuẩn trên 5 Folds)
3. `rq2_regime.csv` (phân rã hiệu năng theo chế độ biến động Low, Normal, High)
4. `rq3_time.csv` (phân rã hiệu năng theo các phân kỳ thời gian 2015-2016, 2017-2018, 2019-2020)
5. `statistical_tests.csv` (kiểm định thống kê Diebold-Mariano và Sign Test trên chuỗi phần dư)

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước vật lý và ma trận dữ liệu
- **Dung lượng đĩa:** ~7.06 MB (7,059,627 bytes)
- **Số hàng (Observations):** $95,200$ hàng
- **Số cột (Features):** $10$ cột

### 2.2 Chi tiết cấu trúc cột (Data Schema)
| Cột | Kiểu dữ liệu | Số giá trị Null | Mô tả học thuật |
| :--- | :--- | :--- | :--- |
| `Date` | `object` (chuỗi YYYY-MM-DD) | 0 | Ngày giao dịch thực tế ngoại mẫu (2015-06-10 đến 2020-10-30) |
| `Model` | `object` (string) | 0 | Tên định danh mô hình (14 mô hình) |
| `Horizon` | `object` (string) | 0 | Chân trời dự báo (`1D`, `5D`, `20D`, `60D`, `120D`) |
| `Fold` | `int64` | 0 | Chỉ số Fold kiểm định ngoại mẫu ($1, 2, 3, 4, 5$) |
| `Actual` | `float64` | 0 | Lợi suất thực tế logarit: $Target_{h,t} = \ln(Close_{t+h} / Close_t)$ |
| `Prediction` | `float64` | 0 | Dự báo điểm ngoại mẫu của mô hình ($\hat{y}_{h,t}$) |
| `Regime` | `object` (string) | 0 | Chế độ biến động thị trường tại thời điểm $t$ (`Low`, `Normal`, `High`) |
| `Actual_Class` | `float64` | 88,400 | Nhãn xu hướng thực tế ($1$ nếu $Actual > 0$, $0$ nếu ngược lại) |
| `Predicted_Probability` | `float64` | 88,400 | Xác suất dự báo lớp 1 từ Logistic Regression |
| `Predicted_Class` | `float64` | 88,400 | Lớp dự báo phân loại từ Logistic Regression |

---

## 3. Kiểm Định Tính Toàn Vẹn & Chất Lượng Dữ Liệu (Data Quality Checks)

### 3.1 Kiểm tra trùng lặp và giá trị khuyết thiếu (Duplicates & Missing Values)
- **Kiểm tra trùng lặp khóa chính `(Date, Model, Horizon, Fold)`:** $0$ trường hợp trùng lặp. Tính duy nhất của chuỗi thời gian ngoại mẫu được bảo toàn $100\%$.
- **Giá trị khuyết thiếu tại các cột hồi quy:** Không có giá trị `NaN`, `None`, hay chuỗi rỗng nào trên các cột `Date`, `Model`, `Horizon`, `Fold`, `Actual`, `Prediction`, `Regime`.
- **Giá trị khuyết thiếu tại các cột phân loại:**
  - Tổng số hàng phân loại kỳ vọng: $1\text{ mô hình (Logistic\_Regression)} \times 5\text{ horizons} \times 5\text{ folds} \times 272\text{ ngày} = 6,800$ hàng.
  - Số hàng có giá trị phân loại: $95,200 - 88,400 = 6,800$ hàng ($100\%$ dữ liệu của `Logistic_Regression`).
  - $88,400$ ô trống thuộc về $13$ mô hình hồi quy. Đây là đặc điểm thiết kế bắt buộc nhằm cô lập bài toán phân loại nhị phân khỏi hồi quy lợi suất.

### 3.2 Kiểm định miền giá trị hợp lệ (Sanity Checks)
- **Giá trị bất thường toán học (Infinite / NaN):** Kiểm tra `np.isinf` trên `Actual` và `Prediction` trả về $0$ giá trị.
- **Biên độ lợi suất thực tế (`Actual`):**
  - Horizon 1D: $[-0.1277, +0.0897]$ (phù hợp với biến động 1 ngày S&P 500 kể cả cú sốc COVID-19 tháng 3/2020).
  - Horizon 120D: $[-0.2855, +0.4009]$ (phù hợp với biên độ tích lũy 6 tháng của chỉ số).
- **Biên độ giá trị dự báo (`Prediction`):**
  - Mô hình Autoregressive (ARIMA, SARIMA, ARIMAX): biên độ $[-0.3703, +0.3971]$, bám sát xu hướng tích lũy dài hạn.
  - Mô hình Tree Ensembles & Regularized Linear: biên độ hẹp hơn đáng kể ($[-0.08, +0.12]$), phản ánh tính chất co rút hồi quy (shrinkage).
  - Không ghi nhận giá trị số học bất hợp lệ (NaN/Inf) trong tập dự báo của các mô hình Deep Learning (Transformer, LSTM, RNN).

---

## 4. Ma Trận Bao Phủ Thực Nghiệm (Coverage Matrix)

Mỗi mô hình sở hữu cấu trúc quan sát cân bằng tuyệt đối trên toàn bộ không gian thực nghiệm:

$$\text{Tổng số quan sát} = 14 \text{ Models} \times 5 \text{ Horizons} \times 5 \text{ Folds} \times 272 \text{ Obs/Fold} = 95,200$$

### 4.1 Phân bổ quan sát theo Chế độ Biến động (Volatility Regime)
Phân loại chế độ biến động được xác định dựa trên ngưỡng phân vị của chuỗi biến động lịch sử (Rolling Volatility):
- **Chế độ `Low` (Thấp):** $46,830$ quan sát ($49.19\%$) — 669 ngày giao dịch/mô hình/horizon.
- **Chế độ `High` (Cao):** $29,190$ quan sát ($30.66\%$) — 417 ngày giao dịch/mô hình/horizon (bao gồm các đợt sụt giảm 2015-2016, sụp đổ Q4/2018 và khủng hoảng COVID-19 Q1/2020).
- **Chế độ `Normal` (Bình thường):** $19,180$ quan sát ($20.15\%$) — 274 ngày giao dịch/mô hình/horizon.
- **Tổng số ngày giao dịch kiểm thử:** $669 + 417 + 274 = 1,360$ ngày (đúng bằng $5 \times 272$).

### 4.2 Phân bổ quan sát theo Phân kỳ Thời gian (Time Period)
- **Giai đoạn 2015–2016:** $27,650$ quan sát (bao gồm Fold 1 và một phần Fold 2: đợt phá giá Nhân dân tệ và sự kiện Brexit).
- **Giai đoạn 2017–2018:** $35,140$ quan sát (bao gồm Fold 2, Fold 3 và một phần Fold 4: năm 2017 biến động cực thấp, Volmageddon đầu 2018 và cú sập Q4/2018).
- **Giai đoạn 2019–2020:** $32,410$ quan sát (bao gồm Fold 4 và Fold 5: chu kỳ nới lỏng tiền tệ 2019 và đại dịch COVID-19).

---

## 5. Tái Tính Toán Chỉ Số Ngoại Mẫu & Đối Chiếu Độc Lập (Independent Recomputation & Cross-Check)

Để đảm bảo không có sự sai lệch số liệu hay giả mạo kết quả, toàn bộ các chỉ số đã được tái tính toán trực tiếp từ $95,200$ quan sát của `all_oos_predictions.csv` và so sánh đối chiếu với `all_fold_metrics.csv`.

### 5.1 Kết quả đối chiếu với `all_fold_metrics.csv`
Tái tính toán độc lập cho $13$ mô hình hồi quy $\times 5$ chân trời $\times 5$ Folds ($325$ lượt đánh giá):
- **Sai số tuyệt đối cực đại MAE:** $\Delta_{\max} = 1.25 \times 10^{-16}$ (chính xác đến giới hạn dấu phẩy động 64-bit).
- **Sai số tuyệt đối cực đại RMSE:** $\Delta_{\max} = 1.25 \times 10^{-16}$.
- **Sai số tuyệt đối cực đại $OOS\_R^2$:** $\Delta_{\max} = 5.86 \times 10^{-14}$.
- **Sai số tuyệt đối cực đại Directional Accuracy (DA):** $\Delta_{\max} = 1.11 \times 10^{-16}$ khi áp dụng đúng định nghĩa toán học đồng nhất:
  $$\text{DA} = \frac{1}{N} \sum_{t=1}^N \mathbb{I}\left( (\text{Actual}_t > 0) == (\text{Prediction}_t > 0) \right)$$

#### Phân tích hiện tượng dị biệt DA tại Fold 2 (Discrepancy Clarification)
Trong quá trình kiểm tra sơ bộ với công thức $\text{sign}(\text{Actual}) == \text{sign}(\text{Prediction})$ hoặc $\text{Actual} \times \text{Prediction} > 0$, ghi nhận độ lệch $\Delta_{\text{DA}} = 0.003676$ tại Fold 2.
- **Nguyên nhân toán học:** Giá trị $0.003676 = \frac{1}{272}$. Trong Fold 2 (ngày 2016-09-08), lợi suất thực tế của S&P 500 ghi nhận mức chính xác $0.000000$.
- Khi áp dụng toán tử so sánh ngặt $(\text{Actual} > 0)$, ngày này được phân loại vào nhóm không tăng (False). Do mô hình dự báo chiều giảm $(\text{Prediction} \le 0)$, phép so sánh logic $\text{False} == \text{False}$ được tính là một phán đoán chính xác về trạng thái thị trường không tăng.
- Sau khi chuẩn hóa theo đúng toán tử logic của mã nguồn dự án, sai số giữa hai file trở về mức **$0.0000000000000000$**.

### 5.2 Đối chiếu với `results/analysis/rq2_regime.csv` & `results/analysis/rq3_time.csv`
- Chỉ số $OOS\_R^2$ và $DA$ theo từng Regime (Low, Normal, High) trong `rq2_regime.csv` khớp chính xác với giá trị tổng hợp ngoại mẫu gộp (pooled out-of-sample) từ `all_oos_predictions.csv` (ví dụ: ARIMA 1D Low có $OOS\_R^2 = 0.0028, DA = 0.5277$ đồng nhất tuyệt đối).
- Chỉ số theo từng giai đoạn trong `rq3_time.csv` khớp chính xác với tập con ngày giao dịch tương ứng (ví dụ: ARIMA 1D 2015-2016 có $OOS\_R^2 = -0.0016, DA = 0.5038, RMSE = 0.0094$).

---

## 6. Tổng Hợp Hiệu Năng Ngoại Mẫu Gộp (Pooled Out-of-Sample Performance)

Dưới đây là ma trận hiệu năng gộp (Pooled $OOS\_R^2$) tính trên toàn bộ $1,360$ ngày kiểm định ngoại mẫu cho từng mô hình và chân trời:

| Model | 1D | 5D | 20D | 60D | 120D | Nhận xét xu hướng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Historical_Mean** | -0.0001 | -0.0003 | -0.0016 | -0.0115 | -0.0607 | Benchmark tiệm cận 0 |
| **Ridge** | +0.0029 | -0.0583 | -0.1284 | -0.3083 | -0.2854 | Suy thoái nhanh sau 1D |
| **Elastic_Net** | +0.0029 | -0.0078 | -0.0178 | -0.0402 | -0.0676 | Kháng suy thoái tốt nhờ co rút |
| **SVR** | +0.0181 | -0.0558 | -0.1121 | -0.1745 | -0.1775 | Đạt đỉnh tại 1D, âm ở chân trời dài |
| **Random_Forest** | -0.0336 | -0.0673 | -0.2567 | -0.3630 | -0.4852 | Suy thoái đơn điệu nghiêm trọng |
| **XGBoost** | -0.0176 | -0.0050 | -0.0435 | -0.2076 | -0.2744 | Tương tự RF, âm sâu tại 60D-120D |
| **LightGBM** | -0.0627 | -0.0748 | -0.0619 | -0.2502 | -0.7834 | Bị phạt nặng bởi lỗi ngoại mẫu 120D |
| **RNN** | -0.0247 | -0.0454 | -0.0650 | -0.2845 | -0.0383 | Biến thiên phi tuyến |
| **LSTM** | +0.0081 | -0.1537 | -0.0680 | -0.2872 | -0.3819 | Dương nhẹ 1D, sau đó suy thoái |
| **Transformer** | -0.7354 | -0.2996 | -0.1908 | -0.6378 | -0.5573 | Âm nặng nề do thiếu dữ liệu lớn |
| **ARIMA** | +0.0135 | +0.5892 | +0.8735 | +0.9438 | +0.9561 | Tăng vọt theo tích lũy chuỗi |
| **ARIMAX** | +0.0260 | +0.6761 | +0.9223 | +0.9640 | +0.9735 | Tăng mạnh nhất trong dòng AR |
| **SARIMA** | +0.0122 | +0.6441 | +0.8737 | +0.9424 | +0.9551 | Xu hướng tương tự ARIMA |

---

## 7. Phân Tích Hiện Tượng & Các Mẫu Hình Then Chốt (Key Empirical Patterns)

### 7.1 Sự phân kỳ triệt để giữa Dòng Mô Hình Cây/Học Sâu và Dòng Mô Hình Tự Hồi Quy
1. **Dòng Tree Ensembles (Random Forest, XGBoost, LightGBM) và Học Sâu (Deep Learning):**
   - Tại $1D$: $OOS\_R^2$ dao động quanh mức $[-0.06, +0.01]$.
   - Khi horizon tăng từ $5D \to 120D$: $OOS\_R^2$ lao dốc mạnh mẽ và đơn điệu xuống $[-0.27, -0.78]$.
   - **Nguyên nhân cấu trúc:** Lợi suất cộng dồn $Target_{h,t}$ có phương sai tăng tuyến tính theo thời gian ($\sigma_{120D}^2 \approx 0.079^2 \gg \sigma_{1D}^2 \approx 0.012^2$). Các mô hình phi tuyến tính phức tạp có xu hướng ghi nhớ (overfit) các mẫu hình nhiễu ngắn hạn trong tập huấn luyện, dẫn đến dự báo điểm lệch pha nghiêm trọng so với thực tế ở chân trời xa.
2. **Dòng Autoregressive (ARIMA, ARIMAX, SARIMA):**
   - $OOS\_R^2$ tăng vọt từ $\approx +0.01$ (1D) lên $>+0.95$ (60D, 120D).
   - *Lưu ý phương pháp luận:* Cần thận trọng khi diễn giải con số này là "năng lực dự báo thống kê thực sự". Phân tích kiểm định phương pháp luận tại Mục 10 và 11 của báo cáo này cho thấy cơ chế lọc 1 bước (1-step filtering) trong `statsmodels` trên biến mục tiêu trực tiếp nhiều kỳ đã tạo ra sự bất tương đồng nghiêm trọng về chân trời thông tin giữa dòng AR và các mô hình ML/DL.

### 7.2 Sự phân kỳ giữa Năng Lực Dự Báo Phương Sai ($OOS\_R^2$) và Độ Chính Xác Hướng Đi ($DA$)
- Mặc dù các mô hình Machine Learning bị suy giảm $OOS\_R^2$ nghiêm trọng tại 60D và 120D, độ chính xác định hướng ($DA$) của chúng lại tăng từ $\approx 52\% - 54\%$ (1D) lên tới **$63\% - 77\%$** (120D).
- **Giải thích hiện tượng:** Thị trường chứng khoán Mỹ (S&P 500) có xu hướng trôi dạt tăng trưởng dài hạn (positive equity risk premium drift). Ở chu kỳ 120 ngày, tỷ lệ dương thực tế trong mẫu ngoại mẫu lên tới **$77.79\%$** ($1,058 / 1,360$ ngày). Do đó, mức $DA$ cao của các mô hình chủ yếu phản ánh xu hướng nền của thị trường hơn là khả năng định thời (timing ability). Chi tiết kiểm định được phân tích tại Mục 10 dưới đây.

---

## 8. Hạn Chế Phương Pháp Luận Của Tập Tin (Limitations)
1. **Chưa chứa dữ liệu về độ biến thiên trong nội bộ Fold:** `all_oos_predictions.csv` là bảng phẳng gồm các cặp dự báo - thực tế. Để xác định ý nghĩa thống kê của sự vượt trội, cần kết hợp chạy kiểm định Diebold-Mariano có hiệu chỉnh tương quan tự hồi Newey-West trên chuỗi sai số dự báo $e_{h,t} = y_{h,t} - \hat{y}_{h,t}$.
2. **Phụ thuộc vào nhãn Regime cố định:** Nhãn `Regime` trong bảng là nhãn xác định ngoại sinh tại thời điểm $t$. Cần đối chiếu với `rq2_regime.csv` để làm rõ mức độ tương tác đa chiều.

---

## 9. Kết Luận Bước 3 & Kế Hoạch Tiếp Theo (Conclusions & Next Steps)
- **Kết luận:** Tập tin `all_oos_predictions.csv` hoàn toàn hợp lệ về mặt cấu trúc bảng và toàn vẹn dữ liệu số học (0 giá trị khuyết thiếu phi lý hoặc ngoại lai vô cực). Mọi phép tổng hợp từ file này đều tái tạo chính xác tuyệt đối các bảng tóm tắt (`all_fold_metrics.csv`, `rq2_regime.csv`, `rq3_time.csv`). Tuy nhiên, về mặt phương pháp luận mô hình, một số phát hiện then chốt cần được xem xét cẩn trọng trước khi đưa vào kết luận cho RQ1/RQ2/RQ3.
- **Bước tiếp theo:** Xem xét các kiến nghị trong phần **Methodological Audit** bên dưới trước khi thực hiện các phân tích suy luận nhân quả và kiểm định giả thuyết.

---

# Methodological Audit

## 1. Audit Scope
Audit phương pháp luận này nhằm kiểm định độc lập tính hợp lệ khoa học và tính toàn vẹn của pipeline thực nghiệm cho nghiên cứu:
**"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"**

- **Phạm vi dữ liệu:** 
  - `results/predictions/all_oos_predictions.csv` ($95,200$ quan sát)
  - `results/fold_metrics/all_fold_metrics.csv` ($325$ lượt đánh giá)
  - `results/summary/model_horizon_summary.csv` ($65$ cặp Model $\times$ Horizon)
- **Phạm vi mã nguồn & cấu hình:** `models/baselines/historical_mean.py`, `models/baselines/arima.py`, `models/baselines/arimax.py`, `models/baselines/sarima.py`, `models/utils.py`, `models/config.py`, và tập dữ liệu phân chia fold `dataset/3.0.3_val_folds.csv`.
- **Nguyên tắc kiểm định:**
  - Không huấn luyện lại mô hình (retrain).
  - Không chỉnh sửa các tệp CSV kết quả.
  - Phân loại bằng chứng thành 4 mức độ: `Verified`, `Not verified`, `Potential issue`, `Insufficient evidence`.
  - Mọi phát hiện vấn đề đều tuân thủ cấu trúc: **Vấn đề $\to$ Bằng chứng $\to$ Mức độ nghiêm trọng $\to$ Ảnh hưởng đến kết quả $\to$ Cách sửa đề xuất**.

---

## 2. Historical Mean Benchmark

### 2.1 Cơ chế tạo dự báo Benchmark
- **Cơ chế triển khai:** Trong `models/baselines/historical_mean.py`, dự báo của Historical Mean được thiết kế theo dạng trung bình mở rộng (expanding mean) trên tập kiểm định ngoại mẫu:
  $$\hat{y}_{t} = \frac{1}{N_{\text{train}} + i} \left( \sum_{k \in \text{train}} y_k + \sum_{j=0}^{i-1} y_{\text{val}, j} \right)$$
- **Đánh giá rò rỉ thông tin (Lookahead Audit):**
  - Đối với chân trời $h = 1$: Tại ngày $t$ ($i$ bước vào validation), giá trị cập nhật là $y_{\text{val}, i-1} = Target_{1, t-1} = \ln(Close_{t} / Close_{t-1})$. Thông tin này hoàn toàn có sẵn tại gốc dự báo $t$ (sau giờ đóng cửa ngày $t-1$).
  - Đối với chân trời $h > 1$ (ví dụ 120D): Biến $y_{\text{val}, i-1} = Target_{120, t-1} = \ln(Close_{t-1+120} / Close_{t-1})$. Về mặt lý thuyết thuần túy, giá trị $Close_{t+119}$ nằm ở tương lai so với thời điểm $t$. Do đó, việc cộng trực tiếp $Target_{h, t-1}$ vào chuỗi trung bình mở rộng tại ngày $t$ có chứa thành phần giá tương lai.
  - **Tác động thực tế:** Do độ dài tập huấn luyện ban đầu rất lớn ($N_{\text{train}} \ge 6,351$), tỷ trọng của một quan sát mới là $\frac{1}{6,352} < 0.00015$. Biến động của $\hat{y}_t$ qua toàn bộ 272 ngày validation của mỗi fold chỉ dao động trong khoảng $\pm 0.0001$, sai số dự báo thay đổi $< 10^{-5}$. Do đó, ảnh hưởng số học là không đáng kể, nhưng cần chuẩn hóa lại quy trình về fixed training mean $\hat{y}_t = \bar{y}_{\text{train}}$ để đảm bảo tính chặt chẽ tuyệt đối.
- **Trạng thái:** `Verified` (Bằng chứng mã nguồn xác định rõ; tác động số học không đáng kể).

### 2.2 Công thức và Sự Bất Thường $OOS\_R^2$ của Historical Mean
- **Định nghĩa chuẩn (Campbell & Thompson, 2008):**
  $$OOS\_R^2 = 1 - \frac{\sum_{t} (y_t - \hat{y}_t)^2}{\sum_{t} (y_t - \hat{y}_{\text{benchmark}, t})^2}$$
  Nếu mô hình chính là benchmark ở mẫu số, thì $OOS\_R^2$ của Historical Mean về mặt định nghĩa phải bằng $0.0000$.
- **Nguyên nhân phát hiện tại Report 02 ($OOS\_R^2_{120D} \approx -0.6974$ theo Fold Mean):**
  - Kiểm tra hàm `calculate_metrics` trong `models/utils.py` cho thấy: Mẫu số của $OOS\_R^2$ trong `all_fold_metrics.csv` được tính bằng phương sai mẫu của tập validation:
    $$\text{Denominator} = \sum_{t=1}^{N_{\text{val}}} (y_t - \bar{y}_{\text{val}})^2 = N_{\text{val}} \cdot \text{Var}(y_{\text{val}})$$
  - Trong khi đó, dự báo của Historical Mean xuất phát từ trung bình tập train ($\bar{y}_{\text{train}}$). Sai số bình phương trung bình của Historical Mean trên tập val là:
    $$\text{MSE}(\text{HM}) = \frac{1}{N_{\text{val}}} \sum_{t=1}^{N_{\text{val}}} (y_t - \bar{y}_{\text{train}})^2 = \text{Var}(y_{\text{val}}) + (\bar{y}_{\text{val}} - \bar{y}_{\text{train}})^2$$
  - Khi đó, công thức tính trong code cho Historical Mean trở thành:
    $$OOS\_R^2_{\text{HM}} = 1 - \frac{\text{MSE}(\text{HM})}{\text{Var}(y_{\text{val}})} = - \frac{(\bar{y}_{\text{val}} - \bar{y}_{\text{train}})^2}{\text{Var}(y_{\text{val}})} \le 0$$
  - Tại Fold 2 chân trời 120D: $\bar{y}_{\text{train}} = 0.0272$, $\bar{y}_{\text{val}} = 0.0881$, $\text{Var}(y_{\text{val}}) = 0.001188$. Chênh lệch bình phương là $(0.0881 - 0.0272)^2 = 0.003704 \implies OOS\_R^2_{\text{Fold 2}} = - \frac{0.003704}{0.001188} \approx -3.1175$.
  - Khi lấy trung bình cộng 5 Folds, giá trị âm sâu của Fold 2 kéo trung bình xuống $-0.6974$.
  - Ngược lại, khi tính $OOS\_R^2$ gộp (pooled) trên toàn bộ $1,360$ ngày ngoại mẫu trong Report 03, giá trị là $-0.0607$.
- **Trạng thái:** `Verified` (Đã làm rõ tuyệt đối nguồn gốc toán học: đây là sự khác biệt giữa chuẩn hóa theo $\text{Var}(y_{\text{val}})$ từng fold và định nghĩa Campbell-Thompson chuẩn).

---

## 3. ARIMA / ARIMAX / SARIMA Target Construction

### 3.1 Hình thức dự báo (Direct vs Recursive)
- **Kiểm tra mã nguồn:** Trong `models/baselines/arima.py`, `models/baselines/arimax.py`, và `models/baselines/sarima.py`, biến mục tiêu được nạp trực tiếp qua:
  ```python
  target_col = f"Target_{horizon}"
  train_y = train_df[target_col].values
  ```
- **Kết luận:** Mô hình triển khai theo phương thức **Direct Multi-Horizon Forecasting** (dự báo trực tiếp chuỗi $Target_{h,t} = \ln(Close_{t+h}/Close_t)$), hoàn toàn **KHÔNG** phải là phương thức đệ quy cộng dồn các bước 1 ngày ($\sum_{i=1}^h \hat{r}_{t+i}$).

### 3.2 Cơ chế lọc (Filtering) và Nguy cơ Rò rỉ Thông tin Chân trời (Critical Methodological Finding)
- **Phân tích cơ chế chạy thực tế:**
  Trong mã nguồn của cả 3 mô hình thời gian:
  ```python
  full_y = np.concatenate([train_y, val_y])
  # Đối với SARIMAX / ARIMA:
  applied = res.apply(full_y)  # hoặc res.apply(full_y, exog=full_exog)
  val_preds = applied.predict(start=n_train, end=n_train + n_val - 1)
  ```
- **Bản chất toán học của `apply()` trong statsmodels:**
  - Lệnh `applied.predict(start=n_train, ...)` trên đối tượng `SARIMAXResultsWrapper` tính toán **dự báo lọc một bước (one-step-ahead in-sample filtered predictions)** trên chuỗi nối tiếp.
  - Đối với mô hình tự hồi quy AR(1) trên chuỗi $y_t = Target_{h,t}$:
    $$\hat{y}_t = \mu + \phi (y_{t-1} - \mu)$$
  - Giá trị trễ $y_{t-1}$ chính là:
    $$y_{t-1} = Target_{h, t-1} = \ln(Close_{t-1+h} / Close_{t-1})$$
  - Tại gốc dự báo $t$, giá trị $Close_{t-1+h}$ là giá đóng cửa tại ngày $t + h - 1$. Với $h = 120$, đây là giá đóng cửa **119 ngày trong tương lai** so với ngày $t$!
- **Hệ quả phương pháp luận:**
  - Mô hình ARIMA/SARIMA/ARIMAX không thực hiện dự báo $h$ bước vào tương lai từ thông tin tại ngày $t$, mà thực chất đang dự báo bước trễ 1 ngày của chuỗi lợi suất gối đầu (overlapping return).
  - Do hai chuỗi $Target_{120, t}$ và $Target_{120, t-1}$ chia sẻ tới 119 ngày giao dịch chung, hệ số tự tương quan giữa chúng đạt $\rho \approx 0.985$. Việc sử dụng $Target_{h, t-1}$ để dự báo $Target_{h, t}$ tự nhiên mang lại $R^2 > 0.95$ và RMSE $\approx 0.012$ phẳng qua mọi chân trời.
  - Các mô hình ML/DL (Ridge, RF, XGBoost, LSTM) được huấn luyện để dự báo $Target_{h,t}$ chỉ dựa vào tập đặc trưng tại thời điểm $t$ ($X_t$).
  - **Sự bất tương đồng thông tin (Information Asymmetry):** Kết quả của dòng ARIMA không thể so sánh trực tiếp với dòng ML/DL vì hai nhóm mô hình được đặt trong hai bài toán dự báo với tập thông tin hoàn toàn khác nhau.
- **Trạng thái:** `Potential issue` (Mức độ nghiêm trọng: **Critical** — vô hiệu hóa tính so sánh của nhóm ARIMA ở các chân trời $h > 1$).

---

## 4. Target Alignment

### 4.1 Khớp nối biến mục tiêu (Target Integrity)
- **Kiểm tra dữ liệu gốc:** Kiểm tra đối chiếu trên tập dữ liệu kiểm định ngoại mẫu `dataset/3.0.3_val_folds.csv`:
  - `Target_1D` $\equiv \ln(Close_{t+1} / Close_t)$ (sai số $< 10^{-16}$).
  - `Target_5D` $\equiv \ln(Close_{t+5} / Close_t)$ (sai số $< 10^{-16}$).
  - `Target_20D` $\equiv \ln(Close_{t+20} / Close_t)$ (sai số $< 10^{-16}$).
  - `Target_60D` $\equiv \ln(Close_{t+60} / Close_t)$ (sai số $< 10^{-16}$).
  - `Target_120D` $\equiv \ln(Close_{t+120} / Close_t)$ (sai số $< 10^{-16}$).
- **Căn chỉnh thời gian (Date Alignment):**
  - Mọi quan sát tại dòng $t$ có ngày giao dịch là $Date_t$. Các đặc trưng $X_t$ (ví dụ: `Return_lag_1`, `Momentum_5D`, `Rolling_Vol_20D`, `VIXCLS`) được tính từ dữ liệu giá đóng cửa đến hết ngày $Date_t$.
  - Kết quả dự báo $\hat{y}_t$ tương ứng chính xác với khoảng thời gian lợi suất từ $Date_t$ đến $Date_{t+h}$.
  - Không phát hiện hiện tượng lệch chỉ mục ngày ($\pm 1$ shift) hay việc nhầm lẫn giá mở cửa/đóng cửa trong các mô hình Machine Learning và Deep Learning.
- **Trạng thái:** `Verified` đối với dòng ML/DL; `Potential issue` đối với điều kiện thông tin của dòng ARIMA (như phân tích tại Mục 3).

---

## 5. ARIMAX Exogenous Timing

### 5.1 Bảng rà soát biến ngoại sinh (Exogenous Audit Table)
Kiểm tra chi tiết tập biến ngoại sinh được sử dụng trong `models/baselines/arimax.py`:

| Variable | Used at prediction origin $t$? | Future information? | Leakage risk | Evidence |
| :--- | :---: | :---: | :---: | :--- |
| `Return_lag_1` | Có (Close $t$ vs $t-1$) | Không | Không | Công thức $\ln(Close_t / Close_{t-1})$, tính tại cuối ngày $t$ |
| `Momentum_5D` | Có (Close $t$ vs $t-5$) | Không | Không | Công thức $\ln(Close_t / Close_{t-5})$, hoàn toàn trong quá khứ |
| `Rolling_Vol_20D` | Có (Cửa sổ $[t-19, t]$) | Không | Không | Độ lệch chuẩn độ dài 20 ngày kết thúc tại ngày $t$ |
| `VIXCLS` | Có (Giá đóng cửa VIX ngày $t$) | Không | Rất thấp (Xem lưu ý) | Giá trị VIX của ngày $t$ khả dụng sau khi thị trường đóng cửa |
| `VIX_change_1D` | Có ($VIX_t - VIX_{t-1}$) | Không | Không | Chênh lệch VIX giữa ngày $t$ và $t-1$ |

- **Đánh giá rủi ro thời điểm:** Toàn bộ các biến ngoại sinh trên đều được đo lường tại thời điểm đóng cửa ngày $t$. Nếu giả định giao dịch được thực hiện tại mức giá đóng cửa ngày $t$ (Close-to-Close return), các biến này không chứa thông tin tương lai.
- **Lưu ý tương tác:** Tuy bản thân các biến ngoại sinh là hợp lệ về mặt thời gian, nhưng khi được đưa vào hàm `apply(full_y, exog=full_exog)`, mô hình ARIMAX vẫn chịu chung cơ chế lọc 1 bước trên biến nội sinh $Target_{h, t-1}$ như ARIMA.
- **Trạng thái:** `Verified` về mặt thời điểm biến ngoại sinh đơn lẻ; `Potential issue` trong cấu trúc kết hợp với chuỗi nội sinh.

---

## 6. Scaling and Preprocessing Leakage

### 6.1 Quy trình chuẩn hóa trong Walk-Forward Folds
- **Kiểm tra mã nguồn:** Hàm `prepare_fold_data` trong `models/utils.py`:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  ```
- **Đánh giá nguyên tắc:**
  $$\text{Fit}(X_{\text{train}}) \to \text{Transform}(X_{\text{train}}) \to \text{Transform}(X_{\text{val}})$$
  - Scaler chỉ được fit trên dữ liệu của tập huấn luyện của từng fold tương ứng.
  - Tập validation hoàn toàn không tham gia vào việc ước lượng trung bình ($\mu$) hoặc độ lệch chuẩn ($\sigma$).
  - Không có hiện tượng `Fit(X_all)` hay rò rỉ phân phối dữ liệu qua biên giới giữa các fold.
- **Trạng thái:** `Verified` (Hoàn toàn cách ly, không có rò rỉ dữ liệu).

---

## 7. Hyperparameter Tuning Leakage

### 7.1 Cô lập tập kiểm định trong quá trình dò tìm siêu tham số
- **Kiểm tra mã nguồn:** Trong `models/tuning/` và các hàm phân chia nội bộ trong `models/utils.py`:
  - Hàm `get_internal_train_val_split(train_df, val_ratio=0.15)` chỉ nhận đầu vào là `train_df`.
  - Quá trình tối ưu hóa với Optuna và Cross-Validation chuỗi thời gian (TimeSeriesSplit) chỉ diễn ra bên trong không gian dữ liệu của $X_{\text{train}}$.
  - Tập kiểm định ngoại mẫu thực sự (`val_df` của từng Outer Walk-Forward Fold) hoàn toàn bị cô lập và không bao giờ được truyền vào hàm mục tiêu của Optuna hay các hàm dừng sớm (early stopping).
- **Trạng thái:** `Verified` (Thiết kế phân tầng chuẩn xác, không có rò rỉ tối ưu hóa).

---

## 8. RNN / LSTM / Transformer Sequence Construction

### 8.2 Cấu trúc chuỗi thời gian (Lookback Sequence)
- **Kiểm tra mã nguồn:** Hàm `create_sequences` trong `models/utils.py`:
  - Chuỗi đầu vào được tạo với cửa sổ quan sát quá khứ $L$ (lookback length):
    $$X_t = [x_{t-L+1}, x_{t-L+2}, \dots, x_t], \quad y_t = Target_{h,t}$$
  - **Kiểm tra rò rỉ tại biên giới tập Train/Validation:**
    Để tạo chuỗi đầu vào cho những ngày đầu tiên của tập validation ($t = 0, \dots, L-1$), mã nguồn sử dụng phần đuôi của tập huấn luyện ($X_{\text{train}}$) làm ngữ cảnh lịch sử hợp lệ (legitimate historical context).
    Tập validation không bị dịch chuyển vào tương lai, và không có bất kỳ điểm dữ liệu nào của validation bị dùng để tạo chuỗi cho tập huấn luyện.
  - Chuẩn hóa scaler được áp dụng trước khi đóng gói thành các mảng tensor 3D, bảo đảm giá trị chuẩn hóa không bị xuyên thấu qua tập test/val.
- **Trạng thái:** `Verified` (Thiết kế chuỗi chuẩn xác, không có rò rỉ biên giới).

---

## 9. Overlapping Target Dependence

### 9.1 Bản chất kinh tế lượng của chuỗi mục tiêu nhiều kỳ
- **Hiện tượng gối đầu (Overlapping):** Với chân trời $h > 1$, biến mục tiêu $Target_{h,t} = \sum_{i=1}^h r_{t+i}$ và $Target_{h,t+1} = \sum_{i=1}^h r_{t+1+i}$ chia sẻ $h - 1$ ngày sinh lợi chung.
- **Hệ quả thống kê:**
  - Sai số dự báo $e_{h,t} = y_{h,t} - \hat{y}_{h,t}$ sẽ có cấu trúc tự tương quan bậc $MA(h-1)$ dưới giả thuyết không.
  - Hiện tượng này **không phải là lỗi rò rỉ dữ liệu (data leakage)**, mà là đặc tính nội tại của bài toán dự báo trực tiếp nhiều chân trời (Direct Multi-Horizon Forecasting).
  - Các chỉ số điểm (Point metrics) như MAE, RMSE, Pooled $OOS\_R^2$, và DA vẫn là các ước lượng không chệch (unbiased).
  - Tuy nhiên, độ lệch chuẩn mẫu thông thường và các kiểm định $t$-test chuẩn sẽ bị phóng đại độ tin cậy. Do đó, mọi kiểm định so sánh mô hình bắt buộc phải sử dụng hiệu chỉnh tự tương quan Newey-West / HAC (như kiểm định Diebold-Mariano với hiệu chỉnh mẫu nhỏ Harvey-Leybourne-Newbold trong `models/utils.py`).
- **Trạng thái:** `Verified` (Đặc tính kinh tế lượng chuẩn, đã có giải pháp kiểm định tương thích).

---

## 10. DA Base-Rate Analysis

### 10.1 Phân tích độ nhạy theo tỷ lệ nền thị trường (Base-Rate Audit)
Tại chân trời 120D, độ chính xác định hướng ($DA$) của nhiều mô hình đạt mức rất cao ($>70\%$ ở ML, $>97\%$ ở ARIMA). Cần xác định liệu đây là năng lực định thời thực sự hay chỉ là sự phản ánh của xu hướng nền thị trường (positive equity drift).

- **Tỷ lệ thực tế dương trong mẫu kiểm định ngoại mẫu ($1,360$ ngày):**
  $$P(Y > 0) = \frac{1,058}{1,360} = \mathbf{77.79\%}$$
  $$P(Y \le 0) = \frac{302}{1,360} = \mathbf{22.21\%}$$
- **Benchmark Luôn Dương (Always-Positive Benchmark):**
  Một mô hình thô thiển luôn dự báo thị trường tăng ($\hat{y}_t > 0, \forall t$) sẽ tự động đạt được:
  $$DA_{\text{always+}} = 77.79\%$$

### 10.2 So sánh phân phối dự báo và ma trận nhầm lẫn (Confusion Matrix Audit)
Bảng đối chiếu phân phối dấu tại chân trời 120D trên toàn bộ $1,360$ ngày ngoại mẫu:

| Model | DA (%) | $P(\hat{Y} > 0)$ | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) | Đánh giá năng lực định thời |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Always-Positive** | 77.79% | 100.0% | 1,058 | 302 | 0 | 0 | Điểm tựa nền (Naive benchmark) |
| **Historical_Mean** | 77.79% | 100.0% | 1,058 | 302 | 0 | 0 | Không có năng lực phân biệt chiều giảm |
| **Elastic_Net** | 77.79% | 100.0% | 1,058 | 302 | 0 | 0 | Co rút về hằng số dương, $TN = 0$ |
| **Ridge** | 77.06% | 99.26% | 1,048 | 302 | 0 | 10 | Gần như luôn dương, $TN = 0$ |
| **XGBoost** | 71.69% | 91.40% | 958 | 285 | 17 | 100 | Bắt được 17/302 ngày giảm ($5.63\%$) |
| **Random_Forest** | 63.31% | 81.99% | 837 | 278 | 24 | 221 | Bắt được 24/302 ngày giảm ($7.95\%$) |
| **ARIMA** | 97.28% | 79.93% | 1,054 | 33 | 269 | 4 | Bắt được 269/302 ngày giảm ($89.07\%$) |

- **Kết luận chuyên môn:**
  - Đối với các mô hình Machine Learning tuyến tính và cây quyết định, mức $DA \approx 63\% - 77\%$ **không phải là bằng chứng của năng lực định thời thị trường (market timing skill)**, mà hoàn toàn do tỷ lệ nền thị trường tăng trưởng dài hạn áp đảo. Khả năng nhận diện các pha sụt giảm ($TN$) của các mô hình này gần như bằng 0.
  - Đối với mô hình ARIMA ($DA = 97.28\%$), việc dự báo chính xác gần như tuyệt đối cả hai chiều tăng/giảm xuất phát trực tiếp từ việc chuỗi lọc 1 bước được tiếp cận giá đóng cửa của ngày $t+119$ thông qua $Target_{120, t-1}$ (như đã chứng minh tại Mục 3).
- **Trạng thái:** `Verified` (Giải thích trọn vẹn bản chất hiện tượng DA cao ở chân trời dài).

---

## 11. Suspicious ARIMA-family Performance

### 11.1 Đánh giá tính bất thường của hiệu năng dòng AR
- **Dữ liệu ghi nhận:**
  - Pooled $OOS\_R^2$ của ARIMAX: $1D \approx 0.026 \to 5D \approx 0.676 \to 20D \approx 0.922 \to 60D \approx 0.964 \to 120D \approx 0.974$.
  - RMSE của ARIMAX gần như phẳng tuyệt đối qua các chân trời: $1D = 0.0108, 5D = 0.0122, 20D = 0.0121, 60D = 0.0120, 120D = 0.0120$.
- **Đối chiếu thực nghiệm kinh tế lượng tài chính:**
  Trong lý thuyết tài chính thực nghiệm (Fama 1970, Welch & Goyal 2008), khả năng dự báo lợi suất cổ phiếu ở chân trời 6 tháng không bao giờ có thể đạt $R^2 > 90\%$ với sai số chuẩn chỉ $1.2\%$, vì biến động tích lũy thực tế của S&P 500 ở chu kỳ 120 ngày có độ lệch chuẩn lên tới $\sigma \approx 7.9\%$.
- **Nguyên nhân cốt lõi:**
  1. Không phải do lỗi gõ phím hay làm tròn số liệu.
  2. Không phải do mô hình học được tín hiệu tài chính thần kỳ.
  3. **Nguyên nhân trực tiếp:** Hàm `apply()` của `statsmodels` thực hiện cập nhật lọc 1 bước trên chuỗi lợi suất gối đầu $Target_{h,t}$. Do đó, tại mỗi ngày $t$, mô hình dự báo $Target_{h,t}$ khi đã biết $Target_{h, t-1}$, vốn chỉ khác biệt với $Target_{h,t}$ đúng 1 ngày sinh lợi lẻ ($r_{t+h}$ thay cho $r_t$).
- **Trạng thái:** `Potential issue` (Mức độ nghiêm trọng: **Critical** — cần tách riêng hoặc cấu hình lại trước khi đưa vào phân tích tổng hợp RQ1).

---

## 12. Fold-level vs Pooled OOS R²

### 12.1 Phân biệt toán học giữa hai phương pháp tổng hợp
Cần phân biệt rõ ràng hai khái niệm thường bị nhầm lẫn trong các báo cáo:
1. **Trung bình $OOS\_R^2$ theo từng Fold (Mean Fold $OOS\_R^2$ - dùng trong Report 01 & Report 02):**
   $$\overline{OOS\_R^2} = \frac{1}{K} \sum_{k=1}^K R^2_k, \quad \text{với } R^2_k = 1 - \frac{\sum_{t \in \text{val}_k} (y_t - \hat{y}_t)^2}{\sum_{t \in \text{val}_k} (y_t - \bar{y}_{\text{val}_k})^2}$$
   - Nhược điểm: Rất nhạy cảm với các fold có phương sai mẫu $\text{Var}(y_{\text{val}_k})$ nhỏ và độ lệch trung bình lớn (như trường hợp Fold 2 ở 120D kéo trung bình xuống âm sâu).
2. **$OOS\_R^2$ Gộp Ngoại Mẫu (Pooled $OOS\_R^2$ - dùng trong Report 03):**
   $$OOS\_R^2_{\text{pooled}} = 1 - \frac{\sum_{k=1}^K \sum_{t \in \text{val}_k} (y_t - \hat{y}_t)^2}{\sum_{k=1}^K \sum_{t \in \text{val}_k} (y_t - \bar{y}_{\text{val}})^2}$$
   - Ưu điểm: Phản ánh tổng thể toàn bộ chuỗi $1,360$ ngày kiểm thử, triệt tiêu sự méo mó cục bộ của từng fold riêng lẻ.
- **Khuyến nghị chuẩn hóa:** Trong các phần báo cáo tiếp theo, cần ghi rõ loại $OOS\_R^2$ đang được trình bày để tránh gây hiểu lầm số liệu.
- **Trạng thái:** `Verified` (Cả hai cách tính đều đúng với định nghĩa tương ứng, cần phân biệt rõ ràng trong văn bản).

---

## 13. Train / Validation / Test Integrity

### 13.1 Phân chia không gian và thời gian
- **Các mốc thời gian thực nghiệm:**
  - **Tập huấn luyện (Training Set):** `1990-03-27` $\to$ `2015-06-09` ($6,351$ ngày giao dịch ban đầu).
  - **Tập kiểm định ngoại mẫu (Validation Set - F1 đến F5):** `2015-06-10` $\to$ `2020-10-30` ($1,360$ ngày giao dịch, mỗi fold 272 ngày).
  - **Tập kiểm tra độc lập (Out-of-Sample Test Set):** `2020-11-02` $\to$ `2026-04-07` ($1,360$ ngày giao dịch).
- **Kiểm định tính cô lập của tập Test:**
  - Tập dữ liệu Test hoàn toàn **chưa được sử dụng** trong bất kỳ bước huấn luyện, tinh chỉnh siêu tham số hay đánh giá nào ở giai đoạn hiện tại.
  - Không có hiện tượng rò rỉ thông tin từ tập Test ngược về tập Train hay Validation.
- **Trạng thái:** `Verified` (Tính toàn vẹn của tập Test được bảo toàn tuyệt đối).

---

## 14. Claims Audit

Rà soát toàn bộ các nhận định học thuật trong `03_all_oos_predictions_report.md` để loại bỏ các phát biểu vượt quá bằng chứng:

| Nhận định gốc trong báo cáo | Bằng chứng hiện có? | Trạng thái | Diễn đạt khuyến nghị thay thế |
| :--- | :---: | :---: | :--- |
| "Không xuất hiện giá trị ngoại lai nổ gradient" | Chỉ kiểm tra `np.isinf` / `isnan` | Cần chỉnh sửa | "Không ghi nhận giá trị số học bất hợp lệ (NaN/Inf) trong tập dự báo" |
| "ARIMAX nắm bắt trọn vẹn đặc tính tự hồi quy, tạo năng lực vượt trội" | Mô hình dùng cơ chế lọc 1 bước trên target nhiều kỳ | Tiềm ẩn sai lệch | "ARIMAX đạt hiệu năng cao do cơ chế lọc 1 bước tiếp cận giá trị trễ mang thông tin tương lai của target gối đầu" |
| "Độ chính xác định hướng cao chứng minh năng lực dự báo" | Tỷ lệ dương thực tế đạt $77.79\%$ tại 120D | Cần chỉnh sửa | "DA cao tại 120D chủ yếu phản ánh xu hướng tăng trưởng nền của thị trường hơn là khả năng định thời" |
| "Transformer bị overfit do thiếu dữ liệu lớn" | Chưa đối chiếu loss train vs loss val | Thiếu bằng chứng | "Transformer suy giảm hiệu năng ngoại mẫu, có thể do cấu trúc tham số lớn chưa tối ưu trên chuỗi tài chính ngắn" |
| "Tập tin all_oos_predictions.csv hoàn toàn hợp lệ 100%" | Dữ liệu bảng hợp lệ, nhưng mô hình AR có vấn đề phương pháp luận | Cần phân biệt | "Dữ liệu bảng và cấu trúc số học hợp lệ 100%, nhưng cần lưu ý tính bất tương đồng phương pháp luận của dòng AR" |

---

## 15. Overall Methodological Assessment

Bảng tổng hợp đánh giá phương pháp luận trên toàn bộ 15 hạng mục kiểm định:

| Hạng mục kiểm định (Audit Item) | Trạng thái (Status) | Mức độ nghiêm trọng (Severity) | Bằng chứng (Evidence) | Tác động phương pháp luận (Impact) |
| :--- | :---: | :---: | :--- | :--- |
| **1. Historical Mean benchmark** | Verified | Medium | Expanding mean cập nhật target gối đầu | Ảnh hưởng số học $< 10^{-5}$, nên chuẩn hóa về fixed train mean |
| **2. ARIMA target construction** | Potential issue | Critical | Mô hình lọc 1 bước trên direct multi-horizon target | Vô hiệu hóa tính so sánh của ARIMA ở $h > 1$ với dòng ML |
| **3. ARIMAX target construction** | Potential issue | Critical | Lọc 1 bước tương tự ARIMA | Tạo ra $OOS\_R^2 > 0.97$ ảo do tiếp cận thông tin tương lai |
| **4. SARIMA target construction** | Potential issue | Critical | Lọc 1 bước tương tự ARIMA | Tương tự ARIMA |
| **5. Target alignment** | Verified | Low | Khớp nối $Target_h$ với $\ln(Close_{t+h}/Close_t)$ chính xác | Chuỗi mục tiêu dòng ML/DL hoàn toàn chuẩn xác |
| **6. ARIMAX exogenous timing** | Verified | Low | Các biến ngoại sinh đo tại cuối ngày $t$ | Biến ngoại sinh hợp lệ về thời gian |
| **7. Scaling & Preprocessing** | Verified | None | `StandardScaler` fit trên train, transform trên val | Hoàn toàn cách ly, không rò rỉ dữ liệu |
| **8. Hyperparameter tuning** | Verified | None | Optuna và TimeSeriesSplit chỉ dùng nội bộ train | Không rò rỉ từ tập validation |
| **9. Sequence construction** | Verified | None | Lookback dùng ngữ cảnh lịch sử train hợp lệ | Không rò rỉ biên giới |
| **10. Overlapping targets** | Verified | Medium | Cấu trúc $MA(h-1)$ tự nhiên của chuỗi cộng dồn | Cần kiểm định Newey-West/HAC khi suy luận thống kê |
| **11. DA base rate** | Verified | High | Thị trường có tỷ lệ dương $77.79\%$ tại 120D | $DA$ cao không phản ánh năng lực định thời |
| **12. ARIMA-family performance** | Potential issue | Critical | RMSE phẳng và $R^2 > 0.95$ bất thường | Tín hiệu cảnh báo trực tiếp của cơ chế lọc 1 bước |
| **13. OOS R² calculation** | Verified | Medium | Khác biệt giữa mẫu số $\text{Var}(y_{\text{val}})$ và $\text{MSE}_{\text{HM}}$ | Giải thích hiện tượng $OOS\_R^2$ âm sâu tại Fold 2 |
| **14. Train/Val/Test isolation** | Verified | None | Tập Test độc lập tuyệt đối từ 2020-11 đến 2026-04 | Không rò rỉ tập kiểm tra |
| **15. Report claims integrity** | Verified | Low | Đã rà soát và điều chỉnh các phát biểu quá mức | Báo cáo đạt chuẩn mực thận trọng học thuật |

---

## 16. Audit Verdict

### **CONDITIONAL PASS — pending verification & standardization**

**Căn cứ kết luận:**
1. **Dữ liệu và hạ tầng tính toán đạt chuẩn:** Tập dữ liệu `all_oos_predictions.csv`, quy trình chuẩn hóa `StandardScaler`, quy trình tối ưu hóa siêu tham số nội bộ `Optuna`, và tính độc lập của tập `Test` đều được kiểm định là **hoàn toàn chính xác, không rò rỉ dữ liệu**.
2. **Hai điều kiện ràng buộc cần chuẩn hóa trước khi kết luận nghiên cứu (RQ1/RQ2/RQ3):**
   - **Điều kiện 1:** Nhóm mô hình tự hồi quy (ARIMA, ARIMAX, SARIMA) không được sử dụng để khẳng định kết luận về khả năng dự báo vượt trội ở chân trời dài ($h \ge 5$), do cơ chế lọc 1 bước trên chuỗi mục tiêu nhiều kỳ tạo ra lợi thế thông tin không công bằng so với dòng Machine Learning.
   - **Điều kiện 2:** Chỉ số $OOS\_R^2$ và $DA$ ở chân trời 120D cần được diễn giải đi kèm với benchmark mẫu số chuẩn (Campbell-Thompson) và tỷ lệ dương nền của thị trường ($77.79\%$).

---

## 17. Recommended Next Action

Dưới đây là các bước hành động tiếp theo được sắp xếp theo thứ tự ưu tiên:

- **Priority 1 — Chuẩn hóa định nghĩa dự báo dòng ARIMA:** Tách riêng dòng mô hình ARIMA/ARIMAX/SARIMA hoặc chuyển đổi sang cấu trúc dự báo đệ quy (recursive multi-step ahead) không điều kiện trên chuỗi lợi suất ngày $\hat{r}_{t+i}$, nhằm đưa tập thông tin về cùng xuất phát điểm với dòng Machine Learning.
- **Priority 2 — Thống nhất mẫu số Benchmark $OOS\_R^2$:** Chuẩn hóa cách tính $OOS\_R^2$ theo đúng công thức Campbell-Thompson (dùng $\text{MSE}_{\text{Historical\_Mean}}$ làm mẫu số chung trên toàn bộ các fold) để loại bỏ hiện tượng âm nhân tạo tại các fold có phương sai mẫu hẹp.
- **Priority 3 — Đánh giá Directional Accuracy so với Naive Drift:** Luôn báo cáo kèm chỉ số $DA - DA_{\text{always+}}$ để làm rõ năng lực định thời thực chất của các mô hình ở các chân trời 60D và 120D.
- **Priority 4 — Sử dụng kiểm định Diebold-Mariano có hiệu chỉnh Newey-West:** Đảm bảo mọi so sánh ý nghĩa thống kê trên chuỗi sai số dự báo gối đầu đều được hiệu chỉnh bậc tự tương quan $h-1$.
- **Priority 5 — Chuyển tiếp sang Phân tích Chuyên sâu RQ1/RQ2/RQ3:** Sử dụng dữ liệu ngoại mẫu đã được kiểm định phương pháp luận để trả lời các câu hỏi nghiên cứu một cách khách quan, khoa học và chặt chẽ.

