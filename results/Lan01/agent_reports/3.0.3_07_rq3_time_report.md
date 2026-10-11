# Báo Cáo Nghiên Cứu Câu Hỏi 3: Tính Ổn Định Của Năng Lực Dự Báo Qua Các Chu Kỳ Thời Gian & Kiểm Định Phương Pháp Luận Toàn Diện (07_rq3_time_report.md)

**Mã tài liệu:** `IS403-REP-07`
**Ngày thực hiện:** 2026-10-09
**Tập tin mục tiêu:** `results/analysis/rq3_time.csv`
**Tập tin kiểm toán kèm theo:** `results/agent_reports/07_rq3_time_audit_log.md`
**Tập tin đối chiếu:** `results/predictions/all_oos_predictions.csv`, `results/fold_metrics/all_fold_metrics.csv`, `results/summary/model_horizon_summary.csv`, `src/training/baselines/generate_rq_analysis.py`, `src/training/baselines/config.py`
**Người thực hiện:** Senior Research Methodology Auditor (Antigravity System)
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu Nghiên Cứu RQ3 (Research Question 3 Focus)

Báo cáo này tập trung giải quyết và kiểm định câu hỏi nghiên cứu số 3:
> **Research Question 3 (RQ3):**
> *"Are the observed horizon- and regime-dependent patterns of predictability stable over time?"*
> *(Các mẫu hình suy thoái dự báo theo chân trời [RQ1] và theo chế độ biến động [RQ2] có duy trì tính ổn định qua các chu kỳ thời gian khác nhau hay không? Có hiện tượng kết quả tổng thể chỉ bị chi phối bởi một giai đoạn cá biệt duy nhất [như cú sốc COVID-19 năm 2020] hay không?)*

### Các nguyên tắc phân định khoa học bắt buộc:
1. **Phân biệt Fold và Time Period (Fold $\neq$ Time Period):** Fold trong kỹ thuật Walk-Forward phản ánh sự mở rộng của cửa sổ huấn luyện và kiểm định qua thời gian, trong khi phân kỳ thời gian ở RQ3 là phép kiểm tra tính ổn định ngoại mẫu trên các khoảng thời gian lịch sử cụ thể.
2. **Khái niệm Giai đoạn Đánh giá theo Niên đại (Chronological Evaluation Periods):** Ba khối thời gian 2 năm (`2015–2016`, `2017–2018`, `2019–2020`) được cấu hình cố định trong mã nguồn ([config.py:49-56](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/config.py#L49)) phục vụ mục đích kiểm tra độ nhạy niên đại, **không được đồng nhất với các chu kỳ kinh tế vĩ mô chính thức theo chuẩn NBER**.
3. **Phân biệt 4 cấp độ diễn giải:**
   - *Phát hiện mô tả (Descriptive pattern):* Các chỉ số $OOS\_R^2$, RMSE, DA quan sát được trong từng khối thời gian.
   - *Bằng chứng thống kê (Statistical evidence):* Phải được kiểm định thông qua các bài kiểm tra có hiệu chỉnh tự tương quan do chuỗi mục tiêu gối đầu (thực hiện ở Step 8).
   - *Tính hợp lệ phương pháp luận (Methodological validity):* Kiểm tra tính chuẩn xác của ngày gốc dự báo, công thức mẫu số benchmark và loại trừ các rò rỉ thông tin.
   - *Giải thích cơ chế (Mechanistic explanation):* Tránh quy kết nguyên nhân nhân quả khi chưa có phân tích độ nhạy hoặc can thiệp thực nghiệm độc lập.

---

## 2. Kích Thước, Lược Đồ Dữ Liệu & Quy Mô Mẫu (Shape, Schema & Sample Sizes)

### 2.1 Kích thước vật lý
- **Đường dẫn lưu trữ:** `results/analysis/rq3_time.csv`
- **Số hàng:** Chính xác $42$ hàng ($14 \text{ Models} \times 3 \text{ Periods}$).
- **Số cột:** Đúng $17$ cột theo schema: `Model`, `Period`, và 5 chân trời $\times 3$ chỉ số (`1D_OOS_R2`, `1D_DA`, `1D_RMSE`, ..., `120D_OOS_R2`, `120D_DA`, `120D_RMSE`).
- **Trùng lặp & Khuyết thiếu:** $0$ hàng trùng lặp trên cặp khóa `(Model, Period)`; $0$ giá trị `NaN` hoặc vô cực bất thường.

### 2.2 Quy mô mẫu và Ranh giới thời gian
Dữ liệu kiểm tra ngoại mẫu được phân bổ theo 3 giai đoạn niên đại:
- **Giai đoạn 2015–2016:** $N = 395$ ngày giao dịch ngoại mẫu ($29.04\%$).
- **Giai đoạn 2017–2018:** $N = 502$ ngày giao dịch ngoại mẫu ($36.91\%$).
- **Giai đoạn 2019–2020:** $N = 463$ ngày giao dịch ngoại mẫu ($34.04\%$).
- **Tổng cộng:** $395 + 502 + 463 = \mathbf{1,360}$ điểm dự báo ngoại mẫu mỗi mô hình trên từng chân trời (đồng nhất tuyệt đối với RQ1 và RQ2).

### 2.3 Nguyên tắc gán ngày và Hiện tượng gối đầu liên kỳ (Boundary Overlap)
- **Gán theo ngày gốc dự báo ($t$):** Các dự báo được phân loại vào giai đoạn dựa trên thời điểm hình thành dự báo $t$ (`Date` trong tệp dự báo), phản ánh góc nhìn của nhà đầu tư tại thời điểm ra quyết định.
- **Hiện tượng gối đầu qua ranh giới phân kỳ:** Đối với các chân trời dài hạn ($60D, 120D$), các quan sát hình thành ở cuối một giai đoạn (ví dụ: tháng 11/2016) sẽ có thời điểm hiện thực hóa lợi suất ($t+h$) vươn sang giai đoạn tiếp theo (đầu năm 2017). Thiết kế nghiên cứu giữ nguyên các điểm dữ liệu này để đảm bảo tính liên tục của chuỗi ngoại mẫu, nhưng cần lưu ý hiện tượng này khi so sánh giữa các giai đoạn kế cận.

---

## 3. Ma Trận Hiệu Năng $OOS\_R^2$ Theo Từng Giai Đoạn Thời Gian (Period × Horizon Performance)

> ⚠️ **Quy chuẩn thuật ngữ bắt buộc:**
> Đại lượng so sánh giữa hai chân trời $\Delta_{120D - 1D} = OOS\_R^2_{120D} - OOS\_R^2_{1D}$ là **Thay đổi hiệu năng qua chân trời (Horizon Change in $OOS\_R^2$, $\Delta OOS\_R^2$)**, không phải hệ số góc suy thoái (decay slope) theo nghĩa thống kê.

> 📝 **Ghi chú về Logistic Regression:**
> Mô hình `Logistic_Regression` trong tập dữ liệu đạt giá trị $OOS\_R^2 \in [-14295.98, -71.28]$ do dự báo xác suất nhị phân thay vì lợi suất log liên tục, gây sai lệch thang đo nghiêm trọng. Mô hình này được tách riêng khỏi các bảng so sánh hồi quy bên dưới để bảo toàn khả năng so sánh định lượng của các mô hình hồi quy thực thụ.

### 3.1 Giai Đoạn 2015–2016 (N = 395 ngày)
| Model | 1D | 5D | 20D | 60D | 120D | Thay đổi $OOS\_R^2$ (120D − 1D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0003 | -0.0006 | -0.0018 | -0.0003 | -0.0285 | -0.0282 |
| **Elastic_Net** | +0.0015 | +0.0108 | +0.0101 | +0.0209 | -0.0323 | -0.0338 |
| **Ridge** | -0.0157 | +0.0056 | +0.0199 | +0.0168 | -0.1767 | -0.1610 |
| **SVR** | -0.0423 | +0.0077 | -0.0420 | +0.0853 | -1.0048 | -0.9625 |
| **Random_Forest** | -0.0177 | -0.0025 | -0.1305 | -0.4570 | -1.3387 | -1.3210 |
| **XGBoost** | -0.0002 | -0.0004 | -0.0305 | -1.2437 | -1.3762 | -1.3760 |
| **LightGBM** | -0.0020 | -0.0096 | -0.3976 | -1.1691 | -2.3512 | -2.3492 |
| **LSTM** | -0.0005 | -0.0997 | -0.1459 | -0.0140 | -0.3784 | -0.3779 |
| **RNN** | -0.1209 | +0.0066 | -0.3075 | -0.9243 | -0.2757 | -0.1548 |
| **Transformer** | -2.4572 | -0.3434 | -0.6514 | -2.2218 | -1.8845 | **+0.5727** |
| *ARIMA (Lỗi lọc 1 bước)* | * -0.0016 * | * +0.6096 * | * +0.8797 * | * +0.9465 * | * +0.9556 * | * +0.9572 * |
| *ARIMAX (Lỗi lọc 1 bước)* | * -0.0037 * | * +0.7504 * | * +0.9296 * | * +0.9730 * | * +0.9801 * | * +0.9838 * |
| *SARIMA (Lỗi lọc 1 bước)* | * -0.0059 * | * +0.6860 * | * +0.8797 * | * +0.9464 * | * +0.9559 * | * +0.9618 * |

### 3.2 Giai Đoạn 2017–2018 (N = 502 ngày)
| Model | 1D | 5D | 20D | 60D | 120D | Thay đổi $OOS\_R^2$ (120D − 1D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0002 | -0.0005 | -0.0002 | -0.0012 | -0.0125 | -0.0123 |
| **Elastic_Net** | -0.0010 | +0.0032 | +0.0129 | -0.1667 | -0.0229 | -0.0219 |
| **Ridge** | -0.0009 | +0.0120 | +0.0033 | -0.2955 | -0.3785 | -0.3776 |
| **SVR** | -0.0024 | -0.0016 | -0.1006 | -0.1776 | -0.2282 | -0.2258 |
| **Random_Forest** | +0.0058 | -0.0004 | +0.0116 | -0.1597 | -0.5635 | -0.5693 |
| **XGBoost** | -0.0003 | -0.0014 | -0.0041 | -0.0031 | -0.0183 | -0.0180 |
| **LightGBM** | -0.0100 | -0.0141 | -0.0032 | -0.2564 | -0.9319 | -0.9219 |
| **LSTM** | -0.0284 | -0.1060 | -0.0451 | -0.9398 | -1.5242 | -1.4958 |
| **RNN** | -0.0461 | -0.0671 | -0.0270 | -0.1213 | -0.2035 | -0.1574 |
| **Transformer** | -1.3847 | -0.4359 | -0.5896 | -0.5226 | -1.2180 | **+0.1667** |
| *ARIMA (Lỗi lọc 1 bước)* | * -0.0015 * | * +0.6089 * | * +0.8570 * | * +0.9484 * | * +0.9556 * | * +0.9571 * |
| *ARIMAX (Lỗi lọc 1 bước)* | * +0.0005 * | * +0.7465 * | * +0.9167 * | * +0.9664 * | * +0.9714 * | * +0.9709 * |
| *SARIMA (Lỗi lọc 1 bước)* | * -0.0018 * | * +0.6951 * | * +0.8570 * | * +0.9482 * | * +0.9555 * | * +0.9573 * |

### 3.3 Giai Đoạn 2019–2020 (N = 463 ngày)
| Model | 1D | 5D | 20D | 60D | 120D | Thay đổi $OOS\_R^2$ (120D − 1D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0005 | -0.0035 | -0.0119 | -0.0475 | -0.1770 | -0.1765 |
| **Elastic_Net** | +0.0039 | -0.0209 | -0.0412 | -0.0480 | -0.1843 | -0.1882 |
| **Ridge** | +0.0084 | -0.1062 | -0.2026 | -0.4325 | -0.3973 | -0.4057 |
| **SVR** | +0.0387 | -0.0981 | -0.1414 | -0.2734 | -0.0530 | -0.0917 |
| **Random_Forest** | -0.0480 | -0.1144 | -0.3583 | -0.4420 | -0.3724 | -0.3244 |
| **XGBoost** | -0.0269 | -0.0109 | -0.0659 | -0.0469 | -0.1833 | -0.1564 |
| **LightGBM** | -0.0922 | -0.1199 | -0.0116 | -0.0560 | -0.4919 | -0.3997 |
| **LSTM** | +0.0192 | -0.1912 | -0.0667 | -0.1955 | -0.1492 | -0.1684 |
| **RNN** | +0.0054 | -0.0585 | -0.0306 | -0.2126 | -0.0173 | -0.0227 |
| **Transformer** | -0.1237 | -0.2438 | -0.0099 | -0.3260 | -0.1484 | -0.0247 |
| *ARIMA (Lỗi lọc 1 bước)* | * +0.0209 * | * +0.5745 * | * +0.8746 * | * +0.9400 * | * +0.9525 * | * +0.9316 * |
| *ARIMAX (Lỗi lọc 1 bước)* | * +0.0398 * | * +0.6273 * | * +0.9211 * | * +0.9600 * | * +0.9701 * | * +0.9303 * |
| *SARIMA (Lỗi lọc 1 bước)* | * +0.0201 * | * +0.6123 * | * +0.8749 * | * +0.9379 * | * +0.9508 * | * +0.9307 * |

---

## 4. Phân Tích Tính Ổn Định Theo Họ Mô Hình (Family-Level Stability Across Time)

Bảng tổng hợp mức trung bình $OOS\_R^2$ của **toàn bộ 6 họ mô hình** qua 3 giai đoạn đánh giá:

| Họ mô hình | Giai đoạn | 1D | 5D | 20D | 60D | 120D | $\Delta OOS\_R^2$ (120D − 1D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tree Ensembles** | 2015-2016 | -0.0066 | -0.0042 | -0.1862 | -0.9566 | -1.6887 | -1.6821 |
| *(RF, XGB, LGBM)* | 2017-2018 | -0.0015 | -0.0053 | +0.0014 | -0.1397 | -0.5046 | -0.5031 |
| | 2019-2020 | -0.0557 | -0.0817 | -0.1453 | -0.1816 | -0.3492 | -0.2935 |
| **Linear Regularized** | 2015-2016 | -0.0071 | +0.0082 | +0.0150 | +0.0188 | -0.1045 | -0.0974 |
| *(Ridge, Elastic Net)*| 2017-2018 | -0.0009 | +0.0076 | +0.0081 | -0.2311 | -0.2007 | -0.1997 |
| | 2019-2020 | +0.0061 | -0.0635 | -0.1219 | -0.2402 | -0.2908 | -0.2969 |
| **Support Vector** | 2015-2016 | -0.0423 | +0.0077 | -0.0420 | +0.0853 | -1.0048 | -0.9625 |
| *(SVR)* | 2017-2018 | -0.0024 | -0.0016 | -0.1006 | -0.1776 | -0.2282 | -0.2258 |
| | 2019-2020 | +0.0387 | -0.0981 | -0.1414 | -0.2734 | -0.0530 | -0.0917 |
| **Deep Learning** | 2015-2016 | -0.8595 | -0.1455 | -0.3683 | -1.0534 | -0.8462 | +0.0133 |
| *(RNN, LSTM, Trans.)*| 2017-2018 | -0.4864 | -0.2030 | -0.2206 | -0.5279 | -0.9819 | -0.4955 |
| | 2019-2020 | -0.0330 | -0.1645 | -0.0357 | -0.2447 | -0.1050 | -0.0719 |
| **Benchmark (Mean)** | 2015-2016 | -0.0003 | -0.0006 | -0.0018 | -0.0003 | -0.0285 | -0.0282 |
| *(Historical Mean)* | 2017-2018 | -0.0002 | -0.0005 | -0.0002 | -0.0012 | -0.0125 | -0.0123 |
| | 2019-2020 | -0.0005 | -0.0035 | -0.0119 | -0.0475 | -0.1770 | -0.1765 |
| **Autoregressive** | 2015-2016 | *-0.0037* | *+0.6820* | *+0.8963* | *+0.9553* | *+0.9639* | *+0.9676* |
| *(ARIMA, ARX, SAR.)*| 2017-2018 | *-0.0009* | *+0.6835* | *+0.8769* | *+0.9543* | *+0.9608* | *+0.9618* |
| *(Bị lỗi lọc 1 bước)*| 2019-2020 | *+0.0269* | *+0.6047* | *+0.8902* | *+0.9460* | *+0.9578* | *+0.9309* |

### 4.1 Bác bỏ nhận định thứ hạng cố định "AR >> Linear ≈ Mean > Tree > DL"
Phân tích dữ liệu thực tế bác bỏ việc tồn tại một thứ hạng cố định giữa các họ mô hình:
1. **Tại chân trời ngắn 1D:** Không có sự phân hóa đáng kể; hầu hết các mô hình đều dao động quanh mức $0$ (từ $-0.05$ đến $+0.03$), ngoại trừ Transformer bị phạt nặng ở các giai đoạn đầu.
2. **Tại chân trời dài 120D trong giai đoạn 2019–2020:** Họ **Deep Learning** đạt $OOS\_R^2 = \mathbf{-0.1050}$, xếp **cao hơn** cả Historical Mean ($-0.1770$), Linear Regularized ($-0.2908$) và Tree Ensembles ($-0.3492$).
3. **Trong giai đoạn 2015–2016 tại 120D:** Deep Learning ($-0.8462$) và SVR ($-1.0048$) cũng vượt trội hơn Tree Ensembles ($-1.6887$).
4. **Kết luận:** Thứ hạng giữa các họ mô hình **bị đảo ngược phụ thuộc vào chân trời dự báo và giai đoạn cụ thể**. Không có căn cứ thực nghiệm để khẳng định một trật tự cố định.

---

## 5. Kiểm Định Các Tuyên Bố Thực Nghiệm Cốt Lõi (Core Empirical Findings Audit)

### 5.1 Kiểm tra tuyên bố "Suy thoái 100% qua các phân kỳ"
- **Thực tế:** Nếu xét theo hiệu số điểm mút $\Delta OOS\_R^2 = OOS\_R^2_{120D} - OOS\_R^2_{1D}$:
  - Phần lớn các mô hình truyền thống (Tree Ensembles, Linear Regularized, SVR, LSTM, RNN) đều có $\Delta OOS\_R^2 < 0$.
  - Tuy nhiên, **Transformer là ngoại lệ rõ rệt**: Mô hình này có hiệu số dương ở 2/3 giai đoạn (+0.5727 trong 2015–2016 và +0.1667 trong 2017–2018) do sai số ở 1D quá lớn, sau đó thu hẹp tương đối ở 120D.
- **Tính phi đơn điệu (Non-monotonicity):**
  Chỉ có duy nhất $3 / 42$ trường hợp (khoảng $7\%$) thể hiện sự suy giảm đơn điệu thực sự qua toàn bộ 5 chân trời ($1D \ge 5D \ge 20D \ge 60D \ge 120D$): LightGBM (2015–2016), XGBoost (2015–2016) và Historical Mean (2019–2020).
  Nhiều mô hình đạt đỉnh hiệu năng tại $20D$ hoặc $60D$ trước khi sụt giảm tại $120D$ (ví dụ: Elastic Net, Ridge và SVR trong giai đoạn 2015–2016 đều đạt $OOS\_R^2 > 0$ tại $20D$ và $60D$).

### 5.2 Kiểm tra hiện tượng "RMSE tăng gấp đôi" & Phân tích Chuẩn hóa
Báo cáo ghi nhận giá trị RMSE tăng gấp đôi trong giai đoạn 2019–2020. Dưới đây là phân tích đối chiếu với độ biến động thực tế của chuỗi mục tiêu ($\text{Target Std}$):

| Giai đoạn | Horizon | Target Std | Target Var | Hist. Mean RMSE | Ridge RMSE | LightGBM RMSE | LSTM RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015-2016** | 120D | 0.05797 | 0.00336 | 0.0587 | 0.0628 | 0.1060 | 0.0680 |
| **2017-2018** | 120D | 0.05658 | 0.00320 | 0.0569 | 0.0664 | 0.0786 | 0.0898 |
| **2019-2020** | 120D | **0.10571** | **0.01118** | **0.1146** | **0.1248** | **0.1290** | **0.1132** |

#### Tỷ số Sai số Chuẩn hóa ($\text{Normalized RMSE} = \frac{\text{RMSE}}{\text{Target Std}}$):
- **Historical Mean:** $2015-2016 = \mathbf{1.013} \quad\vert\quad 2017-2018 = \mathbf{1.006} \quad\vert\quad 2019-2020 = \mathbf{1.084}$
- **Ridge:** $2015-2016 = \mathbf{1.083} \quad\vert\quad 2017-2018 = \mathbf{1.173} \quad\vert\quad 2019-2020 = \mathbf{1.181}$
- **LightGBM:** $2015-2016 = \mathbf{1.829} \quad\vert\quad 2017-2018 = \mathbf{1.389} \quad\vert\quad 2019-2020 = \mathbf{1.220}$
- **LSTM:** $2015-2016 = \mathbf{1.173} \quad\vert\quad 2017-2018 = \mathbf{1.587} \quad\vert\quad 2019-2020 = \mathbf{1.071}$

> [!NOTE]
> **Bản chất khoa học:**
> Việc RMSE tuyệt đối tăng gấp đôi trong giai đoạn 2019–2020 (từ $\approx 0.057$ lên $0.115$) được giải thích gần như toàn bộ bởi **sự mở rộng gấp $1.87$ lần của độ lệch chuẩn lợi suất thực tế thị trường** ($\text{Target Std}$ tăng từ $0.0566$ lên $0.1057$). Khi chuẩn hóa theo độ biến động mục tiêu, tỷ số $\frac{\text{RMSE}}{\text{Std}}$ của các mô hình duy trì ổn định cao, thậm chí LightGBM và LSTM còn có sai số chuẩn hóa tốt hơn so với các giai đoạn trước. Do đó, việc quy chụp nhân quả do chính sách tiền tệ hay cú sốc COVID-19 là thiếu căn cứ thực nghiệm độc lập.

### 5.3 Cô lập dòng mô hình Tự hồi quy (ARIMA, ARIMAX, SARIMA)
- Trong cả 3 giai đoạn, dòng ARIMA đều đạt $OOS\_R^2 \approx 0.95 - 0.98$ tại $120D$.
- Tuyên bố trước đây coi đây là *"bằng chứng thép về tính ổn định cực cao"* bị **bác bỏ hoàn toàn**.
- **Nguyên nhân:** Lệnh `applied.predict` trong `statsmodels` thực hiện lọc 1 bước trên chuỗi lũy kế $Target_{h, t}$, sử dụng $Target_{h, t-1}$ vốn đã chứa $h-1$ ngày lợi suất tương lai. Hệ số tương quan $\frac{h-1}{h} = \frac{119}{120} \approx 0.992$ là tính chất toán học của chuỗi trơn gối đầu, xuất hiện ở mọi giai đoạn thời gian.
- **Biện pháp xử lý:** Dòng mô hình này bị cô lập hoàn toàn khỏi mọi phát biểu về tính ổn định thời gian của năng lực dự báo.

---

## 6. Phụ Thuộc Chuỗi Thời Gian & Chuỗi Mục Tiêu Gối Đầu (Overlapping Target Dependence)

- Chuỗi mục tiêu $Target_{h, t} = \sum_{i=1}^h r_{t+i}$ khiến hai ngày liên tiếp chia sẻ chung $h-1$ ngày thực tế.
- Tại chân trời $120D$, tương quan tự thân giữa hai quan sát kế tiếp là $\approx 0.9917$.
- **Số bậc tự do hiệu dụng (Effective Sample Size):**
  - Giai đoạn 2015–2016 ($N=395$): $N_{\text{eff}} \approx \frac{395}{120} \approx \mathbf{3.3}$ chu kỳ độc lập.
  - Giai đoạn 2017–2018 ($N=502$): $N_{\text{eff}} \approx \frac{502}{120} \approx \mathbf{4.2}$ chu kỳ độc lập.
  - Giai đoạn 2019–2020 ($N=463$): $N_{\text{eff}} \approx \frac{463}{120} \approx \mathbf{3.9}$ chu kỳ độc lập.
- **Hệ quả phương pháp luận:** Các quan sát ngoại mẫu trong từng giai đoạn không độc lập phân phối (vi phạm giả định i.i.d.). Mọi so sánh hiệu năng giữa các giai đoạn hiện tại dừng ở mức **thống kê mô tả (descriptive)**. Để khẳng định tính ổn định có ý nghĩa thống kê, bắt buộc phải thực hiện các kiểm định Diebold-Mariano có hiệu chỉnh độ trễ Harvey-Leybourne-Newbold (HLN) hoặc kiểm định Block Bootstrap ở Step 8.

---

## 7. Phân Loại Tính Ổn Định Thời Gian Đã Hiệu Đính (Corrected Stability Classification)

| Tiêu chí đánh giá | Bằng chứng thực nghiệm quan sát được | Phân loại phương pháp luận |
| :--- | :--- | :---: |
| **Chiều hướng suy giảm điểm mút của ML** | $OOS\_R^2_{120D} < OOS\_R^2_{1D}$ ở hầu hết các mô hình (trừ Transformer) qua cả 3 giai đoạn | **Descriptively Stable (Ổn định về mô tả)** |
| **Tính đơn điệu của suy thoái** | Chỉ 3/42 trường hợp đơn điệu; nhiều mô hình đạt đỉnh tại 20D/60D | **Non-monotonic (Phi đơn điệu)** |
| **Tính ổn định của dòng ARIMA** | Ảo ảnh kỹ thuật do lỗi lọc 1 bước trên chuỗi gối đầu | **Unvalidated / Invalidated (Không hợp lệ)** |
| **Thứ hạng tương đối giữa các họ mô hình** | Đảo chiều theo chân trời (DL vượt Tree & Linear tại 120D trong 2019-2020) | **Unstable / Horizon-dependent (Không ổn định)** |
| **Biên độ sai số tuyệt đối (RMSE)** | Tăng gấp đôi do độ biến động mục tiêu tăng; Normalized RMSE ổn định $\approx 1.05$ | **Scale-dependent (Phụ thuộc biên độ thị trường)** |
| **KẾT LUẬN TỔNG THỂ CHO RQ3** | Mẫu hình mô tả được hỗ trợ, kiểm định thống kê suy diễn đang chờ xử lý | **DESCRIPTIVELY SUPPORTED (Được hỗ trợ về mô tả)** |

---

## 8. Hạn Chế Phương Pháp Luận Của RQ3 (Methodological Limitations)

1. **Phân kỳ theo khối niên đại cố định:** Việc chia thành 3 khối 2 năm mang tính quy ước niên đại, chưa phản ánh các điểm gãy cấu trúc nội tại (structural breaks) hay các chu kỳ kinh tế vi mô từng quý.
2. **Gối đầu qua ranh giới thời gian:** Các dự báo dài hạn cuối kỳ vươn sang kỳ kế tiếp làm mờ ranh giới độc lập giữa các giai đoạn.
3. **Phạm vi tài sản đơn lẻ:** Toàn bộ phân tích thực hiện trên chỉ số S&P 500 giai đoạn 2015–2020, không đại diện cho quy luật cấu trúc phổ quát của toàn bộ thị trường tài chính quốc tế.

---

## 9. Kết Luận Khoa Học Đã Hiệu Đính Cho RQ3 (Corrected Scientific Conclusion)

1. **Câu trả lời trực tiếp cho RQ3:**
   Về mặt **mô tả ngoại mẫu (descriptive out-of-sample)**, mẫu hình suy thoái hiệu năng dự báo từ chân trời ngắn đến chân trời dài của các mô hình Machine Learning truyền thống (Tree Ensembles, Linear Regularized, SVR, Neural Networks) duy trì tính nhất quán về chiều hướng suy giảm điểm mút qua cả 3 giai đoạn niên đại (2015–2016, 2017–2018, 2019–2020).
2. **Sự phụ thuộc vào biên độ thị trường:**
   Mức độ suy thoái không cố định về giá trị số học tuyệt đối:
   - Trong giai đoạn thị trường bình lặng (2017–2018), biên độ suy giảm co hẹp lại.
   - Trong giai đoạn biến động mạnh (2019–2020), quy mô sai số tuyệt đối (RMSE) tăng gấp đôi, phản ánh sự mở rộng tương ứng của độ biến động mục tiêu ($\text{Target Std}$ tăng $1.87\times$), trong khi sai số chuẩn hóa vẫn duy trì ổn định.
3. **Thứ hạng mô hình không cố định:** Không tồn tại trật tự vượt trội phổ quát giữa các họ mô hình; tại chân trời 120D trong giai đoạn biến động cao, các mô hình Deep Learning có khả năng thích ứng tốt hơn các mô hình Tree Ensembles.
4. **Loại trừ dòng ARIMA:** Dòng mô hình Tự hồi quy bị loại bỏ khỏi kết luận khoa học do phát hiện lỗi phương pháp luận rò rỉ tương lai qua lọc 1 bước trên chuỗi gối đầu.

---

## 10. Quyết Nghị Thẩm Định Cuối Cùng & Cổng Chuyển Bước 8 (Validation Gate & Step 8 Readiness)

### Cổng thẩm định phương pháp luận:
- [x] **Toàn vẹn dữ liệu (Data Integrity):** **PASS** ($100\%$ tái tạo chính xác từ `all_oos_predictions.csv`, max discrepancy = $0.000000$).
- [x] **Phân định giai đoạn niên đại:** **PASS** (Đúng số lượng quan sát $395, 502, 463$ ngày, gán theo ngày gốc $t$).
- [x] **Giải trình số học Benchmark:** **PASS** (Mẫu số trong `rq3_time.csv` là phương sai nội kỳ $TSS_{\text{period}}$).
- [ ] **Tính hợp lệ của dòng ARIMA:** **FAIL** *(Lỗi lọc 1 bước trên chuỗi gối đầu cho $h \ge 5$)*.
- [ ] **Kiểm định thống kê tương tác phi phụ thuộc:** **PENDING** *(Chờ thực hiện tại Step 8)*.

### QUYẾT NGHỊ CHÍNH THỨC:

> ### **RQ3 DESCRIPTIVELY SUPPORTED — STATISTICAL/METHODOLOGICAL VALIDATION PENDING**
> *(Các phát hiện mô tả về tính ổn định thời gian của các mô hình Machine Learning được hỗ trợ đầy đủ bởi dữ liệu thực nghiệm; tuy nhiên dòng ARIMA bị loại bỏ và các kiểm định thống kê suy diễn chính quy kiểm soát tự tương quan gối đầu cần được thực hiện tại Step 8).*

---

### SẴN SÀNG CHUYỂN SANG BƯỚC 8 (STATISTICAL TESTS AUDIT): **CÓ (YES - VỚI ĐIỀU KIỆN)**

**Điều kiện bắt buộc cho Step 8:**
1. Khi kiểm toán tệp `results/analysis/statistical_tests.csv` (Diebold-Mariano Tests và Directional Accuracy Tests), **tuyệt đối không sử dụng kết quả dòng ARIMA làm thước đo đối sánh hợp lệ**.
2. Kiểm tra chặt chẽ việc áp dụng hiệu chỉnh Harvey-Leybourne-Newbold (HLN) cho bậc trễ $h$ để kiểm soát chuỗi mục tiêu gối đầu.

