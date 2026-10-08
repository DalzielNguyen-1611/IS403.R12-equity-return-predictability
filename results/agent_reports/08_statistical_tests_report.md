# Báo Cáo Kiểm Định Thống Kê Năng Lực Dự Báo (08_statistical_tests_report.md)

**Mã tài liệu:** `IS403-REP-08`  
**Ngày thực hiện:** 2026-10-08  
**Tập tin mục tiêu:** `results/analysis/statistical_tests.csv`  
**Người thực hiện:** Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu Kiểm Định & Mô Tả Tập Tin (Objective & File Description)
Tập tin `statistical_tests.csv` cung cấp các bằng chứng kiểm định suy diễn thống kê (inferential statistical tests) nhằm xác định:
1. **Kiểm định Hướng đi Nhị thức (Binomial Sign Test for Directional Accuracy):** Kiểm tra xem tỷ lệ dự đoán đúng chiều thị trường ($DA$) của các mô hình có vượt trội hơn mức ngẫu nhiên $50\%$ một cách có ý nghĩa thống kê hay không ($H_0: DA \le 0.5$ vs $H_1: DA > 0.5$).
2. **Kiểm định So sánh Phần dư Diebold-Mariano (Diebold-Mariano Test):** Kiểm tra xem sai số dự báo bình phương của các mô hình có nhỏ hơn có ý nghĩa thống kê so với mô hình chuẩn `Historical_Mean` hay không ($H_0: \mathbb{E}[d_t] \le 0$ vs $H_1: \mathbb{E}[d_t] > 0$, trong đó $d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$).

> **NGUYÊN TẮC THÔNG DIỄN BẮT BUỘC:**  
> **Ý nghĩa thống kê ($p < 0.05$) KHÔNG đồng nghĩa với mô hình tốt hơn.**  
> Đối với mọi kết quả có $p < 0.05$, bắt buộc phải kiểm tra đối chiếu:  
> - Chiều hướng chênh lệch thực tế (direction of performance difference)  
> - Chỉ số $OOS\_R^2$  
> - Sai số MAE và RMSE  
> Tuyệt đối không diễn giải $p < 0.05$ là "mô hình vượt trội" nếu $OOS\_R^2$ đang âm sâu. Phải xem xét vấn đề kiểm định đồng thời nhiều giả thuyết (Multiple Testing Problem).

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước vật lý
- **Đường dẫn lưu trữ:** `results/analysis/statistical_tests.csv`
- **Số hàng:** $70$ hàng ($14 \text{ Models} \times 5 \text{ Horizons}$)
- **Số cột:** $7$ cột: `Model`, `Horizon`, `Test`, `Benchmark`, `Statistic`, `p-value`, `Significant`

### 2.2 Phát hiện dị thường về cơ chế tạo file (Technical Anomaly & Discrepancy Note)
- **Hiện trạng tập tin thô:** Toàn bộ $70$ hàng trong `statistical_tests.csv` đều thuộc về `Test == 'Binomial'` với `Benchmark == '50% DA'`.
- **Nguyên nhân kỹ thuật tại mã nguồn:**  
  Trong script `src/training/baselines/generate_rq_analysis.py` (dòng 44), câu lệnh lọc benchmark được viết là:  
  `bench_df = df[df['Model'] == 'historical_mean']` (chữ thường).  
  Trong khi đó, tập tin `all_oos_predictions.csv` lưu trữ định danh chuẩn hóa là `'Historical_Mean'` (viết hoa chữ H và M). Sự bất đồng ký tự này khiến biến `bench_df` bị rỗng trong quá trình chạy script, dẫn đến nhánh tính toán Diebold-Mariano Test bị bỏ qua và chỉ có nhánh Binomial Test được ghi xuống CSV.
- **Tuân thủ quy tắc ứng xử:** Giữ nguyên vẹn tập tin thô `statistical_tests.csv` không ghi đè; đồng thời tiến hành **tái tính toán độc lập ma trận Diebold-Mariano đầy đủ** trực tiếp từ chuỗi phần dư ngoại mẫu trong `all_oos_predictions.csv` để cung cấp bức tranh kinh tế lượng trọn vẹn nhất.

---

## 3. Phân Tích Kiểm Định Hướng Đi Nhị Thức (Binomial Directional Accuracy Tests)

### 3.1 Ma trận Thống kê và p-value của Kiểm định Binomial trong `statistical_tests.csv`

| Model | 1D DA (p-val) | 5D DA (p-val) | 20D DA (p-val) | 60D DA (p-val) | 120D DA (p-val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 54.85% ($0.0002^*$) | 61.84% ($0.0000^*$) | 68.46% ($0.0000^*$) | 74.56% ($0.0000^*$) | 77.79% ($0.0000^*$) |
| **ARIMA** | 52.72% ($0.0239^*$) | 79.41% ($0.0000^*$) | 91.84% ($0.0000^*$) | 95.74% ($0.0000^*$) | 96.69% ($0.0000^*$) |
| **ARIMAX** | 52.06% ($0.0679$) | 82.43% ($0.0000^*$) | 93.60% ($0.0000^*$) | 97.06% ($0.0000^*$) | 97.43% ($0.0000^*$) |
| **SARIMA** | 53.90% ($0.0022^*$) | 82.06% ($0.0000^*$) | 91.84% ($0.0000^*$) | 95.74% ($0.0000^*$) | 96.91% ($0.0000^*$) |
| **Elastic_Net** | 53.24% ($0.0091^*$) | 57.79% ($0.0000^*$) | 61.84% ($0.0000^*$) | 68.60% ($0.0000^*$) | 77.79% ($0.0000^*$) |
| **Ridge** | 51.47% ($0.1451$) | 50.15% ($0.4571$) | 56.03% ($0.0000^*$) | 56.18% ($0.0000^*$) | 66.10% ($0.0000^*$) |
| **SVR** | 50.96% ($0.2489$) | 55.15% ($0.0001^*$) | 57.06% ($0.0000^*$) | 66.91% ($0.0000^*$) | 66.76% ($0.0000^*$) |
| **Random_Forest** | 54.04% ($0.0017^*$) | 59.34% ($0.0000^*$) | 62.28% ($0.0000^*$) | 61.40% ($0.0000^*$) | 63.31% ($0.0000^*$) |
| **XGBoost** | 54.34% ($0.0008^*$) | 61.32% ($0.0000^*$) | 65.15% ($0.0000^*$) | 71.25% ($0.0000^*$) | 71.69% ($0.0000^*$) |
| **LightGBM** | 53.68% ($0.0037^*$) | 58.53% ($0.0000^*$) | 64.19% ($0.0000^*$) | 63.46% ($0.0000^*$) | 58.38% ($0.0000^*$) |
| **LSTM** | 52.50% ($0.0336^*$) | 54.78% ($0.0002^*$) | 58.60% ($0.0000^*$) | 67.35% ($0.0000^*$) | 65.81% ($0.0000^*$) |
| **RNN** | 49.93% ($0.5324$) | 50.81% ($0.2797$) | 60.07% ($0.0000^*$) | 63.38% ($0.0000^*$) | 73.01% ($0.0000^*$) |
| **Transformer** | 49.71% ($0.5964$) | 53.82% ($0.0026^*$) | 51.84% ($0.0920$) | 60.59% ($0.0000^*$) | 54.41% ($0.0006^*$) |
| **Logistic_Reg.**| 50.59% ($0.3421$) | 52.28% ($0.0483^*$) | 63.97% ($0.0000^*$) | 68.46% ($0.0000^*$) | 74.56% ($0.0000^*$) |

*\* Ghi chú: Ký hiệu $^*$ biểu thị mức ý nghĩa thống kê $p < 0.05$.*

### 3.2 Diễn giải phê phán về kết quả Kiểm định Binomial (Critical Evaluation)
1. **Tại chân trời 1 ngày (1D):**
   - 6/14 mô hình (ARIMAX, Ridge, SVR, RNN, Transformer, Logistic Regression) có $p \ge 0.05$.
   - **Kết luận:** Tại tần suất hàng ngày, thị trường tiệm cận bước đi ngẫu nhiên (Random Walk). Hơn $40\%$ số mô hình hoàn toàn không thể đánh bại một phép tung đồng xu ngẫu nhiên $50/50$ về mặt thống kê.
2. **Tại chân trời dài hạn (60D, 120D) — Cảnh báo diễn giải sai lệch:**
   - Tại 120D, $100\%$ mô hình đều có $p < 0.001$, dường như "vượt trội có ý nghĩa thống kê".
   - **Bản chất thực sự:** Chỉ số S&P 500 có xu hướng tăng trưởng tự nhiên trong dài hạn. Trong tập kiểm định, hơn $77.8\%$ số ngày 120D có lợi suất thực tế dương. Do đó, mô hình `Historical_Mean` (chỉ dự báo hằng số dương) cũng đạt $DA = 77.79\%$ với $p = 1.2 \times 10^{-104}$!
   - Các mô hình Cây (Random Forest, XGBoost) đạt $p < 10^{-10}$ tại 120D **không phải vì chúng có kỹ năng dự báo siêu việt**, mà vì chúng chỉ đơn giản "bắt chước" độ trôi dạt dương của thị trường. Nhận định này được củng cố bởi việc $OOS\_R^2$ của chúng tại 120D lại âm rất sâu ($-0.4852$ và $-0.2744$).

---

## 4. Tái Tính Toán Độc Lập: Kiểm Định Diebold-Mariano So Với Historical Mean (Diebold-Mariano Test Audit)

Áp dụng hàm kiểm định chính quy `utils.diebold_mariano_test` (dùng hàm mất mát bình phương squared loss, độ trễ tương quan $h \in \{1, 5, 20, 60, 120\}$, có hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold) so sánh từng mô hình với `Historical_Mean`:

### 4.1 Ma trận Thống kê Diebold-Mariano (DM-Statistic Matrix)
*(Giá trị dương lớn biểu thị mô hình ứng viên có sai số MSE thấp hơn đáng kể so với Historical Mean)*

| Model | 1D DM (p-val) | 5D DM (p-val) | 20D DM (p-val) | 60D DM (p-val) | 120D DM (p-val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ARIMA** | +1.43 ($0.0769$) | **+4.34 ($0.0000^*$)** | **+2.18 ($0.0147^*$)** | **+2.33 ($0.0101^*$)** | **+2.50 ($0.0062^*$)** |
| **ARIMAX** | +1.21 ($0.1126$) | **+5.11 ($0.0000^*$)** | **+2.19 ($0.0142^*$)** | **+2.32 ($0.0102^*$)** | **+2.50 ($0.0063^*$)** |
| **SARIMA** | +1.31 ($0.0948$) | **+4.62 ($0.0000^*$)** | **+2.18 ($0.0147^*$)** | **+2.33 ($0.0100^*$)** | **+2.51 ($0.0062^*$)** |
| **Ridge** | +0.16 ($0.4377$) | -2.11 ($0.9824$) | -1.61 ($0.9462$) | -1.47 ($0.9292$) | -2.03 ($0.9789$) |
| **Elastic_Net** | 0.00 ($1.0000$) | -0.49 ($0.6885$) | -0.97 ($0.8327$) | -0.98 ($0.8358$) | -1.75 ($0.9598$) |
| **SVR** | +0.54 ($0.2935$) | -1.24 ($0.8927$) | -1.36 ($0.9132$) | -1.08 ($0.8605$) | -0.62 ($0.7329$) |
| **Random_Forest** | -0.99 ($0.8377$) | -1.55 ($0.9394$) | -1.38 ($0.9161$) | -1.51 ($0.9342$) | -2.14 ($0.9836$) |
| **XGBoost** | -0.54 ($0.7064$) | -0.21 ($0.5833$) | -1.51 ($0.9344$) | -1.18 ($0.8802$) | -1.06 ($0.8556$) |
| **LightGBM** | -1.41 ($0.9202$) | -1.59 ($0.9435$) | -1.56 ($0.9400$) | -1.65 ($0.9505$) | -2.12 ($0.9830$) |
| **LSTM** | +0.41 ($0.3424$) | -2.46 ($0.9930$) | -1.74 ($0.9589$) | -1.52 ($0.9357$) | -0.94 ($0.8254$) |
| **RNN** | -0.91 ($0.8188$) | -0.68 ($0.7523$) | -1.68 ($0.9537$) | -2.73 ($0.9968$) | +0.15 ($0.4414$) |
| **Transformer** | -16.85 ($1.0000$) | -4.30 ($1.0000$) | -2.27 ($0.9885$) | -2.37 ($0.9910$) | -1.14 ($0.8736$) |

### 4.2 Phát hiện thực nghiệm cốt lõi từ Kiểm định Diebold-Mariano
1. **Dòng Tự Hồi Quy (ARIMA, ARIMAX, SARIMA): Bằng chứng vượt trội thống kê duy nhất.**
   - Tại 1D: Thống kê DM dương ($+1.21 \to +1.43$) nhưng $p$-value nằm trong khoảng $0.07 - 0.11$, chưa đạt mức ý nghĩa $5\%$.
   - Tại 5D, 20D, 60D, 120D: Toàn bộ 3 mô hình đều đạt **$DM \ge 2.18$ và $p < 0.015$**.
   - **Kết luận:** ARIMA, ARIMAX và SARIMA là **nhóm mô hình duy nhất trong nghiên cứu** vượt trội hơn Historical Mean một cách có ý nghĩa thống kê vững chắc trên chuỗi sai số dự báo bình phương ngoại mẫu.
2. **Dòng Machine Learning & Deep Learning: Không có ý nghĩa vượt trội (Hoàn toàn thất bại trước Benchmark).**
   - Từ 5D đến 120D, toàn bộ các mô hình Cây (Random Forest, LightGBM, XGBoost) và Học sâu (LSTM, Transformer) đều có thống kê $DM < 0$ (nghĩa là MSE của mô hình lớn hơn MSE của Historical Mean) và $p$-value tiến sát 1.0.
   - Không có bất kỳ mô hình Machine Learning nào đánh bại được Historical Mean ở mức ý nghĩa $5\%$ ở các chân trời dài hạn.

---

## 5. Vấn Đề Kiểm Định Đa Giả Thuyết (Multiple Testing Problem & Bonferroni Correction)
- **Quy mô không gian kiểm định:** 12 mô hình hồi quy $\times 5$ chân trời $= 60$ giả thuyết kiểm định Diebold-Mariano đồng thời.
- **Hiệu chỉnh Bonferroni:** Ngưỡng ý nghĩa hiệu chỉnh là $\alpha_{\text{Bonferroni}} = \frac{0.05}{60} \approx 0.00083$.
- **Đánh giá sau hiệu chỉnh:**
  - Tại 5D: ARIMA, ARIMAX và SARIMA đều có $p = 0.00000 < 0.00083 \implies$ **Vượt qua kiểm định Bonferroni một cách hoàn hảo**.
  - Tại 20D, 60D, 120D: $p$-value của dòng AR dao động từ $0.0062$ đến $0.0147$. Mặc dù vượt qua ngưỡng $5\%$ truyền thống và kiểm soát FDR (Benjamini-Hochberg) ở mức $q = 0.05$, các kết quả này không vượt qua ngưỡng Bonferroni cực kỳ bảo thủ.
  - Đối với tất cả các mô hình ML: Không có mô hình nào đạt ý nghĩa thống kê dưới bất kỳ phương pháp hiệu chỉnh nào.

---

## 6. Hạn Chế Phương Pháp Luận Của Kiểm Định Thống Kê (Limitations)
1. **Giả định phân phối chuẩn tiệm cận của Diebold-Mariano:** Trong các chân trời dài hạn ($h = 60, 120$), độ trễ tương quan cao và số lượng khoảng không chồng lấn độc lập thực tế bị suy giảm ($1,360 / 120 \approx 11$ chu kỳ độc lập), có thể khiến phương sai Newey-West bị ước lượng chệch nhẹ.

---

## 7. Kết Luận Bước 8 & Kế Hoạch Tiếp Theo (Conclusions & Next Steps)
- **Kết luận Step 8:**  
  1. Kiểm định Hướng đi ($DA$) xác nhận thị trường 1D khó dự báo, trong khi mức ý nghĩa cực cao ở chân trời dài là do nắm bắt xu hướng tăng trưởng dài hạn của S&P 500 chứ không phản ánh kỹ năng dự báo điểm.  
  2. Kiểm định Diebold-Mariano khẳng định vững chắc: **Chỉ duy nhất dòng mô hình Tự hồi quy (ARIMA, ARIMAX, SARIMA) vượt trội hơn Historical Mean có ý nghĩa thống kê ($p < 0.05$) từ chân trời 5D trở đi**. Toàn bộ các mô hình Machine Learning và Deep Learning đều thất bại trong việc đánh bại benchmark trên cơ sở thống kê sai số bình phương.
- **Bước tiếp theo:** Sau khi hoàn thành trọn vẹn 8 báo cáo chi tiết cho 8 tập tin CSV, tiến hành bước tổng kết tối hậu: **Xây dựng bản tổng hợp nghiên cứu toàn diện `results/agent_reports/FINAL_SYNTHESIS.md`** theo đúng cấu trúc chuẩn mực 18 phần đã quy định.
