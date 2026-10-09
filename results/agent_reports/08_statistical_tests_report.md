# Báo Cáo Kiểm Định Thống Kê Năng Lực Dự Báo (08_statistical_tests_report.md)

**Mã tài liệu:** `IS403-REP-08`  
**Ngày thực hiện:** 2026-10-08  
**Ngày kiểm toán & hiệu chỉnh:** 2026-10-09  
**Tập tin dữ liệu gốc:** `results/analysis/statistical_tests.csv`  
**Tập tin dữ liệu hiệu chỉnh:** `results/analysis/statistical_tests_corrected.csv`  
**Người thực hiện:** Senior Research Methodology Auditor & Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu Kiểm Định & Cơ Sở Phương Pháp Luận (Objective & Methodological Framework)

Báo cáo này cung cấp kết quả kiểm định suy diễn thống kê (inferential statistical tests) nhằm trả lời hai câu hỏi nghiên cứu cốt lõi:
1. **Kiểm định Hướng đi Nhị thức (Binomial Sign Test for Directional Accuracy):** Kiểm tra xem tỷ lệ dự đoán đúng chiều thị trường ($DA$) của các mô hình có cao hơn mức tung đồng xu ngẫu nhiên $50\%$ một cách có ý nghĩa thống kê hay không ($H_0: DA \le 0.5$ vs $H_1: DA > 0.5$).
2. **Kiểm định So sánh Phần dư Diebold-Mariano (Diebold-Mariano Test):** Kiểm tra xem sai số dự báo bình phương của các mô hình ứng viên có nhỏ hơn có ý nghĩa thống kê so với mô hình mốc chuẩn `Historical_Mean` hay không ($H_0: \mathbb{E}[d_t] \le 0$ vs $H_1: \mathbb{E}[d_t] > 0$, trong đó vi sai mất mát được định nghĩa là $d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$).

> [!IMPORTANT]
> **NGUYÊN TẮC THÔNG DIỄN PHƯƠNG PHÁP LUẬN BẮT BUỘC:**  
> 1. **Ý nghĩa thống kê ($p < 0.05$) KHÔNG đồng nghĩa với mô hình có năng lực dự báo vượt trội:**  
>    - Đối với kiểm định nhị thức $DA > 0.5$, nếu thị trường có độ trôi dương mạnh (ví dụ $77.8\%$ số ngày có lợi suất dương), một mô hình ngây thơ luôn đoán chiều dương sẽ đạt $DA = 77.8\%$ với $p < 10^{-90}$ mà hoàn toàn không có kỹ năng dự báo điểm. Bắt buộc phải đối chiếu với tỷ lệ cơ sở (Base Rate) và mốc chuẩn luôn đoán dương (Always-Positive Benchmark).  
> 2. **Kiểm định 1 phía không bác bỏ được $H_0$ ($p > 0.5$) chỉ thể hiện "thiếu bằng chứng vượt trội":**  
>    - Không được quy kết là "mô hình tương đương" hay "hoàn toàn thất bại". Để khẳng định mô hình kém hơn một cách có ý nghĩa thống kê, cần thực hiện kiểm định hai phía ($H_1: \mathbb{E}[d_t] \neq 0$).  
> 3. **Phải kiểm soát kiểm định đa giả thuyết (Multiple Testing Correction):**  
>    - Với 60 cặp so sánh mô hình, việc chỉ nhìn vào $p$-value thô sẽ dẫn đến lạm phát sai lầm loại I (Family-Wise Error Rate). Phải áp dụng hiệu chỉnh Bonferroni và kiểm soát tỷ lệ khám phá sai Benjamini-Hochberg (FDR).  
> 4. **Điều kiện tiên quyết về tính hợp lệ của mô hình:**  
>    - Mọi kết luận thống kê chỉ có giá trị khi quy trình tạo dự báo không vi phạm rò rỉ dữ liệu (data leakage) và căn chỉnh thời gian chuẩn xác.

---

## 2. Kích Thước Dữ Liệu & Kiểm Toán Mã Nguồn (Data Audit & Technical Discrepancy Note)

### 2.1 Hiện trạng tập tin thô `results/analysis/statistical_tests.csv`
- **Số hàng:** $70$ hàng ($14 \text{ Models} \times 5 \text{ Horizons}$).
- **Số cột:** $7$ cột (`Model`, `Horizon`, `Test`, `Benchmark`, `Statistic`, `p-value`, `Significant`).
- **Phát hiện dị thường:** Toàn bộ 70 hàng trong tập tin thô đều là kiểm định `Test == 'Binomial'` so với `Benchmark == '50% DA'`. Tập tin thô **hoàn toàn không chứa bất kỳ hàng kiểm định Diebold-Mariano nào**.

### 2.2 Nguyên nhân kỹ thuật tại mã nguồn pipeline
- Khi kiểm toán tập tin [generate_rq_analysis.py](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/generate_rq_analysis.py#L44), câu lệnh lọc benchmark được viết là:
  ```python
  bench_df = df[df['Model'] == 'historical_mean']  # Dòng 44 & 223
  ```
- Tuy nhiên, trong tập tin dự báo gốc [all_oos_predictions.csv](file:///e:/IS403.R12-equity-return-predictability/results/predictions/all_oos_predictions.csv), tên mô hình chuẩn hóa là `'Historical_Mean'` (viết hoa chữ cái đầu H và M).
- Do sự không khớp chuỗi ký tự này, `bench_df` bị rỗng hoàn toàn, khiến khối mã chạy kiểm định Diebold-Mariano (dòng 218–250) bị bỏ qua trong quá trình sinh tập tin CSV phân tích. Bảng Diebold-Mariano trong các báo cáo sơ bộ trước đây chỉ được trích xuất từ một script ad-hoc mà không lưu trữ vào CSV chính thức.

### 2.3 Giải pháp xử lý và tập tin hiệu chỉnh
- Giữ nguyên vẹn tập tin thô `results/analysis/statistical_tests.csv` để bảo tồn lịch sử thực nghiệm.
- Xuất bản tập tin phân tích thống kê hiệu chỉnh toàn diện: **`results/analysis/statistical_tests_corrected.csv`** với 130 dòng dữ liệu (70 dòng Binomial + 60 dòng Diebold-Mariano), tích hợp đầy đủ các giá trị $p$ không làm tròn, $p$ hiệu chỉnh Bonferroni, $p$ kiểm soát FDR Benjamini-Hochberg, phân phối dấu thực tế và trạng thái phương pháp luận.

---

## 3. Kiểm Định Hướng Đi Nhị Thức (Binomial Directional Accuracy Tests)

### 3.1 Ma trận Thống kê DA và p-value (Kiểm định 1 phía so với $50\%$)

Kiểm định giả thuyết: $H_0: DA \le 0.5$ so với $H_1: DA > 0.5$.  
Số phép thử ngoại mẫu: $N = 1,360$ ngày giao dịch.

| Model | 1D DA ($p$-val) | 5D DA ($p$-val) | 20D DA ($p$-val) | 60D DA ($p$-val) | 120D DA ($p$-val) | Tỷ lệ Dự báo Dương (120D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 54.85% ($1.89 \times 10^{-4*}$) | 61.84% ($1.11 \times 10^{-18*}$) | 68.46% ($2.69 \times 10^{-43*}$) | 74.56% ($1.40 \times 10^{-76*}$) | **77.79%** ($8.04 \times 10^{-99*}$) | **100.0%** |
| **ARIMA** | 52.72% ($0.0239^*$) | 79.41% ($2.93 \times 10^{-111*}$) | 91.84% ($1.63 \times 10^{-244*}$) | 95.74% ($2.87 \times 10^{-307*}$) | 96.69% ($0.0000^*$) | 77.87% |
| **ARIMAX** | 52.06% ($0.0679$) | 82.43% ($5.30 \times 10^{-137*}$) | 93.60% ($5.05 \times 10^{-271*}$) | 97.06% ($0.0000^*$) | 97.43% ($0.0000^*$) | 78.46% |
| **SARIMA** | 53.90% ($0.0022^*$) | 82.06% ($5.87 \times 10^{-134*}$) | 91.84% ($1.63 \times 10^{-244*}$) | 95.74% ($2.87 \times 10^{-307*}$) | 96.91% ($0.0000^*$) | 77.94% |
| **Elastic_Net** | 53.24% ($0.0091^*$) | 57.79% ($4.94 \times 10^{-9*}$) | 61.84% ($1.11 \times 10^{-18*}$) | 68.60% ($5.64 \times 10^{-44*}$) | **77.79%** ($8.04 \times 10^{-99*}$) | **100.0%** |
| **Ridge** | 51.47% ($0.1451$) | 50.15% ($0.4571$) | 56.03% ($4.62 \times 10^{-6*}$) | 56.18% ($3.25 \times 10^{-6*}$) | 66.10% ($2.55 \times 10^{-32*}$) | 80.96% |
| **SVR** | 50.96% ($0.2489$) | 55.15% ($9.62 \times 10^{-5*}$) | 57.06% ($2.91 \times 10^{-7*}$) | 66.91% ($1.19 \times 10^{-36*}$) | 66.76% ($1.76 \times 10^{-35*}$) | 82.06% |
| **Random_Forest** | 54.04% ($0.0017^*$) | 59.34% ($1.91 \times 10^{-12*}$) | 62.28% ($1.15 \times 10^{-19*}$) | 61.40% ($1.05 \times 10^{-17*}$) | 63.31% ($4.85 \times 10^{-23*}$) | 76.76% |
| **XGBoost** | 54.34% ($0.0008^*$) | 61.32% ($1.91 \times 10^{-17*}$) | 65.15% ($4.02 \times 10^{-29*}$) | 71.25% ($3.35 \times 10^{-57*}$) | 71.69% ($1.34 \times 10^{-59*}$) | 91.40% |
| **LightGBM** | 53.68% ($0.0037^*$) | 58.53% ($7.87 \times 10^{-11*}$) | 64.19% ($2.54 \times 10^{-26*}$) | 63.46% ($2.59 \times 10^{-23*}$) | 58.38% ($5.90 \times 10^{-10*}$) | 70.00% |
| **LSTM** | 52.50% ($0.0346^*$) | 54.78% ($2.32 \times 10^{-4*}$) | 58.60% ($1.20 \times 10^{-10*}$) | 67.35% ($2.12 \times 10^{-38*}$) | 65.81% ($2.69 \times 10^{-31*}$) | 81.32% |
| **RNN** | 49.93% ($0.5324$) | 50.81% ($0.2797$) | 60.07% ($6.50 \times 10^{-14*}$) | 63.38% ($3.93 \times 10^{-23*}$) | 73.01% ($7.67 \times 10^{-68*}$) | 90.74% |
| **Transformer** | 49.71% ($0.5964$) | 53.82% ($0.0026^*$) | 51.84% ($0.0920$) | 60.59% ($5.01 \times 10^{-15*}$) | 54.41% ($5.76 \times 10^{-4*}$) | 64.93% |
| **Logistic_Reg.**| 50.59% ($0.3421$) | 52.28% ($0.0483^*$) | 63.97% ($5.49 \times 10^{-26*}$) | 68.46% ($1.37 \times 10^{-43*}$) | 74.56% ($1.40 \times 10^{-76*}$) | 93.82% |

*\* Ghi chú: Ký hiệu $^*$ biểu thị $p < 0.05$. Giá trị hiển thị $0.0000$ là do giới hạn số học dấu phẩy động ($p < 10^{-300}$). Lưu ý: Dòng ARIMA bị loại trừ khỏi suy diễn khoa học hợp lệ do lỗi rò rỉ dữ liệu.*

### 3.2 Phân tích Phê phán Kiểm định Hướng đi (Critical Evaluation)

1. **Thực tế tại chân trời ngắn (1D):**
   - 6/14 mô hình (ARIMAX, Ridge, SVR, RNN, Transformer, Logistic Regression) có $p \ge 0.05$.
   - Các mô hình còn lại chỉ đạt DA dao động hẹp quanh $52.5\% - 54.8\%$.
   - **Nhận định:** Tại tần suất hàng ngày, tỷ lệ nhiễu cao khiến hầu hết các cấu trúc mô hình không thể hiện được lợi thế vượt trội có ý nghĩa thống kê so với mức đoán ngẫu nhiên.
2. **Cảnh báo sai lệch diễn giải tại chân trời dài (60D, 120D) — Hiệu ứng Tỷ lệ Cơ sở (Base Rate Effect):**
   - Tại 120D, $100\%$ mô hình đều có $p < 0.001$, tạo ấn tượng sai lầm rằng các mô hình sở hữu kỹ năng dự báo siêu việt.
   - **Bản chất thực nghiệm:** Chỉ số S&P 500 giai đoạn 2015–2020 có xu hướng tăng trưởng dài hạn. Tỷ lệ số ngày có lợi suất 120D dương đạt **$77.79\%$** ($1,058 / 1,360$ ngày).
   - Một chiến lược ngây thơ **Luôn Dự báo Dương (Always-Positive Benchmark)** hoặc mô hình `Historical_Mean` (dự báo hằng số dương) tự động đạt đúng **$DA = 77.79\%$** với $p = 8.04 \times 10^{-99}$.
   - **So sánh thực chất:**
     - Mô hình Elastic Net co toàn bộ hệ số về 0, dự báo đúng bằng Historical Mean ($100\%$ dương), nên đạt $DA = 77.79\%$.
     - **Tất cả các mô hình Machine Learning và Deep Learning còn lại đều kém hơn Always-Positive Benchmark:** Logistic Regression ($74.56\%$), RNN ($73.01\%$), XGBoost ($71.69\%$), SVR ($66.76\%$), Ridge ($66.10\%$), LSTM ($65.81\%$), Random Forest ($63.31\%$), LightGBM ($58.38\%$), Transformer ($54.41\%$).
   - **Kết luận:** DA cao ở chân trời dài hoàn toàn là hệ quả cơ học của độ trôi dương của thị trường kết hợp với thiên kiến dự báo dương của mô hình. Không có mô hình học máy nào tạo ra được giá trị gia tăng định hướng so với một quy tắc đoán xu hướng ngây thơ.
3. **Vi phạm giả định quan sát độc lập ($iid$):**
   - Lợi suất lũy kế $Target_{h,t} = \log(Close_{t+h}/Close_t)$ có $h-1$ ngày gối đầu. Tại 120D, hai ngày liên tiếp chia sẻ 119 ngày giá chung.
   - Kiểm định Binomial cổ điển giả định $1,360$ phép thử độc lập là không hợp lệ về mặt kinh tế lượng, thổi phồng thống kê kiểm định và đẩy $p$-value về sát 0. Do đó, các giá trị $p$ trong Bảng 3.1 chỉ được xem như chỉ số mô tả (descriptive metrics), không phải suy diễn thống kê chuẩn mực.

---

## 4. Kiểm Định Diebold-Mariano So Với Historical Mean (Diebold-Mariano Test Audit)

Áp dụng hàm kiểm định chính quy `utils.diebold_mariano_test` với:
- Hàm mất mát bình phương (Squared Loss Differential): $d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$.
- Thống kê $d_t > 0$ nghĩa là mô hình ứng viên có bình phương sai số nhỏ hơn Historical Mean (mô hình tốt hơn).
- Tham số trễ Newey-West: $\text{lag} = h$ (phù hợp và bảo thủ nhẹ so với bậc $MA(h-1)$ của sai số dự báo tối ưu).
- Hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold (HLN).
- Kiểm định so sánh trên toàn bộ 12 mô hình hồi quy (loại trừ Logistic Regression do sai lệch không gian mục tiêu nhị phân / xác suất).

### 4.1 Bảng Ma trận Kiểm định Diebold-Mariano (Thống kê DM và p-value 1 phía)

| Model | 1D DM ($p$-val) | 5D DM ($p$-val) | 20D DM ($p$-val) | 60D DM ($p$-val) | 120D DM ($p$-val) | Đánh giá Phương pháp luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARIMA** | +1.43 ($0.0769$) | **+4.34 ($7.61 \times 10^{-6*}$)** | **+2.18 ($0.0147^*$)** | **+2.33 ($0.0101^*$)** | **+2.50 ($0.0062^*$)** | **Không hợp lệ (Rò rỉ lọc 1 bước)** |
| **ARIMAX** | +1.21 ($0.1126$) | **+5.11 ($1.81 \times 10^{-7*}$)** | **+2.19 ($0.0142^*$)** | **+2.32 ($0.0102^*$)** | **+2.50 ($0.0063^*$)** | **Không hợp lệ (Rò rỉ lọc 1 bước)** |
| **SARIMA** | +1.31 ($0.0948$) | **+4.62 ($2.06 \times 10^{-6*}$)** | **+2.18 ($0.0147^*$)** | **+2.33 ($0.0100^*$)** | **+2.51 ($0.0062^*$)** | **Không hợp lệ (Rò rỉ lọc 1 bước)** |
| **Ridge** | +0.16 ($0.4377$) | -2.11 ($0.9824$) | -1.61 ($0.9462$) | -1.47 ($0.9292$) | -2.03 ($0.9789$) | Hợp lệ (Không vượt trội) |
| **Elastic_Net** | 0.00 ($1.0000$) | -0.49 ($0.6885$) | -0.97 ($0.8327$) | -0.98 ($0.8358$) | -1.75 ($0.9598$) | Hợp lệ (Không vượt trội) |
| **SVR** | +0.54 ($0.2935$) | -1.24 ($0.8927$) | -1.36 ($0.9132$) | -1.08 ($0.8605$) | -0.62 ($0.7329$) | Hợp lệ (Không vượt trội) |
| **Random_Forest** | -0.99 ($0.8377$) | -1.55 ($0.9394$) | -1.38 ($0.9161$) | -1.51 ($0.9342$) | -2.14 ($0.9836$) | Hợp lệ (Không vượt trội) |
| **XGBoost** | -0.54 ($0.7064$) | -0.21 ($0.5833$) | -1.51 ($0.9344$) | -1.18 ($0.8802$) | -1.06 ($0.8556$) | Hợp lệ (Không vượt trội) |
| **LightGBM** | -1.41 ($0.9202$) | -1.59 ($0.9435$) | -1.56 ($0.9400$) | -1.65 ($0.9505$) | -2.12 ($0.9830$) | Hợp lệ (Không vượt trội) |
| **LSTM** | +0.41 ($0.3424$) | -2.46 ($0.9930$) | -1.74 ($0.9589$) | -1.52 ($0.9357$) | -0.94 ($0.8254$) | Hợp lệ (Không vượt trội) |
| **RNN** | -0.91 ($0.8188$) | -0.68 ($0.7523$) | -1.68 ($0.9537$) | -2.73 ($0.9968$) | +0.15 ($0.4414$) | Hợp lệ (Không vượt trội) |
| **Transformer** | -16.85 ($1.0000$) | -4.30 ($1.0000$) | -2.27 ($0.9885$) | -2.37 ($0.9910$) | -1.14 ($0.8736$) | Hợp lệ (Không vượt trội) |

*\* Ghi chú: Thống kê DM dương biểu thị mô hình có MSE nhỏ hơn Benchmark. $p$-value là kiểm định 1 phía cho giả thuyết thay thế $H_1: \mathbb{E}[d_t] > 0$ (Mô hình tốt hơn).*

### 4.2 Kiểm Định Hai Phía: Bằng Chứng Về Sự Kém Hơn Có Ý Nghĩa Thống Kê (Two-Sided DM Tests)

Khi kiểm định giả thuyết hai phía ($H_0: \mathbb{E}[d_t] = 0$ vs $H_1: \mathbb{E}[d_t] \neq 0$), giá trị $p_{2s} = 2 \min(p_{1s}, 1 - p_{1s})$ cho thấy:
- **Nhiều mô hình Machine Learning và Deep Learning hoạt động KÉM HƠN CÓ Ý NGHĨA THỐNG KÊ so với Historical Mean:**
  - **Transformer:** 1D ($p_{2s} < 0.0001^*$), 5D ($p_{2s} < 0.0001^*$), 20D ($p_{2s} = 0.0231^*$), 60D ($p_{2s} = 0.0179^*$).
  - **Ridge:** 5D ($DM = -2.11, p_{2s} = 0.0351^*$), 120D ($DM = -2.03, p_{2s} = 0.0423^*$).
  - **LSTM:** 5D ($DM = -2.46, p_{2s} = 0.0139^*$).
  - **RNN:** 60D ($DM = -2.73, p_{2s} = 0.0064^*$).
  - **Random Forest:** 120D ($DM = -2.14, p_{2s} = 0.0328^*$).
  - **LightGBM:** 120D ($DM = -2.12, p_{2s} = 0.0340^*$).
- **Ý nghĩa kinh tế lượng:** Đối với các mô hình phi tuyến phức tạp (Tree Ensembles, Deep Learning), việc ước lượng quá nhiều tham số trên dữ liệu chuỗi tài chính có tỷ lệ tín hiệu/nhiễu thấp dẫn đến hiện tượng quá khớp ngoại mẫu nghiêm trọng (severe out-of-sample overfitting). Sai số dự báo của chúng không chỉ ngẫu nhiên lớn hơn mà lớn hơn một cách có ý nghĩa thống kê so với một mốc chuẩn trung bình lịch sử giản đơn.

---

## 5. Hiệu Chỉnh Đa Giả Thuyết (Multiple Hypothesis Testing Correction)

### 5.1 Định Nghĩa Không Gian Giả Thuyết (Family of Hypotheses)
- **Họ kiểm định chính quy (Primary Confirmatory Family):** Toàn bộ 60 phép kiểm định Diebold-Mariano giữa 12 mô hình hồi quy và mốc chuẩn Historical Mean trên 5 chân trời dự báo.
- **Mức ý nghĩa toàn họ (Family-Wise Significance Level):** $\alpha = 0.05$.

### 5.2 Hiệu Chỉnh Bonferroni & Kiểm Soát FDR Benjamini-Hochberg

1. **Hiệu chỉnh Bonferroni:**
   - Ngưỡng ý nghĩa hiệu chỉnh: $\alpha_{\text{Bonferroni}} = \frac{0.05}{60} \approx 0.0008333$.
   - *(Đính chính lỗi báo cáo cũ: Công thức trong bản thảo trước viết nhầm thành `0.0560 ≈ 0.00083` do lỗi gõ phím thiếu dấu gạch chéo).*
2. **Kiểm soát Tỷ lệ Khám phá Sai (Benjamini-Hochberg FDR):**
   - Sắp xếp 60 giá trị $p$ theo thứ tự tăng dần: $p_{(1)} \le p_{(2)} \le \dots \le p_{(60)}$.
   - Tìm chỉ số lớn nhất $k$ thỏa mãn: $p_{(k)} \le \frac{k}{60} \times q$, với $q = 0.05$.

#### Bảng Đối Chiếu 15 Kiểm Định Có p-value Thô Nhỏ Nhất:

| Hạng ($k$) | Model | Horizon | $DM$ Statistic | $p$-value thô | Ngưỡng BH ($\frac{k}{60} \times 0.05$) | Vượt Bonferroni? ($\le 0.000833$) | Vượt BH FDR? ($q = 0.05$) | Tình trạng Mô hình |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | ARIMAX | 5D | +5.1136 | **$1.81 \times 10^{-7}$** | $0.000833$ | **Có (Sống sót)** | **Có (Sống sót)** | *Không hợp lệ (Rò rỉ)* |
| **2** | SARIMA | 5D | +4.6237 | **$2.06 \times 10^{-6}$** | $0.001667$ | **Có (Sống sót)** | **Có (Sống sót)** | *Không hợp lệ (Rò rỉ)* |
| **3** | ARIMA | 5D | +4.3413 | **$7.61 \times 10^{-6}$** | $0.002500$ | **Có (Sống sót)** | **Có (Sống sót)** | *Không hợp lệ (Rò rỉ)* |
| 4 | SARIMA | 120D | +2.5053 | $0.006176$ | $0.003333$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 5 | ARIMA | 120D | +2.5040 | $0.006198$ | $0.004167$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 6 | ARIMAX | 120D | +2.5001 | $0.006266$ | $0.005000$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 7 | SARIMA | 60D | +2.3283 | $0.010021$ | $0.005833$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 8 | ARIMA | 60D | +2.3252 | $0.010105$ | $0.006667$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 9 | ARIMAX | 60D | +2.3231 | $0.010159$ | $0.007500$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 10 | ARIMAX | 20D | +2.1941 | $0.014200$ | $0.008333$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 11 | ARIMA | 20D | +2.1817 | $0.014651$ | $0.009167$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 12 | SARIMA | 20D | +2.1813 | $0.014665$ | $0.010000$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 13 | ARIMA | 1D | +1.4267 | $0.076943$ | $0.010833$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 14 | SARIMA | 1D | +1.3124 | $0.094805$ | $0.011667$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |
| 15 | ARIMAX | 1D | +1.2136 | $0.112565$ | $0.012500$ | **Không (Trượt)** | **Không (Trượt)** | *Không hợp lệ (Rò rỉ)* |

> [!CAUTION]
> **ĐÍNH CHÍNH QUAN TRỌNG VỀ KẾT QUẢ FDR:**  
> Bản thảo Report 08 trước đây tuyên bố rằng *"Tại 20D, 60D, 120D: Các mô hình AR kiểm soát FDR (Benjamini-Hochberg) ở mức $q = 0.05$"*.  
> **Kiểm toán thực nghiệm xác nhận tuyên bố này là SAI LỆCH VÀ KHÔNG THỂ TÁI LẬP.**  
> Tại hạng $k = 4$, $p_{(4)} = 0.006176 > \frac{4}{60} \times 0.05 = 0.003333$. Do đó, quá trình Benjamini-Hochberg dừng lại ngay sau hạng 3.  
> **Chỉ có đúng 3 phép kiểm định tại chân trời 5D sống sót sau hiệu chỉnh FDR $q = 0.05$** (trùng khớp hoàn toàn với kết quả của Bonferroni). Toàn bộ các kiểm định tại 20D, 60D và 120D đều thất bại trong việc kiểm soát FDR.

---

## 6. Thẩm Định Tính Hợp Lệ Của Dòng Mô Hình ARIMA (ARIMA Validity Audit)

Dòng mô hình Tự hồi quy (ARIMA, ARIMAX, SARIMA) là nhóm mô hình duy nhất có thống kê DM dương và đạt ý nghĩa thống kê dưới hiệu chỉnh Bonferroni và FDR tại 5D. Tuy nhiên, trước khi xem đây là bằng chứng khoa học, nghiên cứu đã tiến hành kiểm toán mã nguồn toàn diện:

1. **Truy vết mã nguồn pipeline:**  
   Trong tệp [arimax.py](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/arimax/arimax.py#L53-L56), quá trình tạo dự báo được thực hiện bằng lệnh:
   ```python
   applied = res.apply(test_series)
   pred = applied.predict(start=0, end=len(test_series)-1)
   ```
2. **Cơ chế rò rỉ lọc 1 bước trên chuỗi gối đầu (1-Step Filtering Leakage):**
   - Phương thức `res.apply().predict()` trong thư viện `statsmodels` thực hiện lọc 1 bước tiến (`one-step-ahead filtering`) trên toàn bộ chuỗi kiểm định.
   - Khi mục tiêu là lợi suất lũy kế gối đầu $Target_{h,t} = \log(P_{t+h}/P_t)$, tại bước $t$, mô hình được cung cấp giá trị quan sát $Target_{h,t-1} = \log(P_{t-1+h}/P_{t-1})$.
   - Biến số này chứa mức giá $P_{t-1+h}$ nằm sâu trong tương lai $h-1$ ngày so với thời điểm dự báo $t$.
3. **Ảo ảnh toán học của tự tương quan gối đầu:**
   - Hệ số tương quan lý thuyết giữa $Target_{h,t}$ và $Target_{h,t-1}$ xấp xỉ $\frac{h-1}{h}$ (ví dụ $\frac{119}{120} \approx 0.992$ tại 120D).
   - Mô hình ARIMA không cần học bất kỳ quy luật kinh tế nào; nó chỉ đơn giản sao chép giá trị trễ $t-1$ đã rò rỉ tương lai để đạt $OOS\_R^2 \approx 0.96$ và $DM > +2.18$.
4. **Phán quyết phương pháp luận:**
   - **Cách ly hoàn toàn dòng mô hình ARIMA/ARIMAX/SARIMA khỏi các kết luận khoa học về năng lực dự báo.**
   - Không được diễn giải kết quả kiểm định DM của nhóm mô hình này như bằng chứng về tính vượt trội trong dự báo lợi suất chứng khoán thực tế.

---

## 7. Kết Luận Nghiên Cứu Hiệu Chỉnh (Corrected Research Conclusions)

Sau khi kiểm toán và hiệu chỉnh các sai lệch phương pháp luận, các kết luận kiểm định thống kê chính thức của nghiên cứu được xác lập như sau:

1. **Về Năng Lực Dự Báo Của Nhóm Mô Hình Học Máy & Học Sâu (ML/DL Models):**
   - **Không có bất kỳ mô hình Machine Learning hoặc Deep Learning nào thể hiện được năng lực dự báo vượt trội hơn Historical Mean** trên cơ sở bình phương sai số ngoại mẫu ở bất kỳ chân trời nào ($DM \le 0$ và $p_{\text{one-sided}} > 0.5$ tại mọi $h \ge 5$).
   - Hơn nữa, kiểm định hai phía khẳng định nhiều mô hình phức tạp (Tree Ensembles, Deep Learning) có sai số **kém hơn có ý nghĩa thống kê** so với Historical Mean do hiện tượng quá khớp nhiễu.
2. **Về Kiểm Định Hướng Đi Nhất Quán (Directional Accuracy):**
   - Tại chân trời ngắn (1D), thị trường phản ánh tính khó dự báo cao; hơn $40\%$ số mô hình không đánh bại được xác suất đoán ngẫu nhiên $50\%$.
   - Tại chân trời dài (60D, 120D), tỷ lệ DA vượt $50\%$ với $p < 0.001$ hoàn toàn là hệ quả của độ trôi dương thị trường ($77.79\%$). Khi so sánh với chuẩn Always-Positive, **$100\%$ mô hình ML đều không tạo ra giá trị định hướng vượt trội**.
3. **Về Nhóm Mô Hình Tự Hồi Quy (ARIMA Family):**
   - Sự vượt trội thống kê của dòng ARIMA bị bác bỏ do lỗi rò rỉ lọc 1 bước trên chuỗi dữ liệu gối đầu.
4. **Thông Điệp Học Thuật Cốt Lõi Cho RQ1, RQ2 & RQ3:**
   - Bằng chứng kiểm định thống kê củng cố vững chắc phát hiện: **Năng lực dự báo lợi suất chứng khoán ngoại mẫu suy giảm mạnh mẽ theo chân trời dự báo**. Các cấu trúc mô hình phức tạp không thể vượt qua mốc chuẩn kinh tế lượng giản đơn khi được đánh giá bằng các tiêu chuẩn kiểm định suy diễn có kiểm soát tự tương quan và đa giả thuyết.

---

## 8. Trạng Thái Chuyển Bước Sang Step 9 / Tổng Hợp Toàn Diện (Step 9 Readiness Gate)

- **Quyết nghị phương pháp luận:** `CONDITIONAL PASS — SOME CLAIMS REQUIRE QUALIFICATION`.
- **Điều kiện tiên quyết cho Step 9 (`FINAL_SYNTHESIS.md`):**
  1. Sử dụng số liệu kiểm định đã được tái lập và chuẩn hóa trong `results/analysis/statistical_tests_corrected.csv`.
  2. Cách ly triệt để dòng ARIMA; trình bày rõ ràng lỗi rò rỉ lọc 1 bước như một phát hiện phương pháp luận quan trọng.
  3. Thống nhất thông điệp khoa học: Thị trường chứng khoán tuân theo quy luật suy giảm năng lực dự báo, và không có mô hình học máy nào đánh bại được mốc chuẩn ngây thơ trên dữ liệu ngoại mẫu thực tế.
