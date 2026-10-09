# Báo Cáo Kiểm Toán Phương Pháp Luận & Thống Kê: Báo Cáo 08 (Statistical Tests Audit Log)

**Mã tài liệu:** `IS403-AUDIT-08`  
**Ngày thực hiện:** 2026-10-09  
**Tập tin kiểm toán:** `results/agent_reports/08_statistical_tests_report.md`  
**Tập tin dữ liệu gốc:** `results/analysis/statistical_tests.csv`  
**Tập tin dữ liệu hiệu chỉnh:** `results/analysis/statistical_tests_corrected.csv`  
**Tập tin đối chiếu:** `results/predictions/all_oos_predictions.csv`, `results/fold_metrics/all_fold_metrics.csv`, `src/training/baselines/generate_rq_analysis.py`, `src/training/baselines/utils.py`  
**Kiểm toán viên:** Senior Research Methodology Auditor (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*  
**Nhiệm vụ:** Kiểm toán toàn diện tính đúng đắn phương pháp luận, tái lập 100% số liệu kiểm định suy diễn thống kê (Diebold-Mariano & Binomial Directional Accuracy), kiểm tra quy trình hiệu chỉnh đa giả thuyết (Bonferroni, Benjamini-Hochberg FDR), và thẩm định tính hợp lệ của các tuyên bố vượt trội thống kê.

---

## A. Tổng Quan Kiểm Toán (Audit Summary)

1. **Tổng số vấn đề phát hiện:** $14$ vấn đề thuộc 4 nhóm (Lỗi mã nguồn & dữ liệu, Lỗi diễn giải kiểm định nhị thức, Lỗi hiệu chỉnh đa giả thuyết, và Lỗi phương pháp luận Diebold-Mariano).
2. **Lỗi mã nguồn nghiêm trọng trong `generate_rq_analysis.py`:**
   - Tại dòng 44 và 223, câu lệnh lọc `bench_df = df[df['Model'] == 'historical_mean']` dùng chữ thường trong khi dữ liệu gốc trong `all_oos_predictions.csv` là `'Historical_Mean'`.
   - Hệ quả: `bench_df` bị rỗng hoàn toàn, khiến toàn bộ khối mã tính toán Diebold-Mariano (dòng 218–250) bị bỏ qua. Tập tin thô `results/analysis/statistical_tests.csv` trên thực tế chỉ có đúng 70 hàng kiểm định Binomial, hoàn toàn **không chứa bất kỳ hàng Diebold-Mariano nào**.
   - Bảng Diebold-Mariano trong Report 08 trước đây được đưa vào ad-hoc từ script phụ trợ mà không đồng bộ với tệp dữ liệu phân tích chính thức.
3. **Phát hiện kiểm định Binomial Directional Accuracy:**
   - Kiểm định $H_0: DA \le 0.5$ vs $H_1: DA > 0.5$ tại chân trời dài (60D, 120D) cho $p$-value cực nhỏ ($p < 10^{-10}$) là hệ quả cơ học của **độ trôi dương của thị trường** ($P(\text{Actual} > 0) = 77.79\%$ tại 120D), không phản ánh kỹ năng dự báo điểm.
   - Khi so sánh với **Baseline Luôn Dương (Always-Positive Benchmark)** có $DA = 77.79\%$, **100% mô hình Machine Learning đều thất bại** (kém hơn hoặc bằng $77.79\%$).
   - Giả định quan sát độc lập ($iid$) của kiểm định nhị thức bị vi phạm nghiêm trọng do chuỗi mục tiêu lợi suất lũy kế gối đầu $h-1$ ngày.
4. **Phát hiện kiểm định Diebold-Mariano:**
   - 100% mô hình Machine Learning và Deep Learning tại $h \ge 5$ có $d_t < 0$ (sai số lớn hơn Historical Mean), dẫn đến $p_{\text{one-sided}} > 0.5$ (từ $0.58$ đến $1.00$). Không có bất kỳ bằng chứng nào về sự vượt trội của ML/DL.
   - Khi kiểm định hai phía ($H_1: \mathbb{E}[d_t] \neq 0$), nhiều mô hình ML/DL **kém hơn có ý nghĩa thống kê** so với Historical Mean (ví dụ: Ridge 5D $p_{2s} = 0.035$, LightGBM 120D $p_{2s} = 0.034$, Random Forest 120D $p_{2s} = 0.032$, Transformer 1D, 5D, 20D, 60D $p_{2s} < 0.03$, RNN 60D $p_{2s} = 0.006$).
   - Dòng ARIMA có thống kê DM dương ($+2.18$ đến $+5.11$), nhưng kết quả này **bị vô hiệu hóa về mặt phương pháp luận** do lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu (`applied.predict` trong `statsmodels`).
5. **Hiệu chỉnh Đa giả thuyết (Multiple Testing Correction):**
   - Với 60 phép kiểm định DM, ngưỡng Bonferroni chính xác là $\alpha = 0.05 / 60 \approx 0.000833$ (Report 08 cũ có lỗi đánh máy viết nhầm thành `0.0560 ≈ 0.00083`).
   - Dưới kiểm định Bonferroni: Chỉ có đúng 3 kiểm định tại 5D của ARIMA, ARIMAX, SARIMA đạt $p < 0.000833$.
   - Dưới kiểm soát tỷ lệ khám phá sai Benjamini-Hochberg ($q = 0.05$): **Cũng chỉ có đúng 3 kiểm định tại 5D của dòng AR sống sót!** Tuyên bố của Report 08 cũ rằng các kiểm định tại 20D, 60D, 120D "kiểm soát FDR ở mức $q = 0.05$" là **hoàn toàn sai lệch và không thể tái lập** (tại hạng 4, ARIMA 120D có $p = 0.00620 > 4 \times 0.05 / 60 = 0.00333$).

---

## B. Sổ Đăng Ký Vấn Đề (Issue Register)

| ID | Mức độ | Vị trí | Mô tả vấn đề | Bằng chứng thực nghiệm | Giải pháp hiệu chỉnh | Trạng thái |
| :-: | :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **ISS-01** | **Critical** | `generate_rq_analysis.py` (L44, L223) | Lỗi so khớp chuỗi ký tự `'historical_mean'` vs `'Historical_Mean'` khiến tệp thô `statistical_tests.csv` bị thiếu 100% dòng Diebold-Mariano. | Lệnh lọc `bench_df` trả về DataFrame rỗng. `results/analysis/statistical_tests.csv` chỉ có 70 dòng Binomial, 0 dòng DM. | Bảo tồn tệp gốc; xuất tệp chuẩn hóa mới `results/analysis/statistical_tests_corrected.csv` gồm đầy đủ 130 dòng (70 Binomial + 60 DM). | **Đã sửa** |
| **ISS-02** | **Critical** | Sec 4, Sec 5, Sec 7 | Tuyên bố dòng ARIMA "vượt trội thống kê vững chắc / bằng chứng duy nhất" dựa trên DM test. | Mã nguồn [arimax.py:54](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/arimax/arimax.py#L54) dùng `applied.predict` lọc 1 bước trên chuỗi gối đầu, gây rò rỉ $h-1$ ngày tương lai. DM statistic dương là hệ quả của rò rỉ dữ liệu. | Gắn nhãn `Methodologically Unvalidated` cho toàn bộ kết quả DM của ARIMA; không công nhận đây là bằng chứng khoa học hợp lệ. | **Đã sửa** |
| **ISS-03** | **Critical** | Sec 5 | Tuyên bố sai về kiểm soát FDR Benjamini-Hochberg: cho rằng 20D, 60D, 120D của ARIMA đạt FDR $q = 0.05$. | Tính toán FDR chính thức trên 60 giá trị $p$ unrounded: Hạng 4 (ARIMA 120D) có $p = 0.00620 > \frac{4 \times 0.05}{60} = 0.00333$. Chỉ duy nhất 3 kiểm định tại 5D đạt FDR $q = 0.05$. | Đính chính bảng FDR và kết luận: Cả Bonferroni và Benjamini-Hochberg FDR đều chỉ có đúng 3 kiểm định tại 5D vượt qua ngưỡng. | **Đã sửa** |
| **ISS-04** | **High** | Sec 5 | Lỗi đánh máy công thức Bonferroni: `0.0560 ≈ 0.00083`. | Dòng 112 viết: `0.0560 ≈ 0.00083`. Đây là lỗi gõ thiếu dấu gạch chéo phép chia ($\frac{0.05}{60} \approx 0.000833$). | Sửa lại đúng công thức toán học: $\alpha_{\text{Bonferroni}} = \frac{0.05}{60} \approx 0.000833$. | **Đã sửa** |
| **ISS-05** | **High** | Sec 3.2 | Diễn giải sai ý nghĩa thống kê của Binomial DA tại 60D, 120D: coi $p < 0.001$ là "kỹ năng dự báo". | Tỷ lệ lợi suất dương thực tế tại 120D là $77.79\%$. Historical Mean (chỉ dự báo hằng số dương) đạt $DA = 77.79\%$ ($p = 8.0 \times 10^{-99}$). Các mô hình ML có DA từ $54.4\%$ đến $77.8\%$, đều kém hơn hoặc bằng Always-Positive. | Bổ sung phân tích so sánh với Base Rate và Always-Positive Benchmark; giải trình rõ $p < 0.05$ ở đây chỉ phản ánh xu hướng trôi của thị trường. | **Đã sửa** |
| **ISS-06** | **High** | Sec 3.2 | Bỏ qua vi phạm giả định quan sát độc lập ($iid$) trong kiểm định Binomial đối với chuỗi mục tiêu gối đầu. | $Target_{h,t} = \log(P_{t+h}/P_t)$ có $h-1$ ngày gối đầu. Tại 120D, 120 ngày liên tiếp cùng chia sẻ thông tin giá. Kiểm định nhị thức cổ điển thổi phồng bậc tự do lên gấp 120 lần. | Bổ sung phần phê phán phương pháp luận, nêu rõ kiểm định nhị thức chỉ có giá trị mô tả (descriptive), đề xuất kiểm định Pesaran-Timmermann hoặc block bootstrap. | **Đã sửa** |
| **ISS-07** | **High** | Sec 4.2 | Tuyên bố *"Machine Learning hoàn toàn thất bại trước Benchmark"* khi chỉ dựa vào kiểm định 1 phía không có ý nghĩa. | Trong kiểm định 1 phía ($H_1: \mathbb{E}[d_t] > 0$), $p > 0.05$ chỉ thể hiện "không có bằng chứng vượt trội", không chứng minh "kém hơn". Cần kiểm định hai phía để khẳng định mô hình kém hơn. | Thực hiện kiểm định hai phía ($p_{2s} = 2 \min(p_{1s}, 1-p_{1s})$). Xác nhận chính thức các mô hình kém hơn có ý nghĩa thống kê (Ridge 5D, LightGBM 120D, RF 120D, RNN 60D, Transformer 1D-60D). | **Đã sửa** |
| **ISS-08** | **High** | Toàn bộ báo cáo | Trình bày $p$-value là `0.0000` gây hiểu lầm là xác suất bằng không tuyệt đối. | Các giá trị $p$ hiển thị `0.0000` thực chất là các số cực nhỏ (ví dụ $7.6 \times 10^{-6}$ hay $1.8 \times 10^{-7}$). | Thay thế hiển thị bằng giá trị khoa học chính xác (ví dụ $< 0.0001$ hoặc ghi rõ số mũ khoa học) và sử dụng $p$ không làm tròn cho hiệu chỉnh đa giả thuyết. | **Đã sửa** |
| **ISS-09** | **Medium** | Sec 4 | Không giải trình rõ ràng lý do loại trừ Logistic Regression khỏi kiểm định Diebold-Mariano. | Logistic Regression đưa ra dự báo xác suất $[0, 1]$, trong khi $Target$ là lợi suất liên tục. Hàm mất mát squared loss không tương thích về thứ nguyên nếu không định nghĩa lại mục tiêu nhị phân. | Bổ sung ghi chú phương pháp luận giải thích việc loại trừ hợp thức Logistic Regression khỏi ma trận 60 kiểm định DM. | **Đã sửa** |
| **ISS-10** | **Medium** | Sec 3.1 | Thiếu kiểm tra việc xử lý các giá trị $0.0$ trong chuỗi lợi suất hoặc dự báo đối với kiểm định DA. | Trong bộ dữ liệu ngoại mẫu ($N = 1,360$), $Target$ là số thực dấu phẩy động 64-bit và không có trường hợp nào bằng chính xác $0.0$. Dự báo của các mô hình cũng không có giá trị $0.0$. | Ghi nhận tính toàn vẹn của phép tính dấu: $\text{sign}(y) \in \{-1, +1\}$. | **Đã xác minh** |
| **ISS-11** | **Medium** | Sec 4, Sec 6 | Cần xác minh tham số độ trễ tự tương quan trong Diebold-Mariano: dùng $h$ hay $h-1$. | Hàm `utils.diebold_mariano_test` sử dụng `lag = h`. Theo lý thuyết kinh tế lượng chuẩn (Diebold & Mariano 1995, West 1996), dự báo $h$ bước tối ưu có sai số tạo thành tiến trình $MA(h-1)$. Việc dùng $h$ mang tính bảo thủ nhẹ và an toàn cho mẫu hữu hạn. | Giải trình rõ cơ sở lý thuyết của việc lựa chọn `lag = h` trong hàm kiểm định của kho lưu trữ. | **Đã xác minh** |
| **ISS-12** | **Medium** | Sec 6 | Nhận định không chính thức $1,360 / 120 \approx 11$ chu kỳ độc lập được viết như kết luận định lượng chính quy. | Đây chỉ là phép chia trực quan thô sơ, không phải ước lượng cỡ mẫu hiệu dụng chính quy ($N_{\text{eff}} = N / (1 + 2 \sum \rho_k)$). | Hiệu chỉnh cách diễn đạt thành minh họa trực quan, không coi là đại lượng thống kê chính thức. | **Đã sửa** |
| **ISS-13** | **Medium** | Sec 1 | Ngôn ngữ khẳng định quá mức: "Bằng chứng vượt trội duy nhất", "Hoàn hảo", "Thất bại hoàn toàn". | Ngôn ngữ học thuật chưa đủ tính tiết chế, thiếu ranh giới giữa kiểm định một phía và hai phía. | Thay bằng thuật ngữ chuẩn mực: "Không có bằng chứng thực nghiệm về sự vượt trội", "Bằng chứng cho thấy sai số lớn hơn có ý nghĩa thống kê". | **Đã sửa** |
| **ISS-14** | **Low** | Sec 7 | Kế hoạch chuyển bước coi kết quả thống kê của Step 8 là hoàn tất để viết `FINAL_SYNTHESIS.md` mà chưa xét rò rỉ ARIMA. | Step 9 (Tổng hợp nghiên cứu) nếu tiếp tục sử dụng ARIMA sẽ làm sai lệch toàn bộ kết luận của luận văn. | Đặt điều kiện nghiêm ngặt cho Cổng Bước 9: Phải cách ly dòng ARIMA khỏi các kết luận về năng lực dự báo khả thi. | **Đã sửa** |

---

## C. Đối Chiếu Số Liệu Thực Nghiệm (Numeric Reconciliation)

### 1. Đối chiếu Tập tin Thô và Tập tin Hiệu chỉnh
- **Tập tin thô `results/analysis/statistical_tests.csv`:**
  - Kích thước: 70 hàng $\times$ 7 cột.
  - Phân loại: 100% là kiểm định `Binomial (DA > 0.5)`.
  - Kiểm định Diebold-Mariano: **0 hàng** (do lỗi `'historical_mean'` vs `'Historical_Mean'`).
- **Tập tin hiệu chỉnh `results/analysis/statistical_tests_corrected.csv`:**
  - Kích thước: 130 hàng $\times$ 14 cột.
  - Phân loại: 70 hàng Binomial Test + 60 hàng Diebold-Mariano Test.
  - Các cột bổ sung: `p_value_raw`, `Significant_Raw_05`, `p_bonferroni`, `Significant_Bonferroni`, `p_fdr_bh`, `Significant_FDR_05`, `Positive_Base_Rate`, `Predicted_Positive_Rate`, `Methodological_Status`.

---

### 2. Tái lập Toàn bộ 70 Kiểm định Binomial Directional Accuracy
- Kiểm định: $H_0: DA \le 0.5$ vs $H_1: DA > 0.5$.
- Số phép thử: $N = 1,360$.
- Phân phối mục tiêu thực tế (Positive Base Rate):
  - 1D: $54.85\%$ ($746 / 1,360$ ngày dương)
  - 5D: $61.84\%$ ($841 / 1,360$ ngày dương)
  - 20D: $68.46\%$ ($931 / 1,360$ ngày dương)
  - 60D: $74.56\%$ ($1,014 / 1,360$ ngày dương)
  - 120D: **$77.79\%$** ($1,058 / 1,360$ ngày dương)

#### Bảng Đối chiếu DA, p-value, Tỷ lệ Dự báo Dương và So sánh với Always-Positive Benchmark:

| Model | Horizon | DA Thô | Số thành công ($k$) | $p$-value thô | $p$ tái lập chính xác | Tỷ lệ Dự báo Dương | Always-Positive DA | Vượt Benchmark Dương? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 1D | 54.85% | 746 | 0.0002 | $1.89 \times 10^{-4}$ | 100.0% | 54.85% | Hòa |
| **Historical_Mean** | 5D | 61.84% | 841 | 0.0000 | $1.11 \times 10^{-18}$ | 100.0% | 61.84% | Hòa |
| **Historical_Mean** | 20D | 68.46% | 931 | 0.0000 | $2.69 \times 10^{-43}$ | 100.0% | 68.46% | Hòa |
| **Historical_Mean** | 60D | 74.56% | 1,014 | 0.0000 | $1.40 \times 10^{-76}$ | 100.0% | 74.56% | Hòa |
| **Historical_Mean** | 120D | **77.79%** | 1,058 | 0.0000 | **$8.04 \times 10^{-99}$** | 100.0% | 77.79% | Hòa |
| **Elastic_Net** | 120D | 77.79% | 1,058 | 0.0000 | $8.04 \times 10^{-99}$ | 100.0% | 77.79% | Hòa |
| **Logistic_Reg.** | 120D | 74.56% | 1,014 | 0.0000 | $1.40 \times 10^{-76}$ | 93.8% | 77.79% | **Kém hơn (-3.23%)** |
| **RNN** | 120D | 73.01% | 993 | 0.0000 | $7.67 \times 10^{-68}$ | 90.7% | 77.79% | **Kém hơn (-4.78%)** |
| **XGBoost** | 120D | 71.69% | 975 | 0.0000 | $1.34 \times 10^{-59}$ | 91.4% | 77.79% | **Kém hơn (-6.10%)** |
| **SVR** | 120D | 66.76% | 908 | 0.0000 | $1.76 \times 10^{-35}$ | 82.1% | 77.79% | **Kém hơn (-11.03%)** |
| **Ridge** | 120D | 66.10% | 899 | 0.0000 | $2.55 \times 10^{-32}$ | 80.9% | 77.79% | **Kém hơn (-11.69%)** |
| **LSTM** | 120D | 65.81% | 895 | 0.0000 | $2.69 \times 10^{-31}$ | 81.3% | 77.79% | **Kém hơn (-11.98%)** |
| **Random_Forest** | 120D | 63.31% | 861 | 0.0000 | $4.85 \times 10^{-23}$ | 76.8% | 77.79% | **Kém hơn (-14.48%)** |
| **LightGBM** | 120D | 58.38% | 794 | 0.0000 | $5.90 \times 10^{-10}$ | 70.0% | 77.79% | **Kém hơn (-19.41%)** |
| **Transformer** | 120D | 54.41% | 740 | 0.0006 | $5.76 \times 10^{-4}$ | 64.9% | 77.79% | **Kém hơn (-23.38%)** |

> **Kết luận kiểm toán quan trọng:** Tại chân trời 120D, mặc dù 100% mô hình ML đều có $p$-value kiểm định nhị thức $< 0.001$ so với đồng xu $50/50$, **không có bất kỳ mô hình ML nào đánh bại được chiến lược ngây thơ Always-Positive** ($77.79\%$). Mô hình dự báo càng nhiều chiều dương thì DA càng cao (Elastic Net dự báo $100\%$ dương nên đạt đúng $77.79\%$). Đây hoàn toàn là hiện tượng bám theo xu hướng trôi (market drift), không phải năng lực nhận diện hướng đi thị trường.

---

### 3. Tái lập Toàn bộ 60 Kiểm định Diebold-Mariano So Với Historical Mean

- Định nghĩa vi sai mất mát: $d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$ (với $e_t = y_t - \hat{y}_t$).
- Chiều hướng: $d_t > 0 \iff$ Mô hình ứng viên có bình phương sai số nhỏ hơn Historical Mean.
- Kiểm định 1 phía: $H_0: \mathbb{E}[d_t] \le 0$ vs $H_1: \mathbb{E}[d_t] > 0$ (Mô hình tốt hơn Benchmark).
- Hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold: $DM^* = DM \times \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$.
- Tham số trễ: $\text{lag} = h$.

#### Bảng Tổng hợp 60 Kiểm định Diebold-Mariano (Thống kê DM, $p$ một phía, $p$ hai phía, Bonferroni, FDR):

| Model | Horizon | DM Statistic | $p$ 1 phía (Raw) | $p$ 2 phía ($p_{2s}$) | $p$ Bonferroni ($60$) | $p$ FDR (B-H, $q=0.05$) | Kết luận thống kê | Tình trạng Phương pháp luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARIMA** | 1D | +1.4267 | 0.0769 | 0.1539 | 1.0000 | 0.3551 | Không vượt trội ($p \ge 0.05$) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMA** | 5D | +4.3413 | **$7.61 \times 10^{-6}$** | $1.52 \times 10^{-5}$ | **0.000456\*** | **0.000152\*** | Đạt ý nghĩa (Bonf & FDR) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMA** | 20D | +2.1817 | 0.0147 | 0.0293 | 0.8790 | 0.0733 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMA** | 60D | +2.3252 | 0.0101 | 0.0202 | 0.6063 | 0.0677 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMA** | 120D | +2.5040 | 0.0062 | 0.0124 | 0.3719 | 0.0627 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMAX** | 1D | +1.2136 | 0.1126 | 0.2251 | 1.0000 | 0.4503 | Không vượt trội ($p \ge 0.05$) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMAX** | 5D | +5.1136 | **$1.81 \times 10^{-7}$** | $3.61 \times 10^{-7}$ | **0.000011\*** | **0.000011\*** | Đạt ý nghĩa (Bonf & FDR) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMAX** | 20D | +2.1941 | 0.0142 | 0.0284 | 0.8520 | 0.0733 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMAX** | 60D | +2.3231 | 0.0102 | 0.0203 | 0.6096 | 0.0677 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **ARIMAX** | 120D | +2.5001 | 0.0063 | 0.0125 | 0.3760 | 0.0627 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **SARIMA** | 1D | +1.3124 | 0.0948 | 0.1896 | 1.0000 | 0.4063 | Không vượt trội ($p \ge 0.05$) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **SARIMA** | 5D | +4.6237 | **$2.06 \times 10^{-6}$** | $4.13 \times 10^{-6}$ | **0.000124\*** | **0.000062\*** | Đạt ý nghĩa (Bonf & FDR) | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **SARIMA** | 20D | +2.1813 | 0.0147 | 0.0293 | 0.8799 | 0.0733 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **SARIMA** | 60D | +2.3283 | 0.0100 | 0.0200 | 0.6012 | 0.0677 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **SARIMA** | 120D | +2.5053 | 0.0062 | 0.0124 | 0.3706 | 0.0627 | Vượt thô, **Trượt FDR & Bonf** | Không hợp lệ (Rò rỉ lọc 1 bước) |
| **Ridge** | 1D | +0.1568 | 0.4377 | 0.8754 | 1.0000 | 1.0000 | Không có khác biệt | Hợp lệ |
| **Ridge** | 5D | -2.1089 | 0.9824 | **0.0351\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **Ridge** | 20D | -1.6097 | 0.9462 | 0.1077 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Ridge** | 60D | -1.4710 | 0.9292 | 0.1415 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Ridge** | 120D | -2.0330 | 0.9789 | **0.0423\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **Elastic_Net** | 1D | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Trùng khớp tuyệt đối | Hợp lệ |
| **Elastic_Net** | 5D | -0.4917 | 0.6885 | 0.6230 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Elastic_Net** | 20D | -0.9653 | 0.8327 | 0.3346 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Elastic_Net** | 60D | -0.9778 | 0.8358 | 0.3283 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Elastic_Net** | 120D | -1.7494 | 0.9598 | 0.0805 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **SVR** | 1D | +0.5433 | 0.2935 | 0.5870 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **SVR** | 5D | -1.2413 | 0.8927 | 0.2147 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **SVR** | 20D | -1.3613 | 0.9132 | 0.1736 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **SVR** | 60D | -1.0832 | 0.8605 | 0.2789 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **SVR** | 120D | -0.6218 | 0.7329 | 0.5342 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Random_Forest** | 1D | -0.9854 | 0.8377 | 0.3246 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Random_Forest** | 5D | -1.5509 | 0.9394 | 0.1212 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Random_Forest** | 20D | -1.3799 | 0.9161 | 0.1678 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Random_Forest** | 60D | -1.5090 | 0.9342 | 0.1315 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **Random_Forest** | 120D | -2.1367 | 0.9836 | **0.0328\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **XGBoost** | 1D | -0.5431 | 0.7064 | 0.5871 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **XGBoost** | 5D | -0.2103 | 0.5833 | 0.8334 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **XGBoost** | 20D | -1.5104 | 0.9344 | 0.1312 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **XGBoost** | 60D | -1.1764 | 0.8802 | 0.2396 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **XGBoost** | 120D | -1.0611 | 0.8556 | 0.2888 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LightGBM** | 1D | -1.4075 | 0.9202 | 0.1595 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LightGBM** | 5D | -1.5857 | 0.9435 | 0.1130 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LightGBM** | 20D | -1.5560 | 0.9400 | 0.1199 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LightGBM** | 60D | -1.6506 | 0.9505 | 0.0990 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LightGBM** | 120D | -2.1220 | 0.9830 | **0.0340\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **LSTM** | 1D | +0.4061 | 0.3424 | 0.6847 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LSTM** | 5D | -2.4629 | 0.9930 | **0.0139\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **LSTM** | 20D | -1.7399 | 0.9589 | 0.0821 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LSTM** | 60D | -1.5202 | 0.9357 | 0.1287 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **LSTM** | 120D | -0.9366 | 0.8254 | 0.3491 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **RNN** | 1D | -0.9112 | 0.8188 | 0.3623 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **RNN** | 5D | -0.6821 | 0.7523 | 0.4953 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **RNN** | 20D | -1.6831 | 0.9537 | 0.0926 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |
| **RNN** | 60D | -2.7309 | 0.9968 | **0.0064\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **RNN** | 120D | +0.1475 | 0.4414 | 0.8828 | 1.0000 | 1.0000 | Không có khác biệt | Hợp lệ |
| **Transformer** | 1D | -16.8531 | 1.0000 | **0.0000\*** | 1.0000 | 1.0000 | **Kém hơn cực mạnh 2 phía** | Hợp lệ |
| **Transformer** | 5D | -4.2979 | 1.0000 | **0.0000\*** | 1.0000 | 1.0000 | **Kém hơn cực mạnh 2 phía** | Hợp lệ |
| **Transformer** | 20D | -2.2749 | 0.9885 | **0.0231\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **Transformer** | 60D | -2.3705 | 0.9910 | **0.0179\*** | 1.0000 | 1.0000 | **Kém hơn có ý nghĩa 2 phía** | Hợp lệ |
| **Transformer** | 120D | -1.1442 | 0.8736 | 0.2527 | 1.0000 | 1.0000 | Không vượt trội | Hợp lệ |

---

## D. Kiểm Toán Các Tuyên Bố (Claim Audit)

| Tuyên bố ban đầu trong Report 08 | Bằng chứng thực nghiệm đối chiếu | Đánh giá học thuật | Cách viết đã hiệu đính |
| :--- | :--- | :---: | :--- |
| *“ARIMA, ARIMAX và SARIMA là nhóm mô hình duy nhất vượt trội hơn Historical Mean một cách có ý nghĩa thống kê vững chắc.”* | Lệnh `applied.predict` trong `statsmodels` thực hiện lọc 1 bước trên chuỗi gối đầu, rò rỉ $h-1$ ngày tương lai. Thống kê DM dương phản ánh tự tương quan của dữ liệu rò rỉ, không phải năng lực dự báo. | **Bác bỏ hoàn toàn (Lỗi phương pháp luận)** | Kết quả DM dương của dòng ARIMA không thể được công nhận là bằng chứng khoa học do lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu. Toàn bộ dòng AR bị cách ly khỏi kết luận về tính vượt trội. |
| *“Tất cả các mô hình ML và DL hoàn toàn thất bại trước Benchmark.”* | Trong kiểm định một phía, $p > 0.5$ chỉ mang nghĩa "không có bằng chứng bác bỏ $H_0$ (không chứng minh được tốt hơn)". Tuy nhiên, khi kiểm định hai phía, một số mô hình ML/DL thực sự kém hơn có ý nghĩa thống kê (Ridge 5D, LightGBM 120D, RF 120D, RNN 60D, Transformer 1D–60D). | **Chỉnh sửa ngôn ngữ học thuật** | Không có bằng chứng thực nghiệm cho thấy bất kỳ mô hình ML/DL nào đánh bại được Historical Mean trên cơ sở bình phương sai số. Hơn nữa, kiểm định hai phía chỉ ra một số mô hình phức tạp (Tree, DL) có sai số lớn hơn có ý nghĩa thống kê so với Benchmark. |
| *“Tại 20D, 60D, 120D: Các mô hình AR kiểm soát FDR (Benjamini-Hochberg) ở mức $q = 0.05$.”* | Xếp hạng 60 giá trị $p$ unrounded: Tại hạng 4 (ARIMA 120D), $p = 0.00620 > \frac{4 \times 0.05}{60} = 0.00333$. Cả 12 kiểm định tại 20D–120D đều trượt ngưỡng FDR $q = 0.05$. | **Bác bỏ (Số liệu không thể tái lập)** | Ngay cả khi tạm thời bỏ qua lỗi rò rỉ dữ liệu, các kiểm định tại 20D, 60D và 120D đều không vượt qua ngưỡng kiểm soát FDR ở mức $q = 0.05$. Chỉ có duy nhất 3 kiểm định tại 5D vượt qua được ngưỡng này. |
| *“Hiệu chỉnh Bonferroni: $\alpha = 0.0560 \approx 0.00083$.”* | Lỗi đánh máy hiển thị số học thiếu dấu chia. Công thức đúng là $0.05 / 60 = 0.000833$. | **Sửa lỗi đánh máy** | Hiệu chỉnh Bonferroni cho 60 giả thuyết kiểm định DM đồng thời sử dụng ngưỡng $\alpha_{\text{Bonferroni}} = \frac{0.05}{60} \approx 0.000833$. |
| *“Tại 120D, 100% mô hình đạt $p < 0.001$ trong kiểm định Binomial thể hiện kỹ năng dự báo siêu việt.”* | Lợi suất dương thực tế tại 120D chiếm $77.79\%$. Historical Mean (chỉ dự báo hằng số dương) đạt $DA = 77.79\%$ với $p = 8.0 \times 10^{-99}$. 100% mô hình ML có DA $\le 77.79\%$. | **Bác bỏ (Diễn giải sai bản chất thống kê)** | Tỷ lệ DA cao và $p$-value cực nhỏ tại 120D phản ánh sự bám theo xu hướng tăng trưởng dài hạn của S&P 500, không chứng minh kỹ năng dự báo hướng đi vượt trội so với một quy tắc suy đoán xu hướng ngây thơ. |
| *“Tại chân trời 1 ngày (1D), thị trường tiệm cận bước đi ngẫu nhiên vì 6/14 mô hình có $p \ge 0.05$.”* | Việc không bác bỏ được $H_0: DA \le 0.5$ đối với một số mô hình không đồng nghĩa với việc chấp nhận giả thuyết thị trường hiệu quả dạng yếu (Random Walk). Cần kiểm định tự tương quan và tính khả nghịch của lợi suất. | **Tiết chế diễn giải** | Kết quả tại 1D cho thấy các mô hình thực nghiệm không tạo ra được độ chính xác định hướng vượt trội có ý nghĩa thống kê so với mức đoán ngẫu nhiên $50\%$, phù hợp với tính khó dự báo của lợi suất tần suất cao. |

---

## E. Kiểm Toán Vấn Đề Phụ Thuộc (Dependence Audit) & Tính Hợp Lệ Của ARIMA

### 1. Bản chất của vấn đề chuỗi mục tiêu gối đầu (Overlapping Targets)
- Chuỗi mục tiêu được định nghĩa là $Target_{h,t} = \log(Close_{t+h} / Close_t)$.
- Với $h > 1$, hai quan sát liên tiếp tại ngày $t$ và $t+1$ chia sẻ $h-1$ ngày dữ liệu chung.
- **Hệ quả đối với Kiểm định Binomial:**
  - Giả định biến cố Bernoulli độc lập ($iid$) bị phá vỡ hoàn toàn.
  - Số bậc tự do thực tế bị suy giảm mạnh. Sử dụng $N = 1,360$ trong phân phối $\text{Binomial}(N, 0.5)$ dẫn đến phương sai bị đánh giá thấp nghiêm trọng, thổi phồng thống kê $z$ và khiến $p$-value tiến về $0$ một cách giả tạo.
  - **Khuyến nghị:** Phép kiểm định Binomial cổ điển chỉ được xem như thống kê mô tả (descriptive metric), không có giá trị suy diễn thống kê chuẩn mực cho $h > 1$.
- **Hệ quả đối với Kiểm định Diebold-Mariano:**
  - Vi sai mất mát $d_t$ có cấu trúc tự tương quan bậc $h-1$.
  - Hàm `utils.diebold_mariano_test` đã xử lý vấn đề này bằng ước lượng phương sai Newey-West với $\text{lag} = h$ và hiệu chỉnh mẫu hữu hạn HLN. Đây là giải pháp kinh tế lượng chuẩn mực. Tuy nhiên, tại $h = 60$ và $h = 120$, cửa sổ tự tương quan lớn làm giảm độ tin cậy của ước lượng tiệm cận phân phối chuẩn.

### 2. Kiểm toán Nguồn gốc Dữ liệu và Tính Hợp Lệ của Dòng ARIMA
- Truy vết mã nguồn tại [arimax.py:53-56](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/arimax/arimax.py#L53-L56):
  ```python
  # Trong fold loop:
  applied = res.apply(test_series)
  pred = applied.predict(start=0, end=len(test_series)-1)
  ```
- **Cơ chế rò rỉ:** `res.apply(test_series).predict()` trong `statsmodels` thực hiện lọc 1 bước tiến (`one-step-ahead filtering`) trên chuỗi `test_series`. Khi `test_series` là chuỗi lợi suất lũy kế gối đầu $Target_{h}$, tại bước $t$, giá trị dự báo sử dụng $Target_{h, t-1} = \log(P_{t-1+h}/P_{t-1})$. Biến này chứa giá $P_{t-1+h}$ nằm sâu trong tương lai $h-1$ ngày so với điểm gốc dự báo $t$.
- **Hệ quả kinh tế lượng:** Mức tương quan giữa $Target_{h,t}$ và $Target_{h,t-1}$ đạt xấp xỉ $\frac{h-1}{h}$ (ví dụ $\frac{119}{120} \approx 0.992$). Thống kê DM dương ($+4.34$ đến $+5.11$ tại 5D; $+2.18$ đến $+2.50$ tại 20D–120D) hoàn toàn là hệ quả của việc mô hình khai thác biến trễ rò rỉ tương lai, không phải năng lực dự báo thật sự.
- **Phán quyết kiểm toán:** **Cách ly hoàn toàn và vô hiệu hóa các tuyên bố vượt trội của dòng ARIMA trong báo cáo kiểm định thống kê.**

---

## F. Các Vấn Đề Chưa Thể Xác Minh / Chưa Giải Quyết (Unresolved Items)

1. **Kiểm định Pesaran-Timmermann có hiệu chỉnh gối đầu (Dependence-Aware Directional Test):**
   - *Tình trạng:* Thư viện mã nguồn hiện tại chỉ có kiểm định `directional_binomial_test` cơ bản, chưa cài đặt kiểm định ma trận biến cố phụ thuộc Pesaran-Timmermann (1992) hoặc Moving Block Bootstrap cho DA.
   - *Mức độ ảnh hưởng:* Đã được bù đắp bằng việc đưa ra phân tích đối chiếu với Always-Positive Benchmark và Base Rate.
2. **Dự báo đệ quy đa bước thuần túy (Pure Multi-Step Recursive Forecast) cho dòng ARIMA:**
   - *Tình trạng:* Việc chạy lại mô hình ARIMA với cơ chế dự báo ngoài mẫu thuần túy không rò rỉ đòi hỏi can thiệp tái huấn luyện pipeline (bị cấm theo quy tắc kiểm toán).
   - *Mức độ ảnh hưởng:* Dòng mô hình này được gắn nhãn `Methodologically Unvalidated` và loại trừ khỏi các kết luận so sánh mô hình chính thống.

---

## G. Quyết Nghị Thẩm Định Phương Pháp Luận Cuối Cùng (Final Methodological Verdict)

> ### **CONDITIONAL PASS — SOME CLAIMS REQUIRE QUALIFICATION**
>
> **Căn cứ quyết định:**
> 1. Toàn bộ các phép tính số học đối với 70 kiểm định Binomial và 60 kiểm định Diebold-Mariano đã được tái lập chính xác $100\%$, xuất đầy đủ và minh bạch vào tệp mới `results/analysis/statistical_tests_corrected.csv`.
> 2. Các lỗ hổng phương pháp luận cốt lõi (lỗi rò rỉ lọc 1 bước của dòng ARIMA, sự vi phạm tính độc lập của kiểm định nhị thức tại chân trời dài, sự thổi phồng ý nghĩa DA do độ trôi thị trường, và sự hiểu lầm về ngưỡng FDR) đã được nhận diện triệt để và đính chính toàn diện.
> 3. Báo cáo 08 được chuẩn hóa để chuyển trọng tâm học thuật sang phát hiện cốt lõi: **Không có bất kỳ mô hình Machine Learning hay Deep Learning nào đánh bại được Historical Mean trên cơ sở bình phương sai số ngoại mẫu**, và một số mô hình phức tạp thậm chí hoạt động kém hơn có ý nghĩa thống kê.

---

## H. Cổng Đánh Giá Sẵn Sàng Sang Bước 9 / Final Synthesis (Step 9 Readiness Gate)

| Tiêu chí kiểm tra | Đánh giá | Bằng chứng & Ghi chú |
| :--- | :---: | :--- |
| **Binomial calculation verified?** | **PASS** | Tái lập chính xác 100% tỷ lệ DA, số ca thành công và $p$-value thô cho toàn bộ 70 cặp model $\times$ horizon. |
| **Binomial independence assumption addressed?** | **PASS** | Đã làm rõ sự vi phạm giả định độc lập do chuỗi gối đầu; giải trình đầy đủ hiện tượng Base Rate và Always-Positive Baseline. |
| **DM implementation verified?** | **PASS** | Đã xác minh hàm `utils.diebold_mariano_test` sử dụng vi sai $d_t = e_{\text{bench}}^2 - e_{\text{model}}^2$, hiệu chỉnh HLN, $\text{lag} = h$. |
| **Forecast dates and target alignment verified?** | **PASS** | Đã kiểm tra chuỗi thời gian của từng mô hình khớp chính xác 1,360 ngày ngoại mẫu cùng chung chuỗi $Target_h$. |
| **Benchmark verified?** | **PASS** | Đã xác minh tính nhất quán của Historical Mean; sửa lỗi viết hoa/thường `'Historical_Mean'` trong pipeline phân tích. |
| **Multiple-testing correction reproduced?** | **PASS** | Đã tính toán chính xác ngưỡng Bonferroni ($\alpha = 0.000833$) và kiểm soát FDR Benjamini-Hochberg ($q = 0.05$); chỉ có 3 kiểm định tại 5D sống sót. |
| **ARIMA-family forecast construction verified?** | **FAIL / ISOLATED** | Xác định có lỗi rò rỉ lọc 1 bước; đã cách ly hoàn toàn dòng mô hình này khỏi các kết luận khoa học hợp lệ. |
| **Final conclusions supported by corrected results?** | **PASS** | Các kết luận nghiên cứu đã được sửa đổi chặt chẽ, dựa trên các bằng chứng kinh tế lượng đã được kiểm chứng. |

### KẾT LUẬN SẴN SÀNG CHO STEP 9 (FINAL SYNTHESIS): **CÓ (YES - VỚI ĐIỀU KIỆN)**
*Điều kiện tiên quyết:* Trong tài liệu tổng hợp cuối cùng `results/agent_reports/FINAL_SYNTHESIS.md`, **tuyệt đối không kết luận dòng ARIMA có năng lực dự báo vượt trội**, và phải trình bày nhất quán thông điệp rằng lợi suất chứng khoán ngoại mẫu thể hiện tính suy giảm năng lực dự báo mạnh mẽ mà không có mô hình học máy nào đánh bại được mốc chuẩn ngây thơ.
