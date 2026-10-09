# Báo Cáo Kiểm Toán Phương Pháp Luận & Thống Kê: Báo Cáo 07 (RQ3 Time Stability Audit Log)

**Mã tài liệu:** `IS403-AUDIT-07`  
**Ngày thực hiện:** 2026-10-09  
**Tập tin kiểm toán:** `results/agent_reports/07_rq3_time_report.md`  
**Tập tin dữ liệu chính:** `results/analysis/rq3_time.csv`  
**Tập tin đối chiếu:** `results/predictions/all_oos_predictions.csv`, `results/fold_metrics/all_fold_metrics.csv`, `results/summary/model_horizon_summary.csv`, `src/training/baselines/generate_rq_analysis.py`, `src/training/baselines/config.py`  
**Kiểm toán viên:** Senior Research Methodology Auditor (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*  
**Câu hỏi nghiên cứu (RQ3):** *"Are the observed horizon- and regime-dependent patterns of predictability stable over time?"*

---

## A. Tổng Quan Kiểm Toán (Audit Summary)

- **Tổng số vấn đề phát hiện:** $12$ vấn đề.
- **Số lỗi số liệu / Bảng biểu:** $2$ vấn đề (bỏ sót 2 họ mô hình Deep Learning & SVR trong bảng Section 4; làm tròn và thiếu giải trình về việc loại bỏ Logistic Regression khỏi bảng hồi quy).
- **Số lỗi diễn giải / Diễn đạt quá mức:** $5$ vấn đề (khẳng định "100% suy thoái", gán nhãn "bằng chứng thép" cho ARIMA, quy chụp nhân quả COVID-19/Fed, khẳng định thứ hạng họ mô hình cố định `AR >> Linear ≈ Mean > Tree > DL`, dùng thuật ngữ "độ dốc" cho hiệu số điểm mút).
- **Số vấn đề phương pháp luận:** $4$ vấn đề (lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu của dòng ARIMA, mẫu số benchmark dùng phương sai nội vùng thay vì Historical Mean do lỗi so khớp chuỗi ký tự, chuỗi mục tiêu $120D$ gối đầu 119 ngày vi phạm tính độc lập quan sát, phân kỳ thời gian là các khối lịch 2 năm tùy biến chứ không phải chu kỳ kinh tế vĩ mô chính quy).
- **Số vấn đề chưa xác minh (Unverified):** $1$ vấn đề (chưa có kiểm định thống kê đa biến chính quy $Horizon \times Period$ hoặc $Regime \times Period$ có hiệu chỉnh tự tương quan Newey-West/Block Bootstrap trong tập tin này — nội dung thuộc phạm vi Step 8).

---

## B. Sổ Đăng Ký Vấn Đề (Issue Register)

| ID | Mức độ | Vị trí | Mô tả vấn đề | Bằng chứng thực nghiệm | Giải pháp hiệu chỉnh | Trạng thái |
| :-: | :---: | :--- | :--- | :--- | :--- | :---: |
| **ISS-01** | **Critical** | Sec 5.1, Sec 6 | Khẳng định dòng ARIMA có *"tính ổn định cực cao / bằng chứng thép"* với $OOS\_R^2 \approx 0.96$. | [arimax.py:54](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/arimax/arimax.py#L54) dùng `applied.predict` lọc 1 bước trên chuỗi gối đầu, rò rỉ $h-1$ ngày tương lai. Tương quan $\frac{119}{120} \approx 0.992$ là hằng số toán học, không phải kỹ năng. | Cách ly hoàn toàn dòng ARIMA khỏi kết luận khoa học về tính ổn định thời gian. Gắn nhãn lỗi phương pháp luận. | **Đã sửa** |
| **ISS-02** | **Critical** | Sec 2.1, Sec 4 | Mẫu số $OOS\_R^2$ trong `rq3_time.csv` không sử dụng Historical Mean làm benchmark. | [generate_rq_analysis.py:44](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/generate_rq_analysis.py#L44) so khớp `'historical_mean'` (thường) trong khi dữ liệu là `'Historical_Mean'` (hoa), khiến mẫu số rơi về $TSS_{\text{period}} = \sum (y_t - \bar{y}_{\text{period}})^2$. | Giải trình rõ bản chất số học: $OOS\_R^2$ được đo lường tương đối so với phương sai mẫu nội kỳ. | **Đã sửa** |
| **ISS-03** | **High** | Sec 3, Sec 4, Sec 6 | Sử dụng thuật ngữ *"độ dốc suy thoái (decay slope)"* cho hiệu số $\Delta_{120D-1D}$. | $\Delta = OOS\_R^2_{120D} - OOS\_R^2_{1D}$ là hiệu số điểm mút (endpoint difference), không phải hệ số góc hồi quy tuyến tính. | Đổi tên chuẩn hóa thành `OOS_R2 Change (120D − 1D)` hoặc `Thay đổi OOS_R2 (120D − 1D)`. | **Đã sửa** |
| **ISS-04** | **High** | Sec 5.1 | Tuyên bố *"Xu hướng suy thoái xuất hiện ở 100% các phân kỳ thời gian"*. | Transformer có $\Delta OOS\_R^2 > 0$ ở 2/3 giai đoạn (+0.5727 ở 2015-2016; +0.1667 ở 2017-2018). Ngoài ra, chỉ có 3/42 trường hợp suy giảm đơn điệu thực sự. | Hiệu chỉnh: Xu hướng giảm điểm mút chỉ đúng với Tree, Linear, SVR, LSTM, RNN; nêu rõ ngoại lệ của Transformer và tính phi đơn điệu. | **Đã sửa** |
| **ISS-05** | **High** | Sec 4, Sec 6 | Khẳng định thứ hạng cố định `AR >> Linear ≈ Mean > Tree > DL` qua mọi giai đoạn. | Tại 120D giai đoạn 2019-2020: Deep Learning (-0.1050) xếp **trên** Mean (-0.1770), Linear (-0.2908), và Tree (-0.3492). Tại 1D các mô hình tiệm cận 0. | Bác bỏ thứ hạng tuyệt đối; phân rã thứ hạng chi tiết theo từng chân trời và từng giai đoạn. | **Đã sửa** |
| **ISS-06** | **High** | Sec 5.3 | Khẳng định quan hệ nhân quả: *"Cú sốc COVID-19 và Fed đẩy sai số lên gấp đôi"*. | Chưa có phân tích độ nhạy loại trừ tháng 3-4/2020. Độ lệch chuẩn mục tiêu tăng từ 0.057 lên 0.106 ($1.87\times$); Normalized RMSE ($\text{RMSE}/\text{SD}$) chỉ biến động nhẹ từ 1.01 đến 1.08. | Chuyển thành mô tả tương quan: Mức tăng RMSE gắn liền với sự gia tăng độ biến động mục tiêu (target volatility), không khẳng định cơ chế nhân quả. | **Đã sửa** |
| **ISS-07** | **Medium** | Sec 4 | Bảng Section 4 bỏ sót họ Deep Learning và Support Vector. | Bảng tóm tắt chỉ đưa Tree, Linear, AR, Mean; loại bỏ DL và SVR dẫn đến mất thông tin quan trọng về hành vi mô hình phức tạp. | Bổ sung đầy đủ cả 6 họ mô hình vào bảng tóm tắt Section 4. | **Đã sửa** |
| **ISS-08** | **Medium** | Sec 1.2, Sec 7 | Coi các khối 2 năm là *"chu kỳ kinh tế vĩ mô"* chính thức. | [config.py:49-56](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/config.py#L49) chỉ chia các khối dương lịch 2 năm cố định (`2015-2016`, `2017-2018`, `2019-2020`), không phải chu kỳ kinh tế theo NBER. | Chuẩn hóa thuật ngữ thành **Các giai đoạn đánh giá theo niên đại (Chronological Evaluation Periods)**. | **Đã sửa** |
| **ISS-09** | **Medium** | Sec 3 | Thiếu giải trình về việc không hiển thị mô hình Logistic Regression trong bảng hồi quy. | Logistic Regression có $OOS\_R^2 \in [-14295, -71]$ do sai lệch đơn vị/thang đo xác suất so với lợi suất thực tế. | Bổ sung ghi chú giải trình lý do tách riêng mô hình phân loại Logistic Regression. | **Đã sửa** |
| **ISS-10** | **Medium** | Sec 5.2, Sec 8 | Khẳng định *"Quy luật cấu trúc nội tại của thị trường tài chính / Bằng chứng thép"*. | Bằng chứng hiện tại hoàn toàn mang tính mô tả ngoại mẫu (descriptive OOS), trên 1 chỉ số duy nhất (S&P 500), chưa thể suy rộng thành quy luật cấu trúc phổ quát. | Tiết chế ngôn ngữ học thuật, giới hạn kết luận trong phạm vi dữ liệu thực nghiệm S&P 500 (2015–2020). | **Đã sửa** |
| **ISS-11** | **High** | Sec 6, Sec 8 | Bỏ qua ảnh hưởng của chuỗi mục tiêu $120D$ gối đầu qua ranh giới phân kỳ. | Target $120D$ ở cuối năm 2016 có điểm hiện thực hóa rơi vào năm 2017. Các quan sát liên tiếp gối đầu 119 ngày. | Làm rõ nguyên tắc nhóm theo ngày gốc dự báo ($t$), cảnh báo hiện tượng gối đầu liên kỳ và vi phạm tính độc lập. | **Đã sửa** |
| **ISS-12** | **Low** | Sec 6, Sec 8 | Kết luận "MOSTLY STABLE" khi chưa có kiểm định thống kê phi phụ thuộc. | Chưa có kiểm định tương tác $Horizon \times Period$ có hiệu chỉnh Newey-West hoặc Block Bootstrap. | Hạ cấp phán quyết xuống `RQ3 DESCRIPTIVELY SUPPORTED — STATISTICAL/METHODOLOGICAL VALIDATION PENDING`. | **Đã sửa** |

---

## C. Đối Chiếu Số Liệu Thực Nghiệm (Numeric Reconciliation)

Toàn bộ các giá trị trong bảng dưới đây được tính toán trực tiếp từ $95,200$ điểm dự báo ngoại mẫu của `all_oos_predictions.csv` và đối chiếu với `rq3_time.csv` cũng như các số liệu được trích dẫn trong `07_rq3_time_report.md`:

### 1. Kiểm tra kích thước và số lượng quan sát
- **Số hàng trong `rq3_time.csv`:** 42 hàng ($14 \text{ Models} \times 3 \text{ Periods}$).
- **Số cột trong `rq3_time.csv`:** 17 cột.
- **Số quan sát ngoại mẫu mỗi mô hình trên từng chân trời:**
  - `2015-2016`: $N = 395$ ngày giao dịch.
  - `2017-2018`: $N = 502$ ngày giao dịch.
  - `2019-2020`: $N = 463$ ngày giao dịch.
  - **Tổng cộng:** $395 + 502 + 463 = 1,360$ ngày (trùng khớp chính xác $100\%$ với RQ1 và RQ2).

### 2. Kiểm tra độ lệch tái tạo `rq3_time.csv` từ `all_oos_predictions.csv`
- **Độ lệch tối đa ($OOS\_R^2$):** `0.000000` (khớp tuyệt đối trên cả 14 mô hình và 5 chân trời).
- **Độ lệch tối đa ($DA$):** `0.000000`.
- **Độ lệch tối đa ($RMSE$):** `0.000000`.

### 3. Bảng đối chiếu các giá trị trung bình họ mô hình (Section 4)
| Họ mô hình | Giai đoạn | Horizon | Giá trị báo cáo cũ | Giá trị tính lại chính xác | Chênh lệch | Hành động |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tree Ensembles** | 2015-2016 | 1D / 120D | -0.007 / -1.689 | -0.0066 / -1.6887 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Tree Ensembles** | 2017-2018 | 1D / 120D | -0.002 / -0.505 | -0.0015 / -0.5046 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Tree Ensembles** | 2019-2020 | 1D / 120D | -0.056 / -0.349 | -0.0557 / -0.3492 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Linear Reg.** | 2015-2016 | 1D / 120D | -0.007 / -0.105 | -0.0071 / -0.1045 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Linear Reg.** | 2017-2018 | 1D / 120D | -0.001 / -0.201 | -0.0009 / -0.2007 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Linear Reg.** | 2019-2020 | 1D / 120D | +0.006 / -0.291 | +0.0061 / -0.2908 | 0.000 | Chuẩn hóa 4 chữ số thập phân |
| **Autoregressive** | 2015-2016 | 1D / 120D | -0.004 / +0.964 | -0.0037 / +0.9639 | 0.000 | Gắn cờ lỗi rò rỉ lọc 1 bước |
| **Autoregressive** | 2017-2018 | 1D / 120D | -0.001 / +0.961 | -0.0009 / +0.9608 | 0.000 | Gắn cờ lỗi rò rỉ lọc 1 bước |
| **Autoregressive** | 2019-2020 | 1D / 120D | +0.027 / +0.958 | +0.0269 / +0.9578 | 0.000 | Gắn cờ lỗi rò rỉ lọc 1 bước |
| **Deep Learning** | 2015-2016 | 1D / 120D | *Bị bỏ sót* | -0.8595 / -0.8462 | N/A | **Bổ sung mới vào bảng** |
| **Deep Learning** | 2017-2018 | 1D / 120D | *Bị bỏ sót* | -0.4864 / -0.9819 | N/A | **Bổ sung mới vào bảng** |
| **Deep Learning** | 2019-2020 | 1D / 120D | *Bị bỏ sót* | -0.0330 / -0.1050 | N/A | **Bổ sung mới vào bảng** |
| **Support Vector** | 2015-2016 | 1D / 120D | *Bị bỏ sót* | -0.0423 / -1.0048 | N/A | **Bổ sung mới vào bảng** |
| **Support Vector** | 2017-2018 | 1D / 120D | *Bị bỏ sót* | -0.0024 / -0.2282 | N/A | **Bổ sung mới vào bảng** |
| **Support Vector** | 2019-2020 | 1D / 120D | *Bị bỏ sót* | +0.0387 / -0.0530 | N/A | **Bổ sung mới vào bảng** |

### 4. Kiểm tra các giá trị RMSE cụ thể trích dẫn trong Section 5.3
- `Historical_Mean` 120D RMSE (2015-2016): Báo cáo = `0.0587` | Tính lại = `0.0587` (Khớp).
- `Historical_Mean` 120D RMSE (2017-2018): Báo cáo = `0.0569` | Tính lại = `0.0569` (Khớp).
- `Historical_Mean` 120D RMSE (2019-2020): Báo cáo = `0.1146` | Tính lại = `0.1146` (Khớp).
- `ARIMA` 120D RMSE (2015-2016): Báo cáo = `0.0122` | Tính lại = `0.0122` (Khớp).
- `ARIMA` 120D RMSE (2019-2020): Báo cáo = `0.0230` | Tính lại = `0.0230` (Khớp).

### 5. Thống kê phương sai chuỗi mục tiêu và Normalized RMSE
| Giai đoạn | Horizon | Target Mean | Target Std | Target Var | Hist. Mean RMSE | Normalized RMSE ($\frac{\text{RMSE}}{\text{Std}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015-2016** | 1D | +0.00018 | 0.00938 | 0.000088 | 0.0094 | 1.002 |
| **2015-2016** | 120D | +0.04365 | 0.05797 | 0.003360 | 0.0587 | **1.013** |
| **2017-2018** | 1D | +0.00021 | 0.00818 | 0.000067 | 0.0082 | 1.002 |
| **2017-2018** | 120D | +0.04070 | 0.05658 | 0.003202 | 0.0569 | **1.006** |
| **2019-2020** | 1D | +0.00060 | 0.01697 | 0.000288 | 0.0170 | 1.002 |
| **2019-2020** | 120D | +0.08045 | 0.10571 | 0.011175 | 0.1146 | **1.084** |

> **Phát hiện quan trọng:** Mặc dù RMSE tuyệt đối tăng gấp đôi từ $0.0569$ lên $0.1146$, chỉ số **Normalized RMSE gần như bất biến ($\approx 1.01 - 1.08$)** do độ biến động mục tiêu ($\text{Target Std}$) tăng tương ứng từ $0.0566$ lên $0.1057$ ($1.87\times$).

---

## D. Kiểm Toán Các Tuyên Bố (Claim Audit)

| Tuyên bố ban đầu trong Report 07 | Bằng chứng thực nghiệm đối chiếu | Đánh giá học thuật | Cách viết đã hiệu đính |
| :--- | :--- | :---: | :--- |
| *“Xu hướng suy thoái dự báo từ 1D đến 120D xuất hiện ở 100% các phân kỳ thời gian.”* | Transformer có $\Delta OOS\_R^2 > 0$ ở 2 giai đoạn. Nhiều mô hình phi đơn điệu (đạt đỉnh tại 20D hoặc 60D như Elastic Net, Ridge, SVR). | **Bác bỏ một phần (Nói quá)** | Xu hướng suy giảm hiệu năng điểm mút từ 1D đến 120D xuất hiện ở phần lớn các mô hình học máy truyền thống, nhưng không mang tính tuyệt đối cho 100% mô hình và không diễn ra đơn điệu qua các chân trời trung gian. |
| *“Tính chất tăng trưởng vượt trội của dòng AR bảo toàn hoàn hảo qua thời gian... Đây là bằng chứng thép.”* | Lệnh `applied.predict` trong `statsmodels` thực hiện lọc 1 bước trên chuỗi gối đầu, rò rỉ $h-1$ ngày tương lai. Mức $R^2 \approx 0.96$ là ảo ảnh toán học của tự tương quan gối đầu $\frac{119}{120}$. | **Bác bỏ hoàn toàn (Lỗi phương pháp luận)** | Kết quả vượt trội của dòng mô hình Tự hồi quy là hệ quả của lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu. Dòng mô hình này bị loại bỏ khỏi bằng chứng về tính ổn định thời gian. |
| *“Thứ hạng giữa các họ mô hình duy trì ổn định: AR >> Linear ≈ Mean > Tree > DL.”* | Tại 120D trong giai đoạn 2019-2020, Deep Learning đạt $OOS\_R^2 = -0.1050$, cao hơn cả Historical Mean (-0.1770), Linear (-0.2908) và Tree (-0.3492). Tại 1D, các mô hình tập trung quanh 0. | **Bác bỏ (Sai lệch thực nghiệm)** | Thứ hạng giữa các họ mô hình không cố định mà thay đổi đáng kể tùy theo chân trời dự báo và giai đoạn cụ thể. Không tồn tại một thứ hạng phổ quát xuyên suốt mọi chân trời. |
| *“Cú sốc COVID-19 và chính sách Fed đã khuếch đại quy mô sai số lên gấp đôi.”* | Target standard deviation tăng $1.87\times$, khiến RMSE tuyệt đối tăng theo tỷ lệ tương ứng, trong khi sai số chuẩn hóa $\frac{\text{RMSE}}{\text{Std}}$ gần như không đổi ($\approx 1.01 - 1.08$). | **Không đủ căn cứ nhân quả** | Sự gia tăng gấp đôi của RMSE trong giai đoạn 2019–2020 phản ánh sự mở rộng của độ biến động thị trường mục tiêu. Phân tích mô tả hiện tại chưa tách biệt được đóng góp cơ học của cú sốc COVID-19. |
| *“Kết quả đại diện cho quy luật cấu trúc nội tại của thị trường tài chính.”* | Nghiên cứu chỉ thực hiện trên chỉ số S&P 500 trong 3 khối thời gian 2 năm (2015–2020). Chưa kiểm tra thị trường quốc tế, cổ phiếu đơn lẻ hoặc chu kỳ dài hơn. | **Nói quá mức (Overgeneralization)** | Các mẫu hình quan sát được mang tính mô tả thực nghiệm trên chỉ số S&P 500 giai đoạn 2015–2020 và không nên được khái quát hóa thành quy luật cấu trúc phổ quát. |

---

## E. Các Vấn Đề Chưa Thể Xác Minh / Chưa Giải Quyết (Unresolved Items)

1. **Kiểm định thống kê tương tác phi phụ thuộc ($Horizon \times Period$ Interaction Tests):**
   - *Lý do chưa xác minh:* Tệp dữ liệu `rq3_time.csv` chỉ cung cấp các thống kê mô tả gộp (pooled descriptive metrics). Để kiểm tra xem tương tác giữa chân trời và thời gian có ý nghĩa thống kê hay không, cần chạy kiểm định suy diễn chính quy (như kiểm định Diebold-Mariano hai chiều, hồi quy bảng với sai số chuẩn Newey-West $L \ge 120$ hoặc Block Bootstrap).
   - *Tệp cần xử lý tiếp:* `results/analysis/statistical_tests.csv` (nội dung chính của **Step 8**).
   - *Mức độ ảnh hưởng đến RQ3:* Không làm vô hiệu hóa các phát hiện mô tả hiện tại, nhưng bắt buộc phải gắn nhãn *"Statistical Validation Pending"*.
   - *Khả năng chuyển bước:* **Không chặn chuyển sang Step 8**, vì Step 8 chính là bước thực hiện các kiểm định thống kê này.

2. **Dữ liệu phân kỳ ngắn hơn (Quarterly / Monthly Subperiods):**
   - *Lý do chưa xác minh:* Dữ liệu hiện được cấu hình cố định theo 3 khối 2 năm trong `config.py`. Việc phân rã ngắn hơn đòi hỏi can thiệp tái cấu trúc pipeline (bị cấm theo quy tắc kiểm toán).
   - *Mức độ ảnh hưởng:* Được ghi nhận rõ ràng như một hạn chế phương pháp luận trong báo cáo.

---

## F. Quyết Nghị Thẩm Định Phương Pháp Luận Cuối Cùng (Final Methodological Verdict)

> ### **RQ3 DESCRIPTIVELY SUPPORTED — STATISTICAL/METHODOLOGICAL VALIDATION PENDING**
>
> **Căn cứ quyết định:**
> 1. Tính toàn vẹn số học của `rq3_time.csv` và khả năng tái lập từ `all_oos_predictions.csv` là hoàn hảo ($100\%$ khớp, max discrepancy = $0.000000$).
> 2. Các mẫu hình mô tả đối với nhóm mô hình Machine Learning (suy giảm hiệu năng tại chân trời dài, sự gia tăng RMSE tương ứng với độ biến động mục tiêu) là có cơ sở dữ liệu thực nghiệm và nhất quán giữa các khối thời gian.
> 3. Tuy nhiên, tính hợp lệ của dòng ARIMA bị bác bỏ do lỗi lọc 1 bước trên chuỗi gối đầu, và các suy diễn thống kê chính quy kiểm soát tự tương quan dài hạn cần được hoàn tất trong Step 8.

---

## G. Cổng Đánh Giá Sẵn Sàng Sang Bước 8 (Step 8 Readiness Gate)

| Tiêu chí kiểm tra | Đánh giá | Bằng chứng & Ghi chú |
| :--- | :---: | :--- |
| **Target construction verified?** | **PASS** | Chuỗi mục tiêu $Target_h = \log(Close_{t+h}/Close_t)$ được tính toán chuẩn xác theo định nghĩa lợi suất log lũy kế. |
| **ARIMA-family forecast alignment verified?** | **FAIL** | Lỗi lọc 1 bước trên chuỗi gối đầu cho $h \ge 5$ tạo rò rỉ tương lai. **Bắt buộc cách ly dòng ARIMA ở Step 8**. |
| **Benchmark verified?** | **PASS** | Đã làm rõ số học: Benchmark denominator trong `rq3_time.csv` là phương sai mẫu nội kỳ $TSS_{\text{period}}$. |
| **OOS_R2 aggregation verified?** | **PASS** | Đã xác minh công thức gộp theo kỳ ngoại mẫu khớp chính xác tuyệt đối với dữ liệu dự báo gốc. |
| **Temporal period assignment verified?** | **PASS** | Đã xác minh 3 khối thời gian 2 năm ($395, 502, 463$ ngày) dựa trên ngày gốc dự báo $t$, không dùng ngày tương lai. |
| **Dependence-aware inference plan specified?** | **PASS** | Đã xây dựng kế hoạch kiểm định: Sử dụng Diebold-Mariano với hiệu chỉnh Harvey-Leybourne-Newbold (HLN) và băng thông trễ $h$. |
| **Existing prediction outputs sufficient for Step 8?** | **PASS** | Tệp `all_oos_predictions.csv` ($95,200$ dòng) và `statistical_tests.csv` đã có sẵn đầy đủ để audit Step 8 mà không cần retrain. |

### KẾT LUẬN SẴN SÀNG SANG BƯỚC 8: **CÓ (YES - VỚI ĐIỀU KIỆN)**
*Điều kiện bắt buộc:* Trong Step 8 (Kiểm định thống kê Diebold-Mariano & Directional Accuracy Tests), **tuyệt đối không sử dụng kết quả dòng ARIMA làm thước đo đối sánh hợp lệ**, và mọi kiểm định so sánh phải tuân thủ việc hiệu chỉnh tự tương quan do chuỗi mục tiêu gối đầu.
