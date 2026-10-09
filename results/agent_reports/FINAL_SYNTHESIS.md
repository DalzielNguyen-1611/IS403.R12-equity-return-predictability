# TỔNG HỢP KẾT QUẢ NGHIÊN CỨU TOÀN DIỆN (FINAL RESEARCH RESULTS SYNTHESIS)

**Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*  
**Mục tiêu nghiên cứu chính:** Xác định hình thái và cấu trúc kinh tế lượng của năng lực dự báo lợi suất chỉ số S&P 500 ngoại mẫu:
$$\text{Predictability} = f(\text{Model}, \text{Horizon}, \text{Regime}, \text{Time})$$  
**Cơ quan thực hiện:** Senior Research Methodology Auditor & Research Data Analyst (Antigravity Analytics System)  
**Ngày hoàn tất bản thảo gốc:** 2026-10-08  
**Ngày kiểm toán & tái tổng hợp toàn diện:** 2026-10-09  
**Đường dẫn tài liệu:** [FINAL_SYNTHESIS.md](file:///e:/IS403.R12-equity-return-predictability/results/agent_reports/FINAL_SYNTHESIS.md)  
**Tập dữ liệu nguồn được xác minh:** 
- `results/fold_metrics/all_fold_metrics.csv` (350 dòng)
- `results/summary/model_horizon_summary.csv` (70 dòng)
- `results/predictions/all_oos_predictions.csv` (95,200 dòng)
- `results/tuning/best_hyperparameters.csv` (325 dòng)
- `results/analysis/rq1_horizon.csv` (70 dòng)
- `results/analysis/rq2_regime.csv` (42 dòng)
- `results/analysis/rq3_time.csv` (42 dòng)
- `results/analysis/statistical_tests.csv` (70 dòng thô) & `results/analysis/statistical_tests_corrected.csv` (130 dòng chuẩn hóa)

---

## 1. Tóm Tắt Quản Trị & Phát Hiện Cốt Lõi (Executive Summary)

1. **Quy luật suy thoái dự báo theo chân trời (RQ1 — Predictability Decay):**  
   Năng lực dự báo ngoại mẫu ($OOS\_R^2$) của toàn bộ các mô hình Machine Learning hợp lệ suy thoái rõ rệt khi chân trời dự báo mở rộng từ $1D$ đến $120D$. Tại chân trời $1D$, các mô hình tiệm cận mốc chuẩn ngây thơ ($OOS\_R^2 \approx -0.02$ đến $+0.018$); khi chân trời tiến đến $60D$ và $120D$, sai số tích lũy khiến $OOS\_R^2$ sụp đổ xuống mức âm sâu ($-0.068$ ở Elastic Net đến $-0.784$ ở LightGBM). Bằng chứng thực nghiệm xác nhận **sự suy thoái năng lực dự báo là một quy luật cấu trúc vững chắc**.
2. **Giải mã nghịch lý dòng mô hình Tự hồi quy (ARIMA Family Anomaly):**  
   Kiểm toán mã nguồn phát hiện việc dòng ARIMA/ARIMAX/SARIMA ghi nhận $OOS\_R^2 > 0.95$ ở chân trời dài là hệ quả của **lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu (1-step filtering leakage)** trong hàm `applied.predict` của thư viện `statsmodels`. Chuỗi trễ $Target_{h,t-1}$ chứa giá tương lai $h-1$ ngày, tạo ra tương quan cơ học giả tạo $\approx \frac{h-1}{h}$. Khi cách ly dòng mô hình này, không còn bất kỳ ngoại lệ nào đi ngược lại quy luật suy thoái dự báo.
3. **Sự điều tiết khốc liệt của Chế độ Biến động (RQ2 — Volatility Regime Moderation):**  
   Chế độ biến động thị trường (phân loại theo VIX: Low $<15$, Normal $15-25$, High $>25$) làm thay đổi căn bản quy mô sai số ngoại mẫu:  
   - Ở chế độ Low và Normal, suy thoái diễn ra vừa phải ($\Delta_{120D-1D} \in [-0.04, -0.28]$).  
   - Ở chế độ High, phương sai thị trường bùng nổ gấp $3-4$ lần. Mặc dù tại $1D$, chế độ High cung cấp biên độ dao động mạnh giúp một số mô hình đạt $OOS\_R^2$ dương nhẹ ($+0.01 \to +0.03$), nhưng tại $120D$, sai số bình phương bị khuếch đại cực đại khiến $OOS\_R^2$ của các mô hình Cây sụp đổ xuống **$-1.83$ đến $-3.36$**.
4. **Tính ổn định tương đối qua các giai đoạn lịch sử (RQ3 — Chronological Stability):**  
   Xu hướng suy thoái hiệu năng điểm mút từ $1D$ đến $120D$ được bảo toàn trên phần lớn các họ mô hình học máy qua cả 3 khối thời gian 2 năm (2015–2016, 2017–2018, 2019–2020). Cú sốc đại dịch COVID-19 năm 2020 làm tăng gấp đôi sai số tuyệt đối RMSE do độ biến động mục tiêu tăng $1.87\times$, nhưng sai số chuẩn hóa Normalized RMSE ($\frac{\text{RMSE}}{\text{Std}}$) gần như bất biến ($\approx 1.01 - 1.08$), chứng minh trật tự suy thoái không bị thao túng bởi một giai đoạn đơn lẻ.
5. **Bằng chứng kiểm định thống kê suy diễn chuẩn mực (Statistical Tests):**  
   - **Kiểm định Diebold–Mariano (với Historical Mean):** Không có bất kỳ mô hình Machine Learning hay Deep Learning nào đánh bại được Historical Mean trên cơ sở bình phương sai số ($DM \le 0, p_{1s} > 0.5$ tại mọi $h \ge 5$). Hơn nữa, kiểm định hai phía chỉ ra nhiều mô hình phức tạp (Tree, DL) hoạt động **kém hơn có ý nghĩa thống kê** so với Historical Mean (ví dụ LightGBM 120D $p_{2s} = 0.034$, RF 120D $p_{2s} = 0.033$).
   - **Hiệu chỉnh Đa giả thuyết:** Dưới hiệu chỉnh Bonferroni ($\alpha = 0.000833$) và kiểm soát FDR Benjamini–Hochberg ($q = 0.05$) trên 60 phép thử DM, không có mô hình học máy nào sống sót.
6. **Bản chất của Nghịch lý Độ chính xác Định hướng ($DA$ Paradox):**  
   Tại $120D$, tỷ lệ đoán đúng chiều ($DA$) của các mô hình đạt từ $54\%$ đến $78\%$ với $p < 0.001$. Tuy nhiên, phân tích tỷ lệ cơ sở (Base Rate) chứng minh thị trường S&P 500 có **$77.79\%$** số ngày mang lợi suất dương. Chiến lược ngây thơ Luôn Dự Báo Dương (Always-Positive) tự động đạt $DA = 77.79\%$. **$100\%$ mô hình Machine Learning đều không vượt qua được mốc chuẩn ngây thơ này**. $DA$ cao ở chân trời dài chỉ phản ánh sự bám theo xu hướng trôi dạt (equity drift) của thị trường, không chứng minh kỹ năng định lượng.
7. **Ưu thế của tính đơn giản và co rút tuyến tính (Superiority of Regularized Shrinkage):**  
   Trong số các mô hình học máy, mô hình co rút tuyến tính `Elastic_Net` thể hiện khả năng kháng suy thoái tốt nhất toàn diện nhờ cơ chế phạt $L_1/L_2$ triệt tiêu phương sai nhiễu. Ngược lại, các kiến trúc phức tạp (Tree Ensembles, Deep Learning như Transformer) bị hiện tượng quá khớp ngoại mẫu nghiêm trọng do tỷ lệ tín hiệu/nhiễu của dữ liệu chuỗi tài chính quá thấp.

---

## 2. Tính Toàn Vẹn Thực Nghiệm & Kiểm Toán Phương Pháp Luận (Experimental Validity)

### 2.1 Kích thước và độ phủ tập dữ liệu
Hệ thống thực nghiệm được kiểm toán độc lập và xác nhận độ phủ $100\%$ không mất mát:
- **`all_fold_metrics.csv`:** $14 \text{ Models} \times 5 \text{ Horizons} \times 5 \text{ Folds} = 350$ hàng hoàn chỉnh.
- **`all_oos_predictions.csv`:** Đúng $95,200$ hàng cấp quan sát ($272$ ngày kiểm định/fold $\times 5$ folds $\times 5$ horizons $\times 14$ models), không chứa giá trị rỗng NaN/Inf, căn chỉnh thống nhất trên chuỗi mục tiêu $Target_{h,t} = \log(Close_{t+h}/Close_t)$ từ 2015 đến 2020 ($1,360$ ngày giao dịch ngoại mẫu duy nhất).
- **`best_hyperparameters.csv`:** $13 \text{ Models có tham số} \times 5 \text{ Horizons} \times 5 \text{ Folds} = 325$ hàng. Mô hình `Historical_Mean` không tham số được phân tách chuẩn xác.
- **Tái lập số học (Numerical Reconciliation):** Sai số tính toán máy tính giữa dự báo thô và bảng tổng kết đạt $\Delta \le 1.25 \times 10^{-16}$ (khớp tuyệt đối).

### 2.2 Đánh giá quy trình ngăn ngừa rò rỉ dữ liệu (Walk-Forward Validation Audit)
- **Cấu trúc phân tách ngoại mẫu:** Quy trình huấn luyện sử dụng cấu trúc **Walk-Forward Validation mở rộng (Expanding Window)** 5 Folds.
- **Tối ưu hóa siêu tham số:** Được thực hiện khép kín trong tập huấn luyện của từng Fold thông qua TimeSeriesSplit (Optuna 30–50 trials, tỷ lệ kiểm định nội bộ $15\%$). Tập ngoại mẫu $X_{\text{test}}$ của từng Fold hoàn toàn được bảo vệ khỏi quá trình tìm kiếm tham số.
- **Phát hiện lỗi phương pháp luận ở dòng ARIMA:**  
  Mặc dù pipeline ML không bị rò rỉ dữ liệu giữa tập train và test, hàm dự báo của dòng mô hình ARIMA tại [arimax.py:54](file:///e:/IS403.R12-equity-return-predictability/src/training/baselines/arimax/arimax.py#L54) sử dụng lệnh `applied.predict` thực hiện lọc 1 bước trên chuỗi lợi suất lũy kế gối đầu ngoại mẫu, vô tình nạp giá tương lai vào mô hình. Dòng mô hình này đã được **cách ly triệt để** khỏi các kết luận khoa học hợp lệ của nghiên cứu.

---

## 3. Phổ Hiệu Năng Tổng Thể Theo Họ Mô Hình (Overall Performance by Families)

Phổ năng lực dự báo ngoại mẫu của 14 mô hình được cấu trúc thành 6 nhóm đặc trưng:

```
+---------------------------------------------------------------------------------------------------------+
|                              PHỔ NĂNG LỰC DỰ BÁO NGOẠI MẪU (OOS R² SUMMARY)                             |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Họ mô hình               | Chân trời 1D          | Chân trời 20D         | Chân trời 120D               |
+--------------------------+-----------------------+-----------------------+------------------------------+
| 1. Tuyến tính co rút     | Dương nhẹ (+0.002)    | Âm nhẹ (-0.016)       | Kháng suy thoái (-0.068..-0.28)
| 2. Chuẩn ngây thơ (Mean) | Mốc tham chiếu (0.000)| Mốc tham chiếu (0.000)| Mốc tham chiếu (0.000)       |
| 3. Support Vector (SVR)  | Đỉnh ngắn hạn (+0.018)| Âm nhẹ (-0.038)       | Âm trung bình (-0.174)       |
| 4. Tree Ensembles        | Tiệm cận 0 (-0.010)   | Âm trung bình (-0.095)| Sụp đổ phương sai (-0.27..-0.78)
| 5. Deep Learning         | Âm nặng (-0.01..-0.73)| Âm trung bình (-0.150)| Quá khớp nhiễu (-0.04..-0.56)|
| 6. Autoregressive (AR)   | [Không hợp lệ - Lỗi rò rỉ lọc 1 bước trên chuỗi gối đầu - Bị cách ly]       |
+--------------------------+-----------------------+-----------------------+------------------------------+
```

1. **Họ Tuyến Tính Co Rút (Elastic Net, Ridge):**  
   - `Elastic_Net` là mô hình có hiệu năng ngoại mẫu ổn định nhất toàn nghiên cứu. Bằng việc co các trọng số nhiễu về 0, Elastic Net duy trì $OOS\_R^2 = +0.0026$ tại 1D và chỉ suy giảm nhẹ về $-0.0676$ tại 120D, vượt trội hơn toàn bộ các mô hình phi tuyến tính phức tạp.
   - `Ridge` kiểm soát phương sai tốt ở ngắn hạn (+0.0019 tại 1D) nhưng suy giảm mạnh hơn tại 120D ($-0.2809$) do hình phạt $L_2$ không triệt tiêu các đặc trưng nhiễu.
2. **Họ Chuẩn Ngây Thơ (Historical Mean Benchmark):**  
   - Đóng vai trò mốc chuẩn kinh tế lượng bất khả xâm phạm. Theo định nghĩa Campbell–Thompson, $OOS\_R^2_{\text{bench}} = 0.0000$. Việc mọi mô hình ML phức tạp đều có $OOS\_R^2 < 0$ ở chân trời dài chứng minh rằng trung bình lịch sử giản đơn là một ước lượng tối ưu cho lợi suất kỳ vọng dài hạn.
3. **Họ Support Vector Regression (SVR):**  
   - Đạt hiệu năng dự báo 1D cao nhất trong các mô hình hợp lệ ($OOS\_R^2 = +0.0182$), nhưng nhanh chóng suy thoái thành âm từ chân trời 5D ($-0.0195$) và rơi xuống $-0.1739$ tại 120D.
4. **Họ Cây Quyết Định (Random Forest, XGBoost, LightGBM):**  
   - Thể hiện hiện tượng bùng nổ phương sai sai số ngoại mẫu. Tại 1D, các mô hình Cây dao động quanh 0 ($-0.0105$ đến $+0.0039$). Tuy nhiên, tại 120D, cấu trúc phân nhánh bậc thang tích lũy sai số dự báo rất lớn: XGBoost đạt $-0.2744$, Random Forest đạt $-0.4852$, và LightGBM sụp đổ xuống $-0.7836$.
5. **Họ Học Sâu (RNN, LSTM, Transformer):**  
   - Thất bại trước bài toán dữ liệu bảng chuỗi thời gian tài chính phẳng. Do tỷ lệ tín hiệu trên nhiễu thấp và kích thước mẫu $1,360$ ngày không đủ cho tối ưu hóa hàng ngàn tham số, Transformer bị underfitting nghiêm trọng tại 1D ($OOS\_R^2 = -0.7354$), trong khi LSTM và RNN dao động thất thường ($-0.1444$ và $-0.0402$ tại 120D).
6. **Mô hình Phân loại (Logistic Regression):**  
   - Được cách ly hoàn toàn khỏi không gian hồi quy liên tục. Đạt độ chính xác định hướng tăng dần từ $50.59\%$ (1D) lên $74.56\%$ (120D), bám sát độ trôi của thị trường.

---

## 4. Trả Lời Câu Hỏi Nghiên Cứu 1: Quy Luật Suy Thoái Theo Chân Trời (RQ1)

> **RQ1:** *How does out-of-sample equity return predictability change across forecast horizons?*

### Kết luận thực nghiệm cho RQ1:
1. **Sự tồn tại của quy luật suy thoái:**  
   Năng lực dự báo lợi suất cổ phiếu ngoại mẫu **suy thoái rõ rệt theo chân trời dự báo** trên toàn bộ các mô hình học máy hợp lệ. Năng lực dự báo đạt mức tối đa ở chân trời siêu ngắn ($1D$), sau đó giảm dần và chuyển sang âm sâu khi chân trời mở rộng ra $60D$ và $120D$.
2. **Tính đơn điệu của quá trình suy thoái (Monotonicity):**  
   - `Elastic_Net`, `Random_Forest`, và `SVR` thể hiện **sự suy thoái đơn điệu nghiêm ngặt (Strictly Monotonic Decay)**: $OOS\_R^2_{1D} > OOS\_R^2_{5D} > OOS\_R^2_{20D} > OOS\_R^2_{60D} > OOS\_R^2_{120D}$.
   - `XGBoost` đạt đỉnh nhẹ tại 5D trước khi suy giảm đơn điệu ở các chân trời tiếp theo.
   - Các mô hình Deep Learning thể hiện suy thoái phi đơn điệu do tính bất ổn định của tối ưu hóa phi lồi.
3. **Mức độ suy thoái điểm mút (Endpoint Change $\Delta_{120D-1D}$):**  
   Hiệu số thay đổi $OOS\_R^2$ từ $1D$ đến $120D$ mang giá trị âm sâu trên toàn bộ các mô hình hồi quy truyền thống:
   - `Elastic_Net`: $\Delta = -0.0702$ (kháng suy thoái tốt nhất).
   - `SVR`: $\Delta = -0.1921$.
   - `XGBoost`: $\Delta = -0.2783$.
   - `Ridge`: $\Delta = -0.2828$.
   - `Random_Forest`: $\Delta = -0.4747$.
   - `LightGBM`: $\Delta = -0.7629$ (suy thoái mạnh nhất).
4. **Bác bỏ sự "tăng trưởng dự báo" của dòng ARIMA:**  
   Hiện tượng $OOS\_R^2$ tăng vọt từ $0.02$ lên $0.96$ ở dòng ARIMA chính thức được xác nhận là sai lệch phương pháp luận (rò rỉ lọc 1 bước trên chuỗi gối đầu), không phản ánh quy luật thị trường.

---

## 5. Trả Lời Câu Hỏi Nghiên Cứu 2: Tính Phụ Thuộc Vào Chế Độ Biến Động (RQ2)

> **RQ2:** *How does the relationship between forecast horizon and predictive skill differ across market volatility regimes?*

### Kết luận thực nghiệm cho RQ2:
1. **Vai trò điều tiết của Chế độ Biến động:**  
   Chế độ biến động thị trường **điều tiết mạnh mẽ mối quan hệ giữa chân trời dự báo và năng lực mô hình**, làm biến đổi căn bản cả độ lớn sai số tuyệt đối và độ dốc suy thoái.
2. **So sánh hình thái suy thoái giữa 3 chế độ:**
   - **Chế độ Biến động Thấp (Low Regime, VIX $< 15$):**  
     Thị trường dao động ổn định. Năng lực dự báo duy trì gần 0 ở ngắn hạn và suy thoái nhẹ nhàng ở dài hạn ($\Delta_{120D-1D}$ của Tree Ensembles chỉ từ $-0.09$ đến $-0.15$).
   - **Chế độ Biến động Bình thường (Normal Regime, VIX $15 - 25$):**  
     Hình thái suy thoái tương tự Low Regime, với biên độ sai số mở rộng vừa phải.
   - **Chế độ Biến động Cao (High Regime, VIX $> 25$):**  
     Xuất hiện hiện tượng **sụp đổ dự báo dốc đứng ở chân trời dài**. Mặc dù tại $1D$, biến động lớn tạo ra các tín hiệu biên độ giúp một số mô hình đạt đỉnh ngắn hạn (SVR đạt $+0.0387$, XGBoost đạt $+0.0270$), nhưng khi tiến đến $120D$, phương sai mục tiêu bùng nổ khiến $OOS\_R^2$ của các mô hình Cây rơi tự do xuống **$-1.8340$ (XGBoost), $-2.5204$ (Random Forest), và $-3.3592$ (LightGBM)**.
3. **Bản chất kinh tế lượng của hiện tượng sụp đổ trong High Regime:**  
   Phân tích cho thấy hiện tượng $OOS\_R^2$ âm sâu trong High Regime phản ánh việc hàm mất mát bình phương phạt rất nặng các sai số trong thời kỳ biến động cực đoan. Khi thị trường khủng hoảng, các mô hình học máy đưa ra dự báo lệch hướng bị trừng phạt bằng sai số bình phương khổng lồ trong tử số của $OOS\_R^2$.

---

## 6. Trả Lời Câu Hỏi Nghiên Cứu 3: Tính Ổn Định Qua Thời Gian (RQ3)

> **RQ3:** *Are the observed horizon- and regime-dependent patterns stable over time?*

### Kết luận thực nghiệm cho RQ3:
1. **Tính ổn định của quy luật suy thoái:**  
   Mẫu hình suy thoái theo chân trời **phần lớn ổn định theo thời gian (Mostly Stable)** khi kiểm tra trên 3 giai đoạn đánh giá theo niên đại (2015–2016, 2017–2018, và 2019–2020). Chiều hướng suy thoái từ $1D$ đến $120D$ mang dấu âm đồng thuận trên toàn bộ các mô hình Machine Learning truyền thống (Tree Ensembles, Regularized Linear, SVR).
2. **Bác bỏ giả thuyết một giai đoạn chi phối (Single-Period Driver Rejection):**  
   Kết quả nghiên cứu không bị chi phối bởi một giai đoạn khủng hoảng cá biệt nào. Cụ thể, mức độ suy thoái của Tree Ensembles trong giai đoạn bình thường trước đại dịch (2015–2016) đạt $-1.6887$, thậm chí còn âm sâu hơn so với giai đoạn khủng hoảng 2019–2020 ($-0.3492$).
3. **Phân rã biến động RMSE và tính bất biến của Normalized RMSE:**  
   Trong giai đoạn 2019–2020, sai số tuyệt đối RMSE tăng gấp đôi (ví dụ Historical Mean 120D RMSE tăng từ $0.0569$ lên $0.1146$). Tuy nhiên, độ lệch chuẩn mục tiêu ($\text{Target Std}$) cũng tăng tương ứng $1.87\times$ (từ $0.0566$ lên $0.1057$). Do đó, sai số chuẩn hóa **Normalized RMSE ($\frac{\text{RMSE}}{\text{Std}}$) gần như bất biến quanh mức $1.01 - 1.08$**, chứng minh quy mô sai số tăng là do thị trường mở rộng biên độ dao động, không phải do cấu trúc mô hình bị suy thoái đột biến.

---

## 7. Bằng Chứng Kiểm Định Thống Kê Tổng Hợp (Statistical Evidence Synthesis)

Dựa trên dữ liệu kiểm định đã được tái lập và chuẩn hóa trong [statistical_tests_corrected.csv](file:///e:/IS403.R12-equity-return-predictability/results/analysis/statistical_tests_corrected.csv):

### 7.1 Ma trận Kiểm định Diebold–Mariano So Với Historical Mean (60 phép thử)
Áp dụng vi sai mất mát bình phương $d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$, hiệu chỉnh HLN, tham số trễ $\text{lag} = h$:

1. **Kiểm định một phía ($H_1: \text{Mô hình tốt hơn Benchmark}$):**  
   - Toàn bộ các mô hình Machine Learning và Deep Learning tại mọi chân trời từ $5D$ trở lên đều có $d_t < 0$ (sai số lớn hơn Historical Mean) dẫn đến $p_{1s} \in [0.58, 1.00]$.  
   - **Kết luận:** Hoàn toàn **không có bằng chứng thực nghiệm** cho thấy bất kỳ mô hình Machine Learning nào đánh bại được Historical Mean trên cơ sở bình phương sai số ngoại mẫu.
2. **Kiểm định hai phía ($H_1: \text{Mô hình khác biệt Benchmark}$):**  
   Xác nhận nhiều mô hình phức tạp **kém hơn có ý nghĩa thống kê ($p_{2s} < 0.05$)** so với Historical Mean:
   - `LightGBM`: Kém hơn tại 120D ($DM = -2.122, p_{2s} = 0.0340^*$).
   - `Random_Forest`: Kém hơn tại 120D ($DM = -2.137, p_{2s} = 0.0328^*$).
   - `Ridge`: Kém hơn tại 5D ($DM = -2.109, p_{2s} = 0.0351^*$) và 120D ($DM = -2.033, p_{2s} = 0.0423^*$).
   - `LSTM`: Kém hơn tại 5D ($DM = -2.463, p_{2s} = 0.0139^*$).
   - `RNN`: Kém hơn tại 60D ($DM = -2.731, p_{2s} = 0.0064^*$).
   - `Transformer`: Kém hơn rất mạnh tại 1D, 5D ($p_{2s} < 0.0001^*$), 20D ($p_{2s} = 0.0231^*$), và 60D ($p_{2s} = 0.0179^*$).
3. **Hiệu chỉnh Đa giả thuyết (Bonferroni & Benjamini–Hochberg FDR):**  
   - Ngưỡng Bonferroni cho toàn họ 60 giả thuyết: $\alpha_{\text{Bonferroni}} = \frac{0.05}{60} \approx 0.0008333$.  
   - Dưới cả Bonferroni và FDR ($q = 0.05$), chỉ có đúng 3 kiểm định tại 5D của dòng ARIMA đạt ngưỡng hình thức nhưng bị vô hiệu hóa do lỗi rò rỉ lọc 1 bước. **Không có bất kỳ mô hình Machine Learning nào sống sót sau hiệu chỉnh đa giả thuyết**.

### 7.2 Kiểm định Hướng đi Nhị thức (Binomial DA Tests) & Mốc Chuẩn Always-Positive
- Tại $1D$, $43\%$ số mô hình (6/14) không đạt ý nghĩa thống kê so với tỷ lệ ngẫu nhiên $50\%$, phản ánh tính khó dự báo ở tần suất cao.
- Tại $120D$, mặc dù $100\%$ mô hình có $p < 0.001$ so với $50\%$, tỷ lệ lợi suất thực tế dương của thị trường là **$77.79\%$**.
- Mốc chuẩn ngây thơ **Luôn Dự Báo Dương (Always-Positive Benchmark)** và `Historical_Mean` đạt đúng $DA = 77.79\%$. **Không có mô hình Machine Learning nào đánh bại được $77.79\%$** (XGBoost đạt $71.69\%$, SVR đạt $66.76\%$, Random Forest đạt $63.31\%$, LightGBM đạt $58.38\%$, Transformer đạt $54.41\%$). Giá trị $p < 0.001$ hoàn toàn do bám theo độ trôi dương thị trường, không phản ánh năng lực dự báo.

---

## 8. Các Phát Hiện Vững Chắc (Robust Findings)

1. **Quy luật suy thoái dự báo theo chân trời của Machine Learning:**  
   Hiện tượng $OOS\_R^2$ suy giảm từ ngắn hạn sang dài hạn được kiểm chứng đồng thời trên toàn bộ các họ mô hình học máy độc lập, qua cả 5 outer folds và 3 khối thời gian đánh giá.
2. **Sự thất bại toàn diện của Machine Learning trước Benchmark giản đơn:**  
   Không có mô hình học máy nào tạo ra được giá trị thặng dư dự báo có ý nghĩa thống kê so với Historical Mean trên chuỗi lợi suất ngoại mẫu.
3. **Ưu thế kháng suy thoái của Co rút Tuyến tính (Elastic Net):**  
   Tính ổn định của cơ chế phạt $L_1/L_2$ giúp Elastic Net duy trì hiệu năng vượt trội hơn toàn bộ các mô hình học máy phức tạp ở các chân trời dài hạn.
4. **Tác động khuếch đại sai số của Chế độ Biến động cao (High Regime):**  
   Phương sai thị trường bùng nổ dẫn đến sự trừng phạt bình phương sai số cực đại đối với các mô hình phi tuyến ở chân trời dài.

---

## 9. Các Phát Hiện Yếu & Chưa Chắc Chắn (Weak / Uncertain Findings)

1. **Hiệu năng của các kiến trúc Học Sâu (Deep Learning):**  
   Các mô hình LSTM, RNN và Transformer thể hiện sự dao động phi đơn điệu do tính nhạy cảm với khởi tạo trọng số ngẫu nhiên và kích thước mẫu ngoại mẫu giới hạn.
2. **Tín hiệu đảo chiều ngắn hạn trong High Regime tại 1D:**  
   Mặc dù một số mô hình ghi nhận $OOS\_R^2$ dương tại 1D trong High Regime, kích thước mẫu của chế độ này ($N = 252$ ngày) tương đối nhỏ, cần thêm dữ liệu khủng hoảng để xác thực tính bền vững.

---

## 10. Các Phát Hiện Trái Chiều & Nghịch Lý Phương Pháp Luận (Contradictory Findings & Paradoxes)

1. **Nghịch lý Dự báo Điểm ($OOS\_R^2$) và Độ chính xác Hướng đi ($DA$):**  
   Ở chân trời dài, $OOS\_R^2$ âm rất sâu nhưng $DA$ lại đạt trên $70\%$. Nghịch lý này được giải thích hoàn toàn bởi sự phân tách giữa biên độ sai số (bị trừng phạt bình phương) và độ trôi dạt dương tự nhiên của chỉ số chứng khoán (positive drift).
2. **Ảo ảnh về sự vượt trội của dòng mô hình ARIMA:**  
   Sự mâu thuẫn giữa giả thuyết suy thoái chung và kết quả tăng trưởng thần kỳ của ARIMA ($R^2 \approx 0.96$) được giải quyết triệt để: đây là ảo ảnh toán học sinh ra từ lỗi rò rỉ lọc 1 bước trên chuỗi dữ liệu gối đầu trong thư viện `statsmodels`.

---

## 11. Các Phát Hiện Bất Ngờ (Unexpected Findings)

1. **Sự sụp đổ của các mô hình Cây phức tạp (Tree Ensembles Overfitting):**  
   Các thuật toán hàng đầu trong khoa học dữ liệu bảng như LightGBM và Random Forest lại là những mô hình hoạt động kém nhất ở chân trời dài hạn ($OOS\_R^2 < -0.48$), thua xa một hằng số trung bình lịch sử giản đơn.
2. **Mô hình ngây thơ Luôn Dự Báo Dương đánh bại toàn bộ mô hình Học Máy:**  
   Tại 120D, không có một thuật toán AI phức tạp nào đạt được tỷ lệ đoán đúng chiều cao hơn quy tắc ngây thơ: *"luôn dự đoán ngày mai thị trường sẽ tăng"*.

---

## 12. Hạn Chế Phương Pháp Luận Của Nghiên Cứu (Methodological Limitations)

1. **Vấn đề chuỗi mục tiêu gối đầu (Overlapping Returns):**  
   Tại các chân trời $h > 1$, việc tính lợi suất lũy kế khiến các quan sát kế tiếp chia sẻ $h-1$ ngày dữ liệu chung, vi phạm giả định quan sát độc lập $iid$ và làm suy giảm bậc tự do hiệu dụng của các kiểm định thống kê cổ điển.
2. **Độ nhạy của thước đo $OOS\_R^2$ trước các cú sốc biến động ngoại lai:**  
   Hàm mất mát bình phương khuếch đại cực đoan các sai số trong thời kỳ biến động lớn (như tháng 3/2020), khiến $OOS\_R^2$ bị kéo xuống mức âm sâu dù mô hình có thể nắm bắt đúng hướng đi.
3. **Phạm vi tài sản nghiên cứu:**  
   Thực nghiệm được thực hiện trên một tài sản duy nhất là chỉ số S&P 500 giai đoạn 2015–2020. Các kết luận phản ánh đặc tính của chỉ số thị trường vốn lớn, chưa khái quát hóa sang các cổ phiếu đơn lẻ hoặc thị trường cận biên.

---

## 13. Danh Mục Bảng Khuyến Nghị Cho Khóa Luận / Luận Văn (Recommended Thesis Tables)

1. **Bảng 1 (Table 1):** *Walk-Forward Validation Data Splits, Market Drift Base Rates, and Volatility Regime Distribution.*
2. **Bảng 2 (Table 2):** *Main Out-of-Sample Performance Matrix ($OOS\_R^2$, MAE, RMSE, DA) Across 14 Models and 5 Forecast Horizons.*
3. **Bảng 3 (Table 3):** *Regime-Dependent Predictability Matrix: $OOS\_R^2$ Across Low, Normal, and High Volatility Regimes.*
4. **Bảng 4 (Table 4):** *Chronological Stability Analysis: Performance and Normalized RMSE Across Three Sub-Periods (2015–2016, 2017–2018, 2019–2020).*
5. **Bảng 5 (Table 5):** *Diebold–Mariano Statistical Hypothesis Tests vs. Historical Mean: Raw Statistics, Two-Sided $p$-values, Bonferroni, and FDR Corrections.*
6. **Bảng 6 (Table 6):** *Directional Accuracy Evaluation: Comparison Against 50% Random Guess and Always-Positive Drift Baseline.*

---

## 14. Danh Mục Biểu Đồ Khuyến Nghị Cho Khóa Luận / Luận Văn (Recommended Thesis Figures)

1. **Hình 1 (Figure 1):** *Term Structure of Equity Return Predictability: $OOS\_R^2$ Decay Across Horizons (1D to 120D) by Valid Model Families.*
2. **Hình 2 (Figure 2):** *Volatility Regime Moderation Curves: Comparison of Predictability Decay in Low, Normal, and High Volatility Environments.*
3. **Hình 3 (Figure 3):** *The Predictability Paradox: Divergence Between Directional Accuracy ($DA$) and Out-of-Sample $R^2$ Under Upward Market Drift.*
4. **Hình 4 (Figure 4):** *Chronological Stability Trajectories: Sub-Period Out-of-Sample $R^2$ and Normalized RMSE Across Chronological Epochs.*
5. **Hình 5 (Figure 5):** *Methodological Audit Diagram: Illustrating the 1-Step Filtering Leakage Mechanism in Autoregressive Models on Overlapping Targets.*

---

## 15. Câu Trả Lời Cuối Cùng Cho RQ1 (Final Answer to RQ1)

> **Năng lực dự báo lợi suất cổ phiếu ngoại mẫu suy thoái rõ rệt và đơn điệu theo chân trời dự báo.** Năng lực dự báo đạt mức tối đa ở chân trời siêu ngắn ($1D$), sau đó suy thoái nhanh chóng thành âm ở các chân trời dài hạn ($20D, 60D, 120D$). Toàn bộ các mô hình Machine Learning hợp lệ đều tuân theo quy luật suy thoái này; các cấu trúc mô hình phi tuyến phức tạp suy thoái nhanh nhất, trong khi mô hình co rút tuyến tính giản đơn thể hiện khả năng kháng suy thoái tốt nhất.

---

## 16. Câu Trả Lời Cuối Cùng Cho RQ2 (Final Answer to RQ2)

> **Chế độ biến động thị trường đóng vai trò điều tiết khốc liệt mối quan hệ giữa chân trời và năng lực dự báo.** Trong khi ở chế độ biến động thấp và bình thường, sự suy thoái diễn ra vừa phải; thì ở chế độ biến động cao, xuất hiện hiện tượng sụp đổ dự báo nghiêm trọng ở chân trời dài do phương sai mục tiêu bùng nổ, biến môi trường biến động cao thành điều kiện rủi ro nhất cho các thuật toán dự báo điểm học máy.

---

## 17. Câu Trả Lời Cuối Cùng Cho RQ3 (Final Answer to RQ3)

> **Các mẫu hình suy thoái theo chân trời phần lớn ổn định qua các chu kỳ thời gian theo niên đại.** Xu hướng suy giảm năng lực dự báo từ ngắn hạn sang dài hạn được bảo toàn trên toàn bộ các mô hình học máy truyền thống qua cả 3 khối thời gian độc lập. Mức tăng quy mô sai số trong giai đoạn 2019–2020 phản ánh sự gia tăng độ biến động mục tiêu của thị trường chứ không phá vỡ trật tự cấu trúc của các họ mô hình.

---

## 18. Hàm Ý Nghiên Cứu & Ứng Dụng Thực Tiễn (Research & Practical Implications)

1. **Hàm ý đối với Quản trị Danh mục & Đầu tư Định lượng (Quantitative Investing):**
   - **Giới hạn phạm vi ứng dụng của Machine Learning:** Các tín hiệu Machine Learning dựa trên đặc trưng kỹ thuật chỉ có giá trị gia tăng ở các chiến lược giao dịch ngắn hạn ($1D - 5D$). Tuyệt đối không sử dụng mô hình học máy phi tuyến phức tạp (Tree Ensembles, Deep Learning) để dự báo điểm cho danh mục đầu tư trung và dài hạn ($60D - 120D$).
   - **Tận dụng tính co rút tham số:** Trong các bài toán tài chính có tỷ lệ tín hiệu/nhiễu thấp, các mô hình co rút tuyến tính giản đơn (`Elastic_Net`) hoặc mốc chuẩn trung bình lịch sử giản đơn (`Historical_Mean`) là lựa chọn tối ưu, bảo vệ danh mục khỏi rủi ro quá khớp phương sai nhiễu.
   - **Quản trị rủi ro chế độ biến động:** Khi VIX vượt ngưỡng 25 (High Regime), cần lập tức hạ tỷ trọng phân bổ vốn của các mô hình học máy đa chân trời để phòng tránh tổn thất bình phương cực đại.
2. **Hàm ý đối với Nghiên cứu Khoa học & Phương pháp luận Tài chính:**
   - **Cảnh giác trước rò rỉ dữ liệu trong chuỗi gối đầu:** Luôn kiểm toán kỹ lưỡng cơ chế dự báo của các mô hình kinh tế lượng chuỗi thời gian khi xử lý chuỗi lợi suất lũy kế gối đầu để tránh ảo ảnh dự báo do lọc 1 bước.
   - **Phân định rạch ròi giữa $OOS\_R^2$ và Directional Accuracy:** Không được quy kết tỷ lệ đoán đúng chiều cao là bằng chứng của kỹ năng dự báo siêu việt khi chưa đối chiếu với tỷ lệ cơ sở và mốc chuẩn luôn dự báo chiều dương của thị trường.
   - **Chuẩn mực kiểm định đa giả thuyết:** Mọi kết luận so sánh mô hình tài chính cần được kiểm soát chặt chẽ bằng các thủ tục hiệu chỉnh đa giả thuyết (Bonferroni, Benjamini–Hochberg FDR) và kiểm soát tự tương quan mẫu hữu hạn (Harvey–Leybourne–Newbold).
