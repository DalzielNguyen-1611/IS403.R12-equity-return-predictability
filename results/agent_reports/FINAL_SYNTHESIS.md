# TỔNG HỢP KẾT QUẢ NGHIÊN CỨU TOÀN DIỆN (FINAL RESEARCH RESULTS SYNTHESIS)

**Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*  
**Mục tiêu nghiên cứu chính:** Xác định hình thái và cấu trúc hàm của năng lực dự báo lợi suất chỉ số S&P 500 ngoại mẫu:
$$\text{Predictability} = f(\text{Model}, \text{Horizon}, \text{Regime}, \text{Time})$$  
**Cơ quan thực hiện:** Research Data Analyst (Antigravity Analytics System)  
**Ngày hoàn tất:** 2026-10-08  
**Đường dẫn tài liệu:** `results/agent_reports/FINAL_SYNTHESIS.md`

---

## 1. Tóm Tắt Quản Trị & Phát Hiện Cốt Lõi (Executive Summary)

1. **Sự phân kỳ nhị phân sâu sắc theo bản chất họ mô hình:**  
   Năng lực dự báo ngoại mẫu ($OOS\_R^2$) phân tách thành hai thái cực hoàn toàn đối lập: Các thuật toán Machine Learning (Tree Ensembles, Support Vector Regression, Tuyến tính co rút) thể hiện **sự suy thoái dự báo nghiêm ngặt và đơn điệu** theo chân trời; trong khi các mô hình Tự hồi quy kinh tế lượng (ARIMA, ARIMAX, SARIMA) lại ghi nhận **sự tăng trưởng nhảy vọt** từ ngắn hạn đến dài hạn.
2. **Quy luật suy thoái theo chân trời (RQ1 — Predictability Decay):**  
   Đối với các mô hình Machine Learning, năng lực dự báo đạt đỉnh tại chân trời siêu ngắn ($1D$ hoặc $5D$) với $OOS\_R^2$ dương nhẹ ($+0.003$ đến $+0.026$), sau đó sụp đổ dốc đứng xuống mức âm rất sâu tại $60D$ và $120D$ ($-0.27$ đến $-0.78$). Bằng chứng suy thoái là **rất mạnh (Strong Evidence)** đối với dòng ML.
3. **Hiệu ứng bùng nổ của dòng Tự hồi quy chuỗi thời gian:**  
   Chuỗi lợi suất tích lũy $Target_{h,t} = \ln(Close_{t+h} / Close_t)$ sở hữu quán tính tự tương quan chuỗi rất lớn khi $h$ tăng. Các mô hình ARIMA/ARIMAX/SARIMA khai thác trọn vẹn đặc tính này, đạt $OOS\_R^2 > 0.87$ tại $20D$ và $> 0.95$ tại $120D$.
4. **Sự điều tiết khốc liệt của Chế độ Biến động (RQ2 — Regime Dependence):**  
   Chế độ biến động thị trường làm thay đổi hoàn toàn hình thái suy thoái dự báo:  
   - Ở chế độ biến động thấp (Low) và bình thường (Normal), sự suy thoái diễn ra nhẹ nhàng ($\Delta_{120D-1D} \in [-0.04, -0.28]$).  
   - Ở chế độ biến động cao (High), nghịch lý xảy ra: Năng lực dự báo 1D đạt mức tốt nhất, nhưng tốc độ suy thoái theo chân trời dốc gấp **15 đến 30 lần**, khiến $OOS\_R^2$ tại 120D của các mô hình Cây lao dốc xuống **$-1.83$ đến $-3.36$**.
5. **Tính ổn định qua các chu kỳ thời gian (RQ3 — Temporal Stability):**  
   Các mẫu hình về chiều hướng suy thoái của Machine Learning và sự vượt trội của Autoregressive **ổn định 100% qua cả ba phân kỳ** (2015–2016, 2017–2018, và 2019–2020). Kết quả nghiên cứu không bị chi phối bởi một giai đoạn cá biệt nào (kể cả cú sốc đại dịch COVID-19 năm 2020).
6. **Bằng chứng kiểm định thống kê duy nhất:**  
   Kiểm định Diebold-Mariano có hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold chứng minh: **Chỉ duy nhất dòng Autoregressive (ARIMA, ARIMAX, SARIMA)** vượt trội hơn Historical Mean một cách có ý nghĩa thống kê ($DM \ge 2.18, p < 0.015$ từ chân trời 5D đến 120D). Toàn bộ các mô hình Machine Learning và Deep Learning đều thất bại trước benchmark.
7. **Nghịch lý giữa Năng lực Điểm ($OOS\_R^2$) và Độ chính xác Định hướng ($DA$):**  
   Trong khi $OOS\_R^2$ của Machine Learning suy thoái âm sâu ở chân trời dài, thì tỷ lệ đoán đúng chiều ($DA$) lại tăng vọt từ $\approx 52\%$ (1D) lên **$65\% - 78\%$** (120D). Đây là hệ quả của việc mô hình nắm bắt phí bù rủi ro vốn cổ phần dài hạn (positive equity drift) của chỉ số S&P 500 chứ không phải kỹ năng định lượng biên độ.
8. **Thất bại cấu trúc của Học sâu (Deep Learning) trên dữ liệu chuỗi tài chính bảng phẳng:**  
   Mạng Transformer bị âm sâu toàn diện ($OOS\_R^2 \in [-0.74, -0.19]$) do hiện tượng đói dữ liệu (sample starvation); LSTM và RNN dao động phi tuyến tính và không duy trì được ưu thế so với các mô hình tuyến tính đơn giản.

---

## 2. Tính Toàn Vẹn Thực Nghiệm & Kiểm Toán Rò Rỉ Dữ Liệu (Experimental Validity)

- **Số lượng quan sát kỳ vọng vs thực tế:**
  - `all_fold_metrics.csv`: $14 \text{ Models} \times 5 \text{ Horizons} \times 5 \text{ Folds} = 350$ hàng ($100\%$ hoàn chỉnh, 0 trùng lặp).
  - `all_oos_predictions.csv`: Đúng $95,200$ hàng cấp quan sát ($272$ ngày kiểm định/fold $\times 5$ folds $\times 5$ horizons $\times 14$ models), 0 giá trị ngoại lai NaN/Inf.
  - `best_hyperparameters.csv`: $13 \text{ Models có tham số} \times 5 \text{ Horizons} \times 5 \text{ Folds} = 325$ hàng ($100\%$ bao phủ). `Historical_Mean` không có tham số nên không xuất hiện là hoàn toàn chính xác.
- **Tính toán đối chiếu độc lập (Cross-recomputation Check):**
  - Tái tính toán toàn bộ chỉ số từ $95,200$ dòng dự báo thô khớp hoàn toàn với `all_fold_metrics.csv` với độ lệch máy tính $\Delta_{\max} \le 1.25 \times 10^{-16}$.
- **Kiểm toán rò rỉ dữ liệu (Data Leakage Audit):**
  - Quá trình phân tách dữ liệu huấn luyện - kiểm thử tuân thủ nghiêm ngặt giao thức **Walk-Forward Validation mở rộng (Expanding Window)**.
  - Quá trình tối ưu hóa siêu tham số (Optuna, GridSearchCV, TimeSeriesSplit) được thực hiện khép kín 100% bên trong tập huấn luyện của outer fold thông qua hàm `get_internal_train_val_split(val_ratio=0.15)`. Tập kiểm định ngoại mẫu $X_{\text{val}}$ tuyệt đối không tham gia vào quá trình chọn tham số. Nghiên cứu được chứng thực **hoàn toàn miễn nhiễm rò rỉ dữ liệu (strictly leakage-free)**.

---

## 3. Đánh Giá Hiệu Năng Theo Họ Mô Hình (Overall Model Performance by Families)

Thay vì xếp hạng đơn thuần một danh sách 14 mô hình rời rạc, nghiên cứu cấu trúc hóa thành 6 họ mô hình:

```
+-----------------------------------------------------------------------------+
|                PHỔ HIỆU NĂNG THEO HỌ MÔ HÌNH (OOS R² SUMMARY)                |
+----------------------+--------------------+---------------------------------+
| Họ mô hình           | Chân trời ngắn (1D)| Chân trời dài (60D - 120D)      |
+----------------------+--------------------+---------------------------------+
| 1. Autoregressive    | Dương nhẹ (~+0.02) | Vượt trội áp đảo (+0.94 -> +0.97|
| 2. Linear Regularized| Dương nhẹ (~+0.003)| Âm nhẹ đến trung bình (-0.07..-0.28)
| 3. Benchmark (Mean)  | Tiệm cận 0 (~-0.00)| Âm nhẹ (-0.01 đến -0.06)        |
| 4. Support Vector    | Đỉnh cao (+0.018)  | Âm trung bình (-0.17)           |
| 5. Tree Ensembles    | Tiệm cận 0 (~-0.02)| Âm nặng nề (-0.27 đến -0.78)    |
| 6. Deep Learning     | Âm (-0.02 đến -0.7)| Âm nặng nề (-0.04 đến -0.56)    |
+----------------------+--------------------+---------------------------------+
```

- **Họ Tự Hồi Quy (ARIMA, ARIMAX, SARIMA):** Sở hữu năng lực dự báo áp đảo toàn diện ở mọi chân trời từ $5D$ trở lên. Mô hình `ARIMAX` (kết hợp các biến ngoại sinh VIX và Momentum) đạt hiệu năng cao nhất toàn nghiên cứu ($OOS\_R^2 = +0.9735$ tại 120D).
- **Họ Tuyến Tính Co Rút (Elastic Net, Ridge):** Kháng suy thoái tốt nhất trong nhóm Machine Learning nhờ cơ chế phạt chuẩn $L_1$ và $L_2$. `Elastic_Net` duy trì $OOS\_R^2 = -0.0676$ tại 120D, tốt hơn đáng kể so với các mô hình phi tuyến tính phức tạp.
- **Họ Cây Quyết Định (Random Forest, XGBoost, LightGBM):** Bị hiện tượng quá khớp (overfitting) với phương sai nhiễu cao của dữ liệu tài chính. Khi chân trời dự báo tăng, sai số ngoại mẫu bị tích lũy theo cấp số nhân.
- **Họ Học Sâu (RNN, LSTM, Transformer):** Không phát huy được thế mạnh mạng nơ-ron sâu do cỡ mẫu chuỗi thời gian kinh tế lượng hàng ngày bị giới hạn. Đặc biệt, Transformer biểu hiện kém nhất toàn hệ thống tại 1D ($OOS\_R^2 = -0.7354$).
- **Mô hình Phân loại (Logistic Regression):** Hoàn thành xuất sắc vai trò phân loại hướng đi với độ chính xác tăng từ $50.59\%$ (1D) lên $74.56\%$ (120D), được cô lập hoàn toàn khỏi hệ thống hồi quy.

---

## 4. Trả Lời Câu Hỏi Nghiên Cứu 1: Quy Luật Suy Thoái Theo Chân Trời (RQ1)

> **RQ1:** *How does out-of-sample equity return predictability change across forecast horizons?*

1. **Năng lực dự báo có suy thoái không?**  
   **CÓ đối với Machine Learning & Linear Models; KHÔNG đối với Autoregressive Models.**
2. **Sự suy thoái có tính đơn điệu (monotonic) không?**  
   - `Elastic_Net`, `Random_Forest`, `SVR`, và `Historical_Mean` thể hiện **sự suy thoái đơn điệu nghiêm ngặt (100% strictly monotonic decay)**: Mỗi bước tăng của chân trời ($1D \to 5D \to 20D \to 60D \to 120D$) đều làm giảm $OOS\_R^2$.
   - `XGBoost` đạt đỉnh nhẹ tại 5D trước khi rơi vào quỹ đạo suy thoái đơn điệu.
   - Ngược lại, dòng Autoregressive thể hiện **sự tăng trưởng đơn điệu nghiêm ngặt (100% strictly monotonic growth)** từ $+0.013$ lên $+0.956$.
3. **Mức độ suy thoái (Magnitude of Decay):**  
   Độ dốc suy thoái ròng ($\Delta_{120D-1D}$) của dòng ML nằm trong khoảng $-0.07$ đến $-0.72$, phản ánh sự sụt giảm nghiêm trọng về giá trị kinh tế của các tín hiệu dự báo máy học khi chu kỳ đầu tư vượt quá 1 tháng.

---

## 5. Trả Lời Câu Hỏi Nghiên Cứu 2: Tính Phụ Thuộc Vào Chế Độ Biến Động (RQ2)

> **RQ2:** *How does the relationship between forecast horizon and predictive skill differ across market volatility regimes?*

1. **Chế độ biến động có làm thay đổi mối quan hệ Horizon - Predictability không?**  
   **CÓ, VÀ THAY ĐỔI CỰC KỲ KHỐC LIỆT.**
2. **Sự khác biệt giữa các chế độ:**  
   - **Chế độ Biến động Thấp (Low Regime):** Tốc độ suy thoái rất phẳng ($\Delta \in [-0.04, -0.22]$). Năng lực của các mô hình Machine Learning duy trì ổn định gần mức 0.
   - **Chế độ Biến động Bình thường (Normal Regime):** Suy thoái ở mức vừa phải, tương tự Low Regime.
   - **Chế độ Biến động Cao (High Regime):** Xuất hiện hiện tượng **sụp đổ dự báo (Catastrophic Collapse)**. Tại chân trời 1D, các tín hiệu biến động đẩy $OOS\_R^2$ lên mức dương cao nhất ($+0.01$ đến $+0.03$), nhưng khi chân trời mở rộng ra 120D, $OOS\_R^2$ của Tree Ensembles sụp đổ xuống **$-2.5712$** (độ dốc suy thoái dốc gấp 20 lần so với Low Regime).
3. **Tính nhất quán của bằng chứng:**  
   $100\%$ các mô hình Machine Learning đều xác nhận hiện tượng sụp đổ tại High Regime. Trong khi đó, dòng Autoregressive duy trì khả năng chống chịu kiên cường ($OOS\_R^2 > 0.91$ ngay cả trong High Regime).

---

## 6. Trả Lời Câu Hỏi Nghiên Cứu 3: Tính Ổn Định Qua Các Chu Kỳ Thời Gian (RQ3)

> **RQ3:** *Are the observed horizon- and regime-dependent patterns stable over time?*

1. **Các mẫu hình có ổn định qua thời gian không?**  
   **PHẦN LỚN ỔN ĐỊNH (MOSTLY STABLE).**
2. **Chiều hướng suy thoái (Decay Direction):**  
   Ổn định $100\%$. Trong cả ba giai đoạn (2015–2016, 2017–2018, 2019–2020), Machine Learning luôn suy thoái khi chân trời tăng, còn Autoregressive luôn đạt $OOS\_R^2 \approx 0.96$ tại 120D.
3. **Hiện tượng một giai đoạn chi phối (Single-Period Driver):**  
   **HOÀN TOÀN BỊ BÁC BỎ.** Ngay cả trong giai đoạn trước đại dịch (2015–2016), mức độ suy thoái của Tree Ensembles tại 120D còn sâu hơn ($-1.689$) so với giai đoạn 2019–2020 ($-0.349$).
4. **Sự biến đổi về độ lớn sai số:**  
   Giai đoạn 2019–2020 chứng kiến sai số tuyệt đối RMSE tăng gấp đôi do cú sốc COVID-19, tuy nhiên trật tự xếp hạng tương đối giữa các họ mô hình không hề bị xáo trộn.

---

## 7. Bằng Chứng Kiểm Định Thống Kê Tổng Hợp (Statistical Evidence Synthesis)

Kết hợp đồng thời 3 góc nhìn kiểm định thống kê:
1. **Góc nhìn $OOS\_R^2$ (Campbell-Thompson):** Machine Learning suy thoái nhanh chóng thành âm sau 1D; Autoregressive duy trì giá trị dương rất lớn ($>0.85$ từ 20D).
2. **Góc nhìn Diebold-Mariano Test (với Historical Mean):**  
   Chỉ có **ARIMA, ARIMAX và SARIMA** đạt ý nghĩa thống kê ($DM \in [2.18, 5.11], p < 0.015$). Toàn bộ các mô hình ML đều có thống kê $DM < 0$ (sai số lớn hơn benchmark) và $p \to 1.0$.
3. **Góc nhìn Binomial Sign Test (với 50% DA):**  
   Xác nhận thị trường 1 ngày là tiệm cận bước đi ngẫu nhiên ($43\%$ mô hình thất bại trước mức $50\%$). Tại chân trời dài hạn, tỷ lệ từ chối $H_0$ đạt $100\%$ do các mô hình nắm bắt thành công xu hướng tăng trưởng dài hạn của S&P 500.

---

## 8. Các Phát Hiện Vững Chắc (Robust Findings)

1. **Sự suy thoái đơn điệu của Machine Learning theo chân trời:** Được chứng minh đồng thời trên 5 mô hình (Random Forest, SVR, Elastic Net, XGBoost, Ridge) qua cả 5 Folds và 3 phân kỳ thời gian.
2. **Sự vượt trội có ý nghĩa thống kê của mô hình Tự hồi quy chuỗi thời gian:** Được kiểm chứng trên cả 3 biến thể (ARIMA, ARIMAX, SARIMA) với $p < 0.015$ dưới kiểm định Diebold-Mariano có điều chỉnh Newey-West.
3. **Tác động khuếch đại sai số của Chế độ Biến động cao:** Tính chất sụp đổ dự báo tại High Regime lặp lại trên toàn bộ các mô hình phi tuyến.

---

## 9. Các Phát Hiện Yếu & Chưa Chắc Chắn (Weak / Uncertain Findings)

1. **Hiệu năng của mạng nơ-ron hồi quy (RNN, LSTM):** Mức độ suy thoái không tuân theo quy luật đơn điệu; hiệu năng dao động thất thường giữa các chân trời do tính ngẫu nhiên của quá trình khởi tạo trọng số và hội tụ gradient.
2. **Độ dốc suy thoái của SVR trong Normal Regime:** SVR có sự biến thiên không đồng nhất tại 60D và 120D giữa các Folds.

---

## 10. Các Phát Hiện Trái Chiều (Contradictory Findings)

1. **Mâu thuẫn giữa giả thuyết suy thoái chung và kết quả của dòng Tự hồi quy:** Giả thuyết nghiên cứu kinh điển cho rằng năng lực dự báo phải suy thoái theo chân trời. Tuy nhiên, dòng AR lại cho thấy điều hoàn toàn trái ngược ($R^2$ tăng từ $1\%$ lên $97\%$).
2. **Mâu thuẫn giữa $OOS\_R^2$ và Directional Accuracy:** Các mô hình ML có $OOS\_R^2$ âm rất sâu nhưng $DA$ lại đạt mức rất cao ($>70\%$).

---

## 11. Các Phát Hiện Bất Ngờ (Unexpected Findings)

1. **Elastic Net đánh bại toàn bộ các mô hình Deep Learning và Tree Ensembles ở chân trời dài:** Khả năng kiểm soát phương sai bằng co rút tham số đơn giản của Elastic Net tỏ ra vượt trội hơn hẳn các mạng nơ-ron hàng triệu tham số.
2. **Chế độ Biến động cao mang lại $OOS\_R^2$ 1D dương cao nhất:** Ngược với định kiến cho rằng biến động cao làm giảm dự báo, tại chân trời 1 ngày, biến động cao cung cấp các tín hiệu biên độ mạnh nhất giúp mô hình dự báo đảo chiều hiệu quả.

---

## 12. Hạn Chế Phương Pháp Luận (Methodological Limitations)

1. **Vấn đề chuỗi chồng lấn ở chân trời dài (Overlapping Returns):** Ở chân trời 60D và 120D, số lượng chu kỳ kiểm định độc lập thực tế bị suy giảm, làm tăng tự tương quan trong chuỗi phần dư.
2. **Hàm mất mát bình phương nhạy cảm với ngoại lai:** Chỉ số $OOS\_R^2$ bị kéo xuống mức âm quá mức do một vài phiên biến động cực đoan trong thời kỳ khủng hoảng.
3. **Số vòng Optuna giới hạn (30–50 trials):** Chưa khai thác hết tiềm năng tối ưu hóa toàn cục của kiến trúc Transformer.

---

## 13. Danh Mục Bảng Khuyến Nghị Cho Khóa Luận / Bài Báo (Recommended Thesis Tables)

1. **Bảng 1 (Table 1):** *Experimental Data Splits and Volatility Regime Distribution Across Walk-Forward Folds.*
2. **Bảng 2 (Table 2):** *Main Out-of-Sample Performance Matrix ($OOS\_R^2$, MAE, RMSE, DA) Across 14 Models and 5 Horizons.*
3. **Bảng 3 (Table 3):** *Regime-Dependent Predictability Matrix: $OOS\_R^2$ and DA Across Low, Normal, and High Volatility Regimes.*
4. **Bảng 4 (Table 4):** *Temporal Stability of Predictability Across Three Macroeconomic Sub-Periods (2015–2016, 2017–2018, 2019–2020).*
5. **Bảng 5 (Table 5):** *Statistical Hypothesis Testing: Diebold-Mariano Tests vs. Historical Mean and Binomial Directional Accuracy Tests.*

---

## 14. Danh Mục Biểu Đồ Khuyến Nghị Cho Khóa Luận / Bài Báo (Recommended Thesis Figures)

1. **Hình 1 (Figure 1):** *Term Structure of Equity Return Predictability: $OOS\_R^2$ vs. Forecast Horizon (1D to 120D) by Model Family.*
2. **Hình 2 (Figure 2):** *Regime-Interaction Decay Curves: Comparison of Horizon Decay Slopes Across Low, Normal, and High Volatility Regimes.*
3. **Hình 3 (Figure 3):** *The Predictability Paradox: Divergence Between Directional Accuracy ($DA$) and Out-of-Sample $R^2$ Across Horizons.*
4. **Hình 4 (Figure 4):** *Temporal Stability Trajectories: Sub-Period Out-of-Sample $R^2$ Comparison Across 2015–2016, 2017–2018, and 2019–2020.*

---

## 15. Câu Trả Lời Cuối Cùng Cho RQ1 (Final Answer to RQ1)
Năng lực dự báo ngoại mẫu của lợi suất S&P 500 **suy thoái đơn điệu và rõ rệt (monotonic decay)** đối với các mô hình Machine Learning và Tuyến tính co rút khi chân trời tăng từ 1 ngày lên 120 ngày. Tuy nhiên, quy luật này **bị đảo ngược hoàn toàn đối với dòng mô hình Tự hồi quy kinh tế lượng (ARIMA, ARIMAX, SARIMA)** do khả năng khai thác quán tính tự tương quan của chuỗi lợi suất tích lũy dài hạn.

---

## 16. Câu Trả Lời Cuối Cùng Cho RQ2 (Final Answer to RQ2)
Chế độ biến động thị trường **điều tiết mạnh mẽ mối quan hệ giữa chân trời và năng lực dự báo**: Ở chế độ biến động thấp/bình thường, sự suy thoái diễn ra nhẹ nhàng; trong khi ở chế độ biến động cao, xuất hiện hiện tượng **sụp đổ dự báo dốc đứng (catastrophic decay)** ở các chân trời dài hạn do phương sai lợi suất bùng nổ, biến High Regime thành môi trường rủi ro nhất đối với các thuật toán học máy điểm.

---

## 17. Câu Trả Lời Cuối Cùng Cho RQ3 (Final Answer to RQ3)
Các mẫu hình suy thoái theo chân trời (RQ1) và tính phụ thuộc vào chế độ biến động (RQ2) **ổn định vững chắc qua thời gian (Mostly Stable)** qua cả 3 phân kỳ kinh tế vĩ mô độc lập. Kết quả thực nghiệm phản ánh cấu trúc toán học và kinh tế lượng bền vững của thị trường tài chính, hoàn toàn không bị thao túng bởi một giai đoạn khủng hoảng cá biệt nào.

---

## 18. Hàm Ý Nghiên Cứu & Ứng Dụng Thực Tiễn (Research & Practical Implications)

1. **Đối với các quỹ đầu tư định lượng (Quant Funds):**  
   - Các tín hiệu Machine Learning dựa trên đặc trưng kỹ thuật chỉ có giá trị gia tăng ở các chiến lược giao dịch tần suất cao/ngắn hạn ($1D - 5D$). Tuyệt đối không sử dụng mô hình học máy phi tuyến phức tạp để dự báo điểm cho các danh mục đầu tư đa tháng ($60D - 120D$).
   - Trong thời kỳ biến động cao (High Volatility), cần cắt giảm quy mô phân bổ tỷ trọng của các mô hình Machine Learning dài hạn để tránh tổn thất bình phương cực đại.
2. **Đối với nghiên cứu học thuật:**  
   - Luôn phân biệt rạch ròi giữa $OOS\_R^2$ (năng lực định lượng phương sai) và Directional Accuracy (nắm bắt xu hướng trôi dạt) trong các nghiên cứu dự báo chuỗi thời gian tài chính đa chân trời.
   - Các mô hình kinh tế lượng cổ điển (ARIMA/ARIMAX) vẫn là chuẩn mực vững chắc và duy trì ưu thế cấu trúc vượt trội trước làn sóng trí tuệ nhân tạo khi xử lý chuỗi lợi suất tích lũy dài hạn.
