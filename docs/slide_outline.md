# KỊCH BẢN & CẤU TRÚC 10 SLIDE GIỚI THIỆU ĐỀ TÀI (SLIDE PRESENTATION OUTLINE)

**Học phần:** Phân tích dữ liệu kinh doanh (IS402.R12)  
**Đề tài:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  
*(Dự báo tỷ suất sinh lợi cổ phiếu S&P 500: Khi nào năng lực dự báo suy thoái? Phân tích ngoại mẫu đa chân trời và phụ thuộc chế độ biến động)*  
**Giảng viên hướng dẫn:** ThS. Dương Phi Long  
**Nhóm sinh viên thực hiện:**
- Nguyễn Đoàn Đức Hiếu (Leader) - 24520500
- Nguyễn Thị Phương Hoa - 23520506
- Nguyễn Văn Phát - 24521314
- Trần Anh Thư - 24521732
- Trần Ngọc Thảo - 24521649

---

## 📌 TỔNG QUAN THỨ TỰ TRÌNH BÀY (SLIDE FLOW)

```text
[1. Tiêu đề] 
   └── [2. Động lực & Đặt vấn đề] 
          └── [3. Phát biểu bài toán (Input/Output)] 
                 └── [4. Dữ liệu & Phân chia Walk-Forward] 
                        └── [5. Mốc chuẩn (Benchmark)] 
                               └── [6. Các phương pháp / Mô hình] 
                                      └── [7. Hệ tiêu chí đánh giá] 
                                             └── [8. Ba câu hỏi nghiên cứu (RQ1, RQ2, RQ3)] 
                                                    └── [9. Lộ trình & Đóng góp] 
                                                           └── [10. Q&A & Cảm ơn]
```

---

## 📑 CHI TIẾT NỘI DUNG TỪNG SLIDE

### 🟢 SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề chính:** BÁO CÁO ĐỀ CƯƠNG ĐỒ ÁN MÔN HỌC
* **Tên đề tài:** 
  * Tiếng Anh: *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*
  * Tiếng Việt: *Phân tích sự suy thoái của năng lực dự báo tỷ suất sinh lợi cổ phiếu ngoại mẫu theo đa chân trời thời gian và chế độ biến động thị trường*
* **Thông tin đơn vị:**
  * Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT)
  * Khoa Hệ thống Thông tin
  * Bộ môn Phân tích dữ liệu kinh doanh
* **Thông tin nhóm & GVHD:**
  * Giảng viên hướng dẫn: ThS. Dương Phi Long
  * Nhóm thực hiện: Nhóm 12 — Lớp IS402.R12
* **Ghi chú thuyết trình:** Trình bày ngắn gọn, giới thiệu tên đề tài và các thành viên đại diện báo cáo.

---

### 🟢 SLIDE 2: ĐẶT VẤN ĐỀ & ĐỘNG LỰC NGHIÊN CỨU (MOTIVATION)
* **Bối cảnh lý thuyết:**
  * Giả thuyết Thị trường Hiệu quả (Efficient Market Hypothesis - Fama, 1970): Giá chứng khoán phản ánh toàn bộ thông tin sẵn có, lợi suất có xu hướng vận động ngẫu nhiên (Random Walk) và rất khó dự báo.
  * Sự trỗi dậy của Machine Learning / Deep Learning: Nhiều nghiên cứu khẳng định mô hình học máy có thể khai thác các quy luật phi tuyến tính để dự báo sinh lời.
* **Vấn đề cốt lõi đặt ra:**
  * Liệu năng lực dự báo đó có tồn tại lâu dài không, hay nó sẽ **suy thoái (decay)** khi kỳ hạn dự báo dài ra?
  * Khi thị trường rơi vào các pha biến động cực đoan (khủng hoảng tài chính, đại dịch), mô hình có bị sụp đổ phương sai hay không?
* **Mục tiêu đề tài:** Xây dựng một khung thực nghiệm quy chuẩn, chặt chẽ để kiểm định tính suy thoái của năng lực dự báo lợi suất chỉ số S&P 500 theo các chân trời thời gian và chế độ biến động khác nhau.

---

### 🟢 SLIDE 3: PHÁT BIỂU BÀI TOÁN — INPUT & OUTPUT (PROBLEM FORMULATION)
* **Bản chất bài toán:**
  * **Bài toán dự báo định lượng (Continuous Point Forecasting - Regression):** Dự báo trực tiếp một con số lợi suất thực tế cụ thể (Continuous return), không dừng lại ở việc dự đoán tăng/giảm nhị phân.
  * **Bài toán phụ đối chứng:** Phân loại định hướng (Binary Directional Prediction) để kiểm tra khả năng bắt nhịp chiều thị trường.
* **Thuộc tính dự đoán (OUTPUT / TARGET DUY NHẤT):**
  * Lợi suất logarit tích lũy đa kỳ hạn của chỉ số S&P 500:
    $$Target_{h,t} = \ln\left(\frac{Close_{t+h}}{Close_t}\right)$$
  * Đánh giá đồng thời trên **5 Kỳ hạn dự báo (Forecast Horizons):**
    1. **`1D`** (Daily): Dự báo 1 phiên tiếp theo (Siêu ngắn hạn).
    2. **`5D`** (Weekly): Dự báo 1 tuần tiếp theo (5 phiên giao dịch).
    3. **`20D`** (Monthly): Dự báo 1 tháng tiếp theo (20 phiên giao dịch).
    4. **`60D`** (Quarterly): Dự báo 1 quý tiếp theo (60 phiên giao dịch).
    5. **`120D`** (Semi-annual): Dự báo nửa năm tiếp theo (120 phiên giao dịch).
* **Thuộc tính đầu vào (INPUT FEATURES):**
  * Không gian gồm **hơn 25 đặc trưng định lượng** được tính toán tại cuối ngày $t$ (không chứa thông tin tương lai):
    * Lợi suất quá khứ (Lagged Returns: 1D, 5D, 20D...).
    * Chỉ báo động lượng & xu hướng (Momentum, Moving Averages).
    * Khối lượng giao dịch chuẩn hóa (Volume ratio, Volume anomaly).
    * Độ biến động lịch sử (Rolling Historical Volatility 20D, 60D).
    * Chỉ số tâm lý sợ hãi thị trường CBOE VIX (`VIXCLS`, biến động VIX).

---

### 🟢 SLIDE 4: NGUỒN DỮ LIỆU & PHÂN CHIA WALK-FORWARD (DATASET & SPLITS)
* **Nguồn dữ liệu:**
  * Chỉ số **S&P 500 Index** (`^GSPC`): Đo lường hiệu quả 500 công ty vốn hóa lớn nhất Hoa Kỳ.
  * Chỉ số biến động **CBOE Volatility Index** (`^VIX` / `VIXCLS`): Đo lường kỳ vọng biến động ngắn hạn.
  * **Độ bao phủ thời gian:** **36 năm lịch sử (1990 – 2026)** với hơn **9,000 phiên giao dịch sạch**, bao quát mọi chu kỳ kinh tế lớn (Bong bóng Dot-com 2000, Khủng hoảng Subprime 2008, Covid-19 2020, chu kỳ lạm phát 2022-2024).
* **Quy trình phân chia dữ liệu chống rò rỉ (Strict Chronological Splits):**
  * 🟩 **Split 1 (Initial In-sample Training):** `1990 - 2015` (6,351 phiên) — Dữ liệu nền tảng ban đầu để huấn luyện mô hình.
  * 🟨 **Split 2 (Walk-Forward Validation):** `2015 - 2020` (1,360 phiên) — Chia thành **5 Folds tuần tự** (mỗi fold 272 phiên ~ 1 năm). Áp dụng kỹ thuật **Expanding Window (Cửa sổ mở rộng)** và **Marcos López de Prado Purging ($h-1$ ngày)** để chống rò rỉ dữ liệu gối đầu (Data Leakage) khi tối ưu siêu tham số.
  * 🟥 **Split 3 (Independent Out-of-Sample Test):** `2020 - 2026` (1,360 phiên) — Khóa chặt tuyệt đối, chỉ mở ra một lần duy nhất để kiểm tra khả năng tổng quát hóa cuối cùng.

---

### 🟢 SLIDE 5: ĐỊNH NGHĨA MỐC CHUẨN (BENCHMARK DEFINITION)
* **Tại sao bài toán dự báo tài chính bắt buộc phải có Benchmark?**
  * Trong kinh doanh và tài chính, một mô hình có sai số thấp hay độ chính xác cao chưa chắc đã hữu ích nếu không vượt qua được chiến lược ngây thơ (Naive strategy) của thị trường.
* **Hai mốc chuẩn được thiết lập trong đề tài:**
  1. **Historical Mean Benchmark (Campbell & Thompson, 2008):**
     * Dự báo bằng giá trị trung bình lịch sử mở rộng (Expanding Mean) của lợi suất quá khứ.
     * Mọi mô hình máy học chỉ được công nhận có năng lực dự báo khi và chỉ khi có sai số nhỏ hơn mô hình mốc này ($OOS\_R^2 > 0$).
  2. **Naive Equity Drift (Always-Positive Benchmark):**
     * Thị trường chứng khoán Mỹ có xu hướng tăng trưởng dài hạn (Positive Equity Risk Premium Drift). Ở kỳ hạn 120D, thực tế có tới **$77.79\%$** số ngày thị trường mang lợi suất dương.
     * Một mô hình chỉ đoán mò chiều tăng sẽ tự động đúng $77.79\%$. Do đó, mô hình phải vượt qua ngưỡng này mới được xem là có kỹ năng định thời (timing skill).

---

### 🟢 SLIDE 6: CÁC PHƯƠNG PHÁP & MÔ HÌNH DỰ KIẾN (METHODOLOGY)
* **Tổng thể 14 mô hình được phân thành 5 họ kiến trúc:**
  1. **Nhóm Baseline & Tự hồi quy kinh tế lượng (Econometric Models):**
     * `Historical_Mean`: Mốc chuẩn trung bình mở rộng.
     * `ARIMA`, `SARIMA`: Mô hình tự hồi quy tích hợp trung bình trượt (có thành phần mùa).
     * `ARIMAX`: Tự hồi quy kết hợp biến ngoại sinh (VIX, Volatility).
  2. **Nhóm Tuyến tính điều chuẩn (Regularized Linear Models):**
     * `Ridge Regression`: Điều chuẩn phạt L2 giúp ổn định đa cộng tuyến.
     * `Elastic Net`: Kết hợp đồng thời phạt L1 (chọn đặc trưng) và L2 (chống quá khớp).
  3. **Nhóm Học máy dựa trên cây (Tree-based Ensembles):**
     * `Random Forest`: Học kết hợp dạng đóng bao (Bagging) giảm phương sai.
     * `XGBoost`, `LightGBM`: Tăng cường độ dốc (Gradient Boosting) tối ưu hóa tốc độ và phi tuyến tính.
  4. **Nhóm Support Vector:**
     * `SVR (Support Vector Regression)`: Tối thiểu hóa sai số cấu trúc trong vùng dung sai epsilon.
  5. **Nhóm Học sâu chuỗi thời gian (Deep Learning):**
     * `RNN`, `LSTM`: Mạng nơ-ron hồi quy nắm bắt phụ thuộc dài hạn.
     * `Transformer`: Kiến trúc Self-Attention hiện đại xử lý tương tác chuỗi.
  * *(Mô hình phân loại nhị phân: `Logistic Regression` làm đối chứng định hướng).*

---

### 🟢 SLIDE 7: HỆ TIÊU CHÍ ĐÁNH GIÁ (EVALUATION METRICS & TESTS)

#### 1. THƯỚC ĐO SAI SỐ ĐIỂM (POINT ACCURACY): MAE & RMSE
* **Công thức toán học:**
  $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
* **Tại sao dùng & Dùng để làm gì?**
  * *Tại sao không dùng một chỉ số mà phải dùng cả hai?*
    * **MAE (Mean Absolute Error):** Đo lường biên độ sai số tuyệt đối trung bình theo thang đo gốc (lợi suất). MAE đối xử công bằng với mọi sai số, phản ánh sai số điển hình của một phiên giao dịch bình thường.
    * **RMSE (Root Mean Squared Error):** Do có thao tác bình phương $(y_t - \hat{y}_t)^2$ trước khi lấy căn bậc hai, RMSE phạt cực nặng các sai số lớn (cực trị / Black Swan).
  * *Ý nghĩa trong tài chính:*
    * Nếu một mô hình có MAE rất thấp nhưng RMSE lại vọt lên cao $\rightarrow$ Mô hình thường ngày dự đoán khá sát, nhưng thỉnh thoảng lại có những phiên "đoán lệch một trời một vực". Trong thị trường chứng khoán, một vài phiên đoán sai biên độ lớn có thể thổi bay tài khoản (cháy đòn bẩy). Vì vậy, RMSE thấp là điều kiện tiên quyết để phòng vệ rủi ro đuôi (Tail Risk).

---

#### 2. THƯỚC ĐO KINH TẾ LƯỢNG NGOẠI MẪU: $R^2_{\text{OOS}}$ (Campbell & Thompson, 2008)
* **Công thức toán học:**
  $$R^2_{\text{OOS}} = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N (y_t - \bar{y}_{\text{bench},t})^2}$$
  * $y_t$: Lợi suất thực tế tại thời điểm $t$.
  * $\hat{y}_t$: Giá trị dự báo của mô hình Machine Learning / Kinh tế lượng.
  * $\bar{y}_{\text{bench},t}$: Mốc chuẩn trung bình lịch sử lũy tiến (Historical Mean Benchmark), tính từ tất cả các ngày trong quá khứ trước $t$.
* **Tại sao dùng?**
  * *Tại sao không dùng $R^2$ thông thường trong Machine Learning?*
    * Trong Machine Learning thông thường, $R^2$ đo lường tỷ lệ phương sai giải thích so với trung bình của tập kiểm thử. Tuy nhiên, trong dữ liệu chuỗi thời gian tài chính, tỷ lệ tín hiệu trên nhiễu (Signal-to-Noise Ratio) cực kỳ thấp ($< 1\%$).
    * Nghiên cứu kinh điển của Welch & Goyal (2008) chỉ ra rằng: Hầu hết các mô hình học máy phức tạp khi ra thực tế đều thua một chiến lược ngây thơ (Naïve Strategy) là: *"Đoán lợi suất ngày mai bằng trung bình lịch sử của tất cả các ngày trước đó"*.
    * Do đó, Giáo sư John Campbell và Samuel Thompson (2008, Review of Financial Studies) đã đề xuất chuẩn mực $R^2_{\text{OOS}}$ để bắt buộc mô hình phải so găng trực tiếp với mốc chuẩn Historical Mean.
* **Dùng để làm gì & Diễn giải:**
  * $R^2_{\text{OOS}} > 0$: Mô hình có sai số bình phương nhỏ hơn việc đoán trung bình lịch sử $\rightarrow$ Mô hình thực sự trích xuất được tín hiệu dự báo có giá trị kinh tế.
  * $R^2_{\text{OOS}} \le 0$: Mô hình dự báo tệ hơn hoặc chỉ bằng một người đoán mò theo trung bình quá khứ $\rightarrow$ Không có giá trị thực tiễn.
  * *Quy ước con số trong tài chính:* Khác với các ngành khác đòi hỏi $R^2 > 50\%$, trong tài chính thực nghiệm (Gu, Kelly & Xiu, 2020), một mô hình chỉ cần đạt $R^2_{\text{OOS}} \approx 1\% - 3\%$ ở tần suất tháng đã được coi là hiệu năng vượt trội (economically significant), có thể tạo ra lợi nhuận phi rủi ro khổng lồ cho các quỹ định lượng.

---

#### 3. ĐỘ CHÍNH XÁC PHƯƠNG HƯỚNG (DIRECTIONAL ACCURACY — DA)
* **Công thức toán học:**
  $$\text{DA} = \frac{1}{N} \sum_{t=1}^N \mathbb{I}\left(\text{sign}(y_t) == \text{sign}(\hat{y}_t)\right)$$
  * $\text{sign}(\cdot)$: Hàm lấy dấu (bằng $+1$ nếu lợi suất $> 0$, bằng $-1$ nếu lợi suất $< 0$).
  * $\mathbb{I}(\cdot)$: Hàm chỉ thị (Indicator Function), nhận giá trị $1$ nếu dự đoán đúng chiều (cùng dương hoặc cùng âm), nhận giá trị $0$ nếu đoán ngược chiều.
* **Tại sao dùng & Dùng để làm gì?**
  * *Tại sao dùng?*
    * Một mô hình có thể dự đoán sai số điểm (ví dụ đoán tăng $+0.1\%$ nhưng thực tế tăng $+2.0\%$, sai số tuyệt đối là $1.9\%$), nhưng nó vẫn bắt đúng xu hướng là thị trường TĂNG.
    * Trong giao dịch tài chính (Trading), việc đoán đúng phương hướng (Lên hay Xuống) quyết định bạn mở vị thế Mua (Long) hay Bán khống (Short). Đoán sai phương hướng sẽ dẫn đến lỗ vị thế ngay lập tức.
  * *Dùng để làm gì?*
    * Đánh giá trực tiếp giá trị thực chiến của mô hình: Tỷ lệ phần trăm số ngày mô hình chỉ đúng hướng đi của chỉ số S&P 500.
    * Mốc đối chiếu tối thiểu là $50\%$ (xác suất tung đồng xu ngẫu nhiên). Mô hình phải vượt qua $50\%$ thì mới có kỹ năng phân định hướng.

---

#### 4. KIỂM ĐỊNH THỐNG KÊ SUY DIỄN CHÍNH QUY (INFERENTIAL TESTS)
* **Tại sao phải dùng Kiểm định thống kê mà không dừng lại ở các con số trên?**
  * *Điểm mấu chốt để trả lời hội đồng:* Các chỉ số MAE, RMSE, $R^2_{\text{OOS}}$, DA ở trên chỉ là ước lượng điểm (Point Estimates) trên một mẫu dữ liệu cụ thể. Nếu $R^2_{\text{OOS}} = +0.015$ hay $\text{DA} = 53\%$, làm sao chúng ta biết con số đó là do mô hình thực sự giỏi, hay chỉ là do may mắn ngẫu nhiên (Luck / Noise)? Kiểm định suy diễn thống kê sinh ra để trả lời dứt khoát câu hỏi này.

* **4.1 Diebold-Mariano Test (DM Test — Hiệu chỉnh HLN 1997):**
  * *Bản chất:* Kiểm định xem vi sai hàm mất mát giữa mốc chuẩn và mô hình:
    $$d_t = e_{\text{bench},t}^2 - e_{\text{model},t}^2$$
    có thực sự lớn hơn $0$ một cách có ý nghĩa thống kê hay không ($H_0: \mathbb{E}[d_t] \le 0$ so với $H_1: \mathbb{E}[d_t] > 0$).
  * *Hiệu chỉnh Harvey-Leybourne-Newbold (1997):*
    * Khi dự báo đa kỳ ($h > 1$ ngày, ví dụ 5D, 20D, 60D, 120D), chuỗi sai số dự báo bị tự tương quan (Autocorrelation) do nhãn lợi suất bị gối đầu lên nhau.
    * Công thức DM nguyên bản sẽ bị phồng sai số. Chuẩn mực HLN (1997) hiệu chỉnh bậc tự do mẫu hữu hạn và tính toán ma trận hiệp phương sai trễ $h-1$, đảm bảo giá trị $p$-value hoàn toàn không bị ảo.
  * *Dùng để làm gì:* Khẳng định xem việc mô hình vượt qua Historical Mean ở mức $p < 0.05$ (hoặc $p < 0.10$) là có ý nghĩa thống kê thực sự, không phải do may rủi.

* **4.2 Binomial Sign Test (Kiểm định Nhị thức):**
  * *Bản chất:* Xem mỗi phiên dự đoán đúng chiều ($\text{sign}(y_t) == \text{sign}(\hat{y}_t)$) như một phép thử Bernoulli (thành công / thất bại) với xác suất thành công ngẫu nhiên $p_0 = 0.5$.
  * *Cặp giả thuyết:*
    $$H_0: \text{DA} \le 0.5 \quad \text{vs} \quad H_1: \text{DA} > 0.5$$
  * *Dùng để làm gì:* Chứng minh rằng tỷ lệ đoán đúng chiều của mô hình vượt trội hơn hẳn một người tung đồng xu ngẫu nhiên ở mức ý nghĩa $95\%$ ($p_{\text{Sign}} < 0.05$).

---

### 🟢 SLIDE 8: BA CÂU HỎI NGHIÊN CỨU THEN CHỐT (RESEARCH QUESTIONS — RQ)
* **Nghiên cứu tập trung giải quyết 3 bài toán phân tích thực nghiệm:**
  * 🎯 **RQ1 (Chân trời dự báo - Forecast Horizon):**
    * *Câu hỏi:* Năng lực dự báo biến đổi như thế nào khi mở rộng kỳ hạn từ $1D \to 5D \to 20D \to 60D \to 120D$? Quy luật suy thoái (Predictability Decay) có diễn ra đồng nhất giữa các họ mô hình không?
  * 🎯 **RQ2 (Chế độ biến động thị trường - Volatility Regime):**
    * *Câu hỏi:* Năng lực dự báo bị điều tiết thế nào qua **3 chế độ biến động**:
      1. `Low Volatility`: Thị trường tăng trưởng ổn định (VIX thấp).
      2. `Normal Volatility`: Thị trường dao động bình thường.
      3. `High Volatility`: Thị trường hoảng loạn / khủng hoảng cực đoan (VIX cao).
  * 🎯 **RQ3 (Tính ổn định thời gian - Time Stability):**
    * *Câu hỏi:* Mẫu hình suy thoái có duy trì tính ổn định qua các phân kỳ lịch sử (2015–2016, 2017–2018, 2019–2020) hay chỉ bị méo mó bởi một sự kiện cá biệt (như đại dịch COVID-19 năm 2020)?

---

### 🟢 SLIDE 9: LỘ TRÌNH THỰC HIỆN & ĐÓNG GÓP KỲ VỌNG (ROADMAP & IMPACT)
* **Quy trình triển khai 4 giai đoạn:**
  1. *Giai đoạn 1:* Thu thập, đồng bộ dữ liệu và kỹ thuật đặc trưng (Feature Engineering).
  2. *Giai đoạn 2:* Tinh chỉnh siêu tham số khép kín (Optuna / TimeSeriesSplit) & Huấn luyện lồng Walk-Forward.
  3. *Giai đoạn 3:* Tính toán sai số ngoại mẫu, kiểm định Diebold-Mariano và phân rã Regime.
  4. *Giai đoạn 4:* Trực quan hóa kết quả kinh doanh và tổng hợp báo cáo khuyến nghị.
* **Đóng góp học thuật & Giá trị kinh doanh:**
  * Làm sáng tỏ giới hạn của trí tuệ nhân tạo (AI/ML) trong bài toán định giá tài sản và tài chính hành vi.
  * Cung cấp khuyến nghị thực tế cho các nhà quản lý quỹ đầu tư định lượng (Quant Funds) trong việc lựa chọn mô hình và chân trời tái cân bằng danh mục theo từng chế độ thị trường.

---

### 🟢 SLIDE 10: TỔNG KẾT & HỎI ĐÁP (CONCLUSION & Q&A)
* **Tóm lược:** Đề tài kết hợp giữa lý thuyết kinh tế lượng tài chính chuẩn mực và các thuật toán học máy tiên tiến, với thiết kế thực nghiệm chặt chẽ chống rò rỉ dữ liệu.
* **Lời cảm ơn:** Trân trọng cảm ơn Thầy và các bạn đã lắng nghe.
* **Mở phiên trao đổi:** Sẵn sàng tiếp nhận các câu hỏi góp ý từ Hội đồng/Giảng viên.
