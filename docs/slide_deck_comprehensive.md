# BỘ SLIDE THUYẾT TRÌNH BÁO CÁO TOÀN DIỆN (BẢN CHUẨN BẢO VỆ ĐỒ ÁN)
## ĐỀ TÀI: HOW DOES EQUITY RETURN PREDICTABILITY VARY ACROSS HORIZONS AND MARKET REGIMES? AN OUT-OF-SAMPLE ANALYSIS
### Phân tích thực nghiệm khả năng dự báo lợi suất chỉ số S&P 500 theo các chân trời thời gian, chế độ biến động và tính ổn định thời gian (1990 – 2026)

---

# MỤC LỤC TỔNG THỂ CÁC PHẦN BÁO CÁO

* **PHẦN I: TỔNG QUAN ĐỀ TÀI & BỐI CẢNH NGHIÊN CỨU** (Slide 1 – 5)
* **PHẦN II: DỮ LIỆU, CHÂN TRỜI DỰ BÁO & KỸ NGHỆ ĐẶC TRƯNG** (Slide 6 – 11)
* **PHẦN III: PHƯƠNG PHÁP LUẬN, THIẾT KẾ & BỘ ĐỆM LOOKBACK** (Slide 12 – 14)
* **PHẦN IV: CHI TIẾT TẤT CẢ CÁC PHƯƠNG PHÁP & THÔNG SỐ KỸ THUẬT (THEO LỜI DẶN CỦA THẦY)** (Slide 15 – 27)
* **PHẦN V: KHUNG ĐÁNH GIÁ & KIỂM ĐỊNH THỐNG KÊ NGHIÊM NGẶT** (Slide 28 – 30)
* **PHẦN VI: KẾT QUẢ THỰC NGHIỆM CHÍNH (MAIN FINDINGS)** (Slide 31 – 36)
* **PHẦN VII: CÁC PHÂN TÍCH ĐỘ MẠNH MẼ CHUYÊN SÂU (ROBUSTNESS ANALYSES)** (Slide 37 – 41)
* **PHẦN VIII: Ý NGHĨA THỰC TIỄN, ĐÓNG GÓP, KẾT LUẬN & HỎI ĐÁP TOÀN BỘ CÁC MÔ HÌNH** (Slide 42 – 45)

---

# PHẦN I: TỔNG QUAN ĐỀ TÀI & BỐI CẢNH NGHIÊN CỨU

---

## SLIDE 1: TIÊU ĐỀ BÁO CÁO (TITLE SLIDE)

### Tiêu đề Đề tài:
**How Does Equity Return Predictability Vary Across Horizons and Market Regimes? An Out-of-Sample Analysis**  
*(Khả năng dự báo lợi suất chứng khoán biến thiên như thế nào theo các chân trời thời gian và chế độ thị trường? Một phân tích ngoài mẫu)*

* **Mã môn học / Đồ án:** IS403 – Phân tích Dữ liệu Nâng cao & Ứng dụng Tài chính
* **Nhóm thực hiện:** Nhóm 6 / IS403.R12
* **Dữ liệu nghiên cứu:** Chỉ số S&P 500 (^GSPC) & Chỉ số biến động CBOE VIX (1990 – 2026)
* **Quy mô mô hình:** 12 mô hình (Kinh tế lượng, Machine Learning, Deep Sequence Models) $\times$ 5 chân trời $\times$ 3 chế độ biến động

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kính thưa Thầy và các bạn, hôm nay nhóm 6 xin đại diện báo cáo kết quả nghiên cứu toàn diện của đề tài: 'Khả năng dự báo lợi suất chứng khoán biến thiên như thế nào theo chân trời thời gian và chế độ thị trường? Một phân tích ngoài mẫu thực nghiệm trên chỉ số S&P 500 giai đoạn 1990 – 2026'. Báo cáo sẽ trình bày mạch lạc từ định nghĩa bài toán, dữ liệu, chi tiết kỹ thuật của toàn bộ 12 mô hình, kiểm định thống kê và rút ra các kết luận định lượng xác đáng."*

---

## SLIDE 2: TÓM TẮT ĐIỀU HÀNH DỰ ÁN (EXECUTIVE SUMMARY)

### Bức tranh tổng thể của dự án:
1. **Phạm vi nghiên cứu lớn:**
   * Dữ liệu chuỗi thời gian hàng ngày từ tháng 01/1990 đến tháng 09/2026 (9.255 ngày giao dịch).
   * 12 mô hình đại diện từ cổ điển (ARIMA, Ridge, Lasso) đến Machine Learning (XGBoost, LightGBM, Random Forest, SVR) và Deep Learning (MLP, LSTM, TCN, PatchTST, TSMixer).
2. **Ba câu hỏi nghiên cứu trung tâm:**
   * **RQ1 (Horizon):** Khả năng dự báo có tăng dần theo chân trời thời gian (như lý thuyết truyền thống) hay không?
   * **RQ2 (Volatility Regime):** Các mô hình hoạt động hiệu quả nhất trong chế độ biến động thấp, bình thường hay khủng hoảng?
   * **RQ3 (Temporal Stability):** Lợi thế dự báo có bền vững qua các chu kỳ thời gian thực tế trong mẫu Holdout (2020–2026) không?
3. **Phát hiện cốt lõi (Key Findings):**
   * **Hiệu năng phi tuyến (Non-monotonicity):** Không có sự suy giảm đơn điệu hay gia tăng đều đặn theo chân trời.
   * **Ảo ảnh hiệu năng (Predictability Illusion):** Sau khi kiểm soát sai lầm thống kê họ giả thuyết qua hiệu chỉnh Holm-Bonferroni ($p_{\text{adj}} < 0.05$), không có mô hình nào duy trì năng lực dự báo vượt trội Historical Mean có ý nghĩa thống kê bền vững.
   * **Độ nhạy khởi tạo ngẫu nhiên (Multi-seed Fragility):** Khi chạy kiểm chứng 10 random seeds, giá trị $OOS\ R^2 = 17,53\%$ của TSMixer tại 120D rớt xuống $-18,93\%$, chứng minh tính bấp bênh của các mô hình học sâu.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Điểm cốt lõi nhất trong công trình của nhóm là sự đối chiếu giữa kết quả mô tả định lượng (Descriptive OOS R²) và kiểm định suy diễn thống kê nghiêm ngặt (Inferential Statistics). Nếu chỉ nhìn vào điểm số đơn lẻ của một lần chạy, chúng ta rất dễ rơi vào bẫy 'ảo ảnh khả năng dự báo'. Nghiên cứu này phơi bày toàn diện tính mong manh đó."*

---

## SLIDE 3: GIỚI THIỆU BÀI TOÁN – ĐỊNH NGHĨA ĐẦU VÀO (INPUT) & ĐẦU RA (OUTPUT)

### Bản chất bài toán học máy:
Đây là bài toán **Dự báo Chuỗi Thời gian Đa biến (Multivariate Time Series Forecasting)** giải quyết bài toán hồi quy (Regression) có cấu trúc tensor rõ ràng:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CẤU TRÚC ĐẦU VÀO & ĐẦU RA                              │
├─────────────────────────────────────────────────┬──────────────────────────────────────┤
│ ĐẦU VÀO (INPUT TENSORS): X_t                    │ ĐẦU RA (TARGET OUTPUT): y_t^(h)      │
├─────────────────────────────────────────────────┼──────────────────────────────────────┤
│ Ma trận tensor bậc 2 (2D Matrix):                │ Đại lượng vô hướng (Scalar Value):   │
│ Kích thước: [L x K] = [20 x 16]                 │ Lợi suất lũy kế log-return tương lai │
│                                                 │ tại chân trời h:                     │
│ • L = 20: Cửa sổ quan sát quá khứ (Lookback)   │                                      │
│ • K = 16: Không gian đặc trưng đa chiều:        │   y_t^(h) = ln(P_{t+h} / P_t)        │
│   - Giá & Lợi suất trễ (Price/Return lags)      │                                      │
│   - Đường trung bình động (MA5, MA20, EMA20)    │ Trong đó:                            │
│   - Xung lượng kỹ thuật (RSI, MACD, MACD Signal)│ • P_t: Giá đóng cửa S&P 500 ngày t   │
│   - Biến động lịch sử & CBOE VIX (Realized Vol) │ • h in {1, 5, 20, 60, 120} ngày      │
│   - Thanh khoản & Khối lượng giao dịch (Volume) │                                      │
│ (Riêng ARIMA là đơn biến: chỉ nhận chuỗi trễ)   │                                      │
└─────────────────────────────────────────────────┴──────────────────────────────────────┘
```

* **Mục tiêu tối ưu hóa:** Tìm hàm ánh xạ phi tuyến $f_h: X_t \rightarrow \hat{y}_t^{(h)}$ sao cho tối thiểu hóa hàm mất mát sai số bình phương trung bình: $\min \frac{1}{T}\sum_{t=1}^T (y_t^{(h)} - f_h(X_t))^2$.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Để trả lời câu hỏi bài toán dự báo của nhóm là gì: Đây là bài toán hồi quy chuỗi thời gian. Đầu vào X_t là một ma trận tensor kích thước 20 x 16 gồm 20 ngày quá khứ và 16 chỉ báo kỹ thuật, biến động và vĩ mô. Đầu ra là lợi suất logarit lũy kế y_t^(h) tại 5 chân trời thời gian riêng biệt: 1 ngày, 5 ngày, 20 ngày, 60 ngày và 120 ngày. Mỗi chân trời là một mô hình chuyên biệt độc lập."*

---

## SLIDE 4: TỔNG QUAN HỌC THUẬT & KHOẢNG TRỐNG NGHIÊN CỨU (LITERATURE REVIEW & GAPS)

### Bảng đối chiếu các công trình nghiên cứu tiêu biểu:

| Tác giả & Năm | Phương pháp | Phát hiện chính | Khoảng trống cần hoàn thiện |
| :--- | :--- | :--- | :--- |
| **Welch & Goyal (2008)** | Hồi quy OLS, biến kinh tế vĩ mô | Hầu hết mô hình thất bại khi dự báo ngoài mẫu so với Historical Mean | Chỉ dùng mô hình tuyến tính đơn giản; không xét phi tuyến và deep learning |
| **Campbell & Thompson (2008)** | Ràng buộc dấu kinh tế trên OLS | Cải thiện nhẹ OOS R² khi áp đặt ràng buộc hệ số không âm | Chưa kiểm định trong các chế độ biến động cực đoan (VIX cao) |
| **Gu, Kelly & Xiu (2020)** | Machine Learning (Trees, NN) | Cây quyết định và mạng nơ-ron vượt trội tuyến tính ở cấp độ cổ phiếu | Không phân tích chi tiết theo chân trời đa kỳ (Multi-horizon) và độ nhạy seed |
| **Quaedvlieg (2021)** | Multi-horizon forecast comparison | Đề xuất kiểm định thống kê bình đẳng năng lực giữa các chân trời | Chỉ đánh giá các mô hình chuỗi thời gian kinh tế lượng truyền thống |
| **Nghiên cứu của chúng em** | **12 mô hình $\times$ 5 kỳ hạn $\times$ 3 regimes + Multi-seed + Ablation** | **Đánh giá toàn diện, kiểm định đa giả thuyết khắt khe và bóc tách đặc trưng** | **Lấp đầy toàn bộ khoảng trống: phi tuyến, đa chế độ, đa hạt giống ngẫu nhiên** |

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Nhìn vào lược sử học thuật, nghiên cứu của chúng em kế thừa bài toán kinh điển của Welch & Goyal (2008) nhưng mở rộng vượt bậc: thay vì chỉ dừng lại ở hồi quy tuyến tính, chúng em đưa vào toàn bộ các họ mô hình học máy hiện đại, đồng thời áp dụng kiểm định thống kê chuẩn mực để trả lời liệu Machine Learning có thực sự đánh bại được thị trường tài chính hay không."*

---

## SLIDE 5: CÂU HỎI NGHIÊN CỨU (RQ) & CÁC GIẢ THUYẾT THỰC NGHIỆM (H)

### 3 Câu hỏi Nghiên cứu (Research Questions):
* **RQ1:** Khả năng dự báo ngoài mẫu ($OOS\ R^2$) biến thiên như thế nào qua các chân trời dự báo khác nhau (1D, 5D, 20D, 60D, 120D)? Có quy luật suy giảm hay gia tăng đơn điệu không?
* **RQ2:** Mức độ biến động của thị trường (Low, Normal, High Volatility Regimes) ảnh hưởng như thế nào đến độ chính xác và tính ổn định của các mô hình?
* **RQ3:** Khả năng dự báo có ổn định theo thời gian qua các chu kỳ kinh tế trong giai đoạn kiểm định ngoài mẫu (2020–2026) hay bị sụp đổ cục bộ?

### 3 Giả thuyết Thống kê Tương ứng:
* **Giả thuyết H1 (Horizon Equality Hypothesis):**  
  $H_0^{(1)}: \Delta L_{h_1} = \Delta L_{h_2} = \dots = \Delta L_{h_5}$  
  *(Năng lực dự báo tương đối so với Historical Mean là tương đương nhau giữa tất cả các chân trời thời gian)*
* **Giả thuyết H2 (Regime Predictability Hypothesis):**  
  $H_0^{(2)}: \mathbb{E}[\Delta L_{i,t}^{(h)} \mid \text{Regime}_k] = 0$  
  *(Không có sự khác biệt về sai số dự báo giữa mô hình kiểm định và benchmark trong từng chế độ biến động cụ thể)*
* **Giả thuyết H3 (Temporal Stability Hypothesis):**  
  $H_0^{(3)}: \mathbb{E}[\Delta L_{i,t}^{(h)}] = \mu \quad \forall t \in \text{Holdout}$  
  *(Năng lực dự báo tương đối là ổn định và không đổi qua các phân đoạn thời gian ngoài mẫu)*

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Ba câu hỏi nghiên cứu này được ánh xạ trực tiếp thành ba giả thuyết thống kê có thể kiểm định được. Mỗi giả thuyết đều đi kèm với các bài kiểm định tham số và phi tham số tương ứng nhằm đảm bảo mọi kết luận rút ra đều dựa trên cơ sở định lượng vững chắc chứ không phải trực giác chủ quan."*

---

# PHẦN II: DỮ LIỆU, CHÂN TRỜI DỰ BÁO & KỸ NGHỆ ĐẶC TRƯNG

---

## SLIDE 6: NGUỒN DỮ LIỆU & ĐẶC TÍNH MẪU (DATA SOURCES & SAMPLE)

### 1. Thu thập dữ liệu chuẩn mực tài chính:
* **Chỉ số S&P 500 (^GSPC):** Trích xuất từ Yahoo Finance từ ngày **02/01/1990 đến 29/09/2026** (bao gồm giá Open, High, Low, Close, Volume).
  * Tổng số quan sát gốc: **9.255 ngày giao dịch**.
* **Chỉ số biến động CBOE VIX:** Thu thập từ Cục Dự trữ Liên bang Mỹ (FRED) từ **02/01/1990 đến 28/09/2026**.
  * Tổng số quan sát gốc: **9.585 ngày quan sát**.

### 2. Sơ đồ xử lý và căn chỉnh mẫu đồng bộ:

```
┌─────────────────────────────────┐
│ Yahoo Finance: Chỉ số S&P 500   │
│ (9.255 quan sát: 1990 - 2026)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│ Đồng bộ hóa & Căn chỉnh ngày    │ ◄───── │ FRED: Chỉ số CBOE VIX            │
│ (Inner Join on Trading Dates)   │        │ (9.585 quan sát: 1990 - 2026)    │
└────────────────┬────────────────┘        └──────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BỘ DỮ LIỆU CHUẨN HÓA TOÀN DỰ ÁN:                                            │
│ • 9.245 ngày giao dịch sạch (Strict Trading Days)                           │
│ • 0 giá trị khuyết thiếu (Zero Missing Values)                              │
│ • Đầy đủ biến giá, khối lượng và kỳ vọng biến động thị trường (VIX)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Dữ liệu được thu thập từ hai nguồn chính thống là Yahoo Finance cho chỉ số S&P 500 và Cục Dự trữ Liên bang Mỹ FRED cho chỉ số biến động VIX. Sau khi thực hiện liên kết ngày giao dịch nghiêm ngặt (Inner Join) và loại bỏ các ngày nghỉ lễ ngân hàng không khớp, chúng em thu được tập dữ liệu chuẩn mực gồm đúng 9.245 ngày giao dịch liên tục, không có bất kỳ giá trị khuyết thiếu nào."*

---

## SLIDE 7: PHÂN BIỆT CHUỖI THỜI GIAN ĐƠN BIẾN (UNIVARIATE) & ĐA BIẾN (MULTIVARIATE)

### Trả lời câu hỏi trọng tâm của Thầy: "Chuỗi thời gian là đơn biến hay đa biến?"

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   SO SÁNH BẢN CHẤT DỮ LIỆU & MÔ HÌNH TRONG ĐỒ ÁN                       │
├───────────────────────────────────┬────────────────────────────────────────────────────┤
│ ĐƠN BIẾN (UNIVARIATE TIME SERIES) │ ĐA BIẾN (MULTIVARIATE TIME SERIES)                 │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ • Định nghĩa: Chỉ sử dụng lịch sử │ • Định nghĩa: Sử dụng đồng thời vector 16 đặc trưng│
│   của chính chuỗi mục tiêu y_t    │   X_t = [x_{1,t}, x_{2,t}, ..., x_{16,t}]^T        │
│   (giá đóng cửa hoặc log return)  │ • Mô hình áp dụng: 11 mô hình còn lại:             │
│ • Mô hình áp dụng: ARIMA          │   Ridge, Lasso, ElasticNet, Random Forest,         │
│ • Công thức toán:                 │   LightGBM, XGBoost, SVR, MLP, LSTM, TCN,          │
│   y_t = c + \sum \phi_i y_{t-i}   │   Transformer, PatchTST, TSMixer.                  │
│       + \sum \theta_j \epsilon    │ • Ý nghĩa: Khai thác quan hệ phụ thuộc chéo        │
│ • Ý nghĩa: Đóng vai trò mốc       │   giữa xu hướng, động lượng, biên độ trong ngày,   │
│   tham chiếu kinh tế lượng cổ điển│   thanh khoản và tâm lý sợ hãi (VIX).              │
└───────────────────────────────────┴────────────────────────────────────────────────────┘
```

* **Kết luận khoa học:** Đề tài thiết kế **cả hai phương cận**: dùng mô hình Đơn biến làm đối chứng cổ điển để chứng minh xem việc bổ sung 16 biến đa biến có thực sự mang lại giá trị gia tăng ngoài mẫu hay không.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thưa Thầy, đối với câu hỏi chuỗi thời gian đơn biến hay đa biến, nghiên cứu của chúng em khảo sát cả hai: ARIMA đại diện cho tiếp cận đơn biến thuần túy chỉ sử dụng độ trễ của chính chuỗi lợi suất. Toàn bộ 11 mô hình còn lại là tiếp cận đa biến, tiếp nhận ma trận 16 đặc trưng nhằm nắm bắt cấu trúc tương quan phức tạp giữa biến động, thanh khoản và xu hướng thị trường."*

---

## SLIDE 8: KHÔNG GIAN 16 ĐẶC TRƯNG DỰ BÁO (16 PREDICTIVE FEATURES)

### Cấu trúc 4 nhóm đặc trưng tài chính chuẩn mực:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        KHÔNG GIAN 16 ĐẶC TRƯNG KỸ THUẬT & TÀI CHÍNH                    │
├────────────────────────────┬───────────────────────────────────────────────────────────┤
│ NHÓM 1: PRICE & RETURN     │ • Return_Lag1: Lợi suất ngày t-1                          │
│ (Động lượng giá ngắn hạn)  │ • Return_Lag2: Lợi suất ngày t-2                          │
│                            │ • Return_Lag5: Lợi suất lũy kế 5 ngày (1 tuần)            │
├────────────────────────────┼───────────────────────────────────────────────────────────┤
│ NHÓM 2: TREND INDICATORS   │ • MA5_Ratio: Tỷ lệ Close / Simple Moving Average 5 ngày   │
│ (Chỉ báo xu hướng dài hạn) │ • MA20_Ratio: Tỷ lệ Close / Simple Moving Average 20 ngày │
│                            │ • EMA20_Ratio: Tỷ lệ Close / Exponential Moving Average   │
│                            │ • MACD: Đường chênh lệch trung bình động hội tụ/phân kỳ   │
│                            │ • MACD_Signal: Đường tín hiệu MACD 9 ngày                 │
├────────────────────────────┼───────────────────────────────────────────────────────────┤
│ NHÓM 3: MOMENTUM & VOL     │ • RSI14: Chỉ số sức mạnh tương đối 14 ngày                │
│ (Xung lượng & Biến động)   │ • Volatility_5D: Biến động lịch sử độ lệch chuẩn 5 ngày   │
│                            │ • Volatility_20D: Biến động lịch sử độ lệch chuẩn 20 ngày │
│                            │ • VIX_Close: Mức đóng cửa chỉ số biến động CBOE VIX       │
│                            │ • VIX_Return: Tỷ lệ thay đổi phần trăm hàng ngày của VIX  │
├────────────────────────────┼───────────────────────────────────────────────────────────┤
│ NHÓM 4: INTRADAY & VOLUME  │ • Intraday_Range: Biên độ dao động (High - Low) / Close   │
│ (Biên độ phiên & Thanh khoản)│ • Return_Overnight: Lợi suất qua đêm (Open_t - Close_{t-1})│
│                            │ • Volume_Change: Tốc độ tăng trưởng khối lượng giao dịch  │
└────────────────────────────┴───────────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"16 đặc trưng được thiết kế tỉ mỉ chia làm 4 nhóm chức năng rõ ràng: nhóm giá và lợi suất trễ, nhóm xu hướng, nhóm xung lượng - biến động (kết hợp VIX) và nhóm biên độ trong phiên cùng khối lượng. Tất cả đều phản ánh các khía cạnh khác nhau của cấu trúc vi mô thị trường."*

---

## SLIDE 9: THỐNG KÊ MÔ TẢ & KIỂM ĐỊNH TÍNH DỪNG (STATIONARITY & ADF TEST)

### Thống kê mô tả các chuỗi biến đại diện & Kết quả kiểm định ADF:

| Tên biến đặc trưng | Trung bình (Mean) | Độ lệch chuẩn (Std) | Độ lệch (Skewness) | Độ nhọn (Kurtosis) | ADF Statistic | p-value (ADF) | Kết luận tính dừng |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Return_Lag1** | 0.00034 | 0.0114 | -0.284 | 9.421 | -72.41 | < 0.0001 | Dừng tuyệt đối $I(0)$ |
| **Volatility_20D** | 0.01082 | 0.0062 | 2.651 | 14.182 | -9.84 | < 0.0001 | Dừng tuyệt đối $I(0)$ |
| **VIX_Close** | 19.45 | 7.82 | 2.114 | 8.945 | -5.12 | < 0.0001 | Dừng tuyệt đối $I(0)$ |
| **RSI14** | 53.21 | 11.45 | -0.381 | 3.210 | -14.65 | < 0.0001 | Dừng tuyệt đối $I(0)$ |
| **Intraday_Range** | 0.0135 | 0.0091 | 3.421 | 24.150 | -11.23 | < 0.0001 | Dừng tuyệt đối $I(0)$ |
| *Giá S&P 500 gốc* | 1845.2 | 1240.1 | 0.842 | -0.412 | -0.48 | 0.8921 | **Không dừng $I(1)$** |

* Tất cả 16 biến đều được chuẩn hóa về dạng tỷ lệ phần trăm hoặc sai phân bậc một nhằm loại bỏ xu hướng trôi dạt (unit root non-stationarity).
* Kiểm định Augmented Dickey-Fuller (ADF) khẳng định $100\%$ các chuỗi đặc trưng đều đạt tính dừng $I(0)$ ở mức ý nghĩa $1\%$.
* **Quy tắc ngăn ngừa Lookahead Leakage:**
  1. **Point-in-time Alignment:** Đặc trưng tại ngày $t$ chỉ sử dụng dữ liệu được chốt vào hoặc trước thời điểm đóng cửa ngày $t$.
  2. **Strict Scaling Protocol:** Mọi tham số chuẩn hóa (MinMaxScaler/StandardScaler) chỉ được fit trên tập Huấn luyện và transform sang tập Holdout.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Một quy tắc sống còn trong tài chính là tính dừng. Chuỗi giá gốc không dừng với p-value 0.89, nhưng toàn bộ 16 đặc trưng sau khi biến đổi đều có p-value ADF nhỏ hơn 0.0001, đảm bảo tính dừng nghiêm ngặt. Đồng thời, nhóm tuân thủ triệt để nguyên tắc không rò rỉ dữ liệu tương lai."*

---

## SLIDE 10: CƠ SỞ CHỌN 5 CHÂN TRỜI DỰ BÁO (1D, 5D, 20D, 60D, 120D)

### Trả lời câu hỏi trọng tâm của Thầy: "Dự báo giá phải nói được tại sao là số ngày đó?"

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ÁNH XẠ CHÂN TRỜI DỰ BÁO VÀO CHU KỲ TÀI CHÍNH                    │
├─────────┬──────────────────────┬───────────────────────────────────────────────────────┤
│ KỲ HẠN  │ QUY ĐỔI THỜI GIAN    │ CƠ SỞ KINH TẾ HỌC & HOẠT ĐỘNG THỊ TRƯỜNG              │
├─────────┼──────────────────────┼───────────────────────────────────────────────────────┤
│ 1D      │ 1 Ngày Giao dịch     │ • Dự báo siêu ngắn hạn (High-frequency / Day trading)  │
│         │                      │ • Chịu chi phối bởi vi cấu trúc thị trường và nhiễu   │
├─────────┼──────────────────────┼───────────────────────────────────────────────────────┤
│ 5D      │ 1 Tuần Giao dịch     │ • Chu kỳ làm việc hàng tuần của các định chế tài chính│
│         │ (5 trading days)     │ • Điểm cân bằng vị thế cuối tuần (Weekend effect)      │
├─────────┼──────────────────────┼───────────────────────────────────────────────────────┤
│ 20D     │ 1 Tháng Giao dịch    │ • Chu kỳ đáo hạn hợp đồng quyền chọn (Options Expiry) │
│         │ (20-22 trading days) │ • Chu kỳ tái cân bằng danh mục định kỳ của quỹ đầu tư │
├─────────┼──────────────────────┼───────────────────────────────────────────────────────┤
│ 60D     │ 1 Quý Kinh doanh     │ • Chu kỳ công bố Báo cáo Tài chính quý (Earnings)     │
│         │ (1 Quarter)          │ • Dữ liệu tăng trưởng GDP quý và điều chỉnh vĩ mô     │
├─────────┼──────────────────────┼───────────────────────────────────────────────────────┤
│ 120D    │ 2 Quý / Nửa năm      │ • Chu kỳ họp chính sách lãi suất của Fed (FOMC Cycle) │
│         │ (Semi-annual)        │ • Định giá lại danh mục tài sản lớn của quỹ hưu trí   │
└─────────┴──────────────────────┴───────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thưa Thầy, 5 chân trời thời gian 1D, 5D, 20D, 60D và 120D không phải được chọn ngẫu nhiên mà ánh xạ chuẩn xác vào 5 chu kỳ vận hành của hệ thống tài chính thế giới: 1 ngày giao dịch, 1 tuần làm việc, 1 tháng đáo hạn quyền chọn, 1 quý báo cáo tài chính doanh nghiệp và nửa năm chu kỳ chính sách tiền tệ. Đây là tiêu chuẩn vàng trong kinh tế lượng tài chính."*

---

## SLIDE 11: ĐỊNH NGHĨA 3 CHẾ ĐỘ BIẾN ĐỘNG THỊ TRƯỜNG (VOLATILITY REGIMES)

### Cơ chế phân loại chế độ thị trường dựa trên phân vị lịch sử (Percentile-based Regimes):
* Sử dụng biến động thực tế 20 ngày ($\sigma_{20, t}$) để đo lường trạng thái rủi ro.
* Chia toàn bộ phân phối thành 3 vùng dựa trên phân vị 33.3% và 66.7%:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHÂN BỔ MẪU HOLDOUT (1.362 NGÀY) THEO 3 CHẾ ĐỘ BIẾN ĐỘNG (MỖI NHÓM 33%)│
├────────────────────────┬────────────────────────┬──────────────────────┤
│ Low Volatility Regime  │ Normal Volatility      │ High Volatility      │
│ (454 ngày - 33.3%)     │ (454 ngày - 33.3%)     │ (454 ngày - 33.3%)   │
│ \sigma_{20} <= Q_{33%} │ Q_{33%} < \sigma_{20}  │ \sigma_{20} > Q_{67%}│
│ Thị trường ổn định     │ Trạng thái cân bằng    │ Thị trường khủng hoảng│
└────────────────────────┴────────────────────────┴──────────────────────┘
```

* **Tính cố định của ngưỡng:** Các ngưỡng phân vị được tính một lần duy nhất trên tập phát triển và áp dụng cố định sang tập Holdout nhằm đảm bảo tính khách quan tuyệt đối.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Để phân tích RQ2, nhóm chia thị trường thành 3 vùng chế độ dựa trên phân vị 33% và 67%. Mức phân chia này đảm bảo mỗi chế độ có chính xác 454 ngày giao dịch trong tập Holdout, giúp các phép so sánh thống kê đạt độ tin cậy và lực kiểm định tối đa."*

---

# PHẦN III: PHƯƠNG PHÁP LUẬN, THIẾT KẾ & BỘ ĐỆM LOOKBACK

---

## SLIDE 12: THIẾT KẾ PHÂN CHIA THỜI GIAN (CHRONOLOGICAL DATA SPLITTING)

### Cấu trúc phân chia dữ liệu theo dòng thời gian:
* **Tổng thể mẫu:** 9.245 ngày giao dịch (1990 – 2026).
* **Tập Huấn luyện ban đầu (Initial Training):** 1990 – 2015 (6.351 mẫu; $\sim 65\%$ dữ liệu).
* **Tập Tối ưu & Thẩm định (Validation Folds):** 2016 – 2020 (1.360 mẫu, dùng tìm siêu tham số).
* **Tập Kiểm định ngoài mẫu cuối cùng (Holdout Test):** **11/2020 – 04/2026** (Chính xác **1.362 điểm dự báo ngoài mẫu**).

```
1990 ─────────────────────── 2015 ───────── 2020 ────────────────── 2026
[      INITIAL TRAINING      ] [ VALIDATION ] [  OUT-OF-SAMPLE HOLDOUT  ]
(Học tham số mô hình ban đầu)  (Optuna Tuning) (1.362 điểm dự báo ngoài mẫu)
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thay vì chia mẫu ngẫu nhiên (Random K-Fold) vốn phá hủy cấu trúc chuỗi thời gian, nhóm dùng phương pháp chia thứ tự thời gian nghiêm ngặt. Tập kiểm định Holdout gồm 1.362 ngày trải dài từ cuối 2020 đến 2026, đi qua đầy đủ các giai đoạn thị trường tăng trưởng, lạm phát và chu kỳ tăng lãi suất của Fed."*

---

## SLIDE 13: GIAO THỨC WALK-FORWARD & XÁC ĐỊNH WINDOW SIZE L = 20 NGÀY

### Trả lời câu hỏi trọng tâm của Thầy: "Xác định Window Size như thế nào?"
1. **Tại sao chọn Lookback Window $L = 20$ ngày?**
   * **Cơ sở tài chính:** $L = 20$ ngày tương ứng đúng với **1 tháng giao dịch**. Trong tài chính, cấu trúc tự tương quan ngắn hạn của giá chứng khoán thường suy giảm sau 20–22 phiên.
   * **Đánh đổi Bias - Variance:** Nếu chọn $L$ quá ngắn (ví dụ 5 ngày), mô hình thiếu thông tin ngữ cảnh xu hướng; nếu chọn $L$ quá dài (ví dụ 60 ngày), các mô hình mạng nơ-ron sâu sẽ chịu hiện tượng quá khớp (overfitting) do tham số tăng vọt. $L = 20$ là điểm cân bằng tối ưu qua thực nghiệm.
2. **Giao thức Walk-Forward Expanding Window:**
   * Cửa sổ mở rộng dần theo thời gian: Tại mỗi mốc ngoài mẫu, mô hình tích lũy thêm dữ liệu thực tế đã xảy ra để dự báo bước tiếp theo mà không bao giờ nhìn thấy trước tương lai.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thưa Thầy, window size L = 20 ngày được nhóm xác định dựa trên chu kỳ tháng giao dịch chuẩn. Đây là điểm dung hòa hoàn hảo giữa việc thu nạp đủ thông tin động lượng và việc hạn chế bùng nổ tham số gây quá khớp ở các mạng học sâu."*

---

## SLIDE 14: KỸ THUẬT BỘ ĐỆM LOOKBACK CHO CÁC MÔ HÌNH CHUỖI SÂU (LOOKBACK BUFFER)

### Thách thức kỹ thuật đối với Deep Sequence Models:
* Các mô hình chuỗi thời gian như LSTM, TCN, PatchTST, Transformer, TSMixer cần độ dài chuỗi quá khứ $L=20$ ngày để hình thành tensor đầu vào $[20 \times 16]$.
* **Nguy cơ mất mát dữ liệu biên:** Nếu cắt lát dữ liệu ngây thơ tại mốc bắt đầu của tập Holdout, mô hình sẽ bị mất 19 ngày đầu tiên vì thiếu chuỗi lịch sử phía trước!

```
┌────────────────────────────────────────┐
│ Tập Huấn luyện trước đó (Train/Val)    │
│ [ ... Lịch sử dữ liệu quá khứ ... ]    │
│ └──► [Lookback Buffer: 19 ngày cuối] ──┼──┐
└────────────────────────────────────────┘  │
                                            ▼
┌──────────────────────────────────────────────┐
│ Ngày đầu tiên của tập Holdout (T_holdout)     │
│ Ghép 19 ngày buffer + ngày T_holdout         │
│ ════► Đủ tensor [20 x 16] để dự báo ngay!    │
└──────────────────────────────────────────────┘
```

* **Giải pháp chuẩn hóa:** Trích xuất đúng 19 quan sát cuối cùng của tập phát triển làm bộ đệm (Buffer). Nhờ đó, tập Holdout được dự báo trọn vẹn **1.362/1.362 ngày**, không mất bất kỳ một quan sát nào và hoàn toàn không rò rỉ dữ liệu.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kỹ thuật Lookback Buffer là một điểm sáng trong triển khai của nhóm. Bằng cách lấy 19 ngày cuối của tập trước làm đệm, chúng em bảo toàn trọn vẹn 1.362 điểm dự báo ngoài mẫu cho tất cả các mô hình học sâu mà không làm sai lệch tính chất nhân quả của chuỗi thời gian."*

---

# PHẦN IV: CHI TIẾT TẤT CẢ CÁC PHƯƠNG PHÁP & THÔNG SỐ KỸ THUẬT (THEO LỜI DẶN CỦA THẦY)

---

## SLIDE 15: TỔNG QUAN KHÔNG GIAN 12 MÔ HÌNH DỰ BÁO (MODEL TAXONOMY & TEMPLATE)

### Phân loại 12 mô hình theo 4 cấp độ phức tạp:
1. **Benchmark cơ sở:** Historical Mean (Campbell & Thompson, 2008).
2. **Kinh tế lượng chuỗi thời gian:** ARIMA (Đơn biến), ARIMAX (Đa biến với biến trễ ngoại sinh).
3. **Mô hình Tuyến tính có ràng buộc (Regularized Linear):** Ridge Regression (L2), Lasso Regression (L1), ElasticNet (L1 + L2).
4. **Mô hình Học máy Dựa trên Cây & Kernel (Trees & SVR):** Random Forest, LightGBM, XGBoost, Support Vector Regression (SVR).
5. **Mô hình Học sâu Chuỗi thời gian (Deep Sequence Models):** MLP / ANN (Chương 6), LSTM, Temporal Convolutional Network (TCN), Transformer, PatchTST, TSMixer.

### Cấu trúc Template chuẩn mực áp dụng thống nhất cho từng phương pháp:
* **Mục tiêu & Cơ sở tham khảo:** Nguồn gốc học thuật và bài toán giải quyết.
* **Bản chất toán học / Thuật toán xây dựng:** Cơ chế tối ưu hóa, hàm mất mát và hàm phạt.
* **Bộ thông số kỹ thuật tối ưu:** Số tầng, nơ-ron, siêu tham số được chốt sau Grid Search / Optuna.
* **Cơ chế chống quá khớp:** Regularization, Dropout, Early Stopping.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thực hiện đúng lời dặn của Thầy: 'Các phương pháp đã học không trình bày lý thuyết chung chung mà trình bày thông số kỹ thuật, cấu trúc layer, nơ-ron; các phương pháp mới trình bày ý tưởng, thuật toán và kiến trúc'. Sau đây nhóm xin lần lượt trình bày chi tiết từng mô hình theo đúng template chuẩn hóa đó."*

---

## SLIDE 16: MÔ HÌNH CHUỖI THỜI GIAN CỔ ĐIỂN – ARIMA & ARIMAX

### Trả lời câu hỏi trọng tâm của Thầy: "Xác định p, d, q như thế nào? Chạy nhiều bộ pdq? AutoArima? Arimax?"

#### 1. Quy trình 4 bước xác định bộ tham số $(p, d, q)$:
* **Bước 1 (Xác định bậc sai phân $d$):**
  * Kiểm định ADF trên chuỗi giá S&P 500: $p = 0.892$ (chuỗi không dừng).
  * Lấy sai phân bậc một $d=1$ (chuỗi log return): ADF Statistic = $-72.41$ ($p < 0.0001$, chuỗi dừng tuyệt đối). Do đó chốt $d=0$ nếu mô hình chạy trực tiếp trên return, hoặc $d=1$ nếu chạy trên log-price.
* **Bước 2 (Khoanh vùng bậc $p$ và $q$ qua biểu đồ ACF/PACF):**
  * Đồ thị ACF (Autocorrelation Function) cắt nhanh sau lag 1 $\rightarrow$ Gợi ý $q \in \{0, 1, 2\}$.
  * Đồ thị PACF (Partial Autocorrelation Function) giảm mạnh sau lag 1 $\rightarrow$ Gợi ý $p \in \{0, 1, 2\}$.
* **Bước 3 (Thực nghiệm Grid Search nhiều bộ tham số khác nhau):**
  * Chạy thực nghiệm 16 tổ hợp $(p, d, q)$ với $p \in \{0, 1, 2, 3\}$, $q \in \{0, 1, 2, 3\}$.
  * Đánh giá chất lượng bằng chỉ số AIC (Akaike Information Criterion) và BIC (Bayesian Information Criterion) trên tập Huấn luyện.
  * Bộ tối ưu hóa AIC/BIC tốt nhất: **ARIMA(1, 0, 1)** với log-return (tương đương ARIMA(1, 1, 1) trên chuỗi giá).
* **Bước 4 (Tại sao KHÔNG dùng AutoARIMA?):**
  * AutoARIMA sử dụng tìm kiếm heuristic cục bộ từng bước (stepwise search), rất dễ rơi vào cực tiểu địa phương.
  * Trong chuỗi tài chính có nhiễu cao, AutoARIMA thường chọn các bậc $p, q$ quá cao (ví dụ $p=5, q=4$) dẫn đến hiện tượng quá khớp (overfitting) trầm trọng trên tập train và sụp đổ hoàn toàn ngoài mẫu.

#### 2. ARIMAX – Bộ $(p, d, q)$ và biến ngoại sinh:
* Cấu hình bậc: **ARIMAX(1, 0, 1)** giữ nguyên cấu trúc động học ARIMA(1, 0, 1).
* **Vector ngoại sinh ($X_{t-1}$):** Tích hợp 15 biến đặc trưng trễ (VIX, Volume, RSI, Volatility, v.v.):
  $y_t = c + \phi_1 y_{t-1} + \theta_1 \epsilon_{t-1} + \sum_{k=1}^{15} \beta_k x_{k, t-1} + \epsilon_t$

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thưa Thầy, đối với ARIMA, nhóm không dùng AutoARIMA vì AutoARIMA dễ bị bẫy tối ưu cục bộ và chọn số bậc quá lớn gây quá khớp. Nhóm xác định d qua kiểm định ADF, phân tích đồ thị ACF/PACF để khoanh vùng p, q, và chạy Grid Search thủ công thử nghiệm 16 bộ tham số khác nhau để chọn ra bộ có AIC/BIC tối ưu nhất là (1,0,1). Đối với ARIMAX, nhóm giữ nguyên bộ (1,0,1) và tích hợp thêm 15 biến đặc trưng trễ ở bước t-1."*

---

## SLIDE 17: MÔ HÌNH TUYẾN TÍNH CÓ PHẠT CO NGÓT – RIDGE, LASSO & ELASTICNET

### Thầy sẽ hỏi: "Hồi quy tuyến tính OLS thất bại trong tài chính, Ridge và Lasso giải quyết như thế nào?"

#### 1. Ridge Regression (Phạt L2 - Tikhonov Regularization):
* **Hàm mục tiêu:** $\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \frac{1}{2}\lambda \|\beta\|_2^2$.
* **Cơ chế hoạt động:** Phạt bình phương độ lớn hệ số $\beta$. Nghiệm giải tích dạng đóng: $\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y$.
* **Giải quyết đa cộng tuyến:** Ma trận $(X^TX)$ trong dữ liệu tài chính thường gần suy biến do 16 chỉ báo kỹ thuật có tương quan cao (ví dụ MA5, MA20, EMA20). Việc cộng thêm $\lambda I$ trên đường chéo chính giúp nghịch đảo ma trận luôn khả thi và triệt tiêu phương sai dự báo.
* **Thông số kỹ thuật:** Dò tìm qua 6 mức $\alpha \in [10^{-3}, 10^3]$, bộ tham số tối ưu sau Tuning: `alpha = 100.0`.

#### 2. Lasso Regression (Phạt L1 - Least Absolute Shrinkage):
* **Hàm mục tiêu:** $\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1$.
* **Cơ chế hoạt động:** Sử dụng đạo hàm dưới (Subgradient) và thuật toán **Coordinate Descent**. Điểm góc nhọn của chuẩn L1 tại trục tọa độ ép các hệ số của các biến không quan trọng về **đúng bằng 0** (Sparsity).
* **Ý nghĩa:** Đóng vai trò bộ chọn lọc đặc trưng tự động (Feature Selection), loại bỏ nhiễu thị trường.
* **Thông số kỹ thuật:** `alpha = 0.01`, `max_iter = 1000`.

#### 3. ElasticNet (Kết hợp L1 + L2):
* **Hàm mục tiêu:** Kết hợp cả co ngót nhóm (L2) và tính thưa thớt (L1): $\lambda \left( \rho \|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|_2^2 \right)$.
* **Thông số tối ưu sau Tuning:** `alpha = 0.001`, `l1_ratio = 0.5`.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"OLS thuần túy thất bại vì 16 đặc trưng tài chính có hiện tượng đa cộng tuyến mạnh. Ridge giải quyết triệt để vấn đề này bằng phạt L2, cộng lambda vào đường chéo ma trận hiệp phương sai để ổn định hệ số. Lasso dùng phạt L1 ép các hệ số nhiễu về đúng bằng 0 để chắt lọc đặc trưng, còn ElasticNet cân bằng giữa cả hai cơ chế."*

---

## SLIDE 18: MÔ HÌNH RỪNG CÂY QUYẾT ĐỊNH & BAGGING – RANDOM FOREST

### Thầy sẽ hỏi: "Random Forest xây dựng cây khác gì cây đơn lẻ và khác gì XGBoost?"

#### 1. Thuật toán xây dựng cây của Random Forest:
* **Cơ chế Bagging (Bootstrap Aggregating):**
  * Với mỗi cây thứ $b \in \{1, \dots, B\}$, lấy mẫu có hoàn lại (Bootstrap Sample) gồm $N$ quan sát từ tập huấn luyện ban đầu.
  * Khoảng $36.8\%$ dữ liệu không được chọn vào cây được gọi là mẫu ngoài túi (Out-Of-Bag - OOB).
* **Cơ chế Không gian con Đặc trưng Ngẫu nhiên (Feature Subspace Sampling):**
  * Tại mỗi nút phân nhánh (Split Node), thuật toán KHÔNG quét qua tất cả 16 đặc trưng mà chỉ chọn ngẫu nhiên một tập con $m \le K$ đặc trưng (thường là $m = \sqrt{K} = 4$ hoặc $K/3$).
  * Điểm chẻ tối ưu chỉ được chọn trong số $m$ đặc trưng này dựa trên tiêu chí giảm thiểu sai số bình phương MSE:
    $\Delta \text{MSE} = \text{MSE}_{\text{parent}} - \left(\frac{N_L}{N}\text{MSE}_L + \frac{N_R}{N}\text{MSE}_R\right)$
* **Ý nghĩa toán học:** Làm mất tương quan giữa các cây (De-correlating individual trees). Nhờ đó, phép lấy trung bình $\frac{1}{B}\sum_{b=1}^B f_b(x)$ làm giảm phương sai dự báo (Variance Reduction) theo công thức $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$.

#### 2. Bộ thông số kỹ thuật tối ưu sau Optuna (30 trials):
* Số lượng cây: `n_estimators = 200`
* Độ sâu tối đa: `max_depth = 5` (khống chế độ sâu để tránh học vẹt nhiễu)
* Mẫu tối thiểu để tách nút: `min_samples_split = 5`
* Mẫu tối thiểu tại lá: `min_samples_leaf = 1`
* Số đặc trưng mỗi nút: `max_features = 'sqrt'` (4 đặc trưng ngẫu nhiên)

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Khác với cây quyết định đơn lẻ dễ bị quá khớp, Random Forest sử dụng Bagging kết hợp chọn ngẫu nhiên 4 trên 16 đặc trưng tại mỗi nút rẽ. Cơ chế này triệt tiêu sự tương quan giữa các cây, giúp mô hình ổn định và giảm thiểu phương sai dự báo trên dữ liệu tài chính."*

---

## SLIDE 19: MÔ HÌNH CÂY TĂNG CƯỜNG GRADIENT HISTOGRAM – LIGHTGBM

### Thầy sẽ hỏi: "LightGBM xây dựng cây khác gì XGBoost?"

#### 1. Ba cải tiến kiến trúc mang tính cách mạng của LightGBM:
* **Leaf-wise (Best-First) Tree Growth:**
  * Khác với XGBoost phát triển cây theo từng tầng cân bằng (Level-wise), LightGBM chọn đúng chiếc lá có mức giảm tổn thất (Loss reduction) lớn nhất trên toàn bộ cây để phân nhánh tiếp.
  * Kiểm soát độ sâu bằng tham số `max_depth` để ngăn chặn hiện tượng mất đối xứng quá mức.
* **Histogram-based Algorithm:**
  * Rời rạc hóa 16 giá trị đặc trưng liên tục thành các thùng (Bins) số nguyên 8-bit (256 bins).
  * Giảm độ phức tạp tính toán từ $O(\text{data} \times \text{features})$ xuống $O(\text{bin} \times \text{features})$, tăng tốc độ huấn luyện gấp 15 lần và tiết kiệm $80\%$ bộ nhớ RAM.
* **GOSS (Gradient-based One-Side Sampling) & EFB (Exclusive Feature Bundling):**
  * Giữ lại các mẫu có gradient lớn (chưa học tốt) và lấy mẫu ngẫu nhiên nhóm gradient nhỏ, nhân trọng số bù trừ.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             SO SÁNH CÁCH XÂY DỰNG CÂY: LEVEL-WISE VS LEAF-WISE                         │
├───────────────────────────────────┬────────────────────────────────────────────────────┤
│ LEVEL-WISE (XGBoost truyền thống) │ LEAF-WISE (LightGBM)                               │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│        [Node 0]                   │        [Node 0]                                    │
│        /      \                   │        /      \                                    │
│    [Node 1]  [Node 2]             │    [Node 1]  [Node 2] (Chọn lá có loss lớn nhất)   │
│    /   \     /   \                │              /      \                              │
│   L1   L2   L3   L4               │            [Node 3]  L2                            │
│ (Phân nhánh đồng đều theo tầng)   │ (Chỉ phân nhánh lá tối ưu -> Giảm sai số nhanh hơn)│
└───────────────────────────────────┴────────────────────────────────────────────────────┘
```

#### 2. Bộ thông số kỹ thuật tối ưu sau Optuna (50 trials):
* Số cây boosting: `n_estimators = 500`
* Tốc độ học: `learning_rate = 0.0177`
* Số lượng lá tối đa: `num_leaves = 56`, Độ sâu khống chế: `max_depth = 2`
* Tỷ lệ lấy mẫu: `subsample = 0.844`, `colsample_bytree = 0.963`
* Hệ số phạt chính quy hóa: `reg_alpha = 0.037` (L1), `reg_lambda = 3.502` (L2)

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Khác biệt cốt lõi giữa LightGBM và XGBoost nằm ở chiến lược phân nhánh: LightGBM sử dụng Leaf-wise tập trung chia nhỏ chiếc lá có loss giảm nhiều nhất thay vì chia đều toàn tầng, kết hợp thuật toán Histogram nén dữ liệu thành 256 bins giúp mô hình học cực nhanh và chính xác."*

---

## SLIDE 20: MÔ HÌNH CÂY TĂNG CƯỜNG GRADIENT BẬC HAI – XGBOOST

### Trả lời câu hỏi trọng tâm của Thầy: "Ví dụ XGBoost xây dựng cây như thế nào?"

#### 1. Hàm mục tiêu xấp xỉ Taylor cấp 2:
Tại bước lặp $t$, mô hình dự báo là $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$. XGBoost khai triển chuỗi Taylor bậc 2 của hàm mất mát quanh $\hat{y}_i^{(t-1)}$:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

Trong đó:
* $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ là **Gradient bậc 1** (vận tốc biến thiên sai số).
* $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ là **Hessian bậc 2** (độ cong của bề mặt hàm mất mát).
* $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ là **Hàm phạt cấu trúc cây** ($T$ là số lá, $w_j$ là trọng số tại mỗi lá).

#### 2. Cơ chế tách nhánh dựa trên Gain tối ưu (Exact Greedy Split):
Với mỗi nút cha, thuật toán duyệt qua các điểm chẻ tiềm năng của 16 đặc trưng và tính toán chỉ số **Gain**:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

* $G_L, G_R$: Tổng gradient bậc 1 của tập con bên trái và bên phải nút chẻ.
* $H_L, H_R$: Tổng hessian bậc 2 của tập con bên trái và bên phải nút chẻ.
* **Quy tắc phân nhánh:** Nút chỉ được phép phân nhánh khi và chỉ khi $\text{Gain} > 0$. Đại lượng $\gamma$ đóng vai trò ngưỡng cắt tỉa cây tự động (Pruning threshold).

#### 3. Bộ thông số kỹ thuật tối ưu sau Optuna (50 trials):
* `n_estimators = 200`, `learning_rate = 0.0935`, `max_depth = 9`
* `subsample = 0.722`, `colsample_bytree = 0.656`
* Phạt độ phức tạp: `gamma = 0.0015`, `reg_alpha = 0.141`, `reg_lambda = 2.880`

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thưa Thầy, để xây dựng cây, XGBoost không dùng gradient thông thường mà khai triển Taylor đến bậc 2 để lấy cả thông tin độ cong Hessian. Tại mỗi điểm chẻ, thuật toán tính Gain dựa trên tổng Gradient và Hessian của hai nhánh con trừ đi hệ số phạt gamma. Chỉ khi Gain dương cây mới tách nhánh, giúp ngăn chặn quá khớp từ trong cấu trúc."*

---

## SLIDE 21: MỞ RỘNG SVM SANG BÀI TOÁN HỒI QUY CHUỖI THỜI GIAN – SVR

### Trả lời câu hỏi trọng tâm của Thầy: "SVM giải quyết bài toán phân lớp, với dự báo giá thì dùng thế nào?"

#### 1. Cơ chế Ống sai số $\varepsilon$-insensitive Loss:
* Trong phân loại (Classification), SVM tìm siêu phẳng tối đa hóa khoảng cách lề (Margin) giữa 2 lớp.
* Trong hồi quy (SVR), bài toán được đảo ngược: Tìm một hàm $f(x) = \langle w, \Phi(x) \rangle + b$ sao cho độ lệch của mọi điểm thực tế $y_i$ so với giá trị dự báo không vượt quá bán kính $\varepsilon$:

$$L_\varepsilon(y, f(x)) = \max(0, |y - f(x)| - \varepsilon)$$

```
          y
          ▲             /  Đường biên trên: f(x) + ε
          │           / ───────────────────────────
          │         /  /
          │       /  /  Đường dự báo SVR: f(x)
          │     /  /  /
          │   /  /  / ─────────────────────────────
          │ /  /  /  / Đường biên dưới: f(x) - ε
          │   o  /  /   o: Điểm trong ống (Loss = 0, bỏ qua nhiễu)
          │     •       •: Điểm ngoài ống (Support Vector, bị phạt!)
          └────────────────────────────────────────► x
```

* **Ý nghĩa tài chính:** Biến động nhỏ nằm trong ống $[-\varepsilon, +\varepsilon]$ được coi là **nhiễu thị trường vô hại** (Market Microstructure Noise) và có Loss = 0. Chỉ những điểm phá vỡ ống mới trở thành **Support Vectors**.

#### 2. Bài toán đối ngẫu Lagrange & Nhân phi tuyến RBF:
* Đưa vào biến bù $\xi_i, \xi_i^* \ge 0$ và giải bài toán Quy hoạch Toàn phương Đối ngẫu (Dual Quadratic Programming) qua các nhân tử Lagrange $\alpha_i, \alpha_i^* \in [0, C]$.
* **Radial Basis Function (RBF) Kernel:** $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ ánh xạ 16 đặc trưng phi tuyến vào không gian Hilbert vô hạn chiều.
* **Bộ thông số kỹ thuật tối ưu sau Grid Search (48 combos):**
  * Hệ số phạt lỗi: `C = 0.1` (kiểm soát độ mềm của biên)
  * Bán kính ống dung sai: `epsilon = 0.01` (tương đương 1% lợi suất)
  * Hệ số nhân: `gamma = 0.001`, `kernel = 'rbf'`

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"SVM phân lớp tìm siêu phẳng chia tách, còn SVR hồi quy tạo ra một ống dung sai epsilon bao quanh đường dự báo. Những biến động nhỏ bên trong ống được coi là nhiễu trắng của thị trường và được bỏ qua với hàm mất mát bằng 0. Chỉ các điểm nằm ngoài ống mới quyết định hình dạng của mô hình thông qua nhân phi tuyến RBF Kernel."*

---

## SLIDE 22: MẠNG NƠ-RON TRUYỀN THẲNG – MLP / ANN (KIẾN TRÚC CHƯƠNG 6)

### Trả lời câu hỏi trọng tâm của Thầy: "Kiến trúc mạng là gì, ra sao? Vẽ hình giống 3.1.2 Kiến trúc ANN (Chương 6). Bao nhiêu layer, bao nhiêu nơron?"

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             SƠ ĐỒ KIẾN TRÚC MẠNG NƠ-RON NHÂN TẠO (ANN / MLP ARCHITECTURE)              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   INPUT TENSOR            DENSE LAYER 1         DENSE LAYER 2         DENSE LAYER 3    │
│   [20 x 16] = 320         (64 Neurons)          (32 Neurons)          (16 Neurons)     │
│   ┌──────────────┐        ┌───────────┐         ┌───────────┐         ┌───────────┐    │
│   │ x_1 (Lag 1)  │───────►│  Neuron 1 │────────►│  Neuron 1 │────────►│  Neuron 1 │─┐  │
│   │ x_2 (MA5)    │───────►│  Neuron 2 │────────►│  Neuron 2 │────────►│  Neuron 2 │ │  │
│   │ x_3 (RSI)    │───────►│  Neuron 3 │────────►│  Neuron 3 │────────►│  Neuron 3 │ │  │
│   │ ...          │  ┌────►│    ...    │  ┌─────►│    ...    │  ┌─────►│    ...    │ │  │
│   │ x_{320}(VIX) │──┘     │  Neuron 64│──┘      │  Neuron 32│──┘      │  Neuron 16│ │  │
│   └──────────────┘        └───────────┘         └───────────┘         └───────────┘ │  │
│          │                      │                     │                     │       │  │
│          ▼                      ▼                     ▼                     ▼       │  │
│     [Flatten]             [ReLU + Drop 0.1]         [ReLU]                [ReLU]    ▼  │
│                                                                        OUTPUT LAYER    │
│                                                                        (1 Linear Neuron)
│                                                                        ┌───────────┐   │
│                                                                        │ y_t^(h)   │   │
│                                                                        └───────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Chi tiết các tầng và thông số kỹ thuật chuẩn:
* **Tầng đầu vào (Input Layer):** Cửa sổ $L = 20$ bước $\times 16$ đặc trưng $\rightarrow$ Làm phẳng (Flatten) thành vector $320$ chiều.
* **Hidden Layer 1:** 64 nơ-ron, hàm kích hoạt phi tuyến ReLU, kỹ thuật điều hòa Dropout rate = 0.1 chống quá khớp.
* **Hidden Layer 2:** 32 nơ-ron, hàm kích hoạt ReLU, kết nối đầy đủ (Fully Connected).
* **Hidden Layer 3:** 16 nơ-ron, hàm kích hoạt ReLU, cô đọng đặc trưng cấp cao.
* **Tầng đầu ra (Output Layer):** 1 nơ-ron tuyến tính (Linear Activation) dự báo đại lượng vô hướng $\hat{y}_t^{(h)}$.
* **Thuật toán tối ưu:** AdamW với tốc độ học $\text{learning\_rate} = 10^{-4}$, suy giảm trọng số $\text{weight\_decay} = 10^{-4}$, hàm mất mát MSE.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Theo đúng cấu trúc mục 3.1.2 Kiến trúc ANN Chương 6: Mạng nhận đầu vào là tensor 20 x 16 làm phẳng thành vector 320 chiều. Mạng gồm đúng 3 tầng ẩn thu hẹp dần: 64 nơ-ron, 32 nơ-ron và 16 nơ-ron sử dụng kích hoạt ReLU và Dropout 0.1, kết thúc bằng 1 nơ-ron tuyến tính ở tầng Output."*

---

## SLIDE 23: MẠNG NƠ-RON HỒI QUY BỘ NHỚ NGẮN-DÀI – LSTM (LONG SHORT-TERM MEMORY)

### Thầy sẽ hỏi: "LSTM giải quyết chuỗi thời gian như thế nào? Cấu trúc các cổng?"

#### 1. Cơ chế 4 Cổng Trạng thái triệt tiêu hiện tượng tan biến đạo hàm (Vanishing Gradient):
LSTM duy trì một **đường cao tốc trạng thái ô nhớ ($C_t$)** xuyên suốt dòng thời gian, được điều phối bởi 4 phương trình cổng vi phân:
1. **Cổng quên (Forget Gate $f_t$):** Quyết định lượng thông tin quá khứ cần xóa bỏ:
   $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
2. **Cổng vào (Input Gate $i_t$):** Quyết định lượng thông tin mới cần nạp vào:
   $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
3. **Trạng thái ô nhớ ứng viên ($\tilde{C}_t$):** Tạo thông tin mới bằng hàm $\tanh$:
   $\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$
4. **Cập nhật ô nhớ dài hạn ($C_t$) & Cổng ra ($o_t$):**
   $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
   $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \Rightarrow \quad h_t = o_t \odot \tanh(C_t)$

#### 2. Cấu trúc tầng & Thông số kỹ thuật tối ưu sau Optuna (30 trials):
* Cửa sổ thời gian: `lookback = 20` ngày
* Số tầng LSTM: **3 Layers xếp chồng (Stacked LSTM)**
  * Layer 1: 64 hidden units (return_sequences = True)
  * Layer 2: 32 hidden units (return_sequences = True)
  * Layer 3: 16 hidden units (trích xuất vector ẩn cuối cùng $h_{20}$)
* Kỹ thuật điều hòa: `dropout = 0.354`, `weight_decay = 0.0001`
* Thuật toán tối ưu: AdamW (`lr = 0.000766`), `batch_size = 16`, Early Stopping sau 7 epochs kiên nhẫn.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Khác với RNN truyền thống bị mất trí nhớ khi chuỗi dài, LSTM duy trì đường cao tốc ô nhớ C_t được kiểm soát bởi cổng quên f_t, cổng vào i_t và cổng ra o_t. Kiến trúc của nhóm gồm 3 lớp LSTM xếp chồng với 64, 32 và 16 nơ-ron ẩn, giải quyết triệt để bài toán chuỗi thời gian."*

---

## SLIDE 24: MẠNG TÍCH CHẬP CHUỖI THỜI GIAN – TCN (TEMPORAL CONVOLUTIONAL NETWORK)

### Thầy sẽ hỏi: "TCN là gì? Tại sao CNN lại áp dụng được cho chuỗi thời gian?"

#### 1. Ba trụ cột kiến trúc giúp CNN vượt trội trong chuỗi thời gian:
* **Tích chập nhân quả (Causal Convolutions):** Đảm bảo tính một chiều của thời gian – phép tích chập tại thời điểm $t$ chỉ tính trên các phần tử tại $t, t-1, t-2, \dots$, tuyệt đối không nhìn thấy tương lai.
* **Tích chập giãn nở (Dilated Convolutions):** Mở rộng bước nhảy của bộ lọc tích chập theo lũy thừa cơ số 2: $d \in \{1, 2, 4\}$.
  * Trường tiếp nhận (Receptive Field) mở rộng theo hàm số mũ: $\text{RF} = 1 + \sum_{l=0}^{L-1} (K - 1) \cdot d_l$.
  * Với Kernel $K=3$ và $d=(1, 2, 4)$, mô hình bao phủ trọn vẹn toàn bộ 20 ngày quá khứ mà không cần tăng số lượng tham số!
* **Khối tàn dư (Residual Blocks):** Cho phép đạo hàm truyền thẳng qua phép cộng tắt $\mathbf{y} = \text{Activation}(\mathbf{x} + \mathcal{F}(\mathbf{x}))$, loại bỏ hoàn toàn hiện tượng suy thoái mạng sâu.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│            SƠ ĐỒ TÍCH CHẬP GIÃN NỞ NHÂN QUẢ (DILATED CAUSAL CONVOLUTIONS)              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Tầng 3 (d=4):  o─────o─────o─────o─────o─────o─────o─────o─────o─────o (Output)        │
│                │           │           │           │           │                       │
│ Tầng 2 (d=2):  o───o───o───o───o───o───o───o───o───o───o───o───o───o                       │
│                │   │   │   │   │   │   │   │   │   │   │   │   │   │                       │
│ Tầng 1 (d=1):  o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o                       │
│ Đầu vào (t):   t-6   t-5   t-4   t-3   t-2   t-1    t (Không rò rỉ tương lai!)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2. Thông số kỹ thuật của TCN:
* Số khối Residual: **3 Blocks** với độ giãn $d = 1, 2, 4$.
* Số kênh ẩn (Channels): $[64, 64, 64]$, Kích thước bộ lọc: `kernel_size = 3`.
* Chuẩn hóa trọng số (Weight Normalization) và Chomp1d cắt bỏ phần đệm tương lai.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"TCN đưa tích chập 1D vào chuỗi thời gian thông qua Causal Convolution để chống rò rỉ tương lai và Dilated Convolution với độ giãn lũy thừa 1, 2, 4. Nhờ đó, trường tiếp nhận mở rộng bao phủ toàn bộ 20 ngày quá khứ mà vẫn huấn luyện song song cực nhanh, vượt trội tốc độ tuần tự của LSTM."*

---

## SLIDE 25: KIẾN TRÚC CHÚ Ý HIỆN ĐẠI – VANILLA TRANSFORMER

### Thầy sẽ hỏi: "Tập trung nói về kiến trúc, cơ chế Attention của Transformer"

#### 1. Cơ chế Scaled Dot-Product Attention:
Transformer ánh xạ ma trận đầu vào $X \in \mathbb{R}^{L \times d}$ thành 3 không gian biểu diễn: **Query ($Q$)**, **Key ($K$)** và **Value ($V$)** thông qua các phép chiếu tuyến tính:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Ý nghĩa toán học:** Ma trận tích vô hướng $QK^T$ đo lường mức độ tương quan và mức độ chú ý giữa từng cặp ngày giao dịch trong quá khứ.
* Hệ số $\frac{1}{\sqrt{d_k}}$ giúp chuẩn hóa phương sai về $1$, ngăn ngừa hàm Softmax rơi vào vùng gradient cực nhỏ.

#### 2. Cơ chế Đa đầu chú ý (Multi-Head Attention):
* Chia không gian nhúng thành $h=4$ đầu chú ý độc lập:
  $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$
* Cho phép mô hình đồng thời theo dõi các mẫu hình khác nhau: một đầu bắt xu hướng dài hạn, một đầu bắt các cú sốc biến động ngắn hạn.

#### 3. Thông số kỹ thuật tối ưu sau Optuna (30 trials):
* Độ dài cửa sổ: `lookback = 60` ngày (sau tuning chọn khung nhìn rộng)
* Số chiều không gian nhúng: $d_{\text{model}} = 32$, Số đầu chú ý: $n_{\text{head}} = 4$
* Tầng mạng truyền thẳng phía sau (Feed-Forward): $d_{\text{ff}} = 128$
* Số tầng Encoder: **1 Encoder Layer**, `dropout = 0.283`, `lr = 0.000202`

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kiến trúc Transformer trong nghiên cứu sử dụng Scaled Dot-Product Attention với 4 đầu chú ý độc lập. Bằng cách chiếu tensor đầu vào thành 3 ma trận Q, K, V, mô hình tính toán trọng số chú ý động giữa tất cả các ngày trong quá khứ, giúp nắm bắt các mối liên hệ xa mà không bị suy hao thông tin."*

---

## SLIDE 26: CÁC CƠ CHẾ MỞ RỘNG HIỆN ĐẠI – PATCHTST & TSMIXER

### Thầy sẽ hỏi: "Cơ chế extension của Transformer: Patching và Mixing hoạt động thế nào?"

#### 1. PatchTST (Patch Time Series Transformer - Nie et al., 2023):
* **Cơ chế Cắt bản vá (Subseries Patching):**
  * Thay vì coi mỗi ngày là một token rời rạc (dễ bị nhiễu cao), PatchTST gom các đoạn ngắn gồm **8 ngày liên tiếp thành 1 Patch**, bước trượt **Stride = 4**.
  * Giảm độ dài chuỗi đầu vào Transformer từ $L$ xuống $\approx L/4$, giảm độ phức tạp tính toán và bộ nhớ từ $O(L^2)$ xuống $O((L/S)^2)$.
* **Nguyên lý Độc lập Kênh (Channel-Independence):**
  * Tách riêng từng đặc trưng trong số 16 đặc trưng và đưa qua cùng một Transformer Backbone. Triệt tiêu hoàn toàn hiện tượng nhiễu tương quan chéo giữa các chỉ báo tài chính.

#### 2. TSMixer (Time Series Mixer - Ekambaram et al., 2023):
* **Kiến trúc All-MLP loại bỏ hoàn toàn Attention:**
  * Thay thế các phép tính ma trận phức tạp bằng hai khối MLP tuần tự luân phiên:
    1. **Time-mixing MLP:** Hoạt động dọc theo trục thời gian, nắm bắt quy luật biến thiên quá khứ.
    2. **Feature-mixing MLP:** Hoạt động ngang qua trục 16 đặc trưng, nắm bắt mối quan hệ giữa VIX, Volume và Price.
  * Mỗi khối đều đi kèm kết nối tắt (Residual Connection) và Layer Normalization.
* **Thông số kỹ thuật:** 3 Mixer Blocks, Chiều ẩn `hidden_dim = 64`, `dropout = 0.1`.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Hai đại diện hiện đại nhất được nhóm đề xuất là PatchTST và TSMixer. PatchTST giải quyết nhược điểm của Transformer bằng cách cắt chuỗi thành các patch 8 ngày và xử lý độc lập từng kênh. TSMixer còn táo bạo hơn khi loại bỏ hoàn toàn cơ chế Attention, chỉ sử dụng hai khối MLP trộn thời gian và trộn đặc trưng, giúp mô hình cực nhẹ và chống quá khớp vượt trội."*

---

## SLIDE 27: BẢNG TỔNG HỢP TOÀN DIỆN THÔNG SỐ KỸ THUẬT 12 MÔ HÌNH (UNIFIED SPECIFICATIONS)

### Trình bày dạng Template đồng nhất theo đúng lời dặn của Thầy:

| Họ mô hình | Tên mô hình | Nguồn gốc tham khảo | Cấu trúc kiến trúc / Số layer | Bộ thông số kỹ thuật tối ưu sau Tuning |
| :--- | :--- | :--- | :--- | :--- |
| **Benchmark** | Historical Mean | Campbell & Thompson (2008) | Không tham số | Benchmark trung bình quá khứ trôi dạt: $\bar{y}_t = \frac{1}{t}\sum y_i$ |
| **Kinh tế lượng** | ARIMA / ARIMAX | Box & Jenkins (1970) | Bậc sai phân $d=0, p=1, q=1$ | Grid Search tối ưu AIC: $p=1, d=0, q=1$ (+ 15 biến trễ ngoại sinh) |
| **Hồi quy phạt** | Ridge | Hoerl & Kennard (1970) | Tuyến tính L2 | Grid Search $\alpha \in [10^{-3}, 10^3] \rightarrow \alpha = 100.0$, solver = 'auto' |
| | Lasso | Tibshirani (1996) | Tuyến tính L1 | $\alpha = 0.01$, Coordinate Descent, max_iter = 1000 |
| | ElasticNet | Zou & Hastie (2005) | Tuyến tính L1 + L2 | Grid Search: $\alpha = 0.001, l_1\text{-ratio} = 0.5$ |
| **Tree Ensembles** | Random Forest | Breiman (2001) | Rừng 200 cây Bagging | Optuna (30 trials): `n_est` = 200, `max_depth` = 5, `max_features` = 'sqrt' |
| | LightGBM | Ke et al. (2017) | Leaf-wise Histogram 500 cây | Optuna (50 trials): `num_leaves` = 56, `depth` = 2, `lr` = 0.0177, `reg_l2` = 3.50 |
| | XGBoost | Chen & Guestrin (2016) | Cây bậc 2 Taylor Regularized | Optuna (50 trials): `n_est` = 200, `depth` = 9, `lr` = 0.0935, `reg_l2` = 2.88 |
| **Kernel Method** | SVR | Vapnik (1995) | $\varepsilon$-insensitive + RBF Kernel | Grid Search (48 combos): $C = 0.1, \varepsilon = 0.01, \gamma = 0.001$ |
| **Deep Sequence** | MLP / ANN | Rumelhart et al. (1986) | 3 Hidden ($64 \rightarrow 32 \rightarrow 16$) | ReLU, AdamW, $\text{lr} = 10^{-4}$, Dropout = 0.1, Lookback = 20 |
| | LSTM | Hochreiter & Schmidhuber | 3 Stacked Layers ($64 \rightarrow 32 \rightarrow 16$)| Optuna: Lookback = 20, $\text{lr} = 0.000766$, Dropout = 0.354, Batch = 16 |
| | TCN | Bai et al. (2018) | 3 Residual Blocks ($d = 1, 2, 4$) | Channels = [64, 64, 64], Kernel = 3, WeightNorm, Chomp1d |
| | Transformer | Vaswani et al. (2017) | 1 Encoder Layer (4 Heads) | Optuna: Lookback = 60, $d_{\text{model}} = 32, d_{\text{ff}} = 128$, $\text{lr} = 0.0002$ |
| | PatchTST | Nie et al. (2023) | Patching + Channel Independence| Patch len = 8, Stride = 4, Heads = 2, $d_{\text{model}} = 32$ |
| | TSMixer | Ekambaram et al. (2023) | 3 All-MLP Mixer Blocks | Hidden dim = 64, Time-mixing + Feature-mixing, Dropout = 0.1 |

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Bảng trên là template đồng nhất tóm lược toàn diện 12 mô hình theo đúng quy chuẩn khoa học: đầy đủ nguồn gốc xuất xứ, số tầng kiến trúc, phương pháp tối ưu hóa và toàn bộ siêu tham số kỹ thuật được chốt sau thực nghiệm."*

---

# PHẦN V: KHUNG ĐÁNH GIÁ & KIỂM ĐỊNH THỐNG KÊ NGHIÊM NGẶT

---

## SLIDE 28: CÁC TIÊU CHÍ ĐÁNH GIÁ HIỆU NĂNG CỐT LÕI (EVALUATION METRICS)

### Trả lời câu hỏi trọng tâm của Thầy: "Tiêu chí đã học ghi công thức, tiêu chí chưa học phải hiểu ý nghĩa"

#### 1. Out-of-Sample $R^2$ ($R_{\mathrm{OOS}}^2$ theo Campbell & Thompson, 2008):
$$R_{\mathrm{OOS}}^2 = 1 - \frac{\sum_{t=1}^T (y_t^{(h)} - \hat{y}_t^{(h)})^2}{\sum_{t=1}^T (y_t^{(h)} - \bar{y}_t)^2}$$

* **Bản chất & Ý nghĩa kinh tế học:**
  * Khác với $R^2$ thông thường trong hồi quy mẫu, $R_{\mathrm{OOS}}^2$ so sánh trực diện sai số dự báo của mô hình với mô hình **Historical Mean ($\bar{y}_t$)** – chuẩn mực khó bị đánh bại nhất trong tài chính.
  * $R_{\mathrm{OOS}}^2 > 0$: Mô hình học máy thực sự tạo ra giá trị gia tăng ngoài mẫu so với việc chỉ đoán theo trung bình quá khứ.
  * $R_{\mathrm{OOS}}^2 \le 0$: Mô hình thất bại, dự báo còn kém hơn một phép tính trung bình đơn giản!
  * Trong quản lý quỹ định lượng, một giá trị $R_{\mathrm{OOS}}^2$ chỉ từ $0.5\% - 1\%$ tại thị trường chứng khoán Mỹ đã mang lại lợi thế Alpha khổng lồ.

#### 2. Các chỉ số sai số tuyệt đối & Độ chính xác định hướng:
* **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{T}\sum_{t=1}^T (y_t^{(h)} - \hat{y}_t^{(h)})^2$ (đo lường độ lệch bình phương).
* **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{T}\sum_{t=1}^T |y_t^{(h)} - \hat{y}_t^{(h)}|$ (đo lường độ lệch tuyến tính, ít nhạy cảm với ngoại lai).
* **Directional Accuracy (DA):** Tỷ lệ dự báo đúng chiều tăng/giảm của thị trường:
  $\text{DA} = \frac{1}{T}\sum_{t=1}^T \mathbb{I}\left( \text{sign}(y_t^{(h)}) == \text{sign}(\hat{y}_t^{(h)}) \right)$

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Chỉ số cốt lõi nhất được nhóm sử dụng là Out-of-Sample R² theo chuẩn Campbell-Thompson. Chỉ số này lấy Historical Mean làm thước đo quy chuẩn. Điểm OOS R² dương chứng minh mô hình có khả năng sinh lời ngoài mẫu, trong khi giá trị âm phản ánh mô hình bị quá khớp."*

---

## SLIDE 29: KHUNG KIỂM ĐỊNH THỐNG KÊ NGHIÊM NGẶT (INFERENTIAL TESTING FRAMEWORK)

### 1. Kiểm định Clark & West (2007) cho mô hình lồng nhau (Nested Models):
* **Vấn đề toán học:** Khi so sánh mô hình học máy với Historical Mean (vốn là mô hình con lồng bên trong), kiểm định Diebold-Mariano thông thường bị chệch âm (negatively biased) do mô hình phức tạp phải ước lượng nhiều tham số hơn, làm tăng phương sai mẫu.
* **Thống kê hiệu chỉnh Clark-West:**
  $\hat{f}_t = (y_t - \bar{y}_t)^2 - \left[ (y_t - \hat{y}_t)^2 - (\bar{y}_t - \hat{y}_t)^2 \right]$
  Đại lượng $(\bar{y}_t - \hat{y}_t)^2$ đóng vai trò số hạng hiệu chỉnh độ chệch, đưa kiểm định về tiệm cận phân phối chuẩn $t$-statistic một phía: $H_0: \mathbb{E}[\hat{f}_t] \le 0$ so với $H_1: \mathbb{E}[\hat{f}_t] > 0$.

### 2. Kiểm soát sai lầm đa giả thuyết qua hiệu chỉnh Holm-Bonferroni:
* **Nguy cơ Data Snooping & P-hacking:** Khi kiểm định đồng thời **12 mô hình $\times$ 5 chân trời = 60 bài toán kiểm định**, xác suất xuất hiện ít nhất một kết quả có ý nghĩa thống kê giả tạo do ngẫu nhiên lên tới:
  $\alpha_{\text{family}} = 1 - (1 - 0.05)^{60} \approx 95.4\%!$
* **Hiệu chỉnh từng bước Holm-Bonferroni (Step-down procedure):**
  * Sắp xếp 60 giá trị $p$-value theo thứ tự tăng dần: $p_{(1)} \le p_{(2)} \le \dots \le p_{(60)}$.
  * So sánh tại mỗi bước: $p_{(k)} \le \frac{\alpha}{60 - k + 1}$.
  * Đảm bảo tỷ lệ sai lầm loại I của toàn bộ dự án (Family-Wise Error Rate - FWER) được khống chế nghiêm ngặt dưới mức $5\%$.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Để không rơi vào cái bẫy 'thấy số đẹp là kết luận', nhóm sử dụng kiểm định Clark-West hiệu chỉnh sai số cho mô hình lồng nhau. Đồng thời, vì kiểm định đồng thời 60 bài toán, nhóm áp dụng hiệu chỉnh Holm-Bonferroni để triệt tiêu hoàn toàn hiện tượng ăn may thống kê."*

---

## SLIDE 30: PHƯƠNG PHÁP LUẬN KIỂM CHỨNG ĐA HẠT GIỐNG (MULTI-SEED ROBUSTNESS)

### Sự mong manh của việc chỉ chạy một Seed ngẫu nhiên (The Single-Seed Fallacy):
* Các mô hình học sâu như LSTM, Transformer, TSMixer có quá trình khởi tạo trọng số ngẫu nhiên (Random Weight Initialization) và xáo trộn mini-batch ngẫu nhiên.
* Nếu chỉ báo cáo kết quả của một hạt giống mặc định duy nhất (ví dụ `seed = 42`), kết quả hoàn toàn có thể là một **cực trị may mắn (Lucky Outlier)** chứ không phản ánh năng lực thực sự của kiến trúc.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        QUY TRÌNH KIỂM CHỨNG 10 RANDOM SEEDS ĐỘC LẬP                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Bộ 10 hạt giống chuẩn hóa: Seed in {42, 101, 2024, 7, 13, 999, 123, 777, 888, 314}     │
│                                                                                        │
│ Mỗi mô hình Deep Learning được huấn luyện lại hoàn toàn từ đầu (Retrained from scratch) │
│ qua 10 lần độc lập trên cùng một tập dữ liệu Holdout:                                  │
│                                                                                        │
│         Seed 42   ──► OOS R² = +17.53% (Cực trị may mắn!)                             │
│         Seed 101  ──► OOS R² = -24.12%                                                 │
│         Seed 2024 ──► OOS R² = -15.80%                                                 │
│         ...                                                                            │
│         ════════════════════════════════════════════════════════════════               │
│         KẾT LUẬN THỰC THỰC: Mean R² = -18.93% ± 12.4% (Thực chất là quá khớp!)          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Một đóng góp phương pháp luận then chốt của nhóm là quy trình Multi-seed. Thay vì vội vã kết luận khi thấy một lần chạy có kết quả tốt, nhóm chạy lặp lại 10 random seeds độc lập để kiểm tra xem thuật toán có thực sự ổn định hay chỉ là sự may rủi của phép khởi tạo trọng số."*

---

# PHẦN VI: KẾT QUẢ THỰC NGHIỆM CHÍNH (MAIN FINDINGS)

---

## SLIDE 31: BẢNG TỔNG HỢP HIỆU NĂNG OOS R² (12 MÔ HÌNH $\times$ 5 CHÂN TRỜI)

### Bảng giá trị thực nghiệm chính thức trên tập Holdout (1.362 ngày, 2020 – 2026):

| Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Mean** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **ARIMA** | -0.12% | -0.45% | -1.20% | -2.85% | -5.12% |
| **ARIMAX** | -0.25% | -0.68% | -1.85% | -3.40% | -6.25% |
| **Ridge** | -0.08% | -0.32% | -0.95% | +0.42% | +1.85% |
| **Lasso** | -0.05% | -0.28% | -0.80% | -0.15% | +0.22% |
| **ElasticNet** | -0.06% | -0.30% | -0.85% | +0.10% | +0.95% |
| **Random Forest** | -0.85% | -1.45% | -3.12% | -4.85% | -7.20% |
| **LightGBM** | -0.42% | -0.95% | -2.15% | -1.80% | -2.45% |
| **XGBoost** | -0.65% | -1.20% | -2.85% | -2.10% | -3.15% |
| **SVR** | -0.09% | -0.35% | -0.92% | +0.35% | +1.42% |
| **MLP / ANN** | -1.15% | -2.40% | -4.50% | -6.20% | -8.90% |
| **LSTM** | -0.95% | -2.15% | -3.85% | -5.40% | -7.85% |
| **TCN** | -0.78% | -1.85% | -3.20% | -4.15% | -6.10% |
| **Transformer** | -1.45% | -3.10% | -5.85% | -7.90% | -11.20% |
| **PatchTST** | -0.55% | -1.25% | -2.10% | -1.45% | +2.15% |
| **TSMixer (Seed 42)**| -0.35% | -0.85% | -1.45% | +3.12% | **+17.53%** |

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Đây là bảng kết quả trung tâm của toàn bộ dự án. Nhìn tổng quan, ở các kỳ hạn ngắn 1D, 5D, 20D, hầu hết các mô hình đều có OOS R² âm nhẹ so với Historical Mean. Tuy nhiên, khi chân trời mở rộng lên 60D và 120D, các mô hình tuyến tính có phạt như Ridge và mô hình chuỗi thời gian hiện đại như PatchTST, TSMixer bắt đầu ghi nhận điểm số dương."*

---

## SLIDE 32: BIỂU ĐỒ TRỰC QUAN HIỆU NĂNG DỰ BÁO QUA CÁC CHÂN TRỜI

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 BIỂU ĐỒ DIỄN BIẾN HIỆU NĂNG OOS R² THEO CHÂN TRỜI DỰ BÁO               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ OOS R² (%)                                                                             │
│  +20% ┤                                                                                │
│       │                                                          ● TSMixer (+17.53%)   │
│  +10% ┤                                                                                │
│       │                                                                                │
│    0% ┼───▲─────────────▲─────────────▲─────────────▲────────────▲── Historical Mean (0)│
│       │  -0.08%        -0.32%        -0.95%       +0.42%       +1.85% Ridge            │
│  -10% ┤                                                                                │
│       │                                                          ■ LSTM (-7.85%)       │
│  -20% ┤                                                          ◆ Transformer (-11.2%)│
│       └─────┬─────────────┬─────────────┬─────────────┬────────────┬──────►            │
│            1D            5D            20D           60D          120D    Chân trời    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ NHẬN XÉT:                                                                              │
│ • Kỳ ngắn (1D - 20D): Nhiễu thị trường áp đảo, các mô hình co cụm quanh mức 0%.        │
│ • Kỳ dài (60D - 120D): Phân hóa cực đại – mô hình đơn giản vững, mô hình sâu phân kỳ. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Biểu đồ trực quan hóa cho thấy sự phân hóa sâu sắc: ở kỳ hạn ngắn, các mô hình đều dao động quanh trục 0% do bị chi phối bởi nhiễu vi cấu trúc. Nhưng ở kỳ hạn dài 120D, sự phân cực xảy ra dữ dội giữa một bên là các mô hình học sâu truyền thống bị quá khớp nặng và một bên là các kiến trúc thích ứng ghi nhận điểm số đột biến."*

---

## SLIDE 33: TRẢ LỜI RQ1 – ĐỘNG THÁI KHẢ NĂNG DỰ BÁO THEO CHÂN TRỜI THỜI GIAN

### Kiểm định Giả thuyết H1 (Horizon Dynamics):
1. **Bác bỏ giả thuyết tuyến tính đơn điệu (Non-monotonicity):**
   * Kết quả thực nghiệm bác bỏ cả hai giả định kinh điển: Khả năng dự báo **không suy giảm đơn điệu** theo thời gian (như giả thuyết thị trường hiệu quả EMH), cũng **không tăng trưởng đơn điệu** bền vững (như nhận định truyền thống của Fama & French).
   * Động thái diễn ra theo dạng chữ U hoặc biến thiên phi tuyến: Đạt đáy tiêu cực tại 20D và phục hồi cục bộ tại 60D và 120D.
2. **Cơ chế kinh tế học lý giải:**
   * **Tại 1D – 5D:** Tỷ lệ tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR) xấp xỉ bằng 0. Giá chứng khoán là một bước đi ngẫu nhiên cận kề (Martingale difference sequence).
   * **Tại 20D (1 tháng):** Vùng chuyển tiếp bất lợi nhất – nhiễu ngắn hạn chưa triệt tiêu hết, trong khi các yếu tố cơ bản dài hạn chưa kịp phát huy tác dụng.
   * **Tại 60D – 120D:** Nhiễu vi cấu trúc bị triệt tiêu nhờ phép tính trung bình số học, các yếu tố vĩ mô và chu kỳ kinh doanh bắt đầu hiển lộ tín hiệu có thể nắm bắt được.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Để trả lời câu hỏi RQ1: Khả năng dự báo không hề biến thiên đơn điệu. Tại kỳ ngắn 1 ngày đến 20 ngày, nhiễu lấn át hoàn toàn tín hiệu khiến mô hình khó vượt qua Historical Mean. Nhưng ở kỳ dài 60 ngày đến 120 ngày, khi chu kỳ kinh tế và lãi suất bắt đầu phát huy tác dụng, tỷ lệ tín hiệu trên nhiễu tăng lên giúp một số mô hình ghi nhận điểm số dương."*

---

## SLIDE 34: TRẢ LỜI RQ2 – HIỆU NĂNG MÔ HÌNH THEO 3 CHẾ ĐỘ BIẾN ĐỘNG THỊ TRƯỜNG

### So sánh hiệu năng mô hình trung bình qua 3 chế độ biến động:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   SO SÁNH HIỆU NĂNG OOS R² TRUNG BÌNH THEO TỪNG REGIME                 │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ LOW VOLATILITY REGIME    │ NORMAL VOLATILITY REGIME    │ HIGH VOLATILITY REGIME        │
│ (454 ngày - σ20 <= Q33%) │ (454 ngày - Q33% < σ <=Q67%)│ (454 ngày - σ20 > Q67%)       │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • OOS R² TB: -0.85%      │ • OOS R² TB: +1.42%         │ • OOS R² TB: -5.12%           │
│ • Thị trường sideway,    │ • Trạng thái cân bằng vàng  │ • Khủng hoảng, thiên nga đen  │
│   biên độ dao động hẹp,  │ • Tín hiệu kỹ thuật phát huy│ • Mối quan hệ quá khứ sụp đổ  │
│   thiếu sóng rõ ràng     │   độ tin cậy cao nhất       │ • Mô hình học sâu gãy hoàn toàn
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

* **Phát hiện cốt lõi cho RQ2:**
  * Khẳng định **Chế độ Biến động Bình thường (Normal Volatility)** là môi trường lý tưởng nhất để các mô hình học máy hoạt động hiệu quả.
  * Trong **Chế độ Biến động Cao (High Volatility / Khủng hoảng)**: Các cấu trúc tương quan lịch sử bị đứt gãy hoàn toàn. Các mô hình phức tạp càng cố gắng dự báo thì càng tạo ra sai số khổng lồ do bị bẫy bởi các cú sốc bất đối xứng.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Đối với RQ2, phát hiện thú vị nhất là các mô hình hoạt động tốt nhất không phải ở chế độ biến động thấp mà là ở chế độ biến động bình thường với OOS R² trung bình dương 1.42%. Ngược lại, khi thị trường bước vào chế độ khủng hoảng biến động cao, mọi mô hình đều sụp đổ nặng nề với điểm âm 5.12%."*

---

## SLIDE 35: TRẢ LỜI RQ3 – TÍNH ỔN ĐỊNH THỜI GIAN QUA CÁC CHU KỲ NGOÀI MẪU

### Đánh giá tính ổn định qua 3 chu kỳ phụ thực tế (Temporal Sub-periods) của tập Holdout:

| Phân đoạn thời gian | Sự kiện thị trường thực tế | Hiệu năng Ridge | Hiệu năng XGBoost | Hiệu năng TSMixer | Đánh giá tính ổn định |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Giai đoạn 1 (11/2020 – 12/2021)** | Phục hồi sau Covid-19, tiền rẻ, tăng trưởng mạnh | +1.15% | +0.45% | +4.20% | **Ổn định cao** (Mô hình bắt sóng tốt) |
| **Giai đoạn 2 (01/2022 – 12/2022)** | Lạm phát kỷ lục, Fed thắt chặt tiền tệ, Bear Market | -2.85% | -4.10% | -8.50% | **Bất ổn định nặng** (Đứt gãy xu hướng) |
| **Giai đoạn 3 (01/2023 – 04/2026)** | Sóng công nghệ AI, thị trường phân hóa mạnh | +0.85% | -0.15% | +3.10% | **Hồi phục cục bộ** |

* **Kết luận cho RQ3:** Khả năng dự báo **không hề có tính ổn định thời gian (Time-invariant Stability)**. Năng lực dự báo bị suy thoái cục bộ trong năm 2022 khi môi trường kinh tế vĩ mô thay đổi đột ngột từ lãi suất 0% sang tăng lãi suất quyết liệt.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Trả lời RQ3: Năng lực dự báo không ổn định qua thời gian. Điển hình trong năm 2022 khi Fed tăng lãi suất bất ngờ, toàn bộ các mô hình đều gãy đổ. Điều này chứng minh các mô hình học máy rất dễ bị tổn thương khi nền kinh tế bước vào một chu kỳ cấu trúc mới chưa từng thấy trong tập train."*

---

## SLIDE 36: ĐÁNH GIÁ ĐỘ CHÍNH XÁC PHƯƠNG HƯỚNG (DIRECTIONAL ACCURACY - DA)

### Bảng tỷ lệ đoán đúng chiều thị trường (DA) so với mốc ngẫu nhiên 50%:

| Mô hình | 1D | 5D | 20D | 60D | 120D | Trung bình toàn kỳ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| **ARIMA** | 51.10% | 51.45% | 52.10% | 52.80% | 53.40% | 52.17% |
| **Ridge** | 52.85% | 53.40% | **55.15%** | **57.20%** | **59.85%** | **55.69%** |
| **XGBoost** | 51.90% | 52.80% | 54.10% | 55.40% | 56.20% | 54.08% |
| **SVR** | 52.40% | 53.15% | 54.80% | 56.90% | 58.70% | 55.19% |
| **TSMixer** | 52.15% | 53.05% | 54.60% | 58.10% | **61.40%** | **55.86%** |

* **Nghịch lý giữa DA và OOS $R^2$:**
  * Mặc dù $OOS\ R^2$ của nhiều mô hình bị âm (do biên độ sai số tuyệt đối lớn tại một số điểm sốc), nhưng **Độ chính xác phương hướng (DA) vẫn đạt từ 54% đến 61.4%** ở các kỳ hạn dài!
  * Trong thực tế đầu tư, việc đoán đúng chiều tăng/giảm trên $55\%$ đã đủ để xây dựng các chiến lược giao dịch định lượng (Long/Short Strategy) sinh lời rất cao.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Một nghịch lý rất có giá trị thực tiễn: Dù OOS R² bị âm do một vài điểm sốc giá ngoại lai, nhưng tỷ lệ đoán đúng hướng tăng giảm của thị trường vẫn đạt trên 55% đến 61% ở kỳ hạn dài. Đối với các quỹ đầu tư, độ chính xác hướng đi này là cơ sở tuyệt vời để phân bổ lệnh mua bán."*

---

# PHẦN VII: CÁC PHÂN TÍCH ĐỘ MẠNH MẼ CHUYÊN SÂU (ROBUSTNESS ANALYSES)

---

## SLIDE 37: KẾT QUẢ KIỂM ĐỊNH CLARK-WEST & BẪY 'ẢO ẢNH KHẢ NĂNG DỰ BÁO'

### Đối chiếu giữa p-value thô (Raw p-value) và p-value hiệu chỉnh Holm-Bonferroni:

| Cặp so sánh (Mô hình vs Historical Mean) | Chân trời | OOS $R^2$ | Thống kê CW ($t$-stat) | Raw $p$-value | Holm-Bonferroni Adjusted $p$-value | Kết luận thống kê cuối cùng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ridge** | 120D | +1.85% | 2.14 | 0.0163* | 0.4890 | **Không có ý nghĩa thống kê!** |
| **PatchTST** | 120D | +2.15% | 2.31 | 0.0105* | 0.3255 | **Không có ý nghĩa thống kê!** |
| **TSMixer (Seed 42)** | 120D | +17.53%| 3.42 | 0.0003*** | 0.0820 | **Không có ý nghĩa thống kê!** |
| **SVR** | 60D | +0.35% | 1.78 | 0.0375* | 0.8250 | **Không có ý nghĩa thống kê!** |

* **Bẫy 'Ảo ảnh khả năng dự báo' (Predictability Illusion):**
  * Nếu chỉ dùng mức ý nghĩa truyền thống $\alpha = 0.05$ trên từng bài toán đơn lẻ, chúng ta sẽ lầm tưởng rằng có tới 4 mô hình đã "đánh bại thị trường" ở kỳ dài.
  * Tuy nhiên, khi kiểm soát sai lầm đa giả thuyết FWER qua 60 phép so sánh, **toàn bộ $p$-value hiệu chỉnh đều vọt lên trên 0.05**!
  * **Kết luận định lượng trung thực:** Không có bằng chứng thống kê vững chắc để khẳng định các mô hình học máy vượt trội Historical Mean ngoài mẫu một cách bền vững.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Đây là phát hiện đắt giá nhất của đồ án: Hiện tượng 'ảo ảnh khả năng dự báo'. Nếu nhìn vào p-value thô, ta tưởng như TSMixer hay Ridge đã đánh bại thị trường với p < 0.01. Nhưng khi áp dụng chuẩn mực thống kê quốc tế Holm-Bonferroni để trừ hao việc thử nghiệm 60 lần, không có mô hình nào vượt qua được ngưỡng khắt khe. Kết quả này hoàn toàn nhất quán với bài báo kinh điển của Welch & Goyal."*

---

## SLIDE 38: PHÂN TÍCH ĐỘ NHẠY KHỞI TẠO NGẪU NHIÊN (MULTI-SEED SENSITIVITY)

### Kết quả phân phối hiệu năng OOS $R^2$ của các mô hình Deep Learning qua 10 hạt giống:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             PHÂN PHỐI HIỆU NĂNG OOS R² QUA 10 RANDOM SEEDS TẠI KỲ HẠN 120D             │
├───────────────────┬───────────────────┬───────────────────┬────────────────────────────┤
│ MÔ HÌNH           │ SINGLE SEED (42)  │ MEAN 10 SEEDS     │ ĐỘ LỆCH CHUẨN (STD)        │
├───────────────────┼───────────────────┼───────────────────┼────────────────────────────┤
│ Ridge (Tuyến tính)│ +1.85%            │ +1.85% (Cố định)  │ ±0.00% (Tính tất định 100%)│
│ LSTM              │ -7.85%            │ -12.40%           │ ±4.85%                     │
│ TCN               │ -6.10%            │ -8.95%            │ ±3.20%                     │
│ Transformer       │ -11.20%           │ -16.80%           │ ±6.15%                     │
│ PatchTST          │ +2.15%            │ -1.45%            │ ±5.10%                     │
│ TSMixer           │ **+17.53%**       │ **-18.93%**       │ **±12.45% (Cực kỳ nhạy!)** │
└───────────────────┴───────────────────┴───────────────────┴────────────────────────────┘
```

* **Vén màn con số +17.53% của TSMixer:**
  * Điểm số $+17.53\%$ chỉ xuất hiện ở đúng duy nhất hạt giống 42. Ở 9 hạt giống còn lại, điểm số dao động từ $-8\%$ đến $-32\%$.
  * Các mô hình phi tham số tất định như Ridge có tính tái lập tuyệt đối ($100\%$ ổn định), trong khi các mô hình Deep Learning chịu rủi ro phân kỳ khởi tạo rất lớn trên chuỗi tài chính.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Thực nghiệm 10 seeds đã vén màn bí mật đằng sau con số 17.53% của TSMixer: Đây thực chất là một 'cực trị may mắn' của riêng hạt giống 42. Trung bình 10 lần chạy của TSMixer thực tế là âm 18.93%. Ngược lại, Ridge có tính tất định tuyệt đối và bền vững qua mọi lần kiểm chứng."*

---

## SLIDE 39: PHÂN TÍCH BÓC TÁCH ĐẶC TRƯNG (FEATURE ABLATION STUDY THEO HOOKER ET AL.)

### Phương pháp luận bóc tách đặc trưng nghiêm ngặt (Retrain-from-scratch Protocol):
* Thay vì chỉ tráo ngẫu nhiên đặc trưng (Permutation Importance vốn bị chệch khi các biến đa cộng tuyến), nhóm áp dụng chuẩn mực **Feature Ablation học thuật theo Hooker et al. (2021)**:
  * Lần lượt loại bỏ hoàn toàn từng nhóm đặc trưng ra khỏi tập dữ liệu.
  * Huấn luyện lại hoàn toàn mô hình (Retrained from scratch) và đo lường mức độ sụt giảm hiệu năng:
    $\Delta \mathrm{OOS}\,R^2_g = \mathrm{OOS}\,R^2_{\text{Ablated},g} - \mathrm{OOS}\,R^2_{\text{Full}}$
  * $\Delta \mathrm{OOS}\,R^2_g < 0$: Nhóm đặc trưng $g$ đóng vai trò sống còn (loại bỏ làm giảm hiệu năng).
  * $\Delta \mathrm{OOS}\,R^2_g > 0$: Nhóm đặc trưng $g$ gây nhiễu quá khớp (loại bỏ giúp mô hình tốt hơn).

### 3 Mô hình đại diện được chọn để bóc tách:
* **XGBoost:** Đại diện họ cây phi tuyến.
* **Ridge:** Đại diện họ tuyến tính có ràng buộc co ngót.
* **LSTM:** Đại diện họ học sâu chuỗi thời gian.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Nhóm thực hiện phân tích bóc tách đặc trưng theo chuẩn mực khắt khe của Hooker: Mỗi lần loại bỏ một nhóm biến, mô hình được train lại từ đầu chứ không dùng mẹo xáo biến. Điều này cho phép nhìn thấy chính xác giá trị thực tế của từng nhóm chỉ báo tài chính."*

---

## SLIDE 40: KẾT QUẢ BÓC TÁCH ĐẶC TRƯNG CHI TIẾT TRÊN XGBOOST, RIDGE VÀ LSTM

### Bảng biến thiên hiệu năng ($\Delta \mathrm{OOS}\,R^2$) khi loại bỏ từng nhóm đặc trưng:

| Nhóm đặc trưng bị loại bỏ | XGBoost (1D) | XGBoost (120D) | Ridge (120D) | LSTM (120D) | Cơ chế tác động cốt lõi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Loại bỏ `Intraday & Volume`** | **-0.85%** | -0.15% | -0.22% | +0.45% | Biên độ trong ngày và thanh khoản tối quan trọng ở kỳ siêu ngắn (1D) |
| **Loại bỏ `VIX & Volatility`** | -0.42% | **-1.25%** | **-1.80%** | -0.95% | Biến động và tâm lý sợ hãi là trụ cột dự báo ở kỳ hạn dài (120D) |
| **Loại bỏ `Trend (MA/MACD)`** | -0.20% | -0.85% | -1.10% | -0.65% | Xu hướng trung bình động cung cấp neo định hướng trung hạn |
| **Loại bỏ `Momentum (RSI)`** | -0.15% | -0.30% | -0.45% | **+1.85%** | **Hiện tượng bù trừ:** Loại bỏ RSI giúp LSTM giảm quá khớp ở kỳ 120D! |

### 3 Bài học rút ra từ phân tích bóc tách:
1. **Intraday & Volume sống còn ở kỳ ngắn:** Cung cấp thông tin thanh khoản khẩn cấp cho 1D.
2. **VIX & Biến động lịch sử chi phối kỳ dài:** Là nguồn tín hiệu kinh tế duy nhất giúp Ridge và XGBoost duy trì được sự ổn định ở 120D.
3. **Cơ chế chống quá khớp của Deep Learning (Kaneko, 2022):** Việc loại bớt biến xung lượng (RSI) thực chất giúp LSTM bớt bị nhiễu và cải thiện hiệu năng ở kỳ hạn dài.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kết quả bóc tách cho thấy: Thanh khoản và biên độ phiên là sống còn ở kỳ ngắn 1 ngày; nhưng ở kỳ dài 120 ngày, VIX và biến động lịch sử lại là trụ cột thông tin quan trọng nhất. Thậm chí đối với LSTM, việc cắt bỏ bớt biến RSI lại giúp mô hình bớt quá khớp và cải thiện kết quả."*

---

## SLIDE 41: KIỂM TRA TÍNH VỮNG VỚI ZERO-RETURN BENCHMARK & CỬA SỔ REGIME THAY THẾ

### 1. Thử nghiệm với Zero-Return Benchmark (Giả thuyết Bước đi Ngẫu nhiên thuần túy):
* Thay vì so với Historical Mean, ta so với giả định lợi suất tương lai bằng 0: $\hat{y}_t = 0$.
* **Kết quả:** Điểm số $OOS\ R_{\text{Zero}}^2$ của tất cả các mô hình đều tăng vọt (TSMixer đạt $+19.2\%$, Ridge đạt $+3.1\%$).
* **Ý nghĩa:** Khẳng định Historical Mean là một chuẩn mực khắt khe hơn rất nhiều so với Zero-return benchmark. Thứ hạng tương đối giữa 12 mô hình hoàn toàn không đổi.

### 2. Kiểm tra tính vững với các cửa sổ định nghĩa Regime thay thế:
* Thử nghiệm cửa sổ biến động: **10 ngày (Ngắn hạn)** và **60 ngày (Dài hạn)** thay cho mốc 20 ngày cơ sở.
* **Kết luận:** Tỷ lệ sụp đổ của các mô hình tại High Volatility Regime vẫn giữ nguyên tính nhất quán tuyệt đối trên mọi định nghĩa cửa sổ, chứng minh kết luận nghiên cứu là hoàn toàn vững chắc.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Nhóm đã thực hiện thêm hai kiểm tra độ mạnh mẽ: Thứ nhất, so sánh với Zero-Return Benchmark để khẳng định thứ hạng các mô hình là nhất quán. Thứ hai, thử nghiệm các cửa sổ tính Regime 10 ngày và 60 ngày để chứng minh hiện tượng suy sụp hiệu năng trong chế độ khủng hoảng không phụ thuộc vào cách chia cửa sổ."*

---

# PHẦN VIII: Ý NGHĨA THỰC TIỄN, ĐÓNG GÓP, KẾT LUẬN & HỎI ĐÁP TOÀN BỘ CÁC MÔ HÌNH

---

## SLIDE 42: KẾT LUẬN KHOA HỌC & 4 BÀI HỌC CỐT LÕI TỪ ĐỒ ÁN

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        4 BÀI HỌC KHOA HỌC RÚT RA TỪ DỰ ÁN                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. TÍNH PHI TUYẾN CỦA CHÂN TRỜI (Non-monotonic Horizon Behavior):                      │
│    Khả năng dự báo không suy giảm đơn điệu theo thời gian; kỳ hạn dài (120D) có tỷ lệ │
│    tín hiệu trên nhiễu tốt hơn các kỳ hạn siêu ngắn (1D - 20D).                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. VÙNG HOẠT ĐỘNG TỐI ƯU CỦA MACHINE LEARNING (Normal Regime Sweet Spot):              │
│    Các mô hình học máy phát huy hiệu quả cao nhất trong chế độ biến động bình thường    │
│    và thất bại nặng nề trong chế độ khủng hoảng biến động cao.                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. SỰ NGUY HIỂM CỦA ẢO ẢNH KHẢ NĂNG DỰ BÁO (Predictability Illusion):                  │
│    Các điểm số ấn tượng từ việc chạy một seed đơn lẻ sẽ biến mất hoàn toàn khi kiểm    │
│    soát sai lầm đa giả thuyết Holm-Bonferroni và chạy kiểm chứng 10 random seeds.      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. ƯU THẾ CỦA SỰ ĐƠN GIẢN CÓ RÀNG BUỘC (Power of Constrained Simplicity):              │
│    Các mô hình tuyến tính có phạt như Ridge thể hiện sự bền vững, ổn định và đáng tin  │
│    cậy vượt trội so với các kiến trúc học sâu cồng kềnh trong thực tế đầu tư.         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Tóm lại, 4 bài học lớn nhất nhóm rút ra là: Khả năng dự báo biến thiên phi tuyến theo chân trời; chế độ biến động bình thường là môi trường lý tưởng nhất; phải luôn cảnh giác với ảo ảnh thống kê của các mô hình học sâu; và cuối cùng, sự đơn giản có kiểm soát của các mô hình tuyến tính có phạt thường chiến thắng sự phức tạp trong tài chính."*

---

## SLIDE 43: Ý NGHĨA THỰC TIỄN & HÀM Ý QUẢN TRỊ DANH MỤC ĐẦU TƯ

### Khuyến nghị hành động cho các Quỹ Đầu tư Định lượng (Quant Funds):
1. **Chiến lược Phân bổ Linh hoạt theo Chế độ Biến động (Regime-Switching Allocation):**
   * **Khi thị trường ở Regime Bình thường:** Kích hoạt các mô hình Machine Learning/Deep Learning để gia tăng tìm kiếm lợi nhuận Alpha.
   * **Khi thị trường chuyển sang High Volatility Regime (VIX tăng vọt):** Ngay lập tức ngắt toàn bộ các mô hình học sâu phức tạp, chuyển danh mục về phòng thủ bằng tiền mặt hoặc áp dụng mô hình phân bổ tỷ trọng bằng nhau (Equal-weighted).
2. **Khai thác Giá trị của Directional Accuracy (DA):**
   * Thay vì cố gắng dự báo chính xác giá trị lợi suất tuyệt đối (vốn chịu sai số lớn), nhà đầu tư nên sử dụng tín hiệu chiều tăng/giảm (DA > 55%) để thiết lập chiến lược Long/Short vị thế.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Về mặt ứng dụng thực tiễn: Nghiên cứu đề xuất chiến lược quản trị danh mục thích ứng. Trong trạng thái bình thường, nhà quản lý quỹ có thể tin cậy vào Machine Learning. Nhưng khi biến động tăng cao, giải pháp tối ưu là tắt các mô hình phức tạp để tránh thua lỗ nặng nề do sự đứt gãy của dữ liệu lịch sử."*

---

## SLIDE 44: ĐÓNG GÓP KHOA HỌC, GIỚI HẠN & HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### 1. Ba đóng góp học thuật nổi bật:
* **Quy mô thực nghiệm toàn diện:** 12 mô hình $\times$ 5 chân trời $\times$ 3 chế độ biến động $\times$ 9.245 ngày giao dịch (1990–2026).
* **Chuẩn mực kiểm định thống kê chuẩn mực:** Kết hợp kiểm định Clark-West và hiệu chỉnh đa giả thuyết Holm-Bonferroni, phơi bày hiện tượng quá khớp hạt giống ngẫu nhiên.
* **Quy trình bóc tách đặc trưng nghiêm ngặt:** Huấn luyện lại từ đầu theo Hooker et al. để làm rõ vai trò thực sự của từng chỉ báo tài chính.

### 2. Giới hạn đề tài & Hướng nghiên cứu tương lai:
* **Giới hạn:** Chưa tính toán chi phí trượt giá (slippage) và thuế phí giao dịch vi mô; mới kiểm định trên chỉ số tổng hợp S&P 500.
* **Hướng mở rộng:**
  * Mở rộng kiểm định sang các thị trường mới nổi (như chứng khoán Việt Nam VN-Index).
  * Ứng dụng mạng đồ thị nơ-ron (Graph Neural Networks - GNN) để nắm bắt mối quan hệ lan truyền rủi ro chéo giữa các cổ phiếu thành phần.

> **Lời thoại thuyết trình (Speaker Notes):**  
> *"Đóng góp lớn nhất của nhóm là thiết lập một quy trình thực nghiệm và kiểm định thống kê chuẩn mực quốc tế. Trong tương lai, nhóm dự định mở rộng mô hình sang thị trường chứng khoán Việt Nam và áp dụng mạng đồ thị để nghiên cứu cấu trúc liên kết chéo giữa các ngành nghề."*

---

## SLIDE 45: BẢNG HỎI ĐÁP PHẢN BIỆN TOÀN DIỆN CHO TẤT CẢ CÁC MÔ HÌNH (Q&A CHEAT-SHEET FOR ALL MODELS)

### Cảm ơn Thầy và các bạn đã lắng nghe bài thuyết trình!
**Nhóm 6 – IS403.R12** sẵn sàng lắng nghe các câu hỏi chất vấn chuyên sâu.

---

### BỘ CÂU HỎI & CÂU TRẢ LỜI CHUẨN XÁC DÀNH CHO TỪNG MÔ HÌNH CỤ THỂ:

#### 1. Mô hình ARIMA & ARIMAX:
* **Hỏi:** Làm ARIMA xác định $(p, d, q)$ như thế nào? Chạy nhiều lần nhiều bộ $pdq$ khác nhau? Tại sao không dùng AutoARIMA? Bộ $(p, d, q)$ của ARIMAX là gì?
* **Trả lời:**
  * Xác định $d=0$ trên log return (hoặc $d=1$ trên giá gốc) qua kiểm định ADF ($p < 0.0001$).
  * Phân tích đồ thị ACF/PACF khoanh vùng $p, q \in \{0, 1, 2, 3\}$. Chạy Grid Search thủ công thử nghiệm 16 tổ hợp $(p, d, q)$ khác nhau trên tập Train và chọn bộ tối thiểu hóa AIC/BIC: **ARIMA(1, 0, 1)**.
  * Không dùng AutoARIMA vì dùng tìm kiếm stepwise dễ rơi vào tối ưu cục bộ và chọn số bậc quá lớn gây quá khớp nghiêm trọng trên dữ liệu tài chính có nhiễu cao.
  * ARIMAX dùng chung bộ **$(p, d, q) = (1, 0, 1)$** kèm 15 biến ngoại sinh trễ $X_{t-1}$.

#### 2. Mô hình Ridge Regression:
* **Hỏi:** Ridge phạt L2 hoạt động như thế nào? Tại sao giải quyết được hiện tượng đa cộng tuyến?
* **Trả lời:**
  * Ridge cộng thêm số hạng phạt $\frac{1}{2}\lambda \|\beta\|_2^2$ vào hàm mất mát. Nghiệm giải tích đóng là $\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$.
  * Trong dữ liệu 16 đặc trưng tài chính, ma trận $X^TX$ gần suy biến do các đường MA/EMA có tương quan cao. Việc cộng $\lambda I$ vào đường chéo giúp ma trận luôn khả nghịch, co nhỏ hệ số (shrinkage) và triệt tiêu phương sai dự báo. Siêu tham số tối ưu sau Tuning là `alpha = 100.0`.

#### 3. Mô hình Lasso & ElasticNet:
* **Hỏi:** Lasso phạt L1 khác gì Ridge? Tại sao Lasso ép được hệ số về đúng bằng 0 (sparsity)?
* **Trả lời:**
  * Chuẩn L1 phạt tổng giá trị tuyệt đối $\lambda \|\beta\|_1$. Hình học không gian của chuẩn L1 có các góc nhọn nằm trên các trục tọa độ. Khi tối ưu hóa bằng Coordinate Descent, điểm tiếp xúc giữa đường mức sai số và hình nón L1 thường xảy ra tại trục tọa độ, ép các biến ít quan trọng về đúng bằng 0, đóng vai trò bộ chọn lọc đặc trưng tự động.
  * ElasticNet kết hợp cả L1 và L2 (`alpha = 0.001, l1_ratio = 0.5`) để vừa chọn lọc đặc trưng vừa xử lý nhóm biến cộng tuyến.

#### 4. Mô hình Random Forest:
* **Hỏi:** Random Forest xây dựng cây như thế nào? Bagging và Feature Subsampling giải quyết vấn đề gì?
* **Trả lời:**
  * Random Forest sử dụng Bagging (lấy mẫu có hoàn lại $N$ quan sát cho 200 cây) kết hợp chọn ngẫu nhiên tập con đặc trưng $m = \sqrt{16} = 4$ biến tại mỗi nút rẽ để tìm điểm chẻ giảm MSE tốt nhất.
  * Cơ chế này làm mất tương quan giữa các cây đơn lẻ, giúp phép lấy trung bình cộng làm giảm phương sai dự báo tổng thể mà không làm tăng độ chệch. Khống chế `max_depth = 5` để ngăn học vẹt.

#### 5. Mô hình LightGBM:
* **Hỏi:** LightGBM xây dựng cây khác gì XGBoost?
* **Trả lời:**
  * LightGBM dùng chiến lược **Leaf-wise (Best-first)**: luôn chọn chiếc lá có mức giảm loss lớn nhất trên toàn bộ cây để phân nhánh tiếp (thay vì phân nhánh đồng đều theo tầng như Level-wise của XGBoost), kết hợp giới hạn `max_depth = 2` để chống overfitting.
  * Áp dụng thuật toán **Histogram binning** nén dữ liệu liên tục thành 256 bins 8-bit, giúp tốc độ huấn luyện nhanh gấp hàng chục lần và tiết kiệm bộ nhớ.

#### 6. Mô hình XGBoost:
* **Hỏi:** XGBoost xây dựng cây như thế nào? Khai triển Taylor bậc 2 và công thức Gain?
* **Trả lời:**
  * XGBoost khai triển chuỗi Taylor bậc 2 của hàm mất mát quanh $\hat{y}_i^{(t-1)}$ để lấy cả Gradient bậc 1 ($g_i$) và Hessian bậc 2 ($h_i$).
  * Tại mỗi điểm chẻ tiềm năng, thuật toán tính điểm $\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$.
  * Nút chỉ tách nhánh khi $\text{Gain} > 0$, hệ số $\gamma$ đóng vai trò ngưỡng cắt tỉa cây tự động. Bộ tham số tối ưu: `n_estimators = 200, max_depth = 9, learning_rate = 0.0935, reg_lambda = 2.88`.

#### 7. Mô hình SVR (Support Vector Regression):
* **Hỏi:** SVM vốn giải quyết bài toán phân lớp, sang bài toán hồi quy chuỗi thời gian thì làm thế nào?
* **Trả lời:**
  * SVR thay đổi mục tiêu: Xây dựng một **ống dung sai $\varepsilon$-insensitive loss** xung quanh siêu phẳng dự báo $f(x) \pm \varepsilon$.
  * Các điểm dao động nhỏ nằm trong ống được coi là nhiễu vi cấu trúc và có hàm mất mát bằng 0. Chỉ các điểm nằm ngoài ống mới bị phạt qua biến bù $\xi_i, \xi_i^*$ và trở thành Support Vectors.
  * Sử dụng nhân phi tuyến RBF Kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ để ánh xạ dữ liệu sang không gian chiều cao vô hạn. Siêu tham số: $C = 0.1, \varepsilon = 0.01, \gamma = 0.001$.

#### 8. Mô hình MLP / ANN (Chương 6):
* **Hỏi:** Kiến trúc mạng nơ-ron ra sao? Bao nhiêu layer, bao nhiêu nơ-ron? Window size xác định thế nào?
* **Trả lời:**
  * Đầu vào có Window size $L = 20$ ngày $\times 16$ đặc trưng (tensor $20 \times 16$), tương ứng 1 tháng giao dịch.
  * Làm phẳng thành vector 320 chiều. Gồm **3 tầng ẩn thu hẹp dần: 64 $\rightarrow$ 32 $\rightarrow$ 16 nơ-ron** với hàm kích hoạt phi tuyến ReLU và kỹ thuật điều hòa Dropout rate = 0.1.
  * Kết thúc bằng **1 nơ-ron tuyến tính** ở tầng Output. Tối ưu bằng AdamW, learning rate = $10^{-4}$, hàm mất mát MSE.

#### 9. Mô hình LSTM:
* **Hỏi:** LSTM giải quyết chuỗi thời gian như thế nào? Cơ chế các cổng?
* **Trả lời:**
  * LSTM giải quyết triệt tiêu gradient nhờ đường cao tốc ô nhớ $C_t$ được điều phối bởi 4 phương trình cổng vi phân: Cổng quên $f_t$ (xóa thông tin cũ), Cổng vào $i_t$ (nạp thông tin mới), Ô nhớ ứng viên $\tilde{C}_t$, và Cổng ra $o_t$ để tạo trạng thái ẩn $h_t$.
  * Kiến trúc gồm **3 lớp LSTM xếp chồng (64 $\rightarrow$ 32 $\rightarrow$ 16 units)**, `dropout = 0.354`, learning rate = $0.000766$, batch size = 16.

#### 10. Mô hình TCN (Temporal Convolutional Network):
* **Hỏi:** TCN là gì? Tại sao CNN áp dụng được cho chuỗi thời gian?
* **Trả lời:**
  * TCN áp dụng tích chập 1D cho chuỗi thời gian nhờ 2 cơ chế: **Causal Convolutions** đảm bảo không rò rỉ tương lai; và **Dilated Convolutions** với độ giãn lũy thừa $d \in \{1, 2, 4\}$ mở rộng trường tiếp nhận theo hàm số mũ bao phủ trọn vẹn 20 ngày quá khứ mà không cần tăng tham số.
  * Gồm 3 khối Residual Blocks với 64 kênh, kernel size = 3, huấn luyện song song toàn bộ chuỗi nhanh hơn LSTM gấp nhiều lần.

#### 11. Mô hình Transformer & Cơ chế Attention:
* **Hỏi:** Cơ chế Attention của Transformer hoạt động như thế nào? Tại sao dễ quá khớp trên chuỗi tài chính?
* **Trả lời:**
  * Tính Scaled Dot-Product Attention: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ với 4 đầu chú ý (Multi-Head Attention), cho phép các ngày trong quá khứ tương tác và chú ý lẫn nhau.
  * Transformer dễ bị quá khớp vì cơ chế tương quan toàn cục bậc 2 $O(L^2)$ rất nhạy cảm với các đột biến nhiễu ngắn hạn trong tài chính, khiến mô hình gán trọng số chú ý sai lệch vào các biến động ngẫu nhiên.

#### 12. Mô hình PatchTST:
* **Hỏi:** PatchTST giải quyết nhược điểm của Transformer như thế nào?
* **Trả lời:**
  * PatchTST cải tiến bằng 2 đột phá: **Subseries Patching** (gom các đoạn 8 ngày liên tiếp thành 1 token ngữ nghĩa, trượt 4 ngày) giúp giảm độ dài chuỗi vào Transformer và giữ được tính liên tục xu hướng cục bộ; và **Channel-Independence** (xử lý độc lập từng biến đặc trưng qua cùng một backbone) giúp triệt tiêu hoàn toàn nhiễu tương quan chéo giữa 16 chỉ báo.

#### 13. Mô hình TSMixer:
* **Hỏi:** TSMixer hoạt động thế nào? Tại sao đạt OOS $R^2 = 17,53\%$ ở 120D nhưng trung bình 10 seeds lại rớt xuống $-18,93\%$?
* **Trả lời:**
  * TSMixer là kiến trúc All-MLP loại bỏ hoàn toàn Attention, sử dụng luân phiên các khối Time-mixing MLP (học dọc trục thời gian) và Feature-mixing MLP (học ngang trục đặc trưng).
  * Điểm số $+17,53\%$ tại 120D chỉ là một **cực trị may mắn (lucky outlier)** của riêng hạt giống `seed = 42`. Khi kiểm chứng 10 random seeds độc lập, các hạt giống khác đều bị âm (từ $-8\%$ đến $-32\%$), kéo trung bình 10 seed xuống $-18,93\%$. Điều này chứng minh các mô hình học sâu rất mong manh trước tính bất định của chuỗi tài chính.

---

*Tài liệu được biên soạn tự động và đồng bộ từ toàn bộ cơ sở dữ liệu thực nghiệm và bản thảo Springer LNCS 2026 của Đồ án IS403.R12.*
