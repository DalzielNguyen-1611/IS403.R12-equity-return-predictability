# 📋 BẢNG PHÂN CÔNG NHIỆM VỤ DỰ ÁN CHO 5 THÀNH VIÊN (TEAM TASK ALLOCATION)
## ĐỀ TÀI: HOW DOES EQUITY RETURN PREDICTABILITY VARY ACROSS HORIZONS AND MARKET REGIMES? AN OUT-OF-SAMPLE ANALYSIS
### Môn học: IS403 – Phân tích Dữ liệu Nâng cao & Ứng dụng Tài chính | Nhóm 6 (IS403.R12)

---

## 🎯 NGUYÊN TẮC PHÂN CHIA NHIỆM VỤ
1. **Phân bổ trọng số đóng góp theo trách nhiệm thực tế (Leader 40% – 4 Thành viên 15%):**
   * **Trưởng nhóm (Leader - 40%):** Giữ vai trò đầu tàu và kiến trúc sư hệ thống (Project Architect & Team Lead), khởi tạo và quản trị toàn bộ repository, thiết lập chuẩn mực đường ống dữ liệu, kỹ nghệ đặc trưng cốt lõi, cơ chế chống rò rỉ Lookback Buffer, chủ trì định hướng và cấu trúc bài báo khoa học chuẩn Springer LNCS (47 trang), thiết kế slide tổng thể và điều phối toàn bộ đồ án.
   * **Bốn Thành viên (Members - 15% mỗi người):** Làm chủ sâu từng mảng kỹ thuật chuyên biệt (Kinh tế lượng/Tuyến tính; Cây/SVR; Học sâu/Attention; Đánh giá thực nghiệm/Kiểm định thống kê), chịu trách nhiệm trọn vẹn từ huấn luyện, tinh chỉnh tham số, giải trình thuật toán đến các slide chuyên sâu của mình.
   * **Tổng mức độ đóng góp:** $40\% + 15\% \times 4 = 100\%$.
2. **Chuyên môn hóa & Làm chủ kỹ thuật (Ownership):** Mỗi thành viên nắm sâu bản chất thuật toán và thông số kỹ thuật của các mô hình mình phụ trách để tự tin trả lời trực tiếp các câu hỏi chất vấn, phản biện của Thầy trong buổi bảo vệ đồ án.
3. **Phối hợp liên hoàn (Pipeline Integration):** Đầu ra của thành viên xử lý dữ liệu là đầu vào cho các thành viên huấn luyện mô hình; kết quả của các thành viên mô hình là cơ sở để thành viên kiểm định thống kê và tổng hợp báo cáo.

---

## 👥 BẢNG TỔNG QUAN PHÂN VAI 5 THÀNH VIÊN

| STT | Họ và Tên & MSSV | Vai trò đảm nhiệm | Tỷ lệ đóng góp | Phạm vi trách nhiệm chính | Họ mô hình / Công việc sở hữu |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Nguyễn Đoàn Đức Hiếu**<br>`24520500` | **Trưởng nhóm (Team Leader & Data Architect)** | **40%** | Quản trị Repo & Pipeline tổng thể, thu thập/làm sạch dữ liệu, kiểm định dừng, kỹ nghệ 16 đặc trưng, cơ chế Lookback Buffer, chủ trì cấu trúc bài báo & điều phối | Dữ liệu S&P 500 & VIX, Giao thức Walk-Forward, Window Size $L=20$, Lookback Buffer, Dashboard |
| **2** | **Nguyễn Thị Phương Hoa**<br>`23520506` | **Kỹ sư Kinh tế lượng & Hồi quy Ràng buộc (Linear ML)** | **15%** | Nghiên cứu và tối ưu các mô hình thống kê chuỗi thời gian cổ điển và hồi quy tuyến tính có phạt co ngót hệ số | `Historical Mean`, `ARIMA`, `ARIMAX`, `Ridge`, `Lasso`, `ElasticNet` |
| **3** | **Nguyễn Văn Phát**<br>`24521314` | **Kỹ sư Mô hình Cây & Kernel Phi tuyến (Trees & SVR)** | **15%** | Nghiên cứu cấu trúc phân nhánh, thuật toán tối ưu bậc hai, điều hòa và hàm mất mát vùng dung sai $\varepsilon$ | `Random Forest`, `LightGBM`, `XGBoost`, `SVR` |
| **4** | **Trần Anh Thư**<br>`24521732` | **Kỹ sư Học sâu Chuỗi thời gian & Attention (Deep Sequence)** | **15%** | Thiết kế kiến trúc mạng nơ-ron đa tầng, cơ chế cổng bộ nhớ, tích chập nhân quả và cơ chế chú ý hiện đại | `MLP / ANN (Chương 6)`, `LSTM`, `TCN`, `Transformer`, `PatchTST`, `TSMixer` |
| **5** | **Trần Ngọc Thảo**<br>`24521649` | **Chuyên viên Đánh giá Thực nghiệm & Báo cáo (Evaluation Lead)** | **15%** | Xây dựng khung đánh giá ngoài mẫu, kiểm định Clark-West, hiệu chỉnh FWER, phân tích Robustness, hỗ trợ biên tập bài báo & slide | $OOS\ R^2$, Clark-West, Holm-Bonferroni, Multi-seed (10 seeds), Feature Ablation, Bài báo Springer 47 trang & Slide 45 trang |

---

## 📌 BẢN MÔ TẢ NHIỆM VỤ CHI TIẾT TỪNG THÀNH VIÊN

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SƠ ĐỒ PHỐI HỢP CÔNG VIỆC 5 THÀNH VIÊN                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [THÀNH VIÊN 1: DATA & PIPELINE]                                                       │
│  Thu thập S&P 500 + VIX ──► Căn chỉnh 9.245 ngày ──► 16 Features ──► Lookback Buffer   │
│                                        │                                               │
│             ┌──────────────────────────┼──────────────────────────┐                    │
│             ▼                          ▼                          ▼                    │
│  [THÀNH VIÊN 2: LINEAR/TS]   [THÀNH VIÊN 3: TREES/SVR]  [THÀNH VIÊN 4: DEEP LEARNING]  │
│  • ARIMA & ARIMAX            • Random Forest            • MLP (Chương 6 ANN)           │
│  • Ridge & Lasso             • LightGBM & XGBoost       • LSTM & TCN                   │
│  • ElasticNet                • Support Vector Reg (SVR) • PatchTST & TSMixer           │
│             │                          │                          │                    │
│             └──────────────────────────┼──────────────────────────┘                    │
│                                        ▼                                               │
│  [THÀNH VIÊN 5: EVALUATION, ROBUSTNESS & SCIENTIFIC REPORTING]                         │
│  OOS R² ──► Clark-West & Holm-Bonferroni ──► 10 Seeds ──► Ablation ──► Paper & Slides  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 👤 THÀNH VIÊN 1: NGUYỄN ĐOÀN ĐỨC HIẾU – TRƯỞNG NHÓM & KIẾN TRÚC SƯ DỰ ÁN (PROJECT ARCHITECT & LEAD)
* **Tỷ lệ đóng góp:** **40%** (Đầu tàu kỹ thuật, quản trị mã nguồn, đường ống dữ liệu, định hướng học thuật & điều phối).
* **Mục tiêu:** Quản trị toàn diện repository, đảm bảo toàn bộ hạ tầng dữ liệu và đường ống (pipeline) vận hành chuẩn xác, hoàn toàn không rò rỉ dữ liệu (zero leakage), chủ trì cấu trúc bài báo và điều phối toàn bộ đồ án.
* **Nhiệm vụ cụ thể:**
  1. **Quản trị Repository & Kiến trúc Hệ thống:**
     * Khởi tạo, tổ chức và quản lý Git repository; thiết lập cấu trúc mã nguồn module hóa chuẩn mực (`src/training`, `src/evaluation`, `src/features`, `src/utils`), chuẩn hóa quy ước code và trang giao diện trực quan `dashboard.html`.
  2. **Thu thập dữ liệu chuẩn:**
     * Trích xuất chỉ số S&P 500 (^GSPC) từ Yahoo Finance (1990 – 2026: 9.255 ngày).
     * Trích xuất chỉ số biến động CBOE VIX từ FRED (9.585 ngày quan sát).
  3. **Tiền xử lý & Khớp ngày giao dịch:**
     * Thực hiện *Inner Join* khớp ngày giao dịch, xử lý các ngày nghỉ lễ, giải quyết triệt để lỗi ngoại lai khối lượng (`Volume inf anomaly`).
     * Chốt bộ dữ liệu chuẩn mực **9.245 ngày giao dịch sạch** (100% không khuyết thiếu).
  4. **Kiểm định tính dừng & Biến đổi chuỗi:**
     * Chạy kiểm định ADF và KPSS; chứng minh chuỗi giá gốc không dừng $I(1)$ ($p=0.89$), chuỗi sai phân log-return dừng tuyệt đối $I(0)$ ($p < 0.0001$).
  5. **Kỹ nghệ 16 Đặc trưng tài chính:**
     * Lập trình tính toán 4 nhóm biến: Giá trễ (`Return_Lag1, 2, 5`), Xu hướng (`MA5, MA20, EMA20, MACD`), Động lượng & Biến động (`RSI14, Volatility_5D, 20D, VIX`), Biên độ phiên & Khối lượng (`Intraday_Range, Return_Overnight, Volume_Change`).
  6. **Thiết kế phân chia thời gian & Kỹ thuật Bộ đệm Lookback (Lookback Buffer):**
     * Thiết lập mốc phân chia: Huấn luyện (1990–2015), Thẩm định Walk-forward (2016–2020), Holdout kiểm định cuối (11/2020 – 04/2026: 1.362 ngày).
     * Lập trình cơ chế đệm 19 ngày cuối của tập trước ghép vào ngày đầu Holdout, giúp các mô hình chuỗi sâu dự báo trọn vẹn 1.362/1.362 ngày mà không mất dữ liệu biên.
     * Xác định Window Size $L = 20$ ngày dựa trên chu kỳ 1 tháng giao dịch.
  7. **Chủ trì Định hướng Bài báo khoa học & Bộ Slide:**
     * Chủ trì định hướng học thuật và cấu trúc bài báo chuẩn Springer LNCS 47 trang.
     * Trực tiếp viết các phần: Đặt vấn đề (Introduction), Mô tả bài toán nghiên cứu (Problem Formulation), Tổng quan dữ liệu & Đặc trưng.
     * Biên soạn khung slide và trực tiếp thuyết trình Slide 1 đến Slide 14, đồng thời điều phối toàn bộ phiên vấn đáp Q&A trước Hội đồng.
  8. **Tổng duyệt & Kiểm thử Mã nguồn:**
     * Chủ trì rà soát chất lượng kỹ thuật, kiểm thử hồi quy và đồng bộ hóa toàn bộ pipeline mô hình và dữ liệu của dự án.
* **Câu hỏi phản biện của Thầy chịu trách nhiệm chính:**
  * *Hỏi: Chuỗi thời gian là đơn biến hay đa biến?* (Slide 7 & Slide 45 Q1)
  * *Hỏi: Tại sao lại chọn các số ngày 1D, 5D, 20D, 60D, 120D?* (Slide 10 & Slide 45 Q2)
  * *Hỏi: Xác định Window Size L=20 ngày như thế nào? Kỹ thuật Lookback Buffer giải quyết vấn đề gì?* (Slide 13, 14)

---

### 👤 THÀNH VIÊN 2: NGUYỄN THỊ PHƯƠNG HOA – KỸ SƯ KINH TẾ LƯỢNG & HỒI QUY RÀNG BUỘC (LINEAR ML)
* **Tỷ lệ đóng góp:** **15%**
* **Mục tiêu:** Làm chủ các mô hình kinh tế lượng cổ điển và các kỹ thuật hồi quy tuyến tính có ràng buộc co ngót hệ số để giải quyết bài toán đa cộng tuyến.
* **Nhiệm vụ cụ thể:**
  1. **Mô hình Historical Mean (Benchmark):**
     * Triển khai mô hình trung bình trôi dạt quá khứ $\bar{y}_t = \frac{1}{t}\sum y_i$ theo chuẩn mực Campbell & Thompson (2008) làm mốc quy chuẩn đối đầu với toàn bộ các mô hình học máy.
  2. **Mô hình Chuỗi thời gian ARIMA (Đơn biến):**
     * Thực hiện quy trình 4 bước xác định $(p, d, q)$: ADF test xác định bậc sai phân $d=0$ (trên return) hoặc $d=1$ (trên log-price); phân tích đồ thị ACF/PACF để khoanh vùng bậc trễ $p, q \in \{0, 1, 2, 3\}$.
     * Lập trình Grid Search thủ công thử nghiệm 16 tổ hợp $(p, d, q)$ trên tập Train; tối thiểu hóa AIC/BIC chốt bộ tối ưu **ARIMA(1, 0, 1)**.
     * Phân tích lý giải vì sao **không dùng AutoARIMA** (tránh bẫy tối ưu cục bộ và quá khớp).
  3. **Mô hình Chuỗi thời gian ARIMAX (Đa biến):**
     * Giữ nguyên cấu trúc động học $(1, 0, 1)$ của ARIMA, tích hợp thêm vector 15 biến ngoại sinh trễ $X_{t-1}$ để bắt tín hiệu từ biến động VIX và thanh khoản.
  4. **Họ mô hình Hồi quy Tuyến tính có ràng buộc (Ridge, Lasso, ElasticNet):**
     * **Ridge Regression (L2):** Phạt bình phương độ lớn $\frac{1}{2}\lambda \|\beta\|_2^2$, lập trình nghiệm giải tích dạng đóng $\beta = (X^TX + \lambda I)^{-1}X^Ty$. Dò tìm siêu tham số $\alpha \in [10^{-3}, 10^3]$ chốt `alpha = 100.0` để khắc phục đa cộng tuyến giữa 16 chỉ báo kỹ thuật.
     * **Lasso Regression (L1):** Phạt giá trị tuyệt đối $\lambda \|\beta\|_1$, tối ưu bằng Coordinate Descent, ép các biến nhiễu về đúng bằng 0 (`alpha = 0.01`).
     * **ElasticNet:** Kết hợp cả L1 và L2 (`alpha = 0.001, l1_ratio = 0.5`).
  5. **Phân tích Cơ chế Co ngót & Thẩm định Tuyến tính:**
     * Lập trình trực quan hóa đường co ngót hệ số phạt L1/L2 và phân tích mức độ giảm phương sai so với hồi quy OLS truyền thống.
* **Câu hỏi phản biện của Thầy chịu trách nhiệm chính:**
  * *Hỏi: Làm ARIMA xác định (p, d, q) như thế nào? Chạy nhiều lần nhiều bộ pdq? Tại sao không dùng AutoARIMA? ARIMAX bộ (p, d, q) là gì?* (Slide 16 & Slide 45 Q1)
  * *Hỏi: Hồi quy OLS thất bại trong tài chính vì sao? Ridge giải quyết đa cộng tuyến thế nào? Lasso ép hệ số về 0 ra sao?* (Slide 17 & Slide 45 Q2, Q3)

---

### 👤 THÀNH VIÊN 3: NGUYỄN VĂN PHÁT – KỸ SƯ MÔ HÌNH CÂY & KERNEL PHI TUYẾN (TREES & SVR)
* **Tỷ lệ đóng góp:** **15%**
* **Mục tiêu:** Làm chủ các thuật toán học máy phi tuyến dựa trên cây quyết định và phương pháp vector hỗ trợ, nắm vững toán học tối ưu phân nhánh.
* **Nhiệm vụ cụ thể:**
  1. **Mô hình Random Forest (Bagging):**
     * Lập trình cấu trúc rừng 200 cây; cơ chế Bootstrap lấy mẫu có hoàn lại và chọn ngẫu nhiên $m = \sqrt{16} = 4$ đặc trưng tại mỗi nút rẽ để triệt tiêu tương quan giữa các cây.
     * Tuning siêu tham số qua Optuna (30 trials): `n_estimators = 200, max_depth = 5, min_samples_split = 5`.
  2. **Mô hình LightGBM (Histogram & Leaf-wise):**
     * Cài đặt chiến lược phân nhánh theo lá tốt nhất (**Leaf-wise Best-First Growth**), khống chế `max_depth = 2` để chống quá khớp.
     * Áp dụng kỹ thuật nén dữ liệu Histogram 256 bins, thuật toán GOSS và EFB giúp tăng tốc độ huấn luyện.
     * Tuning siêu tham số Optuna (50 trials): `n_estimators = 500, num_leaves = 56, learning_rate = 0.0177, reg_lambda = 3.502`.
  3. **Mô hình XGBoost (Second-order Taylor Gradient Boosting):**
     * Nắm vững toán học khai triển Taylor bậc 2 quanh $\hat{y}_i^{(t-1)}$ lấy Gradient $g_i$ và Hessian $h_i$.
     * Lập trình công thức tính điểm Gain tại mỗi điểm chẻ tối ưu có trừ đi hàm phạt độ phức tạp $\gamma T + \frac{1}{2}\lambda \sum w_j^2$.
     * Tuning Optuna (50 trials): `n_estimators = 200, max_depth = 9, learning_rate = 0.0935, reg_lambda = 2.88`.
  4. **Mô hình Support Vector Regression (SVR):**
     * Chuyển đổi từ bài toán SVM phân lớp sang hồi quy thông qua **ống sai số $\varepsilon$-insensitive loss** ($L_\varepsilon = \max(0, |y - f(x)| - \varepsilon)$).
     * Bỏ qua các biến động nhỏ trong ống (nhiễu vi cấu trúc), chỉ phạt các điểm ngoài ống kích hoạt biến bù $\xi_i, \xi_i^*$ trở thành Support Vectors.
     * Áp dụng Radial Basis Function (RBF) Kernel ánh xạ phi tuyến; Grid Search chốt: $C = 0.1, \varepsilon = 0.01, \gamma = 0.001$.
  5. **Phân tích Cấu trúc Cây & Vùng Dung sai SVR:**
     * Phân tích độ sâu phân nhánh, tầm quan trọng đặc trưng (Feature Importance) và cơ chế lọc nhiễu vùng biên của ống sai số SVR.
* **Câu hỏi phản biện của Thầy chịu trách nhiệm chính:**
  * *Hỏi: XGBoost xây dựng cây như thế nào? Khai triển Taylor bậc 2 và công thức Gain?* (Slide 20 & Slide 45 Q6)
  * *Hỏi: LightGBM xây dựng cây khác gì XGBoost? Leaf-wise vs Level-wise?* (Slide 19 & Slide 45 Q5)
  * *Hỏi: SVM phân lớp sang hồi quy dự báo giá liên tục (SVR) như thế nào? Ống sai số epsilon và RBF Kernel?* (Slide 21 & Slide 45 Q7)
  * *Hỏi: Random Forest xây dựng cây khác gì cây đơn lẻ? Cơ chế giảm phương sai?* (Slide 18 & Slide 45 Q4)

---

### 👤 THÀNH VIÊN 4: TRẦN ANH THƯ – KỸ SƯ HỌC SÂU CHUỖI THỜI GIAN & ATTENTION (DEEP SEQUENCE)
* **Tỷ lệ đóng góp:** **15%**
* **Mục tiêu:** Xây dựng, huấn luyện và tối ưu các kiến trúc mạng nơ-ron sâu từ cổ điển đến các mô hình chú ý hiện đại nhất (State-of-the-art).
* **Nhiệm vụ cụ thể:**
  1. **Mô hình Mạng Nơ-ron Truyền thẳng (MLP / ANN - Chương 6):**
     * Thiết kế chuẩn theo mục 3.1.2 Kiến trúc ANN Chương 6: Đầu vào tensor $[20 \times 16] = 320$ chiều làm phẳng $\rightarrow$ 3 tầng ẩn thu hẹp dần: **Dense 64 $\rightarrow$ Dense 32 $\rightarrow$ Dense 16** (kích hoạt ReLU, Dropout 0.1) $\rightarrow$ **1 Linear Output**.
     * Tối ưu bằng thuật toán AdamW, learning rate = $10^{-4}$, hàm mất mát MSE.
  2. **Mô hình LSTM (Long Short-Term Memory):**
     * Cài đặt đường cao tốc trạng thái ô nhớ $C_t$ điều phối bởi 4 phương trình cổng vi phân: Cổng quên $f_t$, Cổng vào $i_t$, Ô nhớ ứng viên $\tilde{C}_t$ và Cổng ra $o_t$ để triệt tiêu hiện tượng tan biến đạo hàm.
     * Kiến trúc 3 tầng LSTM xếp chồng ($64 \rightarrow 32 \rightarrow 16$ units), `dropout = 0.354, lr = 0.000766`.
  3. **Mô hình TCN (Temporal Convolutional Network):**
     * Cài đặt tích chập nhân quả (Causal Convolutions) chống rò rỉ tương lai và tích chập giãn nở (Dilated Convolutions) với độ giãn lũy thừa $d \in \{1, 2, 4\}$ bao phủ 20 ngày quá khứ.
     * 3 khối Residual Blocks với 64 kênh, kernel size = 3, huấn luyện song song toàn chuỗi.
  4. **Mô hình Vanilla Transformer & Scaled Dot-Product Attention:**
     * Cài đặt cơ chế chú ý $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ với 4 đầu chú ý (Multi-Head), $d_{\text{model}} = 32, d_{\text{ff}} = 128$.
     * Phân tích nguyên nhân Transformer dễ bị quá khớp do cơ chế tương quan toàn cục bậc 2 quá nhạy cảm với nhiễu tài chính.
  5. **Các kiến trúc đột phá: PatchTST & TSMixer:**
     * **PatchTST:** Cắt chuỗi thành các bản vá (Subseries Patching 8 ngày, stride 4) giảm 75% độ dài tính toán + nguyên lý Channel-Independence (tách riêng từng biến để triệt tiêu nhiễu chéo).
     * **TSMixer:** Kiến trúc All-MLP loại bỏ hoàn toàn Attention, sử dụng luân phiên khối Time-mixing MLP (trộn dọc thời gian) và Feature-mixing MLP (trộn ngang 16 đặc trưng).
  6. **Tối ưu Hóa Huấn luyện & Chống Quá khớp Deep Learning:**
     * Cài đặt cơ chế Early Stopping, Learning Rate Scheduler và kiểm soát gradient cho các mô hình chuỗi sâu.
* **Câu hỏi phản biện của Thầy chịu trách nhiệm chính:**
  * *Hỏi: Kiến trúc mạng nơ-ron ra sao? Bao nhiêu layer, bao nhiêu nơ-ron?* (Slide 22 & Slide 45 Q8)
  * *Hỏi: LSTM giải quyết chuỗi thời gian ntn? Cơ chế 4 cổng vi phân?* (Slide 23 & Slide 45 Q9)
  * *Hỏi: Tại sao CNN áp dụng được cho chuỗi thời gian? TCN khác gì LSTM?* (Slide 24 & Slide 45 Q10)
  * *Hỏi: Cơ chế Attention của Transformer? Tại sao PatchTST và TSMixer vượt trội hơn?* (Slide 25, 26 & Slide 45 Q11, Q12, Q13)

---

### 👤 THÀNH VIÊN 5: TRẦN NGỌC THẢO – CHUYÊN VIÊN ĐÁNH GIÁ THỰC NGHIỆM & BÁO CÁO (EVALUATION LEAD)
* **Tỷ lệ đóng góp:** **15%**
* **Mục tiêu:** Thiết lập chuẩn mực kiểm định thống kê khắt khe, thực hiện các phân tích độ mạnh mẽ, hỗ trợ hoàn thiện bài báo khoa học và bộ slide bảo vệ.
* **Nhiệm vụ cụ thể:**
  1. **Xây dựng hệ thống chỉ số đo lường ngoài mẫu:**
     * Lập trình tính toán $OOS\ R^2$ theo chuẩn Campbell & Thompson (2008), MSE, MAE và Directional Accuracy (DA) cho 12 mô hình $\times$ 5 chân trời $\times$ 3 regimes.
  2. **Thực hiện Kiểm định Thống kê Clark & West (2007):**
     * Lập trình thống kê hiệu chỉnh $\hat{f}_t$ cho mô hình lồng nhau để loại bỏ độ chệch âm khi so sánh mô hình học máy với Historical Mean.
  3. **Kiểm soát sai lầm đa giả thuyết qua hiệu chỉnh Holm-Bonferroni:**
     * Áp dụng thuật toán hiệu chỉnh từng bước (Step-down procedure) trên ma trận 60 bài toán kiểm định; phơi bày hiện tượng **"Ảo ảnh khả năng dự báo" (Predictability Illusion)** khi toàn bộ $p_{\text{adj}} > 0.05$.
  4. **Chủ trì 3 Phân tích Độ mạnh mẽ chuyên sâu (Robustness Analyses):**
     * **Multi-seed Testing:** Chạy kiểm thử 10 random seeds độc lập trên các mô hình Deep Learning; chứng minh con số $OOS\ R^2 = 17,53\%$ của TSMixer ở 120D thực chất chỉ là cực trị may mắn của riêng Seed 42 (trung bình 10 seed rớt xuống $-18,93\%$).
     * **Feature Ablation Study (theo Hooker et al.):** Lập trình quy trình retrain-from-scratch bóc tách từng nhóm đặc trưng trên XGBoost, Ridge, LSTM.
     * **Zero-Return Benchmark & Cửa sổ Regime thay thế:** Kiểm chứng tính vững với mốc $\hat{y}=0$ và cửa sổ biến động 10 ngày, 60 ngày.
  5. **Hỗ trợ xuất bản & hoàn thiện sản phẩm báo cáo:**
     * Hỗ trợ soạn thảo và hoàn thiện bản thảo bài báo khoa học chuẩn Springer LNCS 47 trang (`paper/springer_manuscript/samplepaper.tex`).
     * Thiết kế toàn bộ **Bộ slide thuyết trình 45 trang** (`docs/slide_deck_comprehensive.md`) với đầy đủ sơ đồ box-drawing, công thức KaTeX chuẩn xác, Speaker Notes và Q&A Cheat-sheet.
     * Cập nhật trang giao diện trực quan hóa [dashboard.html](file:///E:/Repo%20Project/IS403.R12-equity-return-predictability/dashboard.html).
  6. **Điều phối Phân tích Thống kê & Báo cáo Định lượng:**
     * Tổng hợp số liệu định lượng, xây dựng bảng đối sánh đa tiêu chí và hỗ trợ rà soát kiểm định giả thuyết.
* **Câu hỏi phản biện của Thầy chịu trách nhiệm chính:**
  * *Hỏi: Tiêu chí OOS R² tính như thế nào? Ý nghĩa kinh tế học của OOS R² > 0?* (Slide 28)
  * *Hỏi: Tại sao TSMixer đạt 17,53% ở 120D mà nhóm kết luận không có ý nghĩa thống kê? Hiện tượng ảo ảnh khả năng dự báo là gì?* (Slide 37, 38 & Slide 45 Q13)
  * *Hỏi: Phân tích bóc tách đặc trưng (Feature Ablation) thực hiện như thế nào? Nhóm biến nào quan trọng nhất ở kỳ ngắn và kỳ dài?* (Slide 39, 40)

---

## 📊 MA TRẬN TRÁCH NHIỆM RACI (RACI MATRIX)

*Ký hiệu:*
* **R (Responsible):** Người trực tiếp thực hiện nhiệm vụ chính.
* **A (Accountable):** Người chịu trách nhiệm phê duyệt và kết quả cuối cùng.
* **C (Consulted):** Người được tham vấn ý kiến kỹ thuật.
* **I (Informed):** Người nhận thông tin để phối hợp.

| Hạng mục công việc dự án | TV 1 (Data) | TV 2 (Linear) | TV 3 (Trees/SVR) | TV 4 (Deep Seq) | TV 5 (Eval/Paper) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Thu thập & Làm sạch dữ liệu (S&P 500, VIX)** | **R / A** | I | I | I | C |
| **2. Kỹ nghệ 16 Đặc trưng & Bộ đệm Lookback** | **R / A** | C | C | C | I |
| **3. Mô hình ARIMA, ARIMAX (p, d, q) & Benchmark** | C | **R / A** | I | I | C |
| **4. Mô hình Tuyến tính phạt Ridge, Lasso, ElasticNet** | C | **R / A** | I | I | C |
| **5. Mô hình Cây Random Forest, LightGBM, XGBoost** | I | I | **R / A** | I | C |
| **6. Mô hình SVR (Support Vector Regression)** | I | C | **R / A** | I | C |
| **7. Mô hình Mạng nơ-ron MLP / ANN (Chương 6)** | I | I | I | **R / A** | C |
| **8. Mô hình LSTM, TCN (Dilated Causal)** | I | I | I | **R / A** | C |
| **9. Mô hình Transformer, PatchTST, TSMixer** | I | I | I | **R / A** | C |
| **10. Khung đánh giá ngoài mẫu ($OOS\ R^2$, DA, MSE)** | I | C | C | C | **R / A** |
| **11. Kiểm định Clark-West & Hiệu chỉnh Holm-Bonferroni** | I | C | C | C | **R / A** |
| **12. Phân tích Multi-seed (10 seeds) & Ablation Study** | I | C (Ridge) | C (XGB) | C (LSTM) | **R / A** |
| **13. Soạn thảo Bài báo khoa học chuẩn Springer LNCS (47 trang)** | C | C | C | C | **R / A** |
| **14. Biên soạn Bộ Slide 45 trang & Speaker Notes** | C | C | C | C | **R / A** |

---

## 🎯 BẢNG PHÂN CÔNG THUYẾT TRÌNH TRONG BUỔI BẢO VỆ ĐỒ ÁN (45 SLIDES)

Để buổi bảo vệ đồ án diễn ra liền mạch, chuyên nghiệp và tất cả 5 thành viên đều có cơ hội thể hiện năng lực trước hội đồng, 45 slide được phân bổ như sau:

* **Thành viên 1 trình bày Slide 1 đến Slide 14 (~6 phút):**
  * Tiêu đề, Executive Summary, Định nghĩa bài toán Input/Output, Tổng quan học thuật, 3 Câu hỏi nghiên cứu (RQ1–RQ3), Nguồn dữ liệu, Chuỗi đơn biến vs Đa biến, 16 Đặc trưng tài chính, 5 Chân trời dự báo, 3 Chế độ biến động, Phân chia thời gian và Kỹ thuật Lookback Buffer.
* **Thành viên 2 trình bày Slide 15 đến Slide 17 (~3 phút):**
  * Phân loại 12 mô hình & Template chuẩn mực, Quy trình xác định $(p, d, q)$ cho ARIMA/ARIMAX (lý do loại bỏ AutoARIMA), Cơ chế phạt co ngót L2 của Ridge và L1 của Lasso/ElasticNet.
* **Thành viên 3 trình bày Slide 18 đến Slide 21 (~4 phút):**
  * Thuật toán xây dựng cây của Random Forest (Bagging), Cải tiến Leaf-wise Histogram của LightGBM, Khai triển Taylor bậc 2 và công thức Gain của XGBoost, Mở rộng SVM sang hồi quy SVR qua ống sai số $\varepsilon$ và Kernel RBF.
* **Thành viên 4 trình bày Slide 22 đến Slide 27 (~5 phút):**
  * Sơ đồ kiến trúc mạng nơ-ron chuẩn Chương 6 ANN, Cơ chế 4 cổng vi phân của LSTM, Tích chập giãn nở nhân quả của TCN, Scaled Dot-Product Self-Attention của Transformer, Cơ chế Patching và Channel-Independence của PatchTST, Kiến trúc All-MLP của TSMixer và Bảng Master thông số kỹ thuật 12 mô hình.
* **Thành viên 5 trình bày Slide 28 đến Slide 45 (~7 phút):**
  * Khung đánh giá ngoài mẫu, Kiểm định Clark-West & Hiệu chỉnh Holm-Bonferroni, Bảng kết quả OOS R², Biểu đồ trực quan, Trả lời thực nghiệm RQ1, RQ2, RQ3, Directional Accuracy, Bẫy ảo ảnh khả năng dự báo, Kiểm chứng 10 seeds, Bóc tách đặc trưng Feature Ablation, Bài học rút ra, Hàm ý quản trị danh mục, Kết luận và điều phối phần Hỏi - Đáp Q&A.

---

*Tài liệu được lưu trữ chính thức tại thư mục:* `docs/team_assignment/PHAN_CONG_NHIEM_VU_5_THANH_VIEN.md`

