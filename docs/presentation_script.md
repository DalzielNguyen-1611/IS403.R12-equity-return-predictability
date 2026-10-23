# KỊCH BẢN THUYẾT TRÌNH SLIDE GIỚI THIỆU ĐỒ ÁN (BÁM SÁT FILE PDF)

**Học phần:** IS403.R12  
**Tên file slide:** `Giới thiệu đồ án IS403.pdf` (Tổng cộng: 9 Slide)  
**Đề tài:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  
**Giảng viên hướng dẫn:** Thầy Dương Phi Long  
**Nhóm thực hiện:** Nhóm 6  
**Thời lượng trình bày gợi ý:** 7 – 9 phút  

---

## ⏱️ PHÂN BỔ THỜI GIAN GỢI Ý

| Trang Slide | Nội dung slide | Thời lượng ước tính |
| :---: | :--- | :---: |
| **Trang 1** | Slide tiêu đề: Tên đề tài, Nhóm 6, GVHD Thầy Dương Phi Long | 30 giây |
| **Trang 2** | Động lực nghiên cứu (Vấn đề & Mục tiêu) | 1.0 – 1.5 phút |
| **Trang 3** | Input & Output (Bản chất bài toán & 5 kỳ hạn dự báo) | 1.5 phút |
| **Trang 4** | Dataset và Phân chia dữ liệu (3 giai đoạn: 1990–2026) | 1.5 phút |
| **Trang 5** | Phương pháp & Mô hình dự kiến (Bảng 5 nhóm mô hình & Tuning) | 1.5 phút |
| **Trang 6** | Tiêu chí đánh giá (MAE, RMSE, DA, $R^2_{\text{OOS}}$, Kiểm định thống kê) | 2.0 phút |
| **Trang 7 & 8** | Tài liệu tham khảo nền tảng (Hudson, Welch & Goyal, Lo...) | 45 giây |
| **Trang 9** | Lời cảm ơn & Mở phiên Q&A | 30 giây |
| **TỔNG CỘNG** | **Toàn bộ bài báo cáo** | **~8.0 – 9.0 phút** |

---

## 🎤 LỜI THOẠI CHI TIẾT TỪNG SLIDE (BÁM SÁT 100% NỘI DUNG PDF)

---

### 🟢 SLIDE 1 (TRANG 1): SLIDE TIÊU ĐỀ
* **Nội dung trên slide:**  
  * Tiêu đề: *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  
  * Nhóm 6 — Ngày: 15/10/2026 — Mã lớp: IS403.R12 — GVHD: Thầy Dương Phi Long  

> 🗣️ **Lời thoại trình bày:**  
> "Kính thưa Thầy Dương Phi Long cùng toàn thể các bạn sinh viên.  
> 
> Hôm nay, đại diện cho **Nhóm 6** lớp **IS403.R12**, em xin phép được trình bày báo cáo giới thiệu đề tài đồ án môn học với tên gọi:  
> **'When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis'**  
> — tạm dịch là: *Khi nào khả năng dự báo lợi suất cổ phiếu suy thoái? Phân tích ngoài mẫu theo đa kỳ hạn và phụ thuộc chế độ biến động thị trường*.  
> 
> Đề tài được thực hiện dưới sự hướng dẫn của Thầy Dương Phi Long. Sau đây, em xin phép đi vào nội dung chi tiết của bài báo cáo."

🔄 *[Chuyển sang Slide 2]*

---

### 🟢 SLIDE 2 (TRANG 2): ĐỘNG LỰC NGHIÊN CỨU
* **Nội dung trên slide:**  
  * Khả năng dự báo lợi suất cổ phiếu ngoài mẫu thiếu ổn định. [1] [2]  
  * Thiếu sự đối chiếu có hệ thống về khả năng dự báo lợi suất đồng thời theo: 5 kỳ hạn, 3 chế độ biến động và các giai đoạn lịch sử.  
  * Mục tiêu: Đánh giá khả năng dự báo lợi suất S&P 500 theo nhiều kỳ hạn và trạng thái thị trường.  

> 🗣️ **Lời thoại trình bày:**  
> "Về **động lực nghiên cứu**, dự báo lợi suất cổ phiếu là một bài toán khó trong tài chính thực nghiệm, bởi khả năng dự báo ngoài mẫu thường yếu và thiếu ổn định. Cụ thể, **Welch và Goyal (2008) [1]** đã chỉ ra những hạn chế trong hiệu quả dự báo phần bù lợi suất cổ phiếu ngoài mẫu. Tiếp nối nghiên cứu này, **Goyal, Welch và Zafirov (2024) [2]** đánh giá lại các biến dự báo trong những nghiên cứu trước và cho thấy nhiều kết quả không duy trì được hiệu quả khi dữ liệu được cập nhật.  
> 
> Bên cạnh đó, các nghiên cứu trước đã xem xét khả năng dự báo theo nhiều kỳ hạn hoặc trong những điều kiện thị trường khác nhau. Tuy nhiên, đề tài hướng đến việc **đối chiếu có hệ thống ba khía cạnh: kỳ hạn dự báo, chế độ biến động và sự thay đổi qua các giai đoạn lịch sử**, trên cùng chuỗi lợi suất S&P 500 và trong một khung đánh giá ngoài mẫu nhất quán. Đây là khoảng trống nghiên cứu mà đề tài cần kiểm chứng thông qua việc đối chiếu với các công trình liên quan.  
> 
> Xuất phát từ đó, nghiên cứu đánh giá **năm kỳ hạn dự báo từ 1 đến 120 ngày**, ba chế độ biến động gồm thấp, bình thường và cao, đồng thời xem xét tính ổn định của kết quả theo thời gian. Nghiên cứu sử dụng quy trình đánh giá ngoài mẫu cuốn chiếu mở rộng (*Expanding Walk-Forward*) và các biện pháp kiểm soát rò rỉ dữ liệu nhằm bảo đảm tính nhất quán và độ tin cậy của kết quả."  

🔄 *[Chuyển sang Slide 3]*

---

### 🟢 SLIDE 3 (TRANG 3): INPUT & OUTPUT
* **Nội dung trên slide:**  
  * Bản chất bài toán: Chính là Regression (dự báo lợi suất liên tục); Phụ là Classification (dự báo chiều tăng/giảm).  
  * INPUT: Hơn 25 đặc trưng định lượng chia thành 5 nhóm (Lợi suất quá khứ, Động lượng & xu hướng, Khối lượng giao dịch, Độ biến động lịch sử, Chỉ số tâm lý thị trường).  
  * OUTPUT: Lợi suất logarit tương lai $Target_{h,t} = \ln(Close_{t+h} / Close_t)$ trên 5 kỳ hạn: 1D, 5D, 20D, 60D, 120D.  

> 🗣️ **Lời thoại trình bày:**  
> "Về mặt thiết kế bài toán, nhóm xác định rõ:  
> * **Bài toán chính:** Là bài toán **Hồi quy (Regression)** — dự báo trực tiếp giá trị lợi suất liên tục của chỉ số S&P 500.  
> * **Bài toán phụ đối chứng:** Là bài toán **Phân loại (Classification)** — nhằm kiểm tra khả năng bắt nhịp chiều hướng tăng hay giảm của thị trường.  
> 
> Ở nhánh **INPUT (Đặc trưng đầu vào)**: Nhóm xây dựng hệ thống gồm **hơn 25 biến số định lượng**, được phân thành 5 nhóm chỉ báo chính:  
> 1. Lợi suất trong quá khứ.  
> 2. Các chỉ báo động lượng và xu hướng giá.  
> 3. Khối lượng giao dịch.  
> 4. Độ biến động lịch sử.  
> 5. Và các chỉ số tâm lý thị trường (tiêu biểu là chỉ số sợ hãi VIX).  
> 
> Ở nhánh **OUTPUT (Mục tiêu dự báo)**: Nhóm tính toán tỷ suất sinh lợi logarit tương lai theo công thức:  
> **$Target_{h,t} = \ln\left(\frac{Close_{t+h}}{Close_t}\right)$**  
> 
> Công thức này có trích dẫn hai công trình học thuật nền tảng:  
> * Thứ nhất, nghiên cứu của **Hudson & Gregoriou (2015) [1]** trên *International Review of Financial Analysis*: Nhóm lựa chọn **Log Return** thay vì Simple Return nhờ **tính chất cộng dồn thời gian (Time-Additivity)**: lợi suất log tích lũy $h$ ngày chính bằng tổng các lợi suất log hàng ngày: $r_{t, t+h} = \sum_{j=1}^h r_{t+j}$. Đặc tính này giúp đảm bảo tính nhất quán tuyệt đối về mặt toán học khi so sánh khả năng dự báo giữa các chân trời khác nhau từ 1 ngày đến 120 ngày.  
> * Thứ hai, công trình tổng quan kinh điển của **Rapach & Zhou (2013) [2]** trong *Handbook of Economic Forecasting*: Đây là tài liệu quy chuẩn đặt nền tảng phương pháp luận cho toàn bộ thiết kế bài toán dự báo lợi suất cổ phiếu, việc thiết lập mốc chuẩn Historical Mean và khung đánh giá ngoại mẫu (OOS).  
> 
> Mục tiêu này được tính toán đồng thời trên **5 chân trời dự báo**: từ siêu ngắn hạn **1 ngày (1D)**, ngắn hạn **1 tuần (5D)**, trung hạn **1 tháng (20D)**, đến dài hạn **1 quý (60D)** và **nửa năm (120D)**."

🔄 *[Chuyển sang Slide 4]*

---

### 🟢 SLIDE 4 (TRANG 4): DATASET VÀ PHÂN CHIA DỮ LIỆU
* **Nội dung trên slide:**  
  * Nguồn: S&P 500 (`^GSPC`) và CBOE VIX (`^VIX`).  
  * Thời gian: 1990 – 2026, quy mô hơn 9.000 phiên giao dịch, bao phủ nhiều giai đoạn biến động kinh tế và tài chính.  
  * Phân chia: Training (1990–2015), Validation (2015–2020: 5 folds tuần tự), OOS Test (2020–2026: Kiểm tra ngoài mẫu độc lập).  

> 🗣️ **Lời thoại trình bày:**  
> "Về dữ liệu thực nghiệm, nhóm thu thập chuỗi thời gian của chỉ số **S&P 500** và chỉ số biến động **CBOE VIX** trong suốt **36 năm (từ 1990 đến 2026)** với quy mô **hơn 9.000 phiên giao dịch**. Chuỗi dữ liệu dài này đảm bảo mô hình được cọ xát qua mọi chu kỳ kinh tế lớn như khủng hoảng Dot-com 2000, khủng hoảng 2008, và biến động Covid-19 năm 2020.  
> 
> Để đảm bảo tính khách quan và chống rò rỉ dữ liệu theo thời gian, dữ liệu được phân chia thành 3 giai đoạn độc lập:  
> 1. **Giai đoạn Training (1990 – 2015):** Dùng để huấn luyện nền tảng ban đầu cho các mô hình.  
> 2. **Giai đoạn Validation (2015 – 2020):** Gồm **5 folds tuần tự** theo phương pháp Walk-Forward để tinh chỉnh siêu tham số và lựa chọn mô hình tối ưu.  
> 3. **Giai đoạn OOS Test (2020 – 2026):** Tập kiểm tra ngoài mẫu độc lập hoàn toàn, khóa kín và chỉ dùng để đánh giá năng lực thực tế cuối cùng."

🔄 *[Chuyển sang Slide 5]*

---

### 🟢 SLIDE 5 (TRANG 5): PHƯƠNG PHÁP & MÔ HÌNH DỰ KIẾN
* **Nội dung trên slide:**  
  * Tổng cộng: 14 mô hình hồi quy; 1 mô hình phân loại đối chứng (Logistic Regression).  
  * Bảng 5 nhóm phương pháp:  
    - 01: Econometric Models (Historical Mean, ARIMA, SARIMA, ARIMAX) $\rightarrow$ Grid Search (Trừ HM)  
    - 02: Regularized Linear Models (Ridge, Elastic Net) $\rightarrow$ Grid Search  
    - 03: Tree-based Ensembles (Random Forest, XGBoost, LightGBM) $\rightarrow$ Optuna  
    - 04: Support Vector (SVR) $\rightarrow$ Grid Search  
    - 05: Deep Learning (RNN, LSTM, Transformer) $\rightarrow$ Optuna  

> 🗣️ **Lời thoại trình bày:**  
> "Trên Slide 5 là bảng tổng hợp các phương pháp và mô hình thực nghiệm của đề tài. Nhóm triển khai **14 mô hình hồi quy** cùng **1 mô hình phân loại đối chứng** là Logistic Regression, được tổ chức thành 5 nhóm phương pháp:  
> 
> * **Nhóm 1 - Econometric Models:** Gồm `Historical Mean` làm mốc chuẩn cơ sở (không tuning), cùng các mô hình chuỗi thời gian `ARIMA`, `SARIMA` và `ARIMAX` được tinh chỉnh bậc bằng Grid Search.  
> * **Nhóm 2 - Regularized Linear Models:** Gồm `Ridge` và `Elastic Net`, sử dụng Grid Search để tối ưu hóa hệ số điều chuẩn phạt L1/L2.  
> * **Nhóm 3 - Tree-based Ensembles:** Gồm `Random Forest`, `XGBoost`, và `LightGBM`. Do không gian tham số của cây rất lớn, nhóm áp dụng framework **Optuna** để tối ưu hóa tự động.  
> * **Nhóm 4 - Support Vector:** Mô hình `SVR` nhân RBF, tối ưu qua Grid Search.  
> * **Nhóm 5 - Deep Learning:** Gồm mạng hồi quy `RNN`, `LSTM` và kiến trúc `Transformer`, toàn bộ đều được tối ưu siêu tham số thông qua **Optuna** nhằm đảm bảo khai thác tối đa năng lực mô hình."

🔄 *[Chuyển sang Slide 6]*

---

### 🟢 SLIDE 6 (TRANG 6): TIÊU CHÍ ĐÁNH GIÁ
* **Nội dung trên slide:**  
  * Đánh giá đa chiều: MAE, RMSE (Sai số); Directional Accuracy (Độ chính xác xu hướng); Diebold-Mariano & Binomial Sign Test ($p < 0.05$).  
  * Tiêu chí cốt lõi: $R^2_{\text{OOS}} = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y}_{\text{bench},t})^2}$ so với Historical Mean ($R^2_{\text{OOS}} > 0$).  
  * Hạn chế rò rỉ: Không dùng dữ liệu tương lai.  
  * Ý nghĩa kinh tế: $R^2_{\text{OOS}}$ từ 1–3% đã có ý nghĩa đầu tư lớn.  

> 🗣️ **Lời thoại trình bày:**  
> "Về hệ thống đánh giá, nhóm thiết lập khung **đánh giá đa chiều (Multi-dimensional Evaluation)**:  
> * Thứ nhất là **Sai số dự báo**: đo bằng **MAE** (sai số trung bình tuyệt đối) và **RMSE** (phạt nặng các sai số lớn để phòng ngừa rủi ro đuôi).  
> * Thứ hai là **Độ chính xác xu hướng (Directional Accuracy - DA)**: đo tỷ lệ phần trăm số ngày dự báo đúng chiều tăng/giảm của thị trường.  
> * Thứ ba là **Kiểm định thống kê**: Nhóm áp dụng kiểm định **Diebold–Mariano** để so sánh sai số với mốc chuẩn và **Binomial Sign Test** để kiểm tra xác suất đoán đúng chiều có vượt trội hơn tung đồng xu ngẫu nhiên ở mức ý nghĩa $p < 0.05$ hay không.  
> 
> Đặc biệt, **tiêu chí cốt lõi nhất** của đề tài là thước đo kinh tế lượng **$R^2_{\text{OOS}}$ (Out-of-Sample $R^2$)**.  
> Khác với $R^2$ thông thường, công thức này so sánh trực tiếp sai số mô hình với mốc chuẩn trung bình lịch sử lũy tiến $\bar{y}_{\text{bench},t}$.  
> Chỉ khi $R^2_{\text{OOS}} > 0$, mô hình mới chứng minh được là có năng lực thực sự vượt qua chiến lược đoán trung bình. Trong tài chính thực nghiệm, do tỷ lệ tín hiệu trên nhiễu rất thấp, một mô hình chỉ cần đạt **$R^2_{\text{OOS}}$ từ 1% đến 3%** đã mang giá trị kinh tế và lợi thế đầu tư vô cùng lớn."

🔄 *[Chuyển sang Slide 7 & 8]*

---

### 🟢 SLIDE 7 & 8 (TRANG 7 & 8): TÀI LIỆU THAM KHẢO
* **Nội dung trên slide:**  
  * **Trang 7 (Cơ sở phương pháp luận bài toán):**  
    - [1] Hudson & Gregoriou (2015): Lựa chọn Log Return và tính chất cộng dồn thời gian (Time-additivity).  
    - [2] Rapach & Zhou (2013): Cẩm nang kinh điển về dự báo lợi suất cổ phiếu, mốc chuẩn Historical Mean và khung OOS (*Handbook of Economic Forecasting*).  
  * **Trang 8 (Cơ sở lý thuyết & thực nghiệm):**  
    - [1] Welch & Goyal (2008) & [2] Goyal, Welch & Zafirov (2024): Đánh giá toàn diện năng lực dự báo và mốc chuẩn OOS trên *RFS*.  
    - [3] Nechvátalová (2024): Dự báo lợi suất đa chân trời bằng Machine Learning.  
    - [4] Lo (2004): Giả thuyết Thị trường Thích nghi (Adaptive Markets Hypothesis) giải thích sự biến thiên theo chế độ.  

> 🗣️ **Lời thoại trình bày:**  
> "Cơ sở lý thuyết và phương pháp luận của đề tài được nhóm xây dựng dựa trên các công trình nghiên cứu kinh điển và uy tín trong tài chính định lượng:  
> * Ở **Trang 7**, nhóm trích dẫn nghiên cứu của **Hudson & Gregoriou (2015)** làm căn cứ chọn thước đo Log Return có tính chất cộng dồn thời gian; kết hợp cùng cẩm nang tổng quan của **Rapach & Zhou (2013)** trong *Handbook of Economic Forecasting* để định hình phương pháp dự báo và đánh giá ngoại mẫu OOS.  
> * Ở **Trang 8**, các bài báo nền tảng của **Welch & Goyal (2008)** và bản cập nhật mới nhất **(2024)** trên tạp chí *Review of Financial Studies* đặt ra tiêu chuẩn khắt khe về việc so sánh mô hình với mốc Historical Mean. Kết hợp cùng nghiên cứu mới của **Nechvátalová (2024)** về dự báo đa kỳ hạn bằng Machine Learning, và **Giả thuyết Thị trường Thích nghi của Andrew Lo (2004)** giải thích tại sao năng lực dự báo lại thay đổi theo các chế độ thị trường khác nhau."

🔄 *[Chuyển sang Slide 9]*

---

### 🟢 SLIDE 9 (TRANG 9): KẾT THÚC & HỎI ĐÁP (THANK YOU)
* **Nội dung trên slide:**  
  * Thank you — Nhóm 6 — Ngày: 15/10/2026 — Mã lớp: IS403.R12 — GVHD: Thầy Dương Phi Long  

> 🗣️ **Lời thoại trình bày:**  
> "Trên đây là toàn bộ phần trình bày đề cương đồ án môn học của **Nhóm 6**.  
> 
> Nhóm xin chân thành cảm ơn Thầy Dương Phi Long và các bạn đã chú ý theo dõi. Nhóm rất mong nhận được những nhận xét, góp ý quý báu từ Thầy để hoàn thiện đề tài trong các bước thực hiện tiếp theo.  
> Em xin trân trọng cảm ơn Thầy!"

---

## 💡 MẸO TRÌNH BÀY CHO TỪNG SLIDE TRONG FILE PDF

1. **Slide 3 & 4 (Bản chất bài toán & Split):** Hãy nhấn mạnh chữ *"Regression liên tục"* và *"Chống rò rỉ dữ liệu qua 3 giai đoạn độc lập"* để Thầy thấy đề tài có thiết kế bài bản.
2. **Slide 5 (Phương pháp & Bảng Tuning):** Giải thích rõ lý do tại sao nhóm Econometric, Linear, SVR dùng **Grid Search** (không gian nhỏ, vét cạn khách quan) còn Tree và Deep Learning dùng **Optuna** (không gian nhiều chiều, tối ưu hóa thích nghi).
3. **Slide 6 (Tiêu chí cốt lõi):** Giữ nhịp nói chậm lại khi giải thích công thức $R^2_{\text{OOS}}$: *"Mô hình chỉ được công nhận có giá trị khi $R^2_{\text{OOS}} > 0$ so với mốc Historical Mean, và trong tài chính chỉ cần 1% đến 3% đã là có ý nghĩa kinh tế rất lớn"*.
