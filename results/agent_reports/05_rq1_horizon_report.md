# Báo Cáo Nghiên Cứu Câu Hỏi 1: Sự Suy Thoái Của Năng Lực Dự Báo Theo Chân Trời Thời Gian (05_rq1_horizon_report.md)

**Mã tài liệu:** `IS403-REP-05`  
**Ngày thực hiện:** 2026-10-08  
**Tập tin mục tiêu:** `results/analysis/rq1_horizon.csv`  
**Người thực hiện:** Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu Nghiên Cứu RQ1 (Research Question 1 Focus)
Báo cáo này tập trung giải quyết câu hỏi nghiên cứu số 1:
> **Research Question 1 (RQ1):**  
> *"How does out-of-sample equity return predictability change across forecast horizons?"*  
> *(Năng lực dự báo ngoại mẫu của lợi suất cổ phiếu biến đổi như thế nào qua các chân trời dự báo: 1D $\to$ 5D $\to$ 20D $\to$ 60D $\to$ 120D? Có tồn tại quy luật suy thoái dự báo (predictability decay) hay không? Quy luật đó có tính đơn điệu (monotonic) và nhất quán giữa các họ mô hình không?)*

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước vật lý
- **Đường dẫn lưu trữ:** `results/analysis/rq1_horizon.csv`
- **Số hàng:** 14 mô hình (bao gồm benchmark `Historical_Mean`, bài toán phân loại `Logistic_Regression`, và 12 mô hình hồi quy)
- **Số cột:** 11 cột ($Model + 5 \times \{OOS\_R^2, DA\}$)

### 2.2 Cấu trúc dữ liệu
- Các chân trời theo thứ tự thời gian chuẩn hóa: `1D` (1 ngày), `5D` (1 tuần), `20D` (1 tháng), `60D` (1 quý), `120D` (nửa năm / 6 tháng).
- Thước đo chính: **$OOS\_R^2$ (Campbell-Thompson Out-of-Sample $R^2$)**.
- Thước đo bổ trợ: **Directional Accuracy ($DA$)**.

---

## 3. Toàn Văn Ma Trận Hiệu Năng Ngoại Mẫu Theo Chân Trời (Full Empirical Matrix)

| Mô hình | Họ mô hình | 1D $OOS\_R^2$ | 5D $OOS\_R^2$ | 20D $OOS\_R^2$ | 60D $OOS\_R^2$ | 120D $OOS\_R^2$ | Biến thiên ròng ($\Delta_{120D - 1D}$) | Tính Đơn Điệu (Monotonicity) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Historical_Mean** | Benchmark | -0.0001 | -0.0003 | -0.0016 | -0.0115 | -0.0607 | -0.0606 | **Giảm đơn điệu ngặt** |
| **Elastic_Net** | Tuyến tính co rút | +0.0029 | -0.0078 | -0.0178 | -0.0402 | -0.0676 | -0.0705 | **Giảm đơn điệu ngặt** |
| **Random_Forest** | Cây quyết định | -0.0336 | -0.0673 | -0.2567 | -0.3630 | -0.4852 | -0.4516 | **Giảm đơn điệu ngặt** |
| **SVR** | Kernel SVM | +0.0181 | -0.0558 | -0.1121 | -0.1745 | -0.1775 | -0.1956 | **Giảm đơn điệu ngặt** |
| **Ridge** | Tuyến tính co rút | +0.0029 | -0.0583 | -0.1284 | -0.3083 | -0.2854 | -0.2883 | Giảm mạnh (đáy tại 60D) |
| **XGBoost** | Cây tăng cường | -0.0176 | -0.0050 | -0.0435 | -0.2076 | -0.2744 | -0.2568 | Giảm mạnh (đỉnh tại 5D) |
| **LightGBM** | Cây tăng cường | -0.0627 | -0.0748 | -0.0619 | -0.2502 | -0.7834 | -0.7207 | Giảm mạnh phi tuyến |
| **LSTM** | Học sâu chuỗi | +0.0081 | -0.1537 | -0.0680 | -0.2872 | -0.3819 | -0.3900 | Phi đơn điệu, xu hướng giảm |
| **RNN** | Học sâu chuỗi | -0.0247 | -0.0454 | -0.0650 | -0.2845 | -0.0383 | -0.0136 | Phi đơn điệu, dao động |
| **Transformer** | Học sâu Attention | -0.7354 | -0.2996 | -0.1908 | -0.6378 | -0.5573 | +0.1781 | Âm sâu toàn diện |
| **ARIMA** | Tự hồi quy chuỗi | +0.0135 | +0.5892 | +0.8735 | +0.9438 | +0.9561 | +0.9426 | **Tăng đơn điệu ngặt** |
| **ARIMAX** | Tự hồi quy ngoại sinh | +0.0260 | +0.6761 | +0.9223 | +0.9640 | +0.9735 | +0.9475 | **Tăng đơn điệu ngặt** |
| **SARIMA** | Tự hồi quy mùa | +0.0122 | +0.6441 | +0.8737 | +0.9424 | +0.9551 | +0.9429 | **Tăng đơn điệu ngặt** |
| *Logistic_Reg.* | *Phân loại (Side-task)*| *-4478.1* | *-1196.9*| *-388.9* | *-172.6* | *-139.4* | *N/A* | *Không so sánh trực tiếp* |

---

## 4. Ma Trận Độ Chính Xác Hướng Đi (Directional Accuracy Matrix)

| Mô hình | Họ mô hình | 1D DA | 5D DA | 20D DA | 60D DA | 120D DA | Xu hướng biến đổi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Historical_Mean** | Benchmark | 54.85% | 61.84% | 68.46% | 74.56% | 77.79% | **Tăng đơn điệu** |
| **Elastic_Net** | Tuyến tính co rút | 53.24% | 57.79% | 61.84% | 68.60% | 77.79% | **Tăng đơn điệu** |
| **XGBoost** | Cây tăng cường | 54.34% | 61.32% | 65.15% | 71.25% | 71.69% | **Tăng đơn điệu** |
| **RNN** | Học sâu chuỗi | 49.93% | 50.81% | 60.07% | 63.38% | 73.01% | **Tăng đơn điệu** |
| **Ridge** | Tuyến tính co rút | 51.47% | 50.15% | 56.03% | 56.18% | 66.10% | Tăng mạnh sau 5D |
| **SVR** | Kernel SVM | 50.96% | 55.15% | 57.06% | 66.91% | 66.76% | Tăng mạnh |
| **LSTM** | Học sâu chuỗi | 52.50% | 54.78% | 58.60% | 67.35% | 65.81% | Tăng mạnh |
| **Random_Forest** | Cây quyết định | 54.04% | 59.34% | 62.28% | 61.40% | 63.31% | Tăng |
| **LightGBM** | Cây tăng cường | 53.68% | 58.53% | 64.19% | 63.46% | 58.38% | Tăng đến 20D, sau đó giảm |
| **Transformer** | Học sâu Attention | 49.71% | 53.82% | 51.84% | 60.59% | 54.41% | Dao động yếu |
| **ARIMA** | Tự hồi quy chuỗi | 52.72% | 79.41% | 91.84% | 95.74% | 96.69% | **Tăng vọt đơn điệu** |
| **ARIMAX** | Tự hồi quy ngoại sinh | 52.06% | 82.43% | 93.60% | 97.06% | 97.43% | **Tăng vọt đơn điệu** |
| **SARIMA** | Tự hồi quy mùa | 53.90% | 82.06% | 91.84% | 95.74% | 96.91% | **Tăng vọt đơn điệu** |
| **Logistic_Reg.**| Phân loại | 50.59% | 52.28% | 63.97% | 68.46% | 74.56% | Tăng đơn điệu |

---

## 5. Phân Tích Chuyên Sâu Các Phát Hiện Thực Nghiệm (In-Depth Findings)

### 5.1 Năng lực dự báo có thực sự suy thoái không? (Does Predictive Skill Decay?)
Câu trả lời phụ thuộc bản chất cấu trúc của họ mô hình:
1. **Đối với Machine Learning (Tree Ensembles, SVR, Regularized Linear):**  
   **CÓ, BẰNG CHỨNG RẤT MẠNH (STRONG EVIDENCE).**
   - Tại chân trời ngắn $1D$, các mô hình này đạt hiệu năng tương đương hoặc nhỉnh hơn nhẹ so với Historical Mean ($OOS\_R^2$ dương nhẹ tại Elastic Net: $+0.0029$, Ridge: $+0.0029$, SVR: $+0.0181$).
   - Tuy nhiên, khi chân trời mở rộng ra $20D \to 60D \to 120D$, năng lực dự báo sụp đổ nghiêm trọng. Đến chân trời $120D$, $OOS\_R^2$ của Random Forest rơi xuống $-0.4852$, LightGBM rớt xuống $-0.7834$, XGBoost là $-0.2744$, và SVR là $-0.1775$.
   - **Mức độ suy thoái (Magnitude of Decay):** Biên độ sụt giảm từ $1D$ đến $120D$ nằm trong khoảng $-0.07$ đến $-0.72$, cho thấy các thuật toán học máy dựa trên đặc trưng kỹ thuật/biến động hoàn toàn mất khả năng ước lượng quy mô lợi suất khi chân trời đầu tư vượt quá 1 tháng.

2. **Đối với Mô Hình Tự Hồi Quy Kinh Tế Lượng (ARIMA, ARIMAX, SARIMA):**  
   **KHÔNG SUY THOÁI, MÀ TĂNG TRƯỞNG ĐƠN ĐIỆU VƯỢT TRỘI (STRONG CONTRADICTORY EVIDENCE TO DECAY).**
   - Thay vì suy giảm, $OOS\_R^2$ tăng vọt từ mức khiêm tốn $+0.0135$ (1D) lên $+0.5892$ (5D), $+0.8735$ (20D), $+0.9438$ (60D) và đạt đỉnh **$+0.9561$ (ARIMA) / $+0.9735$ (ARIMAX)** tại $120D$.
   - **Bản chất kinh tế lượng:** Chuỗi mục tiêu $Target_{h,t} = \ln(Close_{t+h} / Close_t)$ ở các chân trời dài là chuỗi lợi suất gộp có bước trượt liên tục (overlapping returns). Khi $h$ tăng, chuỗi lợi suất trở nên có tính quán tính tự tương quan (autocorrelation persistence) cực kỳ lớn. Các mô hình AR đã khai thác triệt để cấu trúc trễ này, khiến phương sai sai số của mô hình AR thấp hơn rất nhiều so với phương sai mẫu mở rộng của Historical Mean.

3. **Đối với Mạng Nơ-ron Học Sâu (Deep Learning):**  
   **BẰNG CHỨNG YẾU VÀ KHÔNG ĐƠN ĐIỆU (WEAK / NON-MONOTONIC EVIDENCE).**
   - LSTM và RNN có $OOS\_R^2$ âm ở hầu hết các chân trời nhưng biến động không đồng nhất (ví dụ RNN tại $120D$ lại phục hồi lên $-0.0383$ sau khi chạm đáy $-0.2845$ tại $60D$).
   - Transformer chịu ảnh hưởng của hiện tượng "đói dữ liệu" (sample starvation) nghiêm trọng trên tập chuỗi thời gian tabular 1,360 ngày, dẫn đến $OOS\_R^2$ luôn âm sâu ($-0.7354$ tại 1D và $-0.5573$ tại 120D).

### 5.2 Nghịch lý giữa Năng Lực Điểm ($OOS\_R^2$) và Độ Chính Xác Hướng Đi ($DA$)
- Trong khi $OOS\_R^2$ của các mô hình ML suy thoái nghiêm trọng khi $h$ tăng, thì chỉ số $DA$ của tất cả các mô hình (kể cả Benchmark `Historical_Mean`) lại **tăng mạnh theo chân trời**.
- Cụ thể: `Historical_Mean` có $DA$ tăng từ $54.85\%$ (1D) lên $77.79\%$ (120D); `XGBoost` có $DA$ tăng từ $54.34\%$ lên $71.69\%$.
- **Giải thích:** Sự gia tăng của $DA$ ở chân trời dài là do **phí bù rủi ro vốn cổ phần dài hạn (positive equity drift)** của chỉ số S&P 500 (trong 120 ngày, tỷ lệ các giai đoạn thị trường tăng giá chiếm áp đảo). Các mô hình ML chỉ cần đưa ra dự báo thiên lệch dương nhỏ là đã đạt được $DA$ cao, nhưng sai lệch về biên độ độ lớn khiến phần dư bình phương $(\text{Actual} - \text{Prediction})^2$ tăng nhanh, đẩy $OOS\_R^2$ xuống mức âm.

### 5.3 Cô lập bài toán phân loại Logistic Regression
- Giá trị $OOS\_R^2$ của `Logistic_Regression` trong file ghi nhận $-4478.11$ (1D) đến $-139.37$ (120D).
- Đây là hệ quả kỹ thuật khi lấy xác suất hoặc nhãn nhị phân $\{0, 1\}$ so khớp vào công thức phương sai hồi quy lợi suất liên tục $\approx 0.0003$.
- Tuy nhiên, trên không gian bài toán phân loại hướng đi, $DA$ của Logistic Regression tăng ổn định từ $50.59\%$ (1D) lên $74.56\%$ (120D), phù hợp với quy luật trôi dạt thị trường.

---

## 6. Phân Loại Mức Độ Bằng Chứng Khoa Học (Evidence Classification)

| Câu hỏi bộ phận | Họ mô hình liên quan | Phân loại bằng chứng | Tóm tắt kết luận thực nghiệm |
| :--- | :--- | :--- | :--- |
| **Sự suy thoái đơn điệu ở ML/Linear** | Tree Ensembles, SVR, Elastic Net, Ridge | **Strong Evidence (Bằng chứng mạnh)** | $OOS\_R^2$ giảm dốc từ 1D đến 120D; 3 mô hình giảm đơn điệu 100%. |
| **Sự suy thoái ở Deep Learning** | LSTM, RNN, Transformer | **Weak / Non-monotonic Evidence** | Hiệu năng kém nhưng không giảm đều; dao động phi tuyến. |
| **Sự suy thoái ở Time-series AR** | ARIMA, ARIMAX, SARIMA | **Contradictory Evidence (Bằng chứng trái chiều)** | Hiệu năng tăng mạnh đơn điệu theo chân trời do tận dụng quán tính chuỗi gộp. |
| **Xu hướng chung toàn thị trường** | Tổng hợp 13 mô hình hồi quy | **Heterogeneous Pattern (Mẫu hình đa dạng)** | Không thể khái quát hóa một chiều: ML suy thoái dốc trong khi AR tăng dốc. |

---

## 7. Hạn Chế Phương Pháp Luận Khi Đánh Giá RQ1 (Limitations)
1. **Hiệu ứng chồng lấn chuỗi thời gian (Overlapping Horizon Effect):**  
   Ở chân trời $h = 60D$ và $120D$, các quan sát liên tiếp có phần dữ liệu giá trị chồng chéo lên nhau $59$ và $119$ ngày, tạo ra hiện tượng tự tương quan chuỗi bậc cao trong phần dư. Cần bổ sung kiểm định thống kê Diebold-Mariano có điều chỉnh Newey-West (`statistical_tests.csv`) để kiểm tra xem sự vượt trội của ARIMA/ARIMAX có ý nghĩa thống kê hay không.
2. **Thiếu sự điều chỉnh theo độ biến động của mục tiêu:** Khi $h$ tăng, phương sai của $Target_{h}$ tăng gần như tỷ lệ thuận với $h$, khiến sai số dự báo tuyệt đối (RMSE) tự nhiên phình to.

---

## 8. Kết Luận Bước 5 & Kế Hoạch Tiếp Theo (Conclusions & Next Steps)
- **Kết luận RQ1:** Năng lực dự báo ngoại mẫu của lợi suất S&P 500 phân hóa sâu sắc theo họ mô hình:
  - Các mô hình Machine Learning và Hồi quy Tuyến tính co rút thể hiện **sự suy thoái năng lực dự báo rõ rệt và đơn điệu (Predictability Decay)** khi chân trời tăng.
  - Các mô hình Tự hồi quy chuỗi thời gian (ARIMA/ARIMAX) đi ngược quy luật suy thoái, ghi nhận **năng lực dự báo tăng vọt** nhờ quán tính tự tương quan của chuỗi lợi suất gộp.
- **Bước tiếp theo:** Chuyển sang **STEP 6** — Phân tích tập tin trả lời câu hỏi nghiên cứu số 2: `results/analysis/rq2_regime.csv` nhằm làm rõ mối quan hệ giữa chân trời dự báo và các chế độ biến động thị trường (Low, Normal, High Volatility Regimes).
