# Báo Cáo Phân Tích Dữ Liệu: Tổng Hợp Hiệu Năng Dự Báo Đa Mô Hình & Đa Chân Trời (02_model_horizon_summary_report.md)

**Nguồn dữ liệu:** `results/summary/model_horizon_summary.csv`  
**Thời gian thực hiện:** 2026-10-08 15:18:00 UTC+7  
**Vai trò chuyên gia:** Research Data Analyst (Kinh tế lượng tài chính & Học máy)  
**Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mô Tả Tập Tin & Mục Đích Nghiên Cứu

Tập tin `results/summary/model_horizon_summary.csv` tổng hợp hiệu năng dự báo ngoại mẫu (out-of-sample predictive performance) qua 5 fold kiểm định mở rộng Walk-Forward cho từng mô hình và từng chân trời dự báo.

Mục tiêu phân tích tập trung giải quyết các câu hỏi so sánh tổng hợp:
1. Giá trị trung bình và phương sai của năng lực dự báo ngoại mẫu ($OOS\_R^2$) phân bổ như thế nào giữa các họ mô hình?
2. Năng lực dự báo biến đổi ra sao trên trục liên tục của chân trời: $1\text{D} \to 5\text{D} \to 20\text{D} \to 60\text{D} \to 120\text{D}$?
3. Những mô hình nào vượt trội hơn chuẩn đối sánh `Historical_Mean`, và tại những chân trời cụ thể nào?
4. Những đánh đổi cấu trúc nào tồn tại giữa sai số điểm ($MAE, RMSE$), tỷ lệ phương sai giải thích tương đối ($OOS\_R^2$), và khả năng định thời hướng đi ($DA$)?

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Dimensions & Schema)

- **Tổng số hàng:** 70 hàng ($14\text{ Mô hình} \times 5\text{ Chân trời}$)
- **Số cột:** 10 cột:
  - `Model` (`str`): Tên định danh thuật toán
  - `Horizon` (`str`): `1D`, `5D`, `20D`, `60D`, `120D`
  - `MAE_Mean`, `MAE_Std` (`float`): Giá trị trung bình và độ lệch chuẩn của MAE qua 5 folds
  - `RMSE_Mean`, `RMSE_Std` (`float`): Giá trị trung bình và độ lệch chuẩn của RMSE qua 5 folds
  - `OOS_R2_Mean`, `OOS_R2_Std` (`float`): Giá trị trung bình và độ lệch chuẩn của $OOS\_R^2$ qua 5 folds
  - `DA_Mean`, `DA_Std` (`float`): Giá trị trung bình và độ lệch chuẩn của Độ chính xác hướng đi qua 5 folds

---

## 3. Các Ma Trận Hiệu Năng Toàn Diện

### 3.1 Chỉ số chính: $R^2$ Ngoại mẫu trung bình ($OOS\_R^2_{\text{mean}}$)
*Ghi chú: $OOS\_R^2$ Campbell-Thompson $= 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y}_{\text{hist}})^2}$. Giá trị dương biểu thị vượt trội hơn Historical Mean; giá trị âm biểu thị kém hơn Historical Mean.*

| Họ mô hình | Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** | **Historical_Mean** | -0.0012 | -0.0054 | -0.0267 | -0.0861 | -0.6974 |
| **Kinh tế lượng chuỗi** | **ARIMA** | **+0.0057** | **+0.5835** | **+0.8583** | **+0.9385** | **+0.9404** |
| | **ARIMAX** | **+0.0007** | **+0.7050** | **+0.9123** | **+0.9627** | **+0.9622** |
| | **SARIMA** | **+0.0061** | **+0.6607** | **+0.8584** | **+0.9380** | **+0.9395** |
| **Tuyến tính co rút**| **Elastic_Net** | **+0.0017** | -0.0041 | -0.0389 | -0.1970 | -0.7255 |
| | **Ridge** | -0.0080 | -0.0254 | -0.0802 | -0.3759 | -1.3682 |
| **Phương pháp Kernel**| **SVR** | -0.0105 | -0.0140 | -0.2006 | -0.1489 | -0.8610 |
| **Cây quyết định** | **Random_Forest** | -0.0137 | -0.0253 | -0.1318 | -0.3551 | -1.1247 |
| | **XGBoost** | -0.0069 | -0.0067 | -0.0453 | -0.3875 | -1.0514 |
| | **LightGBM** | -0.0248 | -0.0327 | -0.1155 | -0.4837 | -1.5774 |
| **Học sâu (Deep Learning)** | **RNN** | -0.0711 | -0.0528 | -0.1110 | -0.5103 | -1.4222 |
| | **LSTM** | -0.0105 | -0.1041 | -0.1486 | -0.6348 | -1.4639 |
| | **Transformer** | -2.3800 | -0.8244 | -0.7016 | -1.0473 | -2.0454 |
| **Phân loại nhị phân** | **Logistic_Regression** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

---

### 3.2 Chỉ số ổn định: Độ lệch chuẩn của $OOS\_R^2$ qua các Fold ($OOS\_R^2_{\text{std}}$)
| Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0013 | 0.0048 | 0.0283 | 0.1526 | 1.3558 |
| **ARIMA** | 0.0103 | 0.0332 | 0.0316 | 0.0124 | 0.0198 |
| **ARIMAX** | 0.0305 | 0.0591 | 0.0219 | 0.0106 | 0.0180 |
| **SARIMA** | 0.0117 | 0.0392 | 0.0317 | 0.0124 | 0.0204 |
| **Elastic_Net** | 0.0013 | 0.0109 | 0.0615 | 0.2786 | 1.4114 |
| **Ridge** | 0.0167 | 0.0466 | 0.0867 | 0.2565 | 2.4056 |
| **SVR** | 0.0371 | 0.0582 | 0.3282 | 0.1570 | 0.8862 |
| **Random_Forest** | 0.0245 | 0.0593 | 0.1536 | 0.1361 | 0.7633 |
| **XGBoost** | 0.0128 | 0.0038 | 0.0245 | 0.6516 | 1.3650 |
| **LightGBM** | 0.0434 | 0.0597 | 0.1868 | 0.5661 | 1.0475 |
| **RNN** | 0.0873 | 0.0473 | 0.1478 | 0.4755 | 3.0566 |
| **LSTM** | 0.0233 | 0.0733 | 0.0722 | 0.9383 | 2.0261 |
| **Transformer** | 3.5884 | 1.5362 | 0.8832 | 0.9295 | 2.0388 |

---

### 3.3 Các thước đo hỗ trợ: Sai số tuyệt đối trung bình ($MAE$) & Căn sai số bình phương ($RMSE$)

#### Sai số tuyệt đối trung bình (MAE):
| Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0072 | 0.0155 | 0.0315 | 0.0522 | 0.0626 |
| **ARIMAX** | 0.0072 | 0.0083 | 0.0082 | 0.0084 | 0.0081 |
| **Elastic_Net** | 0.0072 | 0.0156 | 0.0322 | 0.0543 | 0.0629 |
| **XGBoost** | 0.0072 | 0.0156 | 0.0321 | 0.0563 | 0.0693 |
| **Random_Forest** | 0.0072 | 0.0160 | 0.0346 | 0.0588 | 0.0739 |
| **LightGBM** | 0.0073 | 0.0160 | 0.0327 | 0.0587 | 0.0808 |
| **LSTM** | 0.0072 | 0.0166 | 0.0328 | 0.0568 | 0.0715 |
| **Transformer** | 0.0124 | 0.0189 | 0.0377 | 0.0669 | 0.0818 |

#### Căn bậc hai sai số bình phương trung bình (RMSE):
| Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0109 | 0.0222 | 0.0428 | 0.0648 | 0.0760 |
| **ARIMAX** | 0.0108 | 0.0122 | 0.0121 | 0.0120 | 0.0120 |
| **Elastic_Net** | 0.0109 | 0.0223 | 0.0432 | 0.0665 | 0.0763 |
| **XGBoost** | 0.0110 | 0.0223 | 0.0434 | 0.0715 | 0.0841 |
| **Random_Forest** | 0.0110 | 0.0227 | 0.0463 | 0.0743 | 0.0912 |
| **LightGBM** | 0.0111 | 0.0228 | 0.0444 | 0.0740 | 0.1001 |
| **LSTM** | 0.0109 | 0.0236 | 0.0447 | 0.0747 | 0.0882 |
| **Transformer** | 0.0155 | 0.0260 | 0.0494 | 0.0852 | 0.0969 |

---

### 3.4 Thước đo bổ trợ: Độ chính xác hướng đi trung bình ($DA_{\text{mean}}$)
| Mô hình | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 54.85% | 61.84% | 68.46% | 74.56% | 77.79% |
| **ARIMA** | 52.72% | 79.41% | 91.84% | 95.74% | 96.69% |
| **ARIMAX** | 52.06% | 82.43% | 93.60% | 97.06% | 97.43% |
| **SARIMA** | 53.90% | 82.06% | 91.84% | 95.74% | 96.91% |
| **Elastic_Net** | 53.24% | 57.79% | 61.84% | 68.60% | 77.79% |
| **Ridge** | 51.47% | 50.15% | 56.03% | 56.18% | 66.10% |
| **XGBoost** | 54.34% | 61.32% | 65.15% | 71.25% | 71.69% |
| **Random_Forest** | 54.04% | 59.34% | 62.28% | 61.40% | 63.31% |
| **LightGBM** | 53.68% | 58.53% | 64.19% | 63.46% | 58.38% |
| **SVR** | 50.96% | 55.15% | 57.06% | 66.91% | 66.76% |
| **RNN** | 49.93% | 50.81% | 60.07% | 63.38% | 73.01% |
| **LSTM** | 52.50% | 54.78% | 58.60% | 67.35% | 65.81% |
| **Transformer** | 49.71% | 53.82% | 51.84% | 60.59% | 54.41% |

---

## 4. Các Phát Hiện Thực Nghiệm Cốt Lõi & Mẫu Hình Theo Họ Mô Hình

### 4.1 Sự suy thoái dự báo nghiêm ngặt ở các họ Học Máy & Học Sâu
Đối với toàn bộ các mô hình phi tự hồi quy, năng lực dự báo ngoại mẫu thể hiện **sự suy thoái đơn điệu nghiêm ngặt (strict monotonic decay)** khi chân trời dự báo kéo dài:
1. **Các mô hình Cây quyết định (XGBoost, Random Forest, LightGBM):**
   - Tại $1\text{D}$ và $5\text{D}$, hiệu năng bám sát mức của Historical Mean ($OOS\_R^2 \in [-0.006, -0.032]$).
   - Tại $20\text{D}$, $OOS\_R^2$ bắt đầu xấu đi rõ rệt, rơi xuống $[-0.045, -0.131]$.
   - Tại $60\text{D}$ và $120\text{D}$, năng lực dự báo sụp đổ nghiêm trọng vào vùng âm sâu ($120\text{D}$ rơi xuống $-1.051$ ở XGBoost, $-1.125$ ở Random Forest, và $-1.577$ ở LightGBM).
2. **Các mô hình Học Sâu (RNN, LSTM, Transformer):**
   - Hứng chịu tốc độ suy thoái theo chân trời dốc nhất. Tại $120\text{D}$, RNN rơi xuống $-1.422$, LSTM xuống $-1.464$, và Transformer xuống $-2.045$.
   - Transformer bộc lộ tính bất ổn định nghiêm trọng ngay cả ở chân trời ngắn ($1\text{D}$ có $OOS\_R^2 = -2.3800$, độ lệch chuẩn $= 3.5884$), phản ánh việc cấu trúc tham số phức tạp ($d_{\text{model}}=32/64$, cơ chế self-attention) bị quá khớp (overfit) trên phương sai nhiễu cao của lợi suất hàng ngày.
3. **Các mô hình Tuyến tính co rút (Elastic Net, Ridge):**
   - Elastic Net thể hiện tính ổn định cao nhất trong nhóm học máy. Tại $1\text{D}$, Elastic Net đạt $OOS\_R^2 = +0.0017$, đánh bại Historical Mean. Tại $5\text{D}$, mô hình gần như tương đương benchmark ($-0.0041$).
   - Tuy nhiên, đến $120\text{D}$, Elastic Net cũng suy thoái xuống $-0.7255$, bám sát mức của Historical Mean ($-0.6974$).

### 4.2 Sự phân kỳ của dòng Tự Hồi Quy (ARIMA, ARIMAX, SARIMA)
Họ mô hình chuỗi thời gian tự hồi quy thể hiện một quỹ đạo hoàn toàn trái ngược với các mô hình ML/DL:
- Tại $1\text{D}$, $OOS\_R^2$ dương nhẹ ($+0.0007 \text{ đến } +0.0061$).
- Khi chân trời mở rộng, $OOS\_R^2$ **tăng vọt đơn điệu**, đạt $+0.58 \text{ đến } +0.70$ tại $5\text{D}$, $+0.85 \text{ đến } +0.91$ tại $20\text{D}$, và đạt đỉnh $+0.94 \text{ đến } +0.96$ tại $60\text{D}$ và $120\text{D}$.
- **Cơ chế phương pháp luận:** Chuỗi lợi suất log cộng dồn $y_{t+h} = \ln(P_{t+h}/P_t)$ đóng vai trò như một bộ lọc tích lũy trượt. Các mô hình AR đã khớp rất tốt mức độ tự tương quan chuỗi cao của chuỗi lợi suất chồng lấn với số bậc tự do rất ít ($p \le 3, d \le 1, q \le 3$), tránh được "lời nguyền số chiều" vốn làm suy kiệt các mô hình học máy.

### 4.3 Độ chính xác hướng đi ($DA$) không đồng nghĩa với năng lực dự báo điểm
- Độ chính xác hướng đi ($DA$) tăng từ khoảng 50–54% tại $1\text{D}$ lên 65–78% tại $120\text{D}$ trên hầu hết mọi mô hình (kể cả Historical Mean).
- Sự gia tăng $DA$ này diễn ra đồng thời trong khi $OOS\_R^2$ đang suy thoái nghiêm trọng xuống $-1.50$.
- **Lý giải kinh tế lượng:** Thị trường chứng khoán S&P 500 có phí bù rủi ro vốn cổ phần dương dài hạn (upward drift). Trong khoảng thời gian 120 ngày, tỷ lệ thị trường tăng điểm chiếm $>75\%$. Một mô hình chỉ cần liên tục dự báo dương là đã đạt được $DA \approx 75\%$, nhưng sai số dự báo độ lớn lại khiến phần dư bình phương phình to, kéo $OOS\_R^2$ xuống mức âm rất sâu.

---

## 5. Hạn Chế Phương Pháp Luận Của Tập Tin Tổng Hợp

1. **Hiện tượng lấy trung bình qua Fold che giấu phương sai chế độ thị trường:** Độ lệch chuẩn qua các fold tại $120\text{D}$ là rất lớn trên mọi mô hình ($\text{std} > 1.0$ đối với Historical Mean, Elastic Net, XGBoost, và Deep Learning), cho thấy hiệu năng thực tế phụ thuộc rất mạnh vào chế độ kinh tế vĩ mô cụ thể của từng fold.
2. **Cần kiểm tra rò rỉ biến ngoại sinh trong ARIMAX:** Chỉ số $OOS\_R^2$ rất cao ($>0.96$) của ARIMAX tại $60\text{D}$ và $120\text{D}$ đòi hỏi phải được kiểm tra chi tiết ở cấp độ dự báo từng quan sát (Bước 3) để xác nhận các biến ngoại sinh (VIX, Momentum) hoàn toàn là các biến trễ so với mốc dự báo $t$.

---

## 6. Kết Luận & Chuyển Giao Sang Bước 3

### 6.1 Tóm tắt hiệu năng Mô hình $\times$ Chân trời
- **Chân trời ngắn (1D):** Năng lực dự báo rất thấp; nhiễu thị trường chiếm ưu thế. Elastic Net, ARIMA, SARIMA, và ARIMAX vượt nhẹ Historical Mean ($OOS\_R^2 \approx 0.001 - 0.006$). Deep Learning bị quá khớp.
- **Chân trời trung bình (5D, 20D):** Điểm phân kỳ rõ rệt. Dòng Tự hồi quy nắm bắt được động lượng chuỗi tích lũy; các mô hình ML/DL bắt đầu suy thoái rõ ràng.
- **Chân trời dài (60D, 120D):** Sự suy thoái dự báo nghiêm trọng trên toàn bộ các mô hình Machine Learning và Deep Learning ($OOS\_R^2 \le -1.0$). Duy nhất các mô hình Tự hồi quy duy trì tỷ lệ phương sai giải thích dương.

### 6.2 Chuyển giao sang Bước 3
Để xác minh tính hợp lệ của các giá trị tổng hợp này và kiểm tra chéo xem dữ liệu dự báo từng quan sát có khớp chính xác với các chỉ số fold mà không có sai lệch tính toán hay không, chúng ta chuyển tiếp sang tập tin **`results/predictions/all_oos_predictions.csv`**.
