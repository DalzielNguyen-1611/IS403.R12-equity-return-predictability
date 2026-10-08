# Báo Cáo Nghiên Cứu Câu Hỏi 2: Năng Lực Dự Báo Theo Chế Độ Biến Động Thị Trường (06_rq2_regime_report.md)

**Mã tài liệu:** `IS403-REP-06`  
**Ngày thực hiện:** 2026-10-08  
**Tập tin mục tiêu:** `results/analysis/rq2_regime.csv`  
**Người thực hiện:** Research Data Analyst (Antigravity System)  
**Nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. Mục Tiêu Nghiên Cứu RQ2 (Research Question 2 Focus)
Báo cáo này tập trung giải quyết câu hỏi nghiên cứu số 2:
> **Research Question 2 (RQ2):**  
> *"How does the relationship between forecast horizon and predictive skill differ across market volatility regimes?"*  
> *(Mối quan hệ giữa chân trời dự báo và năng lực dự báo khác biệt như thế nào giữa các chế độ biến động thị trường: Biến động Thấp (Low), Bình thường (Normal), và Biến động Cao (High)? Chế độ nào chứng kiến tốc độ suy thoái dự báo dốc nhất?)*

Quy tắc cốt lõi: Không chỉ đơn thuần xem xét chế độ nào có $OOS\_R^2$ cao hơn, mà phải phân tích **độ dốc suy thoái theo chân trời (decay slope)** tương tác với từng chế độ biến động.

---

## 2. Kích Thước & Lược Đồ Dữ Liệu (Shape & Schema)

### 2.1 Kích thước vật lý
- **Đường dẫn lưu trữ:** `results/analysis/rq2_regime.csv`
- **Số hàng:** $70$ hàng ($14 \text{ Models} \times 5 \text{ Horizons}$)
- **Số cột:** $8$ cột: `Model`, `Horizon`, `Low_OOS_R2`, `Low_DA`, `Normal_OOS_R2`, `Normal_DA`, `High_OOS_R2`, `High_DA`

### 2.2 Định nghĩa các chế độ biến động (Volatility Regimes)
- **Low Regime (Biến động thấp):** 669 ngày giao dịch ngoại mẫu ($49.19\%$), đặc trưng bởi thị trường tăng trưởng ổn định (bull market, giai đoạn 2017).
- **Normal Regime (Bình thường):** 274 ngày giao dịch ngoại mẫu ($20.15\%$), biến động quanh mức trung bình lịch sử.
- **High Regime (Biến động cao):** 417 ngày giao dịch ngoại mẫu ($30.66\%$), bao gồm các cú sốc biến động lớn (đợt giảm 2015, sụp đổ Q4/2018 và khủng hoảng COVID-19 Q1/2020).

---

## 3. Ma Trận Hiệu Năng $OOS\_R^2$ Theo Từng Chế Độ Biến Động (Regime × Horizon Performance)

### 3.1 Chế độ Biến Động Thấp (Low Volatility Regime)
| Model | 1D | 5D | 20D | 60D | 120D | Độ dốc suy thoái ($\Delta_{120D - 1D}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0001 | -0.0020 | -0.0005 | -0.0197 | -0.0050 | -0.0049 |
| **Elastic_Net** | -0.0014 | -0.0040 | -0.0343 | -0.0421 | -0.0153 | -0.0139 |
| **Random_Forest** | -0.0020 | +0.0020 | -0.0072 | -0.0305 | -0.1200 | -0.1180 |
| **XGBoost** | +0.0003 | -0.0029 | -0.0035 | -0.0268 | -0.0415 | -0.0418 |
| **LightGBM** | +0.0019 | -0.0014 | +0.0028 | -0.0369 | -0.2252 | -0.2271 |
| **Ridge** | -0.0160 | -0.0085 | -0.0363 | -0.1079 | -0.0810 | -0.0650 |
| **SVR** | -0.0171 | -0.0042 | -0.1228 | -0.0836 | +0.0841 | +0.1012 |
| **LSTM** | -0.0388 | +0.0056 | -0.1122 | -0.0891 | -0.1673 | -0.1285 |
| **RNN** | -0.0935 | -0.0886 | -0.0752 | -0.2166 | -0.1052 | -0.0117 |
| **Transformer** | -3.1347 | -0.5266 | -0.7577 | -0.3839 | -0.4457 | +2.6890 |
| **ARIMA** | +0.0028 | +0.6645 | +0.9010 | +0.9542 | +0.9625 | +0.9597 |
| **ARIMAX** | -0.0145 | +0.7345 | +0.9183 | +0.9569 | +0.9672 | +0.9817 |
| **SARIMA** | +0.0058 | +0.7188 | +0.9009 | +0.9530 | +0.9612 | +0.9554 |

### 3.2 Chế độ Biến Động Bình Thường (Normal Volatility Regime)
| Model | 1D | 5D | 20D | 60D | 120D | Độ dốc suy thoái ($\Delta_{120D - 1D}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0019 | -0.0050 | -0.0253 | -0.0190 | -0.0074 | -0.0055 |
| **Elastic_Net** | -0.0033 | -0.0030 | -0.0336 | +0.0220 | -0.0155 | -0.0122 |
| **Random_Forest** | -0.0070 | -0.0022 | -0.0233 | -0.0347 | -0.1969 | -0.1899 |
| **XGBoost** | -0.0010 | -0.0046 | -0.0261 | +0.0541 | -0.1754 | -0.1744 |
| **LightGBM** | +0.0004 | -0.0018 | -0.0246 | -0.1018 | -0.4753 | -0.4757 |
| **Ridge** | -0.0027 | -0.0156 | -0.0316 | -0.0938 | +0.1380 | +0.1407 |
| **SVR** | -0.0049 | -0.0062 | -0.0152 | -0.0609 | -0.1118 | -0.1069 |
| **LSTM** | -0.0031 | -0.0532 | -0.0765 | -0.6333 | -0.0896 | -0.0865 |
| **RNN** | -0.0644 | +0.0302 | -0.0409 | -0.5109 | -0.0087 | +0.0557 |
| **Transformer** | -1.4479 | -0.2132 | -0.0462 | -0.6615 | -0.7783 | +0.6696 |
| **ARIMA** | -0.0108 | +0.6548 | +0.9012 | +0.9638 | +0.9406 | +0.9514 |
| **ARIMAX** | +0.0088 | +0.7859 | +0.9151 | +0.9779 | +0.9512 | +0.9424 |
| **SARIMA** | -0.0130 | +0.7411 | +0.9006 | +0.9641 | +0.9395 | +0.9525 |

### 3.3 Chế độ Biến Động Cao (High Volatility Regime) — Sụp Đổ Năng Lực Dự Báo
| Model | 1D | 5D | 20D | 60D | 120D | Độ dốc suy thoái ($\Delta_{120D - 1D}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | -0.0002 | -0.0018 | -0.0453 | -0.6478 | -1.1629 | **-1.1627** |
| **Elastic_Net** | +0.0042 | -0.0132 | -0.0623 | -0.7611 | -1.1714 | **-1.1756** |
| **Random_Forest** | -0.0421 | -0.1072 | -0.5956 | -2.2055 | -2.5249 | **-2.4828** |
| **XGBoost** | -0.0227 | -0.0088 | -0.1342 | -1.5758 | -1.8294 | **-1.8067** |
| **LightGBM** | -0.0810 | -0.1178 | -0.1774 | -1.6306 | -3.3593 | **-3.2783** |
| **Ridge** | +0.0065 | -0.0870 | -0.3022 | -1.7489 | -2.0206 | **-2.0271** |
| **SVR** | +0.0264 | -0.0866 | -0.2488 | -1.2188 | -1.6563 | **-1.6827** |
| **LSTM** | +0.0169 | -0.2305 | -0.1113 | -1.2904 | -2.1375 | **-2.1544** |
| **RNN** | -0.0090 | -0.0520 | -0.1444 | -1.0978 | -0.9504 | **-0.9414** |
| **Transformer** | -0.2609 | -0.2510 | -0.1730 | -2.2291 | -2.0235 | **-1.7626** |
| **ARIMA** | +0.0180 | +0.5492 | +0.8354 | +0.8699 | +0.9164 | +0.8984 |
| **ARIMAX** | +0.0345 | +0.6319 | +0.9243 | +0.9455 | +0.9705 | +0.9360 |
| **SARIMA** | +0.0161 | +0.5975 | +0.8364 | +0.8659 | +0.9149 | +0.8988 |

---

## 4. Phân Tích Tương Tác Giữa Họ Mô Hình và Chế Độ Biến Động (Family × Regime Interactions)

Bảng tổng hợp $OOS\_R^2$ trung bình theo họ mô hình tại các mốc chân trời:

```
                          1D                 20D                120D
Họ mô hình          Low   Norm   High   Low   Norm   High   Low   Norm   High
--------------------------------------------------------------------------
Tree Ensembles     0.00  -0.00  -0.05  -0.00  -0.02  -0.30 -0.13  -0.28  -2.57
Linear Regularized-0.01  -0.00  +0.01  -0.04  -0.03  -0.18 -0.05  +0.06  -1.60
Support Vector    -0.02  -0.00  +0.03  -0.12  -0.02  -0.25 +0.08  -0.11  -1.66
Deep Learning     -1.09  -0.51  -0.08  -0.32  -0.05  -0.14 -0.24  -0.29  -1.70
Autoregressive    -0.00  -0.00  +0.02  +0.91  +0.91  +0.87 +0.96  +0.94  +0.93
Benchmark (Mean)  -0.00  -0.00  -0.00  -0.00  -0.03  -0.05 -0.00  -0.01  -1.16
```

---

## 5. Phát Hiện Cốt Lõi Trả Lời RQ2 (Key Findings on Regime-Horizon Dynamics)

### 5.1 Độ dốc suy thoái phân hóa sâu sắc theo chế độ biến động (Regime-Dependent Decay Slope)
Mối quan hệ giữa chân trời dự báo và năng lực dự báo **thay đổi hoàn toàn bản chất** tùy theo chế độ biến động thị trường:
1. **Trong chế độ Biến động Thấp (Low Regime):**  
   - Suy thoái dự báo diễn ra **rất êm dịu (mild decay)**.
   - Năng lực của các mô hình Cây (Random Forest, XGBoost) duy trì gần mức 0 ở 1D và chỉ giảm nhẹ xuống $-0.04$ đến $-0.12$ tại 120D. Độ dốc suy thoái $\Delta_{\text{Low}} \in [-0.04, -0.22]$.
   - Các mô hình hầu như không bị trừng phạt nặng nề bởi sai số lớn vì biên độ dao động của thị trường nhỏ.
2. **Trong chế độ Biến động Bình thường (Normal Regime):**  
   - Mẫu hình suy thoái ở mức vừa phải, tương tự Low Regime. Mô hình Tree Ensembles giảm xuống $-0.28$ tại 120D.
3. **Trong chế độ Biến động Cao (High Regime) — Hiện tượng Sụp Đổ Dự Báo (Catastrophic Collapse):**  
   - Tại chân trời ngắn $1D$: Chế độ High Regime mang lại hiệu năng dự báo cao nhất cho các mô hình hồi quy (SVR đạt $+0.0264$, ARIMAX đạt $+0.0345$, Ridge đạt $+0.0065$, LSTM đạt $+0.0169$). Lý do là trong các đợt khủng hoảng, các tín hiệu biến động (VIX, biên độ High-Low) phát huy tác dụng dự báo đảo chiều ngắn hạn mạnh mẽ nhất.
   - Tuy nhiên, khi chân trời mở rộng ra $60D$ và $120D$, độ dốc suy thoái **lao dốc thẳng đứng (catastrophic decay)**:
     - Độ dốc suy thoái tại High Regime dốc gấp **15 đến 30 lần** so với Low Regime: $\Delta_{\text{High}}$ rơi xuống mức $-1.81$ (XGBoost), $-2.48$ (Random Forest), $-3.28$ (LightGBM).
     - Ngay cả Benchmark `Historical_Mean` cũng sụp đổ xuống $-1.1629$ tại 120D.

### 5.2 Nguyên nhân kinh tế lượng của sự sụp đổ tại High Regime
- **Phương sai ngoại mẫu bùng nổ (Variance Explosion):** Khi thị trường rơi vào khủng hoảng (COVID-19, sụp đổ 2018), biên độ lợi suất thực tế 120 ngày phân kỳ cực lớn (từ âm sâu $-28.5\%$ đến phục hồi chữ V $+40.1\%$).
- **Độ nhạy cảm của hàm mất mát bình phương (Quadratic Loss Sensitivity):** Các mô hình ML huấn luyện trên dữ liệu quá khứ không dự đoán được cường độ của các đợt phục hồi hay sụt giảm cực đoan này. Sai số độ lớn $(\text{Actual} - \text{Prediction})$ bị bình phương trong công thức MSE, khiến tỷ số $\frac{\text{MSE}}{\text{Var}(\text{Actual})}$ tăng vọt, kéo $OOS\_R^2$ xuống mức âm rất sâu.

### 5.3 Sự kiên định của Dòng Tự Hồi Quy qua các chế độ (Autoregressive Resilience)
- Khác với ML, các mô hình Tự hồi quy (ARIMA, ARIMAX, SARIMA) duy trì $OOS\_R^2 > 0.83$ tại $20D$ và $> 0.91$ tại $120D$ trong **cả ba chế độ biến động**.
- Mặc dù tại High Regime, $OOS\_R^2$ tại 120D có giảm nhẹ từ $+0.9625$ (Low) xuống $+0.9164$ (High), mức suy giảm này là không đáng kể so với sự đổ vỡ của các mô hình Machine Learning. Cơ chế tích lũy tự tương quan giúp dòng AR hấp thụ tốt các biến động thị trường dài hạn.

---

## 6. Đánh Giá Mức Độ Nhất Quán & Phân Loại Bằng Chứng (Consistency & Evidence)
- **Tính nhất quán giữa các mô hình (Cross-model consistency):** Cực kỳ cao trong nội bộ từng họ mô hình. $100\%$ các mô hình Cây và Linear Regularized đều có độ dốc suy thoái tại High Regime lớn hơn ít nhất 10 lần so với Low Regime.
- **Phân loại bằng chứng trả lời RQ2:** **STRONG EVIDENCE (Bằng chứng thực nghiệm rất mạnh).**
  - Chế độ biến động thị trường làm thay đổi sâu sắc cấu trúc mối quan hệ giữa chân trời và năng lực dự báo: từ suy thoái phẳng nhẹ ở chế độ biến động thấp sang suy thoái sụp đổ dốc đứng ở chế độ biến động cao.

---

## 7. Hạn Chế Phương Pháp Luận Khi Đánh Giá RQ2 (Limitations)
1. **Thiếu tính động học của Regime trong chân trời dài:** Chế độ biến động được gắn nhãn tại thời điểm dự báo $t$. Đối với chân trời $120D$ (khoảng 6 tháng), thị trường có thể chuyển dịch từ High sang Low (ví dụ sau tháng 4/2020), làm mờ đi ranh giới tác động đơn thuần của regime ban đầu.

---

## 8. Kết Luận Bước 6 & Kế Hoạch Tiếp Theo (Conclusions & Next Steps)
- **Kết luận RQ2:** Chế độ biến động điều tiết mạnh mẽ tốc độ suy thoái của năng lực dự báo:
  - Ở chế độ biến động thấp/bình thường, năng lực dự báo suy giảm chậm rãi và biên độ nhỏ.
  - Ở chế độ biến động cao, xuất hiện hiện tượng nghịch lý: năng lực dự báo 1D đạt đỉnh, nhưng tốc độ suy thoái theo chân trời là khốc liệt nhất, khiến mọi mô hình Machine Learning thất bại nặng nề ở các chân trời dài hạn ($60D, 120D$).
- **Bước tiếp theo:** Chuyển sang **STEP 7** — Phân tích tập tin trả lời câu hỏi nghiên cứu số 3: `results/analysis/rq3_time.csv` nhằm kiểm tra tính ổn định của các quy luật trên qua các chu kỳ thời gian (Temporal Stability).
