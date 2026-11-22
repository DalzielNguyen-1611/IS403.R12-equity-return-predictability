# Hướng Dẫn & Tài Liệu Kỹ Thuật: Benchmark Robustness Evaluation

## 1. Mục Đích & Bối Cảnh Nghiên Cứu
Nghiên cứu hiện tại sử dụng **Expanding Historical Mean (HM)** (theo chuẩn của Campbell & Thompson, 2008) làm benchmark chính thống xuyên suốt các bài toán dự báo lợi suất $S\&P\ 500$ trên 5 chân trời ($1D, 5D, 20D, 60D, 120D$).

Theo yêu cầu của GVHD về việc bổ sung **Benchmark Robustness**:
Module này triển khai kiểm tra độ bền thống kê bằng cách đối sánh năng lực dự báo của 12 mô hình giữa:
1. **Baseline Benchmark**: Expanding Historical Mean ($\hat{y}_{t,h}^{\text{HM}} = \bar{y}_{\text{train}, h}$)
2. **Alternative Benchmark**: Zero-Return Benchmark ($\hat{y}_{t,h}^{\text{Zero}} = 0$) (chuẩn Welch & Goyal, 2008; Campbell & Thompson, 2008).

---

## 2. Cơ Sở Lý Thuyết & Công Thức Toán Học

### 2.1. Định Nghĩa Dự Báo & Tổng Bình Phương Phần Dư (SSE)
Với mỗi mô hình $m \in \{1, \dots, 12\}$ và forecasting horizon $h \in \{1D, 5D, 20D, 60D, 120D\}$ trên mẫu kiểm thử ngoài mẫu $T = 1.362$ ngày (2020-11-02 đến 2026-04-07):

* **Dự báo của mô hình đề xuất**: $\hat{y}_{t,h}^m$
  $$SSE_{\text{Model}} = \sum_{t=1}^T \left( y_{t,h} - \hat{y}_{t,h}^m \right)^2$$

* **Dự báo của Historical Mean**: $\hat{y}_{t,h}^{\text{HM}} = \bar{y}_{\text{train}, h}$
  $$SSE_{\text{HM}} = \sum_{t=1}^T \left( y_{t,h} - \bar{y}_{\text{train}, h} \right)^2$$

* **Dự báo của Zero-Return Benchmark**: $\hat{y}_{t,h}^{\text{Zero}} = 0$
  $$SSE_{\text{Zero}} = \sum_{t=1}^T \left( y_{t,h} - 0 \right)^2 = \sum_{t=1}^T y_{t,h}^2$$

### 2.2. Out-of-Sample $R^2$ ($R_{OOS}^2$) & Mức Chênh Lệch ($\Delta R_{OOS}^2$)
* $R_{OOS}^2$ theo Historical Mean:
  $$R_{OOS, \text{HM}}^2 = 1 - \frac{SSE_{\text{Model}}}{SSE_{\text{HM}}}$$

* $R_{OOS}^2$ theo Zero-Return:
  $$R_{OOS, \text{Zero}}^2 = 1 - \frac{SSE_{\text{Model}}}{SSE_{\text{Zero}}}$$

* Mức chênh lệch giữa hai benchmark:
  $$\Delta R_{OOS}^2 = R_{OOS, \text{Zero}}^2 - R_{OOS, \text{HM}}^2$$

### 2.3. Tính Nhất Quán Kinh Tế Lượng (Theoretical Consistency)
Do thị trường chứng khoán Hoa Kỳ trong lịch sử dài hạn có phần bù rủi ro vốn chủ sở hữu dương ($\bar{y}_{\text{train}} > 0$):
$$SSE_{\text{Zero}} = \sum_{t=1}^T y_{t,h}^2 = \sum_{t=1}^T (y_{t,h} - \bar{y}_{\text{train}} + \bar{y}_{\text{train}})^2 = SSE_{\text{HM}} + 2 \bar{y}_{\text{train}} \sum_{t=1}^T (y_{t,h} - \bar{y}_{\text{train}}) + T \bar{y}_{\text{train}}^2$$

Khi chuỗi lợi suất kiểm thử cũng có xu hướng tăng ($\bar{y}_{\text{test}} > 0$), ta có:
$$SSE_{\text{Zero}} > SSE_{\text{HM}}$$

Do mẫu số của $R_{OOS, \text{Zero}}^2$ lớn hơn mẫu số của $R_{OOS, \text{HM}}^2$, ta có hệ quả toán học tất yếu:
$$\frac{SSE_{\text{Model}}}{SSE_{\text{Zero}}} < \frac{SSE_{\text{Model}}}{SSE_{\text{HM}}} \implies R_{OOS, \text{Zero}}^2 > R_{OOS, \text{HM}}^2$$

> **Ý nghĩa thực tiễn đối với bài báo:**
> Historical Mean là một ngưỡng chuẩn **khắt khe và bảo thủ hơn rất nhiều** so với Zero-Return Benchmark (Random Walk không trôi dạt). Nếu mô hình đã vượt qua được Historical Mean ($R_{OOS,\text{HM}}^2 > 0$), nó sẽ đương nhiên đạt $R_{OOS,\text{Zero}}^2 > 0$ với biên độ cao hơn.

### 2.4. Kiểm Định Thống Kê Diebold-Mariano (1995) với HLN (1997) Correction
Tái sử dụng nguyên vẹn hàm `diebold_mariano_test` với hàm tổn thất bình phương ($loss = 'squared'$), độ trễ tự tương quan $h \in \{1, 5, 20, 60, 120\}$, và hệ số hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold:
$$d_t = (y_{t,h} - \hat{y}_{t,h}^{\text{Bench}})^2 - (y_{t,h} - \hat{y}_{t,h}^{\text{Model}})^2$$
Được tính độc lập cho cả hai chuẩn: $\hat{y}^{\text{Bench}} = \hat{y}^{\text{HM}}$ và $\hat{y}^{\text{Bench}} = 0$.

---

## 3. Hướng Dẫn Chạy Thủ Công (Manual Execution)

Mở PowerShell hoặc Command Prompt tại thư mục gốc của repository (`E:\Repo Project\IS403.R12-equity-return-predictability`) và thực hiện các lệnh sau:

### Cách 1: Chạy đánh giá chính thức trên Holdout Test (Khuyến nghị)
```powershell
python src/testing/run_benchmark_robustness.py
```
*Thời gian chạy dự kiến: ~ 2 - 5 giây.*

### Cách 2: Chạy đầy đủ cả Holdout Test và 5 Validation Folds
```powershell
python src/testing/run_benchmark_robustness.py --include-val
```

### Cách 3: Chỉ kiểm tra tính sẵn sàng của dữ liệu predictions (Dry Run)
```powershell
python src/testing/run_benchmark_robustness.py --check-only
```

---

## 4. Danh Mục File Kết Quả Sẽ Được Tạo Ra

Tất cả kết quả được xuất cô lập tại thư mục `results/benchmark_robustness/`:

| Tên File | Định Dạng | Mô Tả Nội Dung |
| :--- | :---: | :--- |
| `benchmark_comparison_summary.csv` | CSV (60 dòng) | Bảng tổng hợp Holdout Test cho 12 mô hình $\times$ 5 horizons: $R_{OOS,\text{HM}}^2$, $R_{OOS,\text{Zero}}^2$, $\Delta R_{OOS}^2$, MAE, RMSE, p-value DM. |
| `benchmark_robustness_detailed.csv` | CSV (240 dòng) | Bảng chi tiết gồm toàn bộ mẫu và phân rã theo 3 Volatility Regimes (Low, Normal, High). |
| `matrix_oos_r2_historical_mean.csv` | CSV ($12 \times 5$) | Ma trận $R_{OOS}^2$ theo Historical Mean (sẵn sàng đưa vào bảng LaTeX). |
| `matrix_oos_r2_zero_return.csv` | CSV ($12 \times 5$) | Ma trận $R_{OOS}^2$ theo Zero-Return Benchmark. |
| `matrix_delta_oos_r2.csv` | CSV ($12 \times 5$) | Ma trận chênh lệch $\Delta R_{OOS}^2 = R_{OOS,\text{Zero}}^2 - R_{OOS,\text{HM}}^2$. |
| `matrix_dm_pvalue_historical_mean.csv` | CSV ($12 \times 5$) | Ma trận giá trị $p$ của kiểm định Diebold-Mariano đối với Historical Mean. |
| `matrix_dm_pvalue_zero_return.csv` | CSV ($12 \times 5$) | Ma trận giá trị $p$ của kiểm định Diebold-Mariano đối với Zero-Return Benchmark. |
| `regime_benchmark_comparison.csv` | CSV (15 dòng) | So sánh giá trị trung bình $R_{OOS}^2$ của các mô hình theo từng Regime (Low, Normal, High) trên 5 horizons. |
| `validation_folds_benchmark_robustness.csv` | CSV (300 dòng) | Kết quả chi tiết trên từng fold validation (khi chạy cờ `--include-val`). |
| `benchmark_robustness_summary_report.md` | Markdown | Báo cáo tóm tắt học thuật trình bày sẵn các bảng và nhận xét lý thuyết. |
| `benchmark_robustness_execution.log` | Log text | Nhật ký ghi nhận chi tiết quá trình thực thi và xác thực mẫu dữ liệu. |

---

## 5. Những Nội Dung Chưa Được Xác Minh Do Chưa Thực Thi

Tuân thủ nghiêm ngặt chỉ thị: **"CHỈ TẠO CODE, KHÔNG TỰ ĐỘNG CHẠY"**, các nội dung sau chưa được tính toán cụ thể trên đĩa:
1. Giá trị số liệu chính xác từng phần trăm của ma trận $R_{OOS,\text{Zero}}^2$ và $\Delta R_{OOS}^2$.
2. Danh sách cụ thể các cặp mô hình-horizon chuyển từ không có ý nghĩa thống kê ($p \ge 0.05$ với HM) sang có ý nghĩa thống kê ($p < 0.05$ với Zero) trong kiểm định Diebold-Mariano.
3. Độ chênh lệch chính xác của MAE/RMSE giữa mô hình với Zero-Return Benchmark trong từng Volatility Regime.

*Sau khi người dùng chạy lệnh `python src/testing/run_benchmark_robustness.py`, các file trên sẽ lập tức được sinh ra đầy đủ.*

