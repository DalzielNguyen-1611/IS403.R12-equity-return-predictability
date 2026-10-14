# 📊 BẢNG THEO DÕI TIẾN TRÌNH & KẾT QUẢ HUẤN LUYỆN (REAL-TIME RESEARCH DASHBOARD)
**Đề tài:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  
**Dữ liệu:** S&P 500 Equity Returns | **Ghi nhận:** 2026-10-13 10:10:00  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi giai đoạn.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & TRẠNG THÁI TỔNG THỂ

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & ĐÁNH GIÁ | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Tiến trình Huấn luyện (Pha 1: Walk-Forward)** | **`14/14 Models (100.0%)`** | **Tổng số Folds hoàn tất** | **`350/350 Folds (100.0%)`** |
| **Số lượt dự báo ngoại mẫu OOS (Pha 1)** | **`95,200 dự báo`** (2015–2020) | **Zero-Data-Leakage Purging** | `✅ Đạt 100% (Marcos López de Prado)` |
| **Tiến trình Kiểm thử (Pha 2: Test độc lập)** | **`14/14 Models (100.0%)`** | **Tổng số lượt Test Split 3** | **`95,340 dự báo`** (2020–2026) |
| **Tập dữ liệu Test độc lập (Split 3)** | 1,362 phiên (02/11/2020 → 07/04/2026) | **Siêu tham số Test** | Frozen từ Fold 5 (Khép kín tuyệt đối) |
| **Kỷ lục OOS R² Test (60D)** | **`+0.0464` (+4.64%)** | **Mô hình giữ kỷ lục Test 60D** | **`SVR (RBF Kernel)`** |
| **Kỷ lục OOS R² Test (120D)** | **`+0.0573` (+5.73%)** | **Mô hình giữ kỷ lục Test 120D** | **`SARIMA`** (bắt độ trôi nền dài hạn) |
| **Mô hình ổn định nhất toàn diện** | **`Ridge Regression`** | **OOS R² Test của Ridge** | Dương ở 4/5 chân trời (1D, 5D, 20D, 60D) |
| **Trạng thái thực nghiệm** | 🎉 **HOÀN TẤT 100% CẢ 2 PHA (WALK-FORWARD & INDEPENDENT TEST)** | **Kiểm toán dữ liệu & Thư mục** | `✅ Đồng nhất duy nhất tại results/Lan01` |

---

## 📋 2. BẢNG TỔNG HỢP TIẾN TRÌNH 14 MÔ HÌNH (CẢ HUẤN LUYỆN & KIỂM THỬ)

| # | Mô hình | Nhóm thuật toán | Pha 1: Walk-Forward (2015–2020) | Pha 2: Test Độc Lập Split 3 (2020–2026) | Trạng thái | Ghi chú & Hiệu năng nổi bật |
| :-: | :--- | :--- | :-: | :-: | :-: | :--- |
| 1 | **`Historical_Mean`** | Benchmark | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Mốc chuẩn Campbell–Thompson ($OOS\_R^2 = 0.0000$) |
| 2 | **`Ridge`** | Linear Regression | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Dẫn đầu 5D ($+0.0179$), bền vững nhất toàn diện |
| 3 | **`Logistic_Regression`** | Classification | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | ROC-AUC đạt đỉnh $0.6264$ tại 60D, DA $74.60\%$ |
| 4 | **`Elastic_Net`** | Linear Regression | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Co cụm tham số triệt để, kháng nhiễu tốt |
| 5 | **`ARIMA`** | Statistical TS | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | $OOS\_R^2 = +0.0195$ tại 120D sau khi Purging |
| 6 | **`ARIMAX`** | Statistical TS | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Ngoại sinh VIX/Mom phát huy tốt ở 1D và 60D ($+0.0252$) |
| 7 | **`SARIMA`** | Statistical TS | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | **Quán quân 120D** ($OOS\_R^2 = +0.0573, DA = 76.14\%$) |
| 8 | **`LightGBM`** | Tree Ensemble | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | **Quán quân 1D** ($+0.0208$), xuất sắc ở 60D ($+0.0289$) |
| 9 | **`SVR`** | Kernel Method | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | **Kỷ lục Test 60D** ($+0.0464$), dẫn đầu 20D ($+0.0305$) |
| 10 | **`Random_Forest`** | Tree Ensemble | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Đạt $OOS\_R^2 > 0$ tại 1D ($+0.0177$) và 60D ($+0.0033$) |
| 11 | **`XGBoost`** | Tree Ensemble | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Ổn định cao, $DA$ đạt $74.96\%$ tại 60D |
| 12 | **`RNN`** | Deep Learning | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Vượt mốc chuẩn tại 20D ($+0.0048$), suy giảm ở 1D/5D |
| 13 | **`LSTM`** | Deep Learning | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Điểm sáng Deep Learning tại 20D ($+0.0148, DM = +1.22$) |
| 14 | **`Transformer`** | Deep Learning | `25/25 Folds` | `5/5 Horizons` (6,810 mẫu) | ✅ Hoàn tất | Quá khớp tham số, sụp đổ ngoại mẫu ($-0.12$ đến $-0.59$) |

---

## 🏆 3. KẾT QUẢ NGHIÊN CỨU & HỆ THỐNG TÀI LIỆU BÁO CÁO TOÀN DIỆN

Hệ thống kết quả nghiên cứu và báo cáo khoa học đã được kiểm toán, chuẩn hóa và lưu trữ đồng nhất tại `results/Lan01/`:

### 3.1 Dữ liệu thực nghiệm gốc (Raw Artifacts)
- 📊 **Walk-Forward Fold Metrics (Pha 1):** `results/Lan01/fold_metrics/3.0.3_all_fold_metrics.csv` (350 dòng)
- 📊 **Model Horizon Summary (Pha 1):** `results/Lan01/summary/3.0.3_model_horizon_summary.csv` (70 dòng)
- 📊 **OOS Predictions (Pha 1):** `results/Lan01/predictions/3.0.3_all_oos_predictions.csv` (95,200 dòng)
- 📊 **Best Hyperparameters (Pha 1):** `results/Lan01/tuning/3.0.3_best_hyperparameters.csv` (325 dòng)
- 📊 **Independent Test Predictions (Pha 2):** `results/Lan01/test_evaluation/3.0.3_test_predictions.csv` (95,340 dòng)
- 📊 **Independent Test Metrics Summary (Pha 2):** `results/Lan01/test_evaluation/3.0.3_test_metrics_summary.csv` (65 dòng)
- 📊 **Independent Test Classification (Pha 2):** `results/Lan01/test_evaluation/3.0.3_test_classification_summary.csv` (5 dòng)

### 3.2 Hệ thống 12 Báo Cáo Chuyên Sâu & Báo Cáo Hợp Nhất
1. [3.0.3_FINAL_SYNTHESIS.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_FINAL_SYNTHESIS.md) — Tổng hợp kết quả nghiên cứu toàn diện.
2. [3.0.3_01_all_fold_metrics_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_01_all_fold_metrics_report.md) — Báo cáo đo lường cấp fold & kiểm toán dữ liệu.
3. [3.0.3_02_model_horizon_summary_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_02_model_horizon_summary_report.md) — Tổng hợp hiệu năng đa mô hình & đa chân trời.
4. [3.0.3_03_all_oos_predictions_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_03_all_oos_predictions_report.md) — Phân tích dữ liệu dự báo ngoại mẫu (OOS).
5. [3.0.3_04_best_hyperparameters_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_04_best_hyperparameters_report.md) — Đánh giá tinh chỉnh siêu tham số.
6. [3.0.3_05_rq1_horizon_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_05_rq1_horizon_report.md) — Suy thoái năng lực dự báo theo chân trời thời gian.
7. [3.0.3_06_rq2_regime_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_06_rq2_regime_report.md) — Năng lực dự báo theo chế độ biến động thị trường.
8. [3.0.3_07_rq3_time_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_07_rq3_time_report.md) — Tính ổn định năng lực dự báo qua các chu kỳ thời gian.
9. [3.0.3_07_rq3_time_audit_log.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_07_rq3_time_audit_log.md) — Kiểm toán phương pháp luận & thống kê RQ3.
10. [3.0.3_08_statistical_tests_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_08_statistical_tests_report.md) — Kiểm định thống kê (Diebold–Mariano, Clark–West, SPA).
11. [3.0.3_08_statistical_tests_audit_log.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_08_statistical_tests_audit_log.md) — Kiểm toán phương pháp luận kiểm định thống kê.
12. [3.0.3_09_test_evaluation_report.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_09_test_evaluation_report.md) — **Đánh giá tập kiểm thử độc lập (Split 3: 2020–2026)**.
13. ⭐ **BÁO CÁO HỢP NHẤT TOÀN DIỆN:** [3.0.3_ALL_REPORTS_COMBINED.md](file:///e:/Repo%20Project/IS403.R12-equity-return-predictability/results/Lan01/agent_reports/3.0.3_ALL_REPORTS_COMBINED.md) (Hợp nhất trọn vẹn cả 12 báo cáo).

---
*Bảng điều khiển phản ánh trạng thái hoàn tất 100% của toàn bộ hệ thống thực nghiệm dự án IS403.*
