# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** Transformer | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** 2026-10-12 06:44:53  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`Transformer`** | **Fold vừa hoàn tất** | Fold 5/5 (Horizon 120D) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `-0.1775` |
| **Tiến trình Mô hình** | **`14/14 Models (100.0%)`** | **Directional Accuracy (DA)** | `60.66%` |
| **Tiến trình Folds tổng** | **`350/350 Folds (100.0%)`** | **RMSE Fold vừa xong** | `0.1448` |
| **Kỷ lục OOS R² cao nhất** | `0.6473` | **Loss / MAE Fold vừa xong** | `0.1330` |
| **Mô hình giữ Kỷ lục** | ARIMAX (H=120D, Fold=5) | **Best Horizon của Model** | Horizon 120D |
| **Thời gian đã chạy (chính nó)** | **`-`** | **Tốc độ chạy (chính nó)** | **`-`** |
| **Tiến độ Nhóm Linear & Stats** | `✅ ĐÃ HOÀN TẤT 7/7 MODELS` | **Trạng thái Nhóm 1** | `Đã lưu Fold Metrics & OOS Preds` |
| **Dự kiến XONG mô hình này** | Đã xong 100% (còn 0 folds) | **Thời gian còn lại (chính nó)** | Đã xong 100% (còn 0 folds) |
| **Trạng thái mô hình** | 🎉 ĐÃ HOÀN TẤT 100% TOÀN BỘ 14/14 MÔ HÌNH (350/350 FOLDS)! | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Zero manual action) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`Transformer`)

> 💡 **Quy luật huấn luyện Time-Series:** Mô hình chạy tuần tự theo Horizon: **`1D` (Fold 1→2→3→4→5)** rồi mới chuyển sang **`5D` (Fold 1→2→3→4→5)**, kế tiếp là `20D`, `60D`, `120D`.  
> Bảng dưới đây trích xuất **5 bước vừa chạy xong gần nhất** theo đúng dòng thời gian:

| BƯỚC | HORIZON | FOLD | GIAI ĐOẠN VALIDATION | TRẠNG THÁI | VAL OOS R² | VAL DA | VAL RMSE | VAL MAE |
| :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#21** | **120D** | Fold 1/5 | `2015-06-10 → 2016-07-07` | ✅ Hoàn thành | `-2.4921` | `35.66%` | `0.1158` | `0.0975` |
| **#22** | **120D** | Fold 2/5 | `2016-07-08 → 2017-08-04` | ✅ Hoàn thành | `-0.4792` | `80.88%` | `0.0609` | `0.0503` |
| **#23** | **120D** | Fold 3/5 | `2017-08-07 → 2018-09-04` | ✅ Hoàn thành | `-1.0479` | `52.57%` | `0.0853` | `0.0662` |
| **#24** | **120D** | Fold 4/5 | `2018-09-05 → 2019-10-03` | ✅ Hoàn thành | `-1.2062` | `47.79%` | `0.1139` | `0.0861` |
| **#25** | **120D** | Fold 5/5 | `2019-10-04 → 2020-10-30` | ✅ Hoàn thành | `-0.1775` | `60.66%` | `0.1448` | `0.1330` |

---

## 📋 3. TIẾN ĐỘ TỔNG HỢP 14 MÔ HÌNH & DỰ BÁO HOÀN THÀNH

> ⚠️ **Nguyên tắc vàng:** Thời gian ước lượng (ETA) **bắt buộc tính dựa trên tốc độ và thời gian quá khứ của CHÍNH model đó**, tuyệt đối không suy đoán chéo giữa các model khác nhau.

| # | Mô hình | Nhóm thuật toán | Tiến độ | Trạng thái | Thời gian thực tế của chính nó | Tốc độ 1 fold | Dự báo hoàn thành (ETA của chính model) |
| :-: | :--- | :--- | :-: | :-: | :---: | :---: | :--- |
| 1 | **`Historical_Mean`** | Benchmark | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 2 | **`Ridge`** | Linear Regression | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 3 | **`Logistic_Regression`** | Classification | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 4 | **`Elastic_Net`** | Linear Regression | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 5 | **`ARIMA`** | Statistical TS | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 6 | **`ARIMAX`** | Statistical TS | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 7 | **`SARIMA`** | Statistical TS | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 8 | **`LightGBM`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 9 | **`SVR`** | Kernel Method | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 10 | **`Random_Forest`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 11 | **`XGBoost`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 12 | **`RNN`** | Deep Learning | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 13 | **`LSTM`** | Deep Learning | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 14 | **`Transformer`** | Deep Learning | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |

---

## 🏆 4. KẾT QUẢ NGHIÊN CỨU & BÁO CÁO LUẬN VĂN (ĐÃ HOÀN TẤT)

✅ **Toàn bộ 14 mô hình (350/350 Folds) đã hoàn tất 100%!** Hệ thống đã hoàn thành trọn vẹn, không còn tác vụ nào đang chờ.

Tất cả các tệp phân tích câu hỏi nghiên cứu (RQ) và kiểm định thống kê đã được tạo hoàn chỉnh trong thư mục `results/`:
- 📄 **RQ1 (Forecasting Horizon):** `results/analysis/rq1_horizon.csv`
- 📄 **RQ2 (Volatility Regime):** `results/analysis/rq2_regime.csv`
- 📄 **RQ3 (Temporal Stability):** `results/analysis/rq3_time.csv`
- 📄 **Kiểm định Thống kê (DM-Test & Sign-Test):** `results/analysis/statistical_tests.csv`
- 📄 **Dự báo ngoài mẫu (OOS Predictions):** `results/predictions/all_oos_predictions.csv` (18,000+ dự báo)
- 📄 **Bảng tổng hợp Model-Horizon Metrics:** `results/summary/model_horizon_summary.csv`

---
*Bảng điều khiển được cập nhật **hoàn toàn tự động** sau mỗi Fold bằng hook nội bộ (Atomic File Swap).*
