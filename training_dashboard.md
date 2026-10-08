# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** Transformer | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** 2026-10-08 13:39:21  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`Transformer`** | **Fold vừa hoàn tất** | Fold 4/5 (Horizon 60D) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `-0.9689` |
| **Tiến trình Mô hình** | **`13/14 Models (98.3%)`** | **Directional Accuracy (DA)** | `49.63%` |
| **Tiến trình Folds tổng** | **`344/350 Folds (98.3%)`** | **RMSE Fold vừa xong** | `0.0891` |
| **Kỷ lục OOS R² cao nhất** | `0.9815` | **Loss / MAE Fold vừa xong** | `0.0662` |
| **Mô hình giữ Kỷ lục** | ARIMAX (H=120D, Fold=5) | **Best Horizon của Model** | Horizon 60D |
| **Thời gian đã chạy (chính nó)** | **`03h 38m`** | **Tốc độ chạy (chính nó)** | **`689.10s/fold`** |
| **Tiến độ Nhóm Linear & Stats** | ✅ ĐÃ HOÀN TẤT 7/7 MODELS | **Trạng thái Nhóm 1** | Đã lưu Fold Metrics & OOS Preds |
| **Dự kiến XONG mô hình này** | ~01h 08m (còn 6 folds) | **Thời gian còn lại (chính nó)** | ~01h 08m (còn 6 folds) |
| **Trạng thái mô hình** | 🟡 Đang huấn luyện: Fold 4/5 (Horizon 60D) — Cập nhật Real-Time | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Zero manual action) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`Transformer`)

> 💡 **Quy luật huấn luyện Time-Series:** Mô hình chạy tuần tự theo Horizon: **`1D` (Fold 1→2→3→4→5)** rồi mới chuyển sang **`5D` (Fold 1→2→3→4→5)**, kế tiếp là `20D`, `60D`, `120D`.  
> Bảng dưới đây trích xuất **5 bước vừa chạy xong gần nhất** theo đúng dòng thời gian:

| BƯỚC | HORIZON | FOLD | GIAI ĐOẠN VALIDATION | TRẠNG THÁI | VAL OOS R² | VAL DA | VAL RMSE | VAL MAE |
| :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#15** | **20D** | Fold 5/5 | `2019-10-04 → 2020-10-30` | ✅ Hoàn thành | `0.0133` | `63.97%` | `0.0860` | `0.0565` |
| **#16** | **60D** | Fold 1/5 | `2015-06-10 → 2016-07-07` | ✅ Hoàn thành | `-2.5917` | `41.18%` | `0.1106` | `0.0854` |
| **#17** | **60D** | Fold 2/5 | `2016-07-08 → 2017-08-04` | ✅ Hoàn thành | `-1.0063` | `82.35%` | `0.0352` | `0.0248` |
| **#18** | **60D** | Fold 3/5 | `2017-08-07 → 2018-09-04` | ✅ Hoàn thành | `-0.4897` | `54.78%` | `0.0551` | `0.0476` |
| **#19** | **60D** | Fold 4/5 | `2018-09-05 → 2019-10-03` | ✅ Hoàn thành | `-0.9689` | `49.63%` | `0.0891` | `0.0662` |

---

## 📋 3. TIẾN ĐỘ TỔNG HỢP 14 MÔ HÌNH & DỰ BÁO HOÀN THÀNH

> ⚠️ **Nguyên tắc vàng:** Thời gian ước lượng (ETA) **bắt buộc tính dựa trên tốc độ và thời gian quá khứ của CHÍNH model đó**, tuyệt đối không suy đoán chéo giữa các model khác nhau.

| # | Mô hình | Nhóm thuật toán | Tiến độ | Trạng thái | Thời gian thực tế của chính nó | Tốc độ 1 fold | Dự báo hoàn thành (ETA của chính model) |
| :-: | :--- | :--- | :-: | :-: | :---: | :---: | :--- |
| 1 | **`Historical_Mean`** | Benchmark | `25/25` | ✅ Hoàn thành | 2.5s | `0.10s/fold` | Đã xong (100%) |
| 2 | **`Ridge`** | Linear Regression | `25/25` | ✅ Hoàn thành | 6.0s | `0.24s/fold` | Đã xong (100%) |
| 3 | **`Logistic_Regression`** | Classification | `25/25` | ✅ Hoàn thành | 16.0s | `0.64s/fold` | Đã xong (100%) |
| 4 | **`Elastic_Net`** | Linear Regression | `25/25` | ✅ Hoàn thành | 12.0s | `0.48s/fold` | Đã xong (100%) |
| 5 | **`ARIMA`** | Statistical TS | `25/25` | ✅ Hoàn thành | 03m 36s | `8.64s/fold` | Đã xong (100%) |
| 6 | **`ARIMAX`** | Statistical TS | `25/25` | ✅ Hoàn thành | 02m 02s | `4.88s/fold` | Đã xong (100%) |
| 7 | **`SARIMA`** | Statistical TS | `25/25` | ✅ Hoàn thành | 50.0s | `2.00s/fold` | Đã xong (100%) |
| 8 | **`LightGBM`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | 04m 35s | `11.00s/fold` | Đã xong (100%) |
| 9 | **`SVR`** | Kernel Method | `25/25` | ✅ Hoàn thành | 49m 00s | `117.60s/fold` | Đã xong (100%) |
| 10 | **`Random_Forest`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | 08m 08s | `19.52s/fold` | Đã xong (100%) |
| 11 | **`XGBoost`** | Tree Ensemble | `25/25` | ✅ Hoàn thành | 02m 49s | `6.76s/fold` | Đã xong (100%) |
| 12 | **`RNN`** | Deep Learning | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 13 | **`LSTM`** | Deep Learning | `25/25` | ✅ Hoàn thành | Hoàn tất | `-` | Đã xong |
| 14 | **`Transformer`** | Deep Learning | `19/25` | 🔄 Đang chạy (19/25) | 03h 38m | `689.10s/fold` | ~01h 08m (còn 6 folds) |

---

## ⚡ 4. THỨ TỰ & LỆNH ĐIỀU PHỐI MÔ HÌNH TIẾP THEO

Chạy toàn bộ 6 mô hình còn lại tự động trong 1 lệnh duy nhất:
```powershell
python run_remaining.py
```

---
*Bảng điều khiển được cập nhật **hoàn toàn tự động** sau mỗi Fold bằng hook nội bộ (Atomic File Swap).*
