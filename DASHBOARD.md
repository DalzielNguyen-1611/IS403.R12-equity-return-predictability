# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** RNN | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** 2026-10-11 20:40:25  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`RNN`** | **Fold vừa hoàn tất** | Fold 4/5 (Horizon 1D) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `-0.0674` |
| **Tiến trình Mô hình** | **`11/14 Models (79.7%)`** | **Directional Accuracy (DA)** | `50.37%` |
| **Tiến trình Folds tổng** | **`279/350 Folds (79.7%)`** | **RMSE Fold vừa xong** | `0.0106` |
| **Kỷ lục OOS R² cao nhất** | `0.6473` | **Loss / MAE Fold vừa xong** | `0.0074` |
| **Mô hình giữ Kỷ lục** | ARIMAX (H=120D, Fold=5) | **Best Horizon của Model** | Horizon 1D |
| **Thời gian đã chạy (chính nó)** | **`1.0s`** | **Tốc độ chạy (chính nó)** | **`0.25s/fold`** |
| **Tiến độ Nhóm Linear & Stats** | `✅ ĐÃ HOÀN TẤT 7/7 MODELS` | **Trạng thái Nhóm 1** | `Đã lưu Fold Metrics & OOS Preds` |
| **Dự kiến XONG mô hình này** | ~5.2s (còn 21 folds) | **Thời gian còn lại (chính nó)** | ~5.2s (còn 21 folds) |
| **Trạng thái mô hình** | 🟡 Đang huấn luyện: Fold 4/5 (Horizon 1D) — Cập nhật Real-Time | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Zero manual action) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`RNN`)

> 💡 **Quy luật huấn luyện Time-Series:** Mô hình chạy tuần tự theo Horizon: **`1D` (Fold 1→2→3→4→5)** rồi mới chuyển sang **`5D` (Fold 1→2→3→4→5)**, kế tiếp là `20D`, `60D`, `120D`.  
> Bảng dưới đây trích xuất **5 bước vừa chạy xong gần nhất** theo đúng dòng thời gian:

| BƯỚC | HORIZON | FOLD | GIAI ĐOẠN VALIDATION | TRẠNG THÁI | VAL OOS R² | VAL DA | VAL RMSE | VAL MAE |
| :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#1** | **1D** | Fold 1/5 | `2015-06-10 → 2016-07-07` | ✅ Hoàn thành | `-0.1253` | `51.47%` | `0.0113` | `0.0083` |
| **#2** | **1D** | Fold 2/5 | `2016-07-08 → 2017-08-04` | ✅ Hoàn thành | `-0.1830` | `50.00%` | `0.0054` | `0.0040` |
| **#3** | **1D** | Fold 3/5 | `2017-08-07 → 2018-09-04` | ✅ Hoàn thành | `0.0112` | `49.26%` | `0.0077` | `0.0053` |
| **#4** | **1D** | Fold 4/5 | `2018-09-05 → 2019-10-03` | ✅ Hoàn thành | `-0.0674` | `50.37%` | `0.0106` | `0.0074` |

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
| 12 | **`RNN`** | Deep Learning | `4/25` | 🔄 Đang chạy (4/25) | 1.0s | `0.25s/fold` | ~5.2s (còn 21 folds) |
| 13 | **`LSTM`** | Deep Learning | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 14 | **`Transformer`** | Deep Learning | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |

---

## ⚡ 4. THỨ TỰ & LỆNH ĐIỀU PHỐI MÔ HÌNH TIẾP THEO

Chạy toàn bộ mô hình còn lại tự động trong 1 lệnh duy nhất:
```powershell
python run_remaining.py
```

---
*Bảng điều khiển được cập nhật **hoàn toàn tự động** sau mỗi Fold bằng hook nội bộ (Atomic File Swap).*
