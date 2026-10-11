# 📊 BẢNG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN (REAL-TIME DASHBOARD)
**Mô hình:** Sẵn sàng chạy Lan01 | **Dataset:** S&P 500 (Walk-Forward 5 Folds) | **Ghi nhận:** 2026-10-11 18:36:05  
💡 *Mẹo VS Code: Bấm `Ctrl + Shift + V` để mở giao diện xem trước (Markdown Preview) — Tự động cập nhật sau mỗi Fold.*

---

## 📌 1. THÔNG SỐ TIẾN ĐỘ & DỰ BÁO HOÀN THÀNH

| THÔNG SỐ TIẾN ĐỘ | GIÁ TRỊ | KẾT QUẢ & DỰ BÁO | GIÁ TRỊ |
| :--- | :--- | :--- | :--- |
| **Mô hình huấn luyện** | **`Sẵn sàng chạy Lan01`** | **Fold vừa hoàn tất** | Fold -/5 (Horizon -) |
| **Bộ dữ liệu** | S&P 500 Equity Returns | **OOS R² Fold vừa xong** | `-` |
| **Tiến trình Mô hình** | **`0/14 Models (0.0%)`** | **Directional Accuracy (DA)** | `-` |
| **Tiến trình Folds tổng** | **`0/350 Folds (0.0%)`** | **RMSE Fold vừa xong** | `-` |
| **Kỷ lục OOS R² cao nhất** | `-` | **Loss / MAE Fold vừa xong** | `-` |
| **Mô hình giữ Kỷ lục** | - | **Best Horizon của Model** | Horizon - |
| **Thời gian đã chạy (chính nó)** | **`-`** | **Tốc độ chạy (chính nó)** | **`-`** |
| **Tiến độ Nhóm Linear & Stats** | `⏳ Đang chờ (0/7 Models)` | **Trạng thái Nhóm 1** | `Chờ khởi động train` |
| **Dự kiến XONG mô hình này** | *Chờ khởi động Fold 1* | **Thời gian còn lại (chính nó)** | *Chờ khởi động Fold 1* |
| **Trạng thái mô hình** | ⏳ Sẵn sàng chạy Lan01 — Toàn bộ kết quả cũ đã được dọn sạch về 0/350 Folds | **Chế độ cập nhật** | 🔄 TỰ ĐỘNG SAU MỖI FOLD (Zero manual action) |

---

## 🔍 2. CHI TIẾT 5 BƯỚC FOLD GẦN NHẤT CỦA MÔ HÌNH HIỆN TẠI (`Sẵn sàng chạy Lan01`)

> 💡 **Quy luật huấn luyện Time-Series:** Mô hình chạy tuần tự theo Horizon: **`1D` (Fold 1→2→3→4→5)** rồi mới chuyển sang **`5D` (Fold 1→2→3→4→5)**, kế tiếp là `20D`, `60D`, `120D`.  
> Bảng dưới đây trích xuất **5 bước vừa chạy xong gần nhất** theo đúng dòng thời gian:

| BƯỚC | HORIZON | FOLD | GIAI ĐOẠN VALIDATION | TRẠNG THÁI | VAL OOS R² | VAL DA | VAL RMSE | VAL MAE |
| :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: |

---

## 📋 3. TIẾN ĐỘ TỔNG HỢP 14 MÔ HÌNH & DỰ BÁO HOÀN THÀNH

> ⚠️ **Nguyên tắc vàng:** Thời gian ước lượng (ETA) **bắt buộc tính dựa trên tốc độ và thời gian quá khứ của CHÍNH model đó**, tuyệt đối không suy đoán chéo giữa các model khác nhau.

| # | Mô hình | Nhóm thuật toán | Tiến độ | Trạng thái | Thời gian thực tế của chính nó | Tốc độ 1 fold | Dự báo hoàn thành (ETA của chính model) |
| :-: | :--- | :--- | :-: | :-: | :---: | :---: | :--- |
| 1 | **`Historical_Mean`** | Benchmark | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 2 | **`Ridge`** | Linear Regression | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 3 | **`Logistic_Regression`** | Classification | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 4 | **`Elastic_Net`** | Linear Regression | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 5 | **`ARIMA`** | Statistical TS | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 6 | **`ARIMAX`** | Statistical TS | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 7 | **`SARIMA`** | Statistical TS | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 8 | **`LightGBM`** | Tree Ensemble | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 9 | **`SVR`** | Kernel Method | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 10 | **`Random_Forest`** | Tree Ensemble | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 11 | **`XGBoost`** | Tree Ensemble | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
| 12 | **`RNN`** | Deep Learning | `0/25` | ⏳ Chờ chạy | Chưa có lượt chạy | `-` | *Chờ fold 1 của chính model này (Không ước lượng từ model khác)* |
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
