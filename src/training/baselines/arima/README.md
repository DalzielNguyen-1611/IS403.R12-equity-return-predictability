# Univariate ARIMA Model

## 1. Phân loại & Vai trò
- **Nhóm:** `Classical / Statistical`
- **Mô hình:** Tự hồi quy tích hợp trung bình trượt đơn biến (Autoregressive Integrated Moving Average).
- **Mục tiêu:** Mô hình hóa cấu trúc tương quan chuỗi thời gian nội tại của chuỗi lợi suất.

## 2. Lựa chọn bậc mô hình (Order Selection)
- Tìm kiếm tự động giữa các cấu hình $ARIMA(1,0,0)$, $ARIMA(0,0,1)$, $ARIMA(1,0,1)$ dựa trên tiêu chuẩn thông tin Akaike (AIC) **hoàn toàn trên dữ liệu Train**.
- Ứng dụng phương pháp Kalman filter cập nhật trạng thái từng bước không rò rỉ dữ liệu tương lai.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/classical_statistical/arima/arima.py
```
