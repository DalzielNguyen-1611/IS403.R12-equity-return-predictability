# Seasonal ARIMA (SARIMA)

## 1. Phân loại & Vai trò
- **Nhóm:** `Classical / Statistical`
- **Mô hình:** ARIMA theo mùa định kỳ (Seasonal ARIMA).
- **Chu kỳ mùa:** $s = 5$ (đại diện cho tuần giao dịch 5 ngày từ Thứ Hai đến Thứ Sáu).

## 2. Cơ sở phương pháp luận & Hạn chế thực nghiệm
- Dưới giả thuyết thị trường hiệu quả (EMH), lợi suất cổ phiếu hàng ngày có tính mùa vụ tuyến tính cực kỳ yếu hoặc tiệm cận 0.
- SARIMA được đưa vào nhằm mục đích thực nghiệm kiểm chứng xem hiệu ứng lịch tuần có tồn tại hay không.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/classical_statistical/sarima/sarima.py
```
