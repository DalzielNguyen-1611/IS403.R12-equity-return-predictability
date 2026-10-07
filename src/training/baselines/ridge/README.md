# Ridge Regression

## 1. Phân loại & Vai trò
- **Nhóm:** `Classical / Statistical`
- **Mô hình:** Hồi quy tuyến tính có điều chuẩn L2 (Tikhonov Regularization).
- **Mục tiêu:** Ngăn chặn hiện tượng đa cộng tuyến (multicollinearity) giữa 16 đặc trưng tài chính.

## 2. Công thức toán học
$$\min_w \frac{1}{2N} \|Xw - y\|_2^2 + \frac{\alpha}{2} \|w\|_2^2$$

## 3. Quy chuẩn Data Leakage
- `StandardScaler` được fit độc quyền trên `X_train` của từng Fold và transform cho cả `X_train`, `X_val`.
- Tham số điều chuẩn $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$ được chọn tự động bằng `TimeSeriesSplit` trên dữ liệu Train.

## 4. Cách chạy độc lập
```bash
python src/training/baselines/classical_statistical/ridge/ridge.py
```
