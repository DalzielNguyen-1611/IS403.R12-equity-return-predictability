# Elastic Net Regression

## 1. Phân loại & Vai trò
- **Nhóm:** `Classical / Statistical`
- **Mô hình:** Hồi quy kết hợp điều chuẩn L1 (Lasso) và L2 (Ridge).
- **Mục tiêu:** Vừa chọn lọc đặc trưng (sparsity từ L1) vừa xử lý tương quan mạnh giữa các nhóm biến (grouping effect từ L2).

## 2. Công thức toán học
$$\min_w \frac{1}{2N} \|Xw - y\|_2^2 + \alpha \cdot \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|_2^2$$
Trong đó $\rho$ là tỷ lệ `l1_ratio`.

## 3. Không gian siêu tham số
- `alphas`: `[0.01, 0.1, 1.0, 10.0]`
- `l1_ratio`: `[0.1, 0.5, 0.7, 0.9]`
- Lựa chọn tham số bằng `TimeSeriesSplit` trên tập Train của từng Fold.

## 4. Cách chạy độc lập
```bash
python src/training/baselines/classical_statistical/elastic_net/elastic_net.py
```
