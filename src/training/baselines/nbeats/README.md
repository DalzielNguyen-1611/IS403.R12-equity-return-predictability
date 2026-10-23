# N-BEATS (Neural Basis Expansion Analysis)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Basis Expansion / MLP)`
- **Mô hình:** Neural Basis Expansion Analysis for Time Series Forecasting (Oreshkin et al., ICLR 2020).
- **Mục tiêu:** Dự báo chuỗi thời gian dựa trên kiến trúc Doubly Residual Stacking. Mỗi khối dự báo thành phần backcast (để trừ vào đầu vào) và thành phần forecast (tổng hợp dự báo cuối cùng), loại bỏ hiện tượng overfitting và không cần xử lý tuần tự.

## 2. Kiến trúc & Siêu tham số
- **Cơ chế Doubly Residual:** Tách bạch phần tín hiệu đã học qua backcast residual $x_{l+1} = x_l - \hat{x}_l$.
- **Basis Expansion:** Các lớp Fully-Connected 4 tầng ánh xạ hệ số đa thức/Fourier sang không gian dự báo.
- **Siêu tham số cơ sở:**
  - `num_stacks`: 3
  - `num_blocks`: 2
  - `hidden_dim`: 128
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/nbeats/nbeats_model.py
```
